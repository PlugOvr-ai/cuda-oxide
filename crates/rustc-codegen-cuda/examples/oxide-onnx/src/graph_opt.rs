/*!
 * Load-time graph rewrites.
 *
 * These run once, on the host, between decoding the ONNX graph and uploading
 * weights to the device. They are pure graph algebra: every rewrite here is
 * exact in infinite precision, so the GPU results stay comparable to the
 * reference runtimes (the correctness gates in `main.rs` are what proves it).
 *
 * The wins are large because inference-time BatchNorm and elementwise
 * activations are bandwidth-bound: each one is a full round trip of the
 * activation tensor through DRAM. Removing the node removes the round trip.
 */

use std::collections::{HashMap, HashSet};

use crate::model::{AttributeProto, AttributeType, NodeProto, attr_f};

/// Fused-activation codes written into the private `oxide_act` attribute and
/// read back by the executor. Kept in sync with `kernels::gpu` ACT_* constants.
pub const ACT_NONE: i64 = 0;
pub const ACT_RELU: i64 = 1;
pub const ACT_CLIP: i64 = 2;

#[derive(Debug, Default, Clone, Copy)]
pub struct OptStats {
    /// BatchNorm nodes folded into the weights of the Conv that feeds them.
    pub bn_folded: usize,
    /// Relu/Clip nodes absorbed into the producing Conv's epilogue.
    pub act_fused: usize,
    /// Relu nodes absorbed into a standalone BatchNorm's epilogue.
    pub bn_act_fused: usize,
    /// BatchNorm nodes whose 5 inputs were collapsed to (scale, shift).
    pub bn_affine: usize,
    /// Residual `Add` nodes folded into the producing Conv's epilogue.
    pub residual_fused: usize,
    /// Bias `Add` nodes folded into the producing MatMul's epilogue.
    pub matmul_bias_fused: usize,
}

impl OptStats {
    /// Nodes removed from the graph. `bn_affine` rewrites a node in place
    /// rather than removing it, so it is not counted here.
    pub fn total(&self) -> usize {
        self.bn_folded + self.act_fused + self.bn_act_fused + self.residual_fused
    }
}

type Weights = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// Run every rewrite, in dependency order, until the graph is stable.
///
/// BatchNorm folding must run before activation fusion: folding rewires the
/// Conv to produce the BatchNorm's output, which is what makes the following
/// Relu directly fusable into the Conv.
pub fn optimize(
    nodes: &mut Vec<NodeProto>,
    weights: &mut Weights,
    graph_outputs: &HashSet<String>,
) -> OptStats {
    let mut stats = OptStats::default();
    stats.bn_folded = fold_conv_batchnorm(nodes, weights, graph_outputs);
    stats.act_fused = fuse_activation_into(nodes, graph_outputs, "Conv");
    stats.bn_act_fused = fuse_activation_into(nodes, graph_outputs, "BatchNormalization");
    stats.bn_affine = precompute_batchnorm_affine(nodes, weights);
    stats.residual_fused = fuse_residual_into_conv(nodes, graph_outputs);
    stats.matmul_bias_fused = fuse_bias_into_matmul(nodes, weights, graph_outputs);
    stats
}

/// Fold a bias `Add` into the `MatMul` that feeds it.
///
/// A transformer's linear layers arrive as `MatMul` followed by `Add` of a
/// 1-D initializer — 48 of them in BERT-base. The `Add` broadcasts a `[768]`
/// vector across `[1, 128, 768]`, so it takes the general N-D broadcast path:
/// four metadata buffers, a per-element loop over the rank, and a full extra
/// round trip of the activation through DRAM. It measured 1.12 ms, 18% of the
/// model, for arithmetic that is free.
///
/// The GEMM's split-K reduction already applies a bias, so the vector is
/// appended as the `MatMul`'s third input and flagged with `oxide_bias`.
/// Applies only when the bias is a load-time constant whose length matches the
/// output's last dimension, the `MatMul` feeds nothing but the `Add`, and the
/// `MatMul` output is not itself a graph output.
fn fuse_bias_into_matmul(
    nodes: &mut Vec<NodeProto>,
    weights: &Weights,
    graph_outputs: &HashSet<String>,
) -> usize {
    struct Fuse {
        mm: usize,
        add: usize,
        bias: String,
    }
    let mut fuses: Vec<Fuse> = Vec::new();
    {
        let counts = consumer_counts(nodes);
        let producer = producers(nodes);
        let mut claimed: HashSet<usize> = HashSet::new();

        for (add_idx, add) in nodes.iter().enumerate() {
            if add.op_type != "Add" || add.input.len() != 2 {
                continue;
            }
            for (a, b) in [(0usize, 1usize), (1, 0)] {
                let from = add.input[a].as_str();
                let bias = add.input[b].as_str();
                let Some(&mm_idx) = producer.get(from) else {
                    continue;
                };
                let mm = &nodes[mm_idx];
                if mm.op_type != "MatMul" || mm.input.len() != 2 || mm.output.len() != 1 {
                    continue;
                }
                if claimed.contains(&mm_idx) {
                    continue;
                }
                if counts.get(from).copied().unwrap_or(0) != 1 || graph_outputs.contains(from) {
                    continue;
                }
                // The bias must be a load-time constant vector as long as the
                // GEMM's N, which is the weight's trailing dimension.
                let (Some(bv), Some(wv)) = (weights.get(bias), weights.get(mm.input[1].as_str()))
                else {
                    continue;
                };
                if bv.1.len() != 1 || wv.1.len() != 2 || bv.1[0] != wv.1[1] {
                    continue;
                }
                claimed.insert(mm_idx);
                fuses.push(Fuse {
                    mm: mm_idx,
                    add: add_idx,
                    bias: bias.to_string(),
                });
                break;
            }
        }
    }

    let mut dead: HashSet<usize> = HashSet::new();
    for f in &fuses {
        let add_out = nodes[f.add].output[0].clone();
        let mm = &mut nodes[f.mm];
        mm.input.push(f.bias.clone());
        set_attr_i(mm, "oxide_bias", 1);
        mm.output[0] = add_out;
        dead.insert(f.add);
    }
    if !dead.is_empty() {
        let mut idx = 0;
        nodes.retain(|_| {
            let keep = !dead.contains(&idx);
            idx += 1;
            keep
        });
    }
    fuses.len()
}

/// Fold a residual `Add` into the convolution that feeds it.
///
/// A ResNet block ends `conv(...) + shortcut`, which costs a whole kernel and
/// a full round trip of the activation through DRAM — 44 MB and 16 launches on
/// ResNet50. The convolution's epilogue already reads the accumulator and
/// writes the output, so it can add the shortcut on the way past. TensorRT
/// fuses the same pattern; its kernel names say so.
///
/// The shortcut tensor is appended as the Conv's fourth input and flagged with
/// `oxide_residual`, with an empty bias slot inserted when the Conv has none so
/// the index is stable. Applies only when the Conv's output feeds the Add and
/// nothing else, and is not a graph output.
fn fuse_residual_into_conv(nodes: &mut Vec<NodeProto>, graph_outputs: &HashSet<String>) -> usize {
    struct Fuse {
        conv: usize,
        add: usize,
        residual: String,
    }
    let mut fuses: Vec<Fuse> = Vec::new();
    {
        let counts = consumer_counts(nodes);
        let producer = producers(nodes);
        let mut claimed: HashSet<usize> = HashSet::new();

        for (add_idx, add) in nodes.iter().enumerate() {
            if add.op_type != "Add" || add.input.len() != 2 {
                continue;
            }
            // Exactly one side must come from a Conv we can absorb into.
            for (a, b) in [(0usize, 1usize), (1, 0)] {
                let from = add.input[a].as_str();
                let other = add.input[b].as_str();
                let Some(&conv_idx) = producer.get(from) else {
                    continue;
                };
                let conv = &nodes[conv_idx];
                if conv.op_type != "Conv" || conv.output.len() != 1 {
                    continue;
                }
                if claimed.contains(&conv_idx) {
                    continue;
                }
                if counts.get(from).copied().unwrap_or(0) != 1 || graph_outputs.contains(from) {
                    continue;
                }
                // The residual must already exist when the Conv runs.
                let Some(&res_idx) = producer.get(other) else {
                    continue;
                };
                if res_idx > conv_idx {
                    continue;
                }
                claimed.insert(conv_idx);
                fuses.push(Fuse {
                    conv: conv_idx,
                    add: add_idx,
                    residual: other.to_string(),
                });
                break;
            }
        }
    }

    let mut dead: HashSet<usize> = HashSet::new();
    for f in &fuses {
        let add_out = nodes[f.add].output[0].clone();
        let conv = &mut nodes[f.conv];
        while conv.input.len() < 3 {
            conv.input.push(String::new());
        }
        conv.input.truncate(3);
        conv.input.push(f.residual.clone());
        set_attr_i(conv, "oxide_residual", 1);
        conv.output[0] = add_out;
        dead.insert(f.add);
    }
    if !dead.is_empty() {
        let mut idx = 0;
        nodes.retain(|_| {
            let keep = !dead.contains(&idx);
            idx += 1;
            keep
        });
    }
    fuses.len()
}

/// Collapse a surviving BatchNormalization's five inputs into the two arrays
/// the kernel actually needs: `y = x * scale[c] + shift[c]`.
///
/// Rewrites `[x, gamma, beta, mean, var]` to `[x, scale, shift]`, which halves
/// the parameter traffic and — more importantly — moves the reciprocal square
/// root from once per element to once per channel, on the host.
///
/// Nodes whose parameters are not load-time constants keep their five inputs
/// and the executor falls back to computing the normalization on device.
fn precompute_batchnorm_affine(nodes: &mut [NodeProto], weights: &mut Weights) -> usize {
    let mut rewritten = 0;
    for node in nodes.iter_mut() {
        if node.op_type != "BatchNormalization" || node.input.len() < 5 {
            continue;
        }
        let params: Option<Vec<Vec<f32>>> = node.input[1..5]
            .iter()
            .map(|n| weights.get(n.as_str()).map(|(data, _)| data.clone()))
            .collect();
        let Some(params) = params else { continue };
        let (gamma, beta, mean, var) = (&params[0], &params[1], &params[2], &params[3]);
        let channels = gamma.len();
        if [beta, mean, var].iter().any(|p| p.len() != channels) {
            continue;
        }

        let eps = attr_f(node, "epsilon", 1e-5);
        let scale: Vec<f32> = (0..channels)
            .map(|c| gamma[c] / (var[c] + eps).sqrt())
            .collect();
        let shift: Vec<f32> = (0..channels)
            .map(|c| beta[c] - mean[c] * scale[c])
            .collect();

        let base = &node.output[0];
        let scale_name = format!("{base}_oxide_bn_scale");
        let shift_name = format!("{base}_oxide_bn_shift");
        weights.insert(scale_name.clone(), (scale, vec![channels]));
        weights.insert(shift_name.clone(), (shift, vec![channels]));

        node.input.truncate(1);
        node.input.push(scale_name);
        node.input.push(shift_name);
        rewritten += 1;
    }
    rewritten
}

/// Count how many node inputs reference each tensor name.
fn consumer_counts(nodes: &[NodeProto]) -> HashMap<&str, usize> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for node in nodes {
        for input in &node.input {
            *counts.entry(input.as_str()).or_insert(0) += 1;
        }
    }
    counts
}

/// Map each tensor name to the index of the node producing it.
fn producers(nodes: &[NodeProto]) -> HashMap<&str, usize> {
    let mut map = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        for output in &node.output {
            map.insert(output.as_str(), idx);
        }
    }
    map
}

/// Set an integer attribute, replacing any existing one with the same name.
fn set_attr_i(node: &mut NodeProto, name: &str, value: i64) {
    node.attribute.retain(|a| a.name != name);
    node.attribute.push(AttributeProto {
        name: name.to_string(),
        r#type: AttributeType::Int as i32,
        i: value,
        ..Default::default()
    });
}

/// Set a float attribute, replacing any existing one with the same name.
fn set_attr_f(node: &mut NodeProto, name: &str, value: f32) {
    node.attribute.retain(|a| a.name != name);
    node.attribute.push(AttributeProto {
        name: name.to_string(),
        r#type: AttributeType::Float as i32,
        f: value,
        ..Default::default()
    });
}

/// Fold inference-time BatchNormalization into the weights of the Conv that
/// produces its input.
///
/// At inference BatchNorm is a per-channel affine map:
///
/// ```text
///   scale[c] = gamma[c] / sqrt(var[c] + eps)
///   shift[c] = beta[c] - mean[c] * scale[c]
///   bn(y)[c] = y[c] * scale[c] + shift[c]
/// ```
///
/// and convolution is linear, so `bn(conv(x))` is itself a convolution:
///
/// ```text
///   W'[o, ...] = W[o, ...] * scale[o]
///   b'[o]      = b[o] * scale[o] + shift[o]      (b = 0 when Conv has no bias)
/// ```
///
/// This is exact — including with padding, because the fold happens *after*
/// the convolution, so padded positions are unaffected. (Folding a BatchNorm
/// into a *following* Conv would not be exact under zero padding, which is why
/// only this direction is implemented.)
///
/// Applies only when the Conv output feeds this BatchNorm and nothing else,
/// and is not itself a graph output — otherwise the pre-normalization value is
/// still observable and cannot be rewritten away.
fn fold_conv_batchnorm(
    nodes: &mut Vec<NodeProto>,
    weights: &mut Weights,
    graph_outputs: &HashSet<String>,
) -> usize {
    // Plan every fold against the original graph, then apply. This keeps the
    // borrow of `nodes` read-only while deciding, and means a rejected
    // candidate can never leave a half-rewritten node behind.
    struct Fold {
        conv: usize,
        bn: usize,
        scale: Vec<f32>,
        shift: Vec<f32>,
    }

    let mut folds: Vec<Fold> = Vec::new();
    {
        let counts = consumer_counts(nodes);
        let producer = producers(nodes);
        // A weight tensor shared by two Convs cannot be scaled in place.
        let mut claimed: HashSet<&str> = HashSet::new();

        for (bn_idx, bn) in nodes.iter().enumerate() {
            if bn.op_type != "BatchNormalization" || bn.input.len() < 5 {
                continue;
            }
            let x = bn.input[0].as_str();
            let Some(&conv_idx) = producer.get(x) else {
                continue;
            };
            let conv = &nodes[conv_idx];
            if conv.op_type != "Conv" || conv.output.len() != 1 {
                continue;
            }
            // The pre-normalization activation must not be observable anywhere
            // else in the graph.
            if counts.get(x).copied().unwrap_or(0) != 1 || graph_outputs.contains(x) {
                continue;
            }
            // BatchNorm's non-X inputs and the Conv weights must all be
            // constants known at load time.
            let w_name = conv.input[1].as_str();
            if !weights.contains_key(w_name) || claimed.contains(w_name) {
                continue;
            }
            if counts.get(w_name).copied().unwrap_or(0) != 1 {
                continue;
            }
            let params: Option<Vec<&Vec<f32>>> = bn.input[1..5]
                .iter()
                .map(|n| weights.get(n.as_str()).map(|(data, _)| data))
                .collect();
            let Some(params) = params else { continue };
            let (gamma, beta, mean, var) = (params[0], params[1], params[2], params[3]);

            let (_, w_shape) = &weights[w_name];
            if w_shape.is_empty() {
                continue;
            }
            let out_channels = w_shape[0];
            if [gamma, beta, mean, var]
                .iter()
                .any(|p| p.len() != out_channels)
            {
                continue;
            }
            // A Conv bias, if present, must also be a load-time constant that
            // no other node reads.
            if conv.input.len() > 2 && !conv.input[2].is_empty() {
                let b_name = conv.input[2].as_str();
                if !weights.contains_key(b_name)
                    || counts.get(b_name).copied().unwrap_or(0) != 1
                    || weights[b_name].0.len() != out_channels
                {
                    continue;
                }
            }

            let eps = attr_f(bn, "epsilon", 1e-5);
            let scale: Vec<f32> = (0..out_channels)
                .map(|c| gamma[c] / (var[c] + eps).sqrt())
                .collect();
            let shift: Vec<f32> = (0..out_channels)
                .map(|c| beta[c] - mean[c] * scale[c])
                .collect();

            claimed.insert(w_name);
            folds.push(Fold {
                conv: conv_idx,
                bn: bn_idx,
                scale,
                shift,
            });
        }
    }

    let mut dead: HashSet<usize> = HashSet::new();
    for fold in &folds {
        let (w_name, bias_name, bn_output) = {
            let conv = &nodes[fold.conv];
            let bias = conv
                .input
                .get(2)
                .filter(|n| !n.is_empty())
                .map(|n| n.to_string());
            (
                conv.input[1].clone(),
                bias,
                nodes[fold.bn].output[0].clone(),
            )
        };

        // W'[o, ...] = W[o, ...] * scale[o]
        let (w_data, w_shape) = weights.get_mut(&w_name).expect("checked above");
        let out_channels = w_shape[0];
        let per_channel = w_data.len() / out_channels;
        for (o, s) in fold.scale.iter().enumerate().take(out_channels) {
            for value in &mut w_data[o * per_channel..(o + 1) * per_channel] {
                *value *= s;
            }
        }

        // b'[o] = b[o] * scale[o] + shift[o], creating the bias if absent.
        match bias_name {
            Some(name) => {
                let (b_data, _) = weights.get_mut(&name).expect("checked above");
                for (o, b) in b_data.iter_mut().enumerate() {
                    *b = *b * fold.scale[o] + fold.shift[o];
                }
            }
            None => {
                let name = format!("{bn_output}_oxide_folded_bias");
                weights.insert(name.clone(), (fold.shift.clone(), vec![out_channels]));
                let conv = &mut nodes[fold.conv];
                while conv.input.len() < 2 {
                    conv.input.push(String::new());
                }
                conv.input.truncate(2);
                conv.input.push(name);
            }
        }

        // The Conv now produces what the BatchNorm used to.
        nodes[fold.conv].output[0] = bn_output;
        dead.insert(fold.bn);
    }

    if !dead.is_empty() {
        let mut idx = 0;
        nodes.retain(|_| {
            let keep = !dead.contains(&idx);
            idx += 1;
            keep
        });
    }
    folds.len()
}

/// Absorb a Relu/Clip that consumes `producer_op`'s output into that node's
/// epilogue, recorded as the private `oxide_act` attribute.
///
/// Saves a full read + write of the activation tensor per fused node: the
/// producing kernel already has the value in a register when it stores it.
///
/// Applies only when the producer's output feeds the activation and nothing
/// else, and is not a graph output.
fn fuse_activation_into(
    nodes: &mut Vec<NodeProto>,
    graph_outputs: &HashSet<String>,
    producer_op: &str,
) -> usize {
    struct Fuse {
        producer: usize,
        act_node: usize,
        kind: i64,
        lo: f32,
        hi: f32,
    }

    let mut fuses: Vec<Fuse> = Vec::new();
    {
        let counts = consumer_counts(nodes);
        let producer_of = producers(nodes);
        let mut claimed: HashSet<usize> = HashSet::new();

        for (act_idx, act) in nodes.iter().enumerate() {
            let kind = match act.op_type.as_str() {
                "Relu" => ACT_RELU,
                "Clip" => ACT_CLIP,
                _ => continue,
            };
            if act.input.is_empty() {
                continue;
            }
            let x = act.input[0].as_str();
            let Some(&prod_idx) = producer_of.get(x) else {
                continue;
            };
            if nodes[prod_idx].op_type != producer_op || claimed.contains(&prod_idx) {
                continue;
            }
            if counts.get(x).copied().unwrap_or(0) != 1 || graph_outputs.contains(x) {
                continue;
            }
            // Only fuse a Clip whose bounds are load-time constants. Opset <= 10
            // carries them as attributes; opset 11+ as optional inputs, which
            // this pass does not resolve, so those stay as separate nodes.
            let (lo, hi) = if kind == ACT_CLIP {
                if act.input.len() > 1 {
                    continue;
                }
                (
                    attr_f(act, "min", f32::NEG_INFINITY),
                    attr_f(act, "max", f32::INFINITY),
                )
            } else {
                (0.0, f32::INFINITY)
            };

            claimed.insert(prod_idx);
            fuses.push(Fuse {
                producer: prod_idx,
                act_node: act_idx,
                kind,
                lo,
                hi,
            });
        }
    }

    let mut dead: HashSet<usize> = HashSet::new();
    for fuse in &fuses {
        let act_output = nodes[fuse.act_node].output[0].clone();
        let node = &mut nodes[fuse.producer];
        set_attr_i(node, "oxide_act", fuse.kind);
        if fuse.kind == ACT_CLIP {
            set_attr_f(node, "oxide_act_lo", fuse.lo);
            set_attr_f(node, "oxide_act_hi", fuse.hi);
        }
        node.output[0] = act_output;
        dead.insert(fuse.act_node);
    }

    if !dead.is_empty() {
        let mut idx = 0;
        nodes.retain(|_| {
            let keep = !dead.contains(&idx);
            idx += 1;
            keep
        });
    }
    fuses.len()
}
