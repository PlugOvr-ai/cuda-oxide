/*!
 * ONNX model loading, protobuf decoding, and graph analysis.
 *
 * Generated prost types are in the `onnx` module (via include! in main.rs).
 */

use anyhow::{Result, anyhow};
use std::collections::{HashMap, HashSet, VecDeque};

// Re-export the prost-generated ONNX types used across this crate.
pub use crate::proto::onnx::{
    AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, attribute_proto::AttributeType,
    tensor_proto::DataType,
};

// ============================================================================
// Model loading
// ============================================================================

/// Decode an ONNX `.onnx` file from disk into a `ModelProto`.
pub fn load_model(path: &str) -> Result<ModelProto> {
    use prost::Message;
    let bytes = std::fs::read(path).map_err(|e| anyhow!("Cannot read model '{}': {}", path, e))?;
    ModelProto::decode(bytes.as_slice())
        .map_err(|e| anyhow!("Proto decode error for '{}': {}", path, e))
}

// ============================================================================
// Initializer (weight) extraction
// ============================================================================

/// Extract all initializer tensors from the graph as flat Vec<f32> + shape.
///
/// Handles both `float_data` (already f32 values) and `raw_data` (little-endian bytes).
pub fn load_initializers(graph: &GraphProto) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let mut map = HashMap::new();
    for t in &graph.initializer {
        let name = t.name.clone();
        let shape: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
        let data = tensor_to_f32(t)?;
        map.insert(name, (data, shape));
    }
    Ok(map)
}

/// Convert a `TensorProto` to a flat `Vec<f32>`.
///
/// Supported dtypes:
///   1 = FLOAT   — stored as-is or decoded from raw_data (4 bytes LE each)
///   7 = INT64   — cast element-wise to f32 (shape tensors; small integers)
///  11 = DOUBLE  — truncated to f32 from raw_data (8 bytes LE each)
pub fn tensor_to_f32(t: &TensorProto) -> Result<Vec<f32>> {
    let dtype = t.data_type;

    // --- FLOAT (1) ---
    if dtype == DataType::Float as i32 {
        if !t.float_data.is_empty() {
            return Ok(t.float_data.clone());
        }
        if !t.raw_data.is_empty() {
            let values: Vec<f32> = t
                .raw_data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            return Ok(values);
        }
        let numel: usize = t.dims.iter().map(|&d| d as usize).product();
        return Ok(vec![0.0f32; numel]);
    }

    // --- INT64 (7) — used for Reshape shape, Gather indices, etc. ---
    if dtype == DataType::Int64 as i32 {
        if !t.int64_data.is_empty() {
            return Ok(t.int64_data.iter().map(|&v| v as f32).collect());
        }
        if !t.raw_data.is_empty() {
            let values: Vec<f32> = t
                .raw_data
                .chunks_exact(8)
                .map(|b| {
                    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect();
            return Ok(values);
        }
        let numel: usize = t.dims.iter().map(|&d| d as usize).product();
        return Ok(vec![0.0f32; numel]);
    }

    // --- DOUBLE (11) ---
    if dtype == DataType::Double as i32 {
        if !t.double_data.is_empty() {
            return Ok(t.double_data.iter().map(|&v| v as f32).collect());
        }
        if !t.raw_data.is_empty() {
            let values: Vec<f32> = t
                .raw_data
                .chunks_exact(8)
                .map(|b| {
                    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect();
            return Ok(values);
        }
        let numel: usize = t.dims.iter().map(|&d| d as usize).product();
        return Ok(vec![0.0f32; numel]);
    }

    // --- INT32 (6) ---
    if dtype == DataType::Int32 as i32 {
        if !t.int32_data.is_empty() {
            return Ok(t.int32_data.iter().map(|&v| v as f32).collect());
        }
        if !t.raw_data.is_empty() {
            let values: Vec<f32> = t
                .raw_data
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
                .collect();
            return Ok(values);
        }
        let numel: usize = t.dims.iter().map(|&d| d as usize).product();
        return Ok(vec![0.0f32; numel]);
    }

    // --- BOOL (9) — 1 byte per element; e.g. GPT-2 causal mask → 0.0/1.0 ---
    if dtype == DataType::Bool as i32 {
        if !t.int32_data.is_empty() {
            return Ok(t
                .int32_data
                .iter()
                .map(|&v| if v != 0 { 1.0 } else { 0.0 })
                .collect());
        }
        if !t.raw_data.is_empty() {
            return Ok(t
                .raw_data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect());
        }
        let numel: usize = t.dims.iter().map(|&d| d as usize).product();
        return Ok(vec![0.0f32; numel]);
    }

    Err(anyhow!(
        "Unsupported tensor dtype {} for '{}'; supported: FLOAT(1), BOOL(9), INT32(6), INT64(7), DOUBLE(11)",
        dtype,
        t.name
    ))
}

// ============================================================================
// Topological sort (Kahn's algorithm)
// ============================================================================

/// Return nodes in a valid execution order for the given graph.
///
/// Nodes with no inputs that are yet to be produced by other nodes are
/// eligible first; we output them in the order they become ready.
pub fn topological_sort(graph: &GraphProto) -> Result<Vec<NodeProto>> {
    // Map from output tensor name → node index that produces it
    let mut producer: HashMap<&str, usize> = HashMap::new();
    for (i, node) in graph.node.iter().enumerate() {
        for out in &node.output {
            if !out.is_empty() {
                producer.insert(out.as_str(), i);
            }
        }
    }

    // Set of names available from the start (graph inputs + initializers)
    let mut available: HashSet<&str> = HashSet::new();
    for inp in &graph.input {
        available.insert(inp.name.as_str());
    }
    for init in &graph.initializer {
        available.insert(init.name.as_str());
    }

    // in-degree: number of *not-yet-available* inputs for each node
    let n = graph.node.len();
    let mut in_degree: Vec<usize> = vec![0; n];
    // consumers: producer node i → list of node indices that consume its outputs
    let mut consumers: Vec<Vec<usize>> = vec![vec![]; n];

    for (j, node) in graph.node.iter().enumerate() {
        for inp in &node.input {
            if inp.is_empty() || available.contains(inp.as_str()) {
                continue;
            }
            if let Some(&prod_idx) = producer.get(inp.as_str()) {
                in_degree[j] += 1;
                consumers[prod_idx].push(j);
            }
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut sorted = Vec::with_capacity(n);
    while let Some(i) = queue.pop_front() {
        sorted.push(graph.node[i].clone());
        for &j in &consumers[i] {
            in_degree[j] -= 1;
            if in_degree[j] == 0 {
                queue.push_back(j);
            }
        }
    }

    if sorted.len() != n {
        return Err(anyhow!("Graph has a cycle — cannot topologically sort"));
    }
    Ok(sorted)
}

// ============================================================================
// Attribute helpers
// ============================================================================

pub fn attr_f(node: &NodeProto, name: &str, default: f32) -> f32 {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.f)
        .unwrap_or(default)
}

pub fn attr_i(node: &NodeProto, name: &str, default: i64) -> i64 {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.i)
        .unwrap_or(default)
}

pub fn attr_ints(node: &NodeProto, name: &str) -> Vec<i64> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
        .unwrap_or_default()
}

pub fn attr_string(node: &NodeProto, name: &str) -> String {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .map(|a| String::from_utf8_lossy(&a.s).into_owned())
        .unwrap_or_default()
}

// ============================================================================
// Shape utilities
// ============================================================================

/// Compute the output shape for a 2D convolution.
pub fn conv2d_output_shape(
    in_h: usize,
    in_w: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dil_h: usize,
    dil_w: usize,
) -> (usize, usize) {
    let eff_kh = dil_h * (kh - 1) + 1;
    let eff_kw = dil_w * (kw - 1) + 1;
    let out_h = (in_h + 2 * pad_h).saturating_sub(eff_kh) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w).saturating_sub(eff_kw) / stride_w + 1;
    (out_h, out_w)
}

/// Compute output shape for MaxPool2D.
pub fn maxpool_output_shape(
    in_h: usize,
    in_w: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> (usize, usize) {
    let out_h = (in_h + 2 * pad_h).saturating_sub(kh) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w).saturating_sub(kw) / stride_w + 1;
    (out_h, out_w)
}

/// Extract `pads` attribute as `[pad_h, pad_w]` (ONNX uses [top,left,bottom,right]).
pub fn parse_pads(node: &NodeProto) -> (usize, usize) {
    let pads = attr_ints(node, "pads");
    if pads.len() >= 2 {
        (pads[0] as usize, pads[1] as usize)
    } else {
        (0, 0)
    }
}

/// Extract `strides` attribute as `[stride_h, stride_w]`.
pub fn parse_strides(node: &NodeProto) -> (usize, usize) {
    let s = attr_ints(node, "strides");
    if s.len() >= 2 {
        (s[0] as usize, s[1] as usize)
    } else {
        (1, 1)
    }
}

/// Extract `dilations` attribute.
pub fn parse_dilations(node: &NodeProto) -> (usize, usize) {
    let d = attr_ints(node, "dilations");
    if d.len() >= 2 {
        (d[0] as usize, d[1] as usize)
    } else {
        (1, 1)
    }
}

/// Extract `kernel_shape` attribute as `[kh, kw]`.
pub fn parse_kernel_shape(node: &NodeProto) -> (usize, usize) {
    let ks = attr_ints(node, "kernel_shape");
    if ks.len() >= 2 {
        (ks[0] as usize, ks[1] as usize)
    } else {
        (1, 1)
    }
}
