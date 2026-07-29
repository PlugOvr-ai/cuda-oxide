/*!
 * OnnxExecutor: loads an ONNX model and dispatches each graph node to the
 * corresponding CUDA kernel(s).
 *
 * Conv2D is implemented via im2col + GEMM.
 * All other ops map to a single kernel or a trivial shape manipulation.
 */

#![allow(clippy::too_many_arguments)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, LaunchConfig};

use std::mem::ManuallyDrop;

use crate::graph_opt;
use crate::kernels::gpu;
use crate::model::{
    GraphProto, NodeProto, attr_f, attr_i, attr_ints, conv2d_output_shape, load_initializers,
    maxpool_output_shape, parse_dilations, parse_kernel_shape, parse_pads, parse_strides,
    topological_sort,
};
use crate::tensor::TensorMap;

// ============================================================================
// OnnxExecutor
// ============================================================================

pub struct OnnxExecutor {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub module: gpu::LoadedModule,
    /// Pre-loaded weight buffers: name → (DeviceBuffer, shape)
    pub weights: HashMap<String, (DeviceBuffer<f32>, Vec<usize>)>,
    /// Host copies of small constant initializers (shape/axes/split tensors).
    /// Read by Reshape/Squeeze/Split to avoid a D2H stream sync per node.
    pub consts: HashMap<String, Vec<f32>>,
    /// Graph nodes in topological order
    pub nodes: Vec<NodeProto>,
    /// Graph-level input names
    pub input_names: Vec<String>,
    /// Graph-level output names
    pub output_names: Vec<String>,
    /// Device copies of small constant metadata arrays (shapes, strides,
    /// permutations), keyed by their contents. See [`Self::meta_buf`].
    meta_cache: RefCell<HashMap<u64, DeviceBuffer<f32>>>,
    /// Intermediate buffers recycled across inferences, keyed by element count.
    /// See [`Self::alloc_buf`].
    buf_pool: RefCell<HashMap<usize, Vec<DeviceBuffer<f32>>>>,
    /// Weight matrices pre-packed as f16 pairs for the tensor-core GEMM, keyed
    /// by (device pointer, rows, k). See [`Self::packed_weights`].
    f16_weights: RefCell<HashMap<(u64, usize, usize), DeviceBuffer<u32>>>,
    /// Same, for weights that sit in the B operand — packed [n][ceil(k/2)].
    /// See [`Self::packed_weights_cols`].
    f16_weights_b: RefCell<HashMap<(u64, usize, usize), DeviceBuffer<u32>>>,
}

impl OnnxExecutor {
    /// Create an executor from an already-parsed GraphProto.
    pub fn from_graph(graph: &GraphProto, ctx: Arc<CudaContext>) -> Result<Self> {
        let stream = ctx.default_stream();
        let module = gpu::load(&ctx).map_err(|e| anyhow!("Failed to load CUDA module: {:?}", e))?;

        let mut host_weights = load_initializers(graph)?;
        let mut nodes = topological_sort(graph)?;

        // Rewrite the graph before anything reaches the device: folding
        // BatchNorm into Conv weights and absorbing activations removes whole
        // kernels, each of which is a full DRAM round trip of its activation
        // tensor.
        let graph_output_names: std::collections::HashSet<String> =
            graph.output.iter().map(|vi| vi.name.clone()).collect();
        let opt = graph_opt::optimize(&mut nodes, &mut host_weights, &graph_output_names);
        if opt.total() > 0 && std::env::var("OXIDE_QUIET").is_err() {
            eprintln!(
                "  [graph-opt] folded {} BatchNorm, fused {} Conv activations, \
                 {} BatchNorm activations ({} nodes removed)",
                opt.bn_folded,
                opt.act_fused,
                opt.bn_act_fused,
                opt.total()
            );
        }

        let mut weights = HashMap::new();
        let mut consts = HashMap::new();
        for (name, (data, shape)) in host_weights {
            // Keep a host copy of small constants (shape/axes/split tensors)
            // so metadata ops never trigger a per-node D2H synchronize.
            if data.len() <= 64 {
                consts.insert(name.clone(), data.clone());
            }
            let buf = DeviceBuffer::from_host(&stream, &data)
                .map_err(|e| anyhow!("H2D weight '{}' failed: {:?}", name, e))?;
            weights.insert(name, (buf, shape));
        }

        let input_names = graph
            .input
            .iter()
            .filter(|vi| !weights.contains_key(&vi.name))
            .map(|vi| vi.name.clone())
            .collect();

        let output_names = graph.output.iter().map(|vi| vi.name.clone()).collect();

        Ok(Self {
            ctx,
            stream,
            module,
            weights,
            consts,
            nodes,
            input_names,
            output_names,
            meta_cache: RefCell::new(HashMap::new()),
            buf_pool: RefCell::new(HashMap::new()),
            f16_weights: RefCell::new(HashMap::new()),
            f16_weights_b: RefCell::new(HashMap::new()),
        })
    }

    /// Run inference on a map of input tensors.
    pub fn run(
        &self,
        inputs: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
        let t_setup = std::time::Instant::now();
        let mut tensors = TensorMap::new();

        for (name, (data, shape)) in inputs {
            // Stream-ordered allocation, not `from_host`: that allocates with
            // cuMemAlloc, whose cuMemFree when the TensorMap drops at end of
            // run synchronises the whole device — and the next inference's
            // allocation then queues behind that. Measured at 6 ms of BERT's
            // ~20 ms, for two tensors of 128 elements.
            let mut buf = self
                .alloc_buf(data.len())
                .map_err(|e| anyhow!("input alloc '{}': {}", name, e))?;
            buf.copy_from_host(&self.stream, data)
                .map_err(|e| anyhow!("H2D input '{}': {:?}", name, e))?;
            tensors.insert(name, buf, shape.clone());
        }

        let t_setup_ms = t_setup.elapsed().as_secs_f64() * 1e3;
        #[allow(unused_assignments)]
        let mut t_loop_ms = 0.0f64;
        let profile_mode = std::env::var("OXIDE_PROFILE").unwrap_or_default();
        let profile = !profile_mode.is_empty();
        // Three modes, because each answers a different question and the first
        // two are easy to misread:
        //   OXIDE_PROFILE=1      host wall time with a sync after every node.
        //                        Measures device time, but charges the sync's
        //                        pipeline drain to whichever node it follows,
        //                        which inflates cheap kernels enormously.
        //   OXIDE_PROFILE=host   no sync: pure host dispatch cost. Shows where
        //                        the host blocks, not what anything costs.
        //   OXIDE_PROFILE=event  CUDA events around each node's launches. The
        //                        pipeline is never drained, so this is the
        //                        node's real device time in situ — the only
        //                        one of the three to trust for attribution.
        let sync_each = profile_mode != "host" && profile_mode != "event";
        if profile_mode == "event" {
            use std::collections::BTreeMap;
            let flags = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
            let mut marks: Vec<(usize, CudaEvent, CudaEvent)> =
                Vec::with_capacity(self.nodes.len());
            for (idx, node) in self.nodes.iter().enumerate() {
                let start = self
                    .ctx
                    .new_event(flags)
                    .map_err(|e| anyhow!("event create: {:?}", e))?;
                let end = self
                    .ctx
                    .new_event(flags)
                    .map_err(|e| anyhow!("event create: {:?}", e))?;
                start
                    .record(&self.stream)
                    .map_err(|e| anyhow!("event record: {:?}", e))?;
                self.dispatch_node(node, &mut tensors)
                    .map_err(|e| anyhow!("op {} (inputs={:?}): {}", node.op_type, node.input, e))?;
                end.record(&self.stream)
                    .map_err(|e| anyhow!("event record: {:?}", e))?;
                marks.push((idx, start, end));
            }
            self.stream
                .synchronize()
                .map_err(|e| anyhow!("sync: {:?}", e))?;

            let mut acc: BTreeMap<String, (f64, u32)> = BTreeMap::new();
            for (idx, start, end) in &marks {
                let ms = start.elapsed_ms(end).unwrap_or(0.0) as f64;
                let e = acc
                    .entry(self.nodes[*idx].op_type.clone())
                    .or_insert((0.0, 0));
                e.0 += ms;
                e.1 += 1;
            }
            let mut rows: Vec<_> = acc.into_iter().collect();
            rows.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            let total: f64 = rows.iter().map(|r| r.1.0).sum();
            eprintln!("  ── OXIDE_PROFILE=event (device time per op-type, ms) ──");
            for (op, (ms, n)) in &rows {
                eprintln!(
                    "  {:>20}  {:>8.2} ms  ({:>3}×, {:>5.1}%)",
                    op,
                    ms,
                    n,
                    100.0 * ms / total
                );
            }
            eprintln!("  {:>20}  {:>8.2} ms  (device busy)", "TOTAL", total);
        } else if profile {
            use std::collections::BTreeMap;
            let mut acc: BTreeMap<String, (f64, u32)> = BTreeMap::new();
            for node in &self.nodes {
                let t0 = std::time::Instant::now();
                self.dispatch_node(node, &mut tensors)
                    .map_err(|e| anyhow!("op {} (inputs={:?}): {}", node.op_type, node.input, e))?;
                if sync_each {
                    self.stream.synchronize().ok();
                }
                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                let e = acc.entry(node.op_type.clone()).or_insert((0.0, 0));
                e.0 += dt;
                e.1 += 1;
            }
            let mut rows: Vec<_> = acc.into_iter().collect();
            rows.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            let total: f64 = rows.iter().map(|r| r.1.0).sum();
            eprintln!("  ── OXIDE_PROFILE (per op-type, ms) ──");
            for (op, (ms, n)) in &rows {
                eprintln!(
                    "  {:>20}  {:>8.2} ms  ({:>3}×, {:>5.1}%)",
                    op,
                    ms,
                    n,
                    100.0 * ms / total
                );
            }
            eprintln!(
                "  {:>20}  {:>8.2} ms  (sum incl. per-op sync)",
                "TOTAL", total
            );
            t_loop_ms = 0.0;
        } else {
            let t_loop = std::time::Instant::now();
            for node in &self.nodes {
                self.dispatch_node(node, &mut tensors)
                    .map_err(|e| anyhow!("op {} (inputs={:?}): {}", node.op_type, node.input, e))?;
            }
            t_loop_ms = t_loop.elapsed().as_secs_f64() * 1e3;
        }

        let mut outputs = HashMap::new();
        let t_out = std::time::Instant::now();
        for name in &self.output_names {
            // Resolves alias chains (e.g. a final Reshape) to the real buffer.
            if let Ok(eb) = Self::get_tensor(&tensors, &self.weights, name) {
                let data = eb
                    .buf()
                    .to_host_vec(&self.stream)
                    .map_err(|e| anyhow!("D2H output '{}': {:?}", name, e))?;
                outputs.insert(name.clone(), (data, eb.shape().clone()));
            }
        }
        if std::env::var("OXIDE_PHASES").is_ok() {
            eprintln!(
                "  [phases] h2d+setup={:.2}ms  node-loop(submit)={:.2}ms  drain+d2h={:.2}ms",
                t_setup_ms,
                t_loop_ms,
                t_out.elapsed().as_secs_f64() * 1e3
            );
        }
        // The outputs are already on the host, so every intermediate is dead;
        // hand them to the pool rather than enqueueing a free for each.
        self.recycle(tensors);
        Ok(outputs)
    }

    fn get_tensor<'t, 'w>(
        tensors: &'t TensorMap,
        weights: &'w HashMap<String, (DeviceBuffer<f32>, Vec<usize>)>,
        name: &str,
    ) -> Result<EitherBuf<'t, 'w>> {
        // Zero-copy view: resolve the alias chain to the owning buffer in
        // tensors.map, but report the alias's (reshaped) view shape.
        if let Some((owner, shape)) = tensors.resolve_alias(name) {
            if let Some((b, _)) = tensors.map.get(owner) {
                return Ok(EitherBuf::FromTensors(b, shape));
            }
        }
        if let Some((b, s)) = tensors.map.get(name) {
            return Ok(EitherBuf::FromTensors(b, s));
        }
        if let Some((b, s)) = weights.get(name) {
            return Ok(EitherBuf::FromWeights(b, s));
        }
        Err(anyhow!("Tensor '{}' not found", name))
    }

    /// Metadata-op result: alias the input buffer with a new shape (zero copy)
    /// when the input is an activation; fall back to a real D2D copy only when
    /// the input is a weight/initializer (rare; keeps EitherBuf lifetimes simple).
    fn alias_or_copy(
        &self,
        tensors: &mut TensorMap,
        out_name: &str,
        in_name: &str,
        new_shape: Vec<usize>,
    ) -> Result<()> {
        if tensors.map.contains_key(in_name) || tensors.alias.contains_key(in_name) {
            tensors.insert_alias(out_name, in_name, new_shape);
            return Ok(());
        }
        // Input is a weight: copy so the result lives in tensors.map.
        let (numel, src_ptr) = {
            let eb = Self::get_tensor(tensors, &self.weights, in_name)?;
            (eb.buf().len(), eb.buf().cu_deviceptr())
        };
        let out = self
            .alloc_buf(numel)
            .map_err(|e| anyhow!("alias_or_copy alloc: {}", e))?;
        self.dtod_copy(&out, src_ptr, numel * 4)?;
        tensors.insert(out_name, out, new_shape);
        Ok(())
    }

    fn dispatch_node(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        match node.op_type.as_str() {
            "Relu" => self.op_relu(node, tensors),
            "Clip" => self.op_clip(node, tensors),
            "Add" => self.op_add(node, tensors),
            "Mul" => self.op_mul(node, tensors),
            "Conv" => self.op_conv(node, tensors),
            "BatchNormalization" => self.op_batchnorm(node, tensors),
            "MaxPool" => self.op_maxpool(node, tensors),
            "GlobalAveragePool" => self.op_global_avg_pool(node, tensors),
            "Gemm" => self.op_gemm(node, tensors),
            "MatMul" => self.op_matmul(node, tensors),
            "Softmax" => self.op_softmax(node, tensors),
            "LayerNormalization" => self.op_layernorm(node, tensors),
            "Erf" => self.op_erf(node, tensors),
            "Tanh" => self.op_tanh(node, tensors),
            "Sub" => self.op_sub(node, tensors),
            "Pow" => self.op_pow(node, tensors),
            "Where" => self.op_where(node, tensors),
            "Div" => self.op_div(node, tensors),
            "Split" => self.op_split(node, tensors),
            "Flatten" => self.op_flatten(node, tensors),
            "Reshape" => self.op_reshape(node, tensors),
            "Dropout" => self.op_passthrough(node, tensors),
            "Cast" => self.op_passthrough(node, tensors),
            "Shape" => self.op_shape(node, tensors),
            "Unsqueeze" => self.op_unsqueeze(node, tensors),
            "Squeeze" => self.op_squeeze(node, tensors),
            "Concat" => self.op_concat(node, tensors),
            "Transpose" => self.op_transpose(node, tensors),
            "Gather" => self.op_gather(node, tensors),
            "Constant" => self.op_constant(node, tensors),
            other => {
                eprintln!(
                    "[oxide_onnx] WARNING: unsupported op '{}' node '{}' — passthrough",
                    other, node.name
                );
                if !node.input.is_empty() && !node.output.is_empty() {
                    self.op_passthrough(node, tensors)
                } else {
                    Ok(())
                }
            }
        }
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Read a small tensor's values on the host. Prefers the cached constant
    /// (zero sync) and only falls back to a D2H copy for non-constant tensors.
    fn host_vals(&self, tensors: &TensorMap, name: &str) -> Result<Vec<f32>> {
        if let Some(v) = self.consts.get(name) {
            return Ok(v.clone());
        }
        let eb = Self::get_tensor(tensors, &self.weights, name)?;
        eb.buf()
            .to_host_vec(&self.stream)
            .map_err(|e| anyhow!("host_vals '{}' d2h: {:?}", name, e))
    }

    fn dtod_copy(&self, dst: &DeviceBuffer<f32>, src_ptr: u64, num_bytes: usize) -> Result<()> {
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                dst.cu_deviceptr(),
                src_ptr,
                num_bytes,
                self.stream.cu_stream(),
            )
            .map_err(|e| anyhow!("dtod copy: {:?}", e))
        }
    }

    fn sync(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow!("stream sync: {:?}", e))
    }

    /// 2D launch config for the 16×16-tiled SGEMM kernels (C is m×n).
    fn sgemm_cfg(m: usize, n: usize) -> LaunchConfig {
        const TILE: u32 = 16;
        LaunchConfig {
            grid_dim: ((n as u32).div_ceil(TILE), (m as u32).div_ceil(TILE), 1),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Launch config for the tensor-core SGEMM: one warp per 16×8 output tile.
    #[allow(dead_code)]
    fn sgemm_mma_cfg(m: usize, n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((n as u32).div_ceil(8), (m as u32).div_ceil(16), 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Launch config for the register-blocked SGEMM: one 256-thread block per
    /// 64×64 output tile, each thread a 4×4 micro-tile.
    #[allow(dead_code)]
    fn sgemm_rb_cfg(m: usize, n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((n as u32).div_ceil(64), (m as u32).div_ceil(64), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Launch config for the N-register-tiled SGEMM (`sgemm_fast`): one
    /// 256-thread block per 16(M)×128(N) tile.
    #[allow(dead_code)]
    fn sgemm_fast_cfg(m: usize, n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((n as u32).div_ceil(128), (m as u32).div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Launch config for the 2-D register-tiled SGEMM (`sgemm_reg`) and the
    /// implicit-GEMM convolution: one 256-thread block per 64×64 output tile,
    /// 4×4 outputs per thread.
    fn sgemm_reg_cfg(m: usize, n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (
                (n as u32).div_ceil(64).max(1),
                (m as u32).div_ceil(64).max(1),
                1,
            ),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        }
    }

    /// How many ways to split the K dimension for a given GEMM shape.
    ///
    /// The 64×64 register-tiled kernel needs roughly two blocks per SM to keep
    /// this card busy; the output alone rarely provides that at batch 1
    /// (M=512, N=49 gives 8). Splitting K makes up the difference, bounded so
    /// that each split still has enough K to amortise its shared-memory
    /// staging, and so the partial buffer stays small.
    pub fn split_factor(m: usize, n: usize, k: usize) -> usize {
        Self::split_factor_for(m, n, k, 164)
    }

    /// As [`Self::split_factor`], with an explicit block target.
    ///
    /// The best target is not universal, which a sweep made plain (ms):
    ///
    ///     target        41     82    164    328    656   1312
    ///     ResNet50    3.69   2.81   2.43   2.24   2.22   2.30
    ///     ViT                        6.01   6.70   6.99   7.30
    ///
    /// Convolution keeps improving well past two blocks per SM, while the
    /// transformer projections degrade past it. Same kernel, opposite
    /// preference — which is the argument for a per-shape autotuner rather
    /// than any single constant.
    pub fn split_factor_for(m: usize, n: usize, k: usize, target: usize) -> usize {
        let target_blocks: usize = std::env::var("OXIDE_SPLIT_TARGET")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(target);
        const MIN_K_PER_SPLIT: usize = 128;
        let base_blocks = m.div_ceil(64) * n.div_ceil(64);
        let wanted = target_blocks.div_ceil(base_blocks.max(1));
        let by_k = (k / MIN_K_PER_SPLIT).max(1);
        wanted.clamp(1, by_k.min(16))
    }

    /// Dispatch `C = act(alpha·A·B + bias)` to the best available kernel.
    ///
    /// Split-K is the default: it measured 2.1× faster than the 16×16 kernel
    /// across every GEMM shape in ResNet50 (3.7× on the worst one), because it
    /// is the only scheme here that gets arithmetic intensity and enough
    /// blocks at the same time. The reduction pass applies alpha, the per-row
    /// bias and the activation, so the epilogue stays fused.
    ///
    /// `OXIDE_GEMM=tiled|reg|splitk` overrides the choice for measurement.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_sgemm_epilogue(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        bias: Option<&DeviceBuffer<f32>>,
        act: u32,
        lo: f32,
        hi: f32,
        // True when the respective operand is a load-time constant, which is
        // what makes its f16 packing cacheable. Conv puts weights in A; Gemm
        // and MatMul put them in B.
        a_static: bool,
        b_static: bool,
        c: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let choice = std::env::var("OXIDE_GEMM").unwrap_or_default();
        let splits = match choice.as_str() {
            "tiled" | "reg" => 1,
            _ => Self::split_factor(m, n, k),
        };
        // The register-tiled kernel needs either a split — which is what makes
        // a small output fill the GPU — or an output big enough to keep its
        // 64×64 tiles busy. Attention MatMuls are neither: 197×64 with K=64 has
        // no K to split and only four tiles, and there the 16×16 kernel with a
        // fused epilogue wins, because split-K would add a whole extra pass
        // over the output for nothing (measured: 2.5 ms -> 6.4 ms on ViT).
        let worth_it = splits >= 2 || m * n >= 65_536;
        let use_splitk = choice != "tiled" && choice != "reg" && worth_it;

        // f16 tensor cores, when `a` is a constant we can keep pre-packed. The
        // GEMM is 1.55x the f32 split-K path on these shapes; the cost is f16's
        // 10-bit mantissa, so it is gated on the correctness harness rather
        // than assumed harmless. OXIDE_F16=0 disables it.
        let f16_shape_ok = k >= 32
            && m * n >= 4096
            && std::env::var("OXIDE_F16").map(|v| v != "0").unwrap_or(true);
        if use_splitk && a_static && f16_shape_ok {
            let kpairs = k.div_ceil(2);
            let a_packed = self.packed_weights(a, m, k)?;
            let k_per_split = k.div_ceil(splits).next_multiple_of(16).max(16);
            let mut partials = self
                .alloc_buf(splits * m * n)
                .map_err(|e| anyhow!("f16 partials alloc: {}", e))?;
            let gemm_cfg = LaunchConfig {
                grid_dim: (
                    (n as u32).div_ceil(64).max(1),
                    (m as u32).div_ceil(64).max(1),
                    splits as u32,
                ),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.module.sgemm_f16_tc_splitk_wpacked(
                    &self.stream,
                    gemm_cfg,
                    m as u32,
                    n as u32,
                    k as u32,
                    k_per_split as u32,
                    &a_packed,
                    kpairs as u32,
                    b,
                    &mut partials,
                )
            }
            .map_err(|e| anyhow!("sgemm_f16_tc launch: {:?}", e))?;
            let bias_operand = bias.unwrap_or(a);
            unsafe {
                self.module.reduce_splits(
                    &self.stream,
                    LaunchConfig::for_num_elems((m * n) as u32),
                    &partials,
                    splits as u32,
                    (m * n) as u32,
                    n as u32,
                    alpha,
                    bias_operand,
                    u32::from(bias.is_some()),
                    bias_operand,
                    0, // no residual on this path
                    act,
                    lo,
                    hi,
                    c,
                )
            }
            .map_err(|e| anyhow!("reduce_splits launch: {:?}", e))?;
            return Ok(());
        }

        // Same, for the operand order ONNX linear layers use: weights in B.
        if use_splitk && !a_static && b_static && f16_shape_ok {
            let kpairs = k.div_ceil(2);
            let b_packed = self.packed_weights_cols(b, k, n)?;
            let k_per_split = k.div_ceil(splits).next_multiple_of(16).max(16);
            let mut partials = self
                .alloc_buf(splits * m * n)
                .map_err(|e| anyhow!("f16 partials alloc: {}", e))?;
            let gemm_cfg = LaunchConfig {
                grid_dim: (
                    (n as u32).div_ceil(64).max(1),
                    (m as u32).div_ceil(64).max(1),
                    splits as u32,
                ),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // Pack the activation once, rather than letting every block that
            // shares a row of A redo the same conversions during staging —
            // 12x over for a ViT projection, and the dominant instruction cost
            // of the B-packed kernel.
            let mut a_packed =
                unsafe { DeviceBuffer::<u32>::uninitialized_async(&self.stream, m * kpairs) }
                    .map_err(|e| anyhow!("f16 A scratch: {:?}", e))?;
            unsafe {
                self.module.pack_f16_rows(
                    &self.stream,
                    LaunchConfig::for_num_elems((m * kpairs) as u32),
                    a,
                    k as u32,
                    kpairs as u32,
                    &mut a_packed,
                )
            }
            .map_err(|e| anyhow!("f16 A pack: {:?}", e))?;
            unsafe {
                self.module.sgemm_f16_tc_splitk_ab_w8(
                    &self.stream,
                    gemm_cfg,
                    m as u32,
                    n as u32,
                    k as u32,
                    k_per_split as u32,
                    &a_packed,
                    &b_packed,
                    kpairs as u32,
                    &mut partials,
                )
            }
            .map_err(|e| anyhow!("sgemm_f16_tc_ab_w8 launch: {:?}", e))?;
            let bias_operand = bias.unwrap_or(a);
            unsafe {
                self.module.reduce_splits(
                    &self.stream,
                    LaunchConfig::for_num_elems((m * n) as u32),
                    &partials,
                    splits as u32,
                    (m * n) as u32,
                    n as u32,
                    alpha,
                    bias_operand,
                    u32::from(bias.is_some()),
                    bias_operand,
                    0, // no residual on this path
                    act,
                    lo,
                    hi,
                    c,
                )
            }
            .map_err(|e| anyhow!("reduce_splits launch: {:?}", e))?;
            return Ok(());
        }

        // One split means the reduction would be a pure copy of the partials
        // into the output — measured at up to 24% of a shape's time. Write the
        // final values straight from the GEMM instead.
        if use_splitk && splits == 1 {
            let bias_operand = bias.unwrap_or(a);
            return unsafe {
                self.module.sgemm_reg(
                    &self.stream,
                    Self::sgemm_reg_cfg(m, n),
                    m as u32,
                    n as u32,
                    k as u32,
                    alpha,
                    a,
                    b,
                    bias_operand,
                    u32::from(bias.is_some()),
                    act,
                    lo,
                    hi,
                    c,
                )
            }
            .map_err(|e| anyhow!("sgemm_reg launch: {:?}", e));
        }

        if use_splitk {
            // Two register-block sizes, chosen per shape from measurement. The
            // 8x8 block has twice the arithmetic intensity but a quarter of the
            // blocks per output, so it only pays when the 128x128 tile is
            // actually filled and K is long enough to amortise the staging:
            //
            //                       4x4 (64x64)   8x8 (128x128)
            //   s2 3x3 (128,784,1152)   52.4 us      38.9 us
            //   s3 3x3 (256,196,2304)   60.5 us      43.4 us
            //   s1 3x3 (64,3136,576)    50.1 us      64.7 us   (M=64 wastes half the tile)
            //   s4 3x3 (512,49,4608)    71.2 us      76.4 us   (N=49 wastes most of it)
            let big_tile = m >= 128 && n >= 128 && k >= 1024;
            let splits = if big_tile {
                let base_blocks = m.div_ceil(128) * n.div_ceil(128);
                (164usize.div_ceil(base_blocks.max(1))).clamp(1, (k / 128).max(1).min(16))
            } else {
                splits
            };
            let k_per_split = k.div_ceil(splits).next_multiple_of(8).max(8);
            let partials = self
                .alloc_buf(splits * m * n)
                .map_err(|e| anyhow!("splitk partials alloc: {}", e))?;
            // `partials` is freed stream-ordered when it drops, i.e. after the
            // reduction that reads it has run.
            let mut partials = partials;

            let tile = if big_tile { 128u32 } else { 64u32 };
            let gemm_cfg = LaunchConfig {
                grid_dim: (
                    (n as u32).div_ceil(tile).max(1),
                    (m as u32).div_ceil(tile).max(1),
                    splits as u32,
                ),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            if big_tile {
                unsafe {
                    self.module.sgemm_reg8_splitk(
                        &self.stream,
                        gemm_cfg,
                        m as u32,
                        n as u32,
                        k as u32,
                        k_per_split as u32,
                        a,
                        b,
                        &mut partials,
                    )
                }
                .map_err(|e| anyhow!("sgemm_reg8_splitk launch: {:?}", e))?;
            } else {
                unsafe {
                    self.module.sgemm_reg_splitk(
                        &self.stream,
                        gemm_cfg,
                        m as u32,
                        n as u32,
                        k as u32,
                        k_per_split as u32,
                        a,
                        b,
                        &mut partials,
                    )
                }
                .map_err(|e| anyhow!("sgemm_reg_splitk launch: {:?}", e))?;
            }

            // The kernel never reads `bias`, so an absent bias can pass any
            // slice; `a` is already resident and correctly typed.
            let bias_operand = bias.unwrap_or(a);
            unsafe {
                self.module.reduce_splits(
                    &self.stream,
                    LaunchConfig::for_num_elems((m * n) as u32),
                    &partials,
                    splits as u32,
                    (m * n) as u32,
                    n as u32,
                    alpha,
                    bias_operand,
                    u32::from(bias.is_some()),
                    bias_operand,
                    0, // no residual on this path
                    act,
                    lo,
                    hi,
                    c,
                )
            }
            .map_err(|e| anyhow!("reduce_splits launch: {:?}", e))?;
            return Ok(());
        }

        // Fallback: the 16×16 kernel with the epilogue fused into its store.
        let bias_operand = bias.unwrap_or(a);
        unsafe {
            self.module.sgemm_bias_act(
                &self.stream,
                Self::sgemm_cfg(m, n),
                m as u32,
                n as u32,
                k as u32,
                alpha,
                a,
                b,
                bias_operand,
                u32::from(bias.is_some()),
                act,
                lo,
                hi,
                c,
            )
        }
        .map_err(|e| anyhow!("sgemm_bias_act launch: {:?}", e))
    }

    /// `C = alpha·A·B` with no epilogue, for the plain Gemm/MatMul paths.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_sgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        beta: f32,
        b_static: bool,
        c: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        // beta != 0 accumulates into C, which the split-K reduction does not
        // model; no current call site uses it, but keep the old kernel honest.
        if beta != 0.0 {
            return unsafe {
                self.module.sgemm_tiled(
                    &self.stream,
                    Self::sgemm_cfg(m, n),
                    m as u32,
                    n as u32,
                    k as u32,
                    alpha,
                    a,
                    b,
                    beta,
                    c,
                )
            }
            .map_err(|e| anyhow!("sgemm_tiled launch: {:?}", e));
        }
        self.dispatch_sgemm_epilogue(
            m,
            n,
            k,
            alpha,
            a,
            b,
            None,
            graph_opt::ACT_NONE as u32,
            0.0,
            0.0,
            // Gemm/MatMul take an activation as `a`; their weights are in `b`.
            false,
            b_static,
            c,
        )
    }

    /// Launch config for the block-per-row reduction kernels (softmax /
    /// layernorm): one 256-thread block per row.
    fn row_block_cfg(rows: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (rows.max(1) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    // =========================================================================
    // Relu
    // =========================================================================
    fn op_relu(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let ea = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let numel = ea.buf().len();
        let shape = ea.shape().clone();
        let mut out = self
            .alloc_buf(numel)
            .map_err(|e| anyhow!("relu alloc: {}", e))?;
        // Fused read→write: no separate D2D copy of the input.
        unsafe {
            self.module.relu_fwd(
                &self.stream,
                LaunchConfig::for_num_elems(numel as u32),
                ea.buf(),
                &mut out,
            )
        }
        .map_err(|e| anyhow!("relu launch: {:?}", e))?;
        drop(ea);
        tensors.insert(&out_name, out, shape);
        Ok(())
    }

    // =========================================================================
    // Clip
    // =========================================================================
    fn op_clip(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();

        // Opset ≤10: min/max are float attributes.
        // Opset ≥11: min/max are optional input tensors at input[1] and input[2].
        let mut lo = attr_f(node, "min", f32::NEG_INFINITY);
        let mut hi = attr_f(node, "max", f32::INFINITY);
        // host_vals prefers the cached-constant path (no D2H sync); min/max
        // are constants in practice (ReLU6 etc.).
        if node.input.len() > 1 && !node.input[1].is_empty() {
            if let Ok(v) = self.host_vals(tensors, &node.input[1]) {
                if !v.is_empty() {
                    lo = v[0];
                }
            }
        }
        if node.input.len() > 2 && !node.input[2].is_empty() {
            if let Ok(v) = self.host_vals(tensors, &node.input[2]) {
                if !v.is_empty() {
                    hi = v[0];
                }
            }
        }

        let ea = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let numel = ea.buf().len();
        let shape = ea.shape().clone();
        let mut out = self
            .alloc_buf(numel)
            .map_err(|e| anyhow!("clip alloc: {}", e))?;
        // Fused read→write: no separate D2D copy of the input.
        unsafe {
            self.module.clip_fwd(
                &self.stream,
                LaunchConfig::for_num_elems(numel as u32),
                ea.buf(),
                lo,
                hi,
                &mut out,
            )
        }
        .map_err(|e| anyhow!("clip launch: {:?}", e))?;
        drop(ea);
        tensors.insert(&out_name, out, shape);
        Ok(())
    }

    // =========================================================================
    // Add
    // =========================================================================
    fn op_add(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        // Borrow both inputs simultaneously (immutable, both from tensors or weights).
        // With NLL the borrow ends after the kernel call, before tensors.insert.
        let ea = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let eb = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
        let na = ea.buf().len();
        let nb = eb.buf().len();
        // Scalar broadcast: one operand is a single element.
        if na != nb && (na == 1 || nb == 1) {
            let (tensor, numel, shape, scalar_name) = if nb == 1 {
                (ea.buf(), na, ea.shape().clone(), node.input[1].as_str())
            } else {
                (eb.buf(), nb, eb.shape().clone(), node.input[0].as_str())
            };
            let s = self.host_vals(tensors, scalar_name)?[0];
            let mut out = self
                .alloc_buf(numel)
                .map_err(|e| anyhow!("add alloc: {}", e))?;
            unsafe {
                self.module.add_scalar(
                    &self.stream,
                    LaunchConfig::for_num_elems(numel as u32),
                    tensor,
                    s,
                    &mut out,
                )
            }
            .map_err(|e| anyhow!("add_scalar launch: {:?}", e))?;
            tensors.insert(&out_name, out, shape);
            return Ok(());
        }
        let a_shape = ea.shape().clone();
        let b_shape = eb.shape().clone();
        // Equal element count AND identical shape → fast elementwise path.
        if na == nb && a_shape == b_shape {
            let mut out = self
                .alloc_buf(na)
                .map_err(|e| anyhow!("add alloc: {}", e))?;
            unsafe {
                self.module.add_elementwise(
                    &self.stream,
                    LaunchConfig::for_num_elems(na as u32),
                    ea.buf(),
                    eb.buf(),
                    &mut out,
                )
            }
            .map_err(|e| anyhow!("add launch: {:?}", e))?;
            tensors.insert(&out_name, out, a_shape);
            return Ok(());
        }
        // General NumPy broadcasting (e.g. attention scores + mask).
        let (out_shape, out_str, a_str, b_str, ndim) = Self::broadcast_meta(&a_shape, &b_shape)?;
        let out_numel: usize = out_shape.iter().product();
        let osh = self.meta_buf(&Self::to_f32(&out_shape))?;
        let ost = self.meta_buf(&Self::to_f32(&out_str))?;
        let ast = self.meta_buf(&Self::to_f32(&a_str))?;
        let bst = self.meta_buf(&Self::to_f32(&b_str))?;
        let mut out = self
            .alloc_buf(out_numel)
            .map_err(|e| anyhow!("add bcast alloc: {}", e))?;
        unsafe {
            self.module.add_bcast(
                &self.stream,
                LaunchConfig::for_num_elems(out_numel as u32),
                ea.buf(),
                eb.buf(),
                &osh,
                &ost,
                &ast,
                &bst,
                ndim as u32,
                &mut out,
            )
        }
        .map_err(|e| anyhow!("add_bcast launch: {:?}", e))?;
        drop(ea);
        drop(eb);
        // add_bcast reads these async — keep alive until the run's final sync.
        tensors.insert(&out_name, out, out_shape);
        Ok(())
    }

    fn to_f32(v: &[usize]) -> Vec<f32> {
        v.iter().map(|&x| x as f32).collect()
    }

    /// Row-major strides of `s`.
    fn row_major(s: &[usize]) -> Vec<usize> {
        let mut st = vec![1usize; s.len()];
        for i in (0..s.len().saturating_sub(1)).rev() {
            st[i] = st[i + 1] * s[i + 1];
        }
        st
    }

    /// Broadcast strides of operand `s` against `out` (right-aligned;
    /// a size-1 or left-padded dim carries stride 0).
    fn bcast_strides(out: &[usize], s: &[usize]) -> Vec<usize> {
        let ndim = out.len();
        let mut padded = vec![1usize; ndim];
        padded[ndim - s.len()..].copy_from_slice(s);
        let full = Self::row_major(&padded);
        (0..ndim)
            .map(|i| if padded[i] == 1 { 0 } else { full[i] })
            .collect()
    }

    /// NumPy broadcast of several shapes into one output shape.
    fn bcast_shape(shapes: &[&[usize]]) -> Result<Vec<usize>> {
        let ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut out = vec![1usize; ndim];
        for s in shapes {
            let off = ndim - s.len();
            for (i, &d) in s.iter().enumerate() {
                let o = &mut out[off + i];
                if *o == 1 {
                    *o = d;
                } else if d != 1 && d != *o {
                    return Err(anyhow!("broadcast: incompatible {:?}", shapes));
                }
            }
        }
        Ok(out)
    }

    /// NumPy-broadcast metadata for two shapes, returned in row-major terms.
    /// Returns (out_shape, out_strides, a_strides, b_strides, ndim) where a
    /// broadcast dimension carries stride 0. Shapes are right-aligned.
    fn broadcast_meta(
        a: &[usize],
        b: &[usize],
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, usize)> {
        let ndim = a.len().max(b.len());
        let pad = |s: &[usize]| -> Vec<usize> {
            let mut p = vec![1usize; ndim];
            p[ndim - s.len()..].copy_from_slice(s);
            p
        };
        let ap = pad(a);
        let bp = pad(b);
        let row_major = |s: &[usize]| -> Vec<usize> {
            let mut st = vec![1usize; s.len()];
            for i in (0..s.len().saturating_sub(1)).rev() {
                st[i] = st[i + 1] * s[i + 1];
            }
            st
        };
        let mut out_shape = vec![0usize; ndim];
        for i in 0..ndim {
            out_shape[i] = if ap[i] == bp[i] || bp[i] == 1 {
                ap[i]
            } else if ap[i] == 1 {
                bp[i]
            } else {
                return Err(anyhow!("Add: incompatible shapes {:?} vs {:?}", a, b));
            };
        }
        let a_full = row_major(&ap);
        let b_full = row_major(&bp);
        let a_str: Vec<usize> = (0..ndim)
            .map(|i| if ap[i] == 1 { 0 } else { a_full[i] })
            .collect();
        let b_str: Vec<usize> = (0..ndim)
            .map(|i| if bp[i] == 1 { 0 } else { b_full[i] })
            .collect();
        Ok((out_shape.clone(), row_major(&out_shape), a_str, b_str, ndim))
    }

    // =========================================================================
    // Mul — element-wise on GPU
    // =========================================================================
    fn op_mul(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let ea = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let eb = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
        let na = ea.buf().len();
        let nb = eb.buf().len();
        if na != nb && (na == 1 || nb == 1) {
            let (tensor, numel, shape, scalar_name) = if nb == 1 {
                (ea.buf(), na, ea.shape().clone(), node.input[1].as_str())
            } else {
                (eb.buf(), nb, eb.shape().clone(), node.input[0].as_str())
            };
            let s = self.host_vals(tensors, scalar_name)?[0];
            let mut out = self
                .alloc_buf(numel)
                .map_err(|e| anyhow!("mul alloc: {}", e))?;
            unsafe {
                self.module.mul_scalar(
                    &self.stream,
                    LaunchConfig::for_num_elems(numel as u32),
                    tensor,
                    s,
                    &mut out,
                )
            }
            .map_err(|e| anyhow!("mul_scalar launch: {:?}", e))?;
            tensors.insert(&out_name, out, shape);
            return Ok(());
        }
        let numel = na;
        let shape = ea.shape().clone();
        let mut out = self
            .alloc_buf(numel)
            .map_err(|e| anyhow!("mul alloc: {}", e))?;
        unsafe {
            self.module.mul_elementwise(
                &self.stream,
                LaunchConfig::for_num_elems(numel as u32),
                ea.buf(),
                eb.buf(),
                &mut out,
            )
        }
        .map_err(|e| anyhow!("mul launch: {:?}", e))?;
        tensors.insert(&out_name, out, shape);
        Ok(())
    }

    // =========================================================================
    // Conv2D — im2col + cuBLAS SGEMM, grouped/depthwise, fully on GPU.
    //
    // No D2H/H2D transfers: the input and weight DeviceBuffers are accessed via
    // ManuallyDrop sub-views (pointer offset + no-op Drop) so each group slice
    // can be passed to im2col and cuBLAS without any PCIe traffic.
    // =========================================================================
    fn op_conv(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();

        // Grab device pointers and shapes — no D2H download.
        let (x_ptr, x_shape) = {
            let ex = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
            (ex.buf().cu_deviceptr(), ex.shape().clone())
        };
        let (w_ptr, w_shape) = {
            let ew = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
            (ew.buf().cu_deviceptr(), ew.shape().clone())
        };

        let (batch_n, c_in, h_in, w_in) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
        let (n_out, c_in_per_group, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
        let group = attr_i(node, "group", 1) as usize;
        let c_out_per_group = n_out / group;

        let (pad_h, pad_w) = parse_pads(node);
        let (stride_h, stride_w) = parse_strides(node);
        let (dil_h, dil_w) = parse_dilations(node);
        let (out_h, out_w) = conv2d_output_shape(
            h_in, w_in, kh, kw, pad_h, pad_w, stride_h, stride_w, dil_h, dil_w,
        );

        let col_rows_g = c_in_per_group * kh * kw;
        let col_cols = out_h * out_w;
        let out_numel = batch_n * n_out * out_h * out_w;

        // Pre-allocate output on GPU (zeroed, written in-place per group).
        let mut result_buf = self
            .alloc_buf(out_numel)
            .map_err(|e| anyhow!("conv result alloc: {}", e))?;

        // ── Fast path: depthwise conv (group == c_in, 1 input chan/group) ──
        // Collapses the O(group) im2col+GEMM launches into one fused kernel.
        let is_depthwise = c_in_per_group == 1 && group == c_in && group > 1;
        // ── Fast path: 1×1 pointwise conv (stride 1, no pad/dilation) ──
        // im2col for a 1×1 kernel is a verbatim copy of the whole input into a
        // same-sized buffer — pure waste. A 1×1 conv *is* a GEMM:
        //   out[b] = W[n_out × c_in] · X[b][c_in × (H·W)]
        // so feed the input straight to SGEMM (no col buffer, no im2col launch).
        let is_pointwise = kh == 1
            && kw == 1
            && group == 1
            && pad_h == 0
            && pad_w == 0
            && stride_h == 1
            && stride_w == 1
            && dil_h == 1
            && dil_w == 1;
        if is_depthwise {
            let x_full = ManuallyDrop::new(unsafe {
                DeviceBuffer::<f32>::from_raw_parts(
                    x_ptr,
                    batch_n * c_in * h_in * w_in,
                    self.ctx.clone(),
                )
            });
            let w_full = ManuallyDrop::new(unsafe {
                DeviceBuffer::<f32>::from_raw_parts(
                    w_ptr,
                    n_out * c_in_per_group * kh * kw,
                    self.ctx.clone(),
                )
            });
            unsafe {
                self.module.depthwise_conv2d(
                    &self.stream,
                    LaunchConfig::for_num_elems(out_numel as u32),
                    &*x_full,
                    &*w_full,
                    c_in as u32,
                    h_in as u32,
                    w_in as u32,
                    n_out as u32,
                    c_out_per_group as u32,
                    kh as u32,
                    kw as u32,
                    pad_h as u32,
                    pad_w as u32,
                    stride_h as u32,
                    stride_w as u32,
                    dil_h as u32,
                    dil_w as u32,
                    out_h as u32,
                    out_w as u32,
                    &mut result_buf,
                )
            }
            .map_err(|e| anyhow!("conv depthwise: {:?}", e))?;
        } else if is_pointwise {
            let hw = h_in * w_in; // == out_h * out_w
            let w_full = ManuallyDrop::new(unsafe {
                DeviceBuffer::<f32>::from_raw_parts(w_ptr, n_out * c_in, self.ctx.clone())
            });
            for b in 0..batch_n {
                let x_b = ManuallyDrop::new(unsafe {
                    DeviceBuffer::<f32>::from_raw_parts(
                        x_ptr + (b * c_in * hw * 4) as u64,
                        c_in * hw,
                        self.ctx.clone(),
                    )
                });
                let mut out_b = ManuallyDrop::new(unsafe {
                    DeviceBuffer::<f32>::from_raw_parts(
                        result_buf.cu_deviceptr() + (b * n_out * hw * 4) as u64,
                        n_out * hw,
                        self.ctx.clone(),
                    )
                });
                // C[n_out × hw] = W[n_out × c_in] · X_b[c_in × hw]
                self.dispatch_sgemm(n_out, hw, c_in, 1.0, &w_full, &x_b, 0.0, false, &mut out_b)
                    .map_err(|e| anyhow!("conv 1x1 sgemm b={}: {}", b, e))?;
            }
        } else if !is_depthwise
            && group == 1
            && std::env::var("OXIDE_F16").map(|v| v != "0").unwrap_or(true)
            && col_rows_g >= 32
            && n_out * col_cols >= 4096
        {
            // ── Implicit-GEMM on f16 tensor cores ───────────────────────────
            // One launch per (batch, split): the column matrix is never
            // materialised, which is where the DRAM gap to TensorRT sat.
            let (act, act_lo, act_hi) = Self::fused_act(node);
            let has_bias = node.input.len() > 2 && !node.input[2].is_empty();
            let bias_ptr = if has_bias {
                let eb = Self::get_tensor(tensors, &self.weights, &node.input[2])?;
                Some(eb.buf().cu_deviceptr())
            } else {
                None
            };
            let has_residual = attr_i(node, "oxide_residual", 0) != 0 && node.input.len() > 3;
            let residual_ptr = if has_residual {
                let er = Self::get_tensor(tensors, &self.weights, &node.input[3])?;
                Some(er.buf().cu_deviceptr())
            } else {
                None
            };
            let m = n_out;
            let kk = col_rows_g;
            let nn = col_cols;
            // Convolution wants more parallelism than the Gemm path; see
            // `split_factor_for`.
            let splits = Self::split_factor_for(m, nn, kk, 328);
            let k_per_split = kk.div_ceil(splits).next_multiple_of(16).max(16);
            // Must match the stride `packed_weights` packs with.
            let kpairs = kk.div_ceil(2);

            let w_full = ManuallyDrop::new(unsafe {
                DeviceBuffer::<f32>::from_raw_parts(w_ptr, m * kk, self.ctx.clone())
            });
            let a_packed = self.packed_weights(&w_full, m, kk)?;

            for b in 0..batch_n {
                let x_b = ManuallyDrop::new(unsafe {
                    DeviceBuffer::<f32>::from_raw_parts(
                        x_ptr + (b * c_in * h_in * w_in * 4) as u64,
                        c_in * h_in * w_in,
                        self.ctx.clone(),
                    )
                });
                let mut out_b = ManuallyDrop::new(unsafe {
                    DeviceBuffer::<f32>::from_raw_parts(
                        result_buf.cu_deviceptr() + (b * m * nn * 4) as u64,
                        m * nn,
                        self.ctx.clone(),
                    )
                });
                let mut partials = self
                    .alloc_buf(splits * m * nn)
                    .map_err(|e| anyhow!("conv f16 partials: {}", e))?;
                let cfg = LaunchConfig {
                    grid_dim: (
                        (nn as u32).div_ceil(64).max(1),
                        (m as u32).div_ceil(64).max(1),
                        splits as u32,
                    ),
                    // 8 warps over the same 64x64 tile: see conv2d_f16_tc_w8.
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.module.conv2d_f16_tc_w8(
                        &self.stream,
                        cfg,
                        m as u32,
                        nn as u32,
                        kk as u32,
                        k_per_split as u32,
                        &a_packed,
                        kpairs as u32,
                        &x_b,
                        c_in as u32,
                        h_in as u32,
                        w_in as u32,
                        kh as u32,
                        kw as u32,
                        pad_h as u32,
                        pad_w as u32,
                        stride_h as u32,
                        stride_w as u32,
                        dil_h as u32,
                        dil_w as u32,
                        out_w as u32,
                        &mut partials,
                    )
                }
                .map_err(|e| anyhow!("conv2d_f16_tc_w8 launch: {:?}", e))?;

                let bias_view = ManuallyDrop::new(unsafe {
                    match bias_ptr {
                        Some(ptr) => DeviceBuffer::<f32>::from_raw_parts(ptr, m, self.ctx.clone()),
                        None => DeviceBuffer::<f32>::from_raw_parts(w_ptr, m, self.ctx.clone()),
                    }
                });
                // A residual Add folded in by the graph rewrite is added here,
                // before the activation, instead of costing its own kernel.
                let residual_view = ManuallyDrop::new(unsafe {
                    match residual_ptr {
                        Some(ptr) => DeviceBuffer::<f32>::from_raw_parts(
                            ptr + (b * m * nn * 4) as u64,
                            m * nn,
                            self.ctx.clone(),
                        ),
                        None => DeviceBuffer::<f32>::from_raw_parts(w_ptr, m, self.ctx.clone()),
                    }
                });
                unsafe {
                    self.module.reduce_splits(
                        &self.stream,
                        LaunchConfig::for_num_elems((m * nn) as u32),
                        &partials,
                        splits as u32,
                        (m * nn) as u32,
                        nn as u32,
                        1.0,
                        &bias_view,
                        u32::from(has_bias),
                        &residual_view,
                        u32::from(has_residual),
                        act,
                        act_lo,
                        act_hi,
                        &mut out_b,
                    )
                }
                .map_err(|e| anyhow!("conv f16 reduce: {:?}", e))?;
            }

            tensors.insert(&out_name, result_buf, vec![batch_n, n_out, out_h, out_w]);
            return Ok(());
        } else if std::env::var("OXIDE_CONV_IMPLICIT").is_ok() {
            // ── Implicit-GEMM path: the column matrix never reaches DRAM ────
            // Off by default: measured 2× slower than im2col + SGEMM at batch
            // 1, because a 64×64 output tile leaves late ResNet stages
            // (M=512, N=49) with 8 blocks for 82 SMs. Needs split-K to expose
            // enough parallelism before it can win here; kept behind the env
            // var so that work has a starting point.
            let (act, act_lo, act_hi) = Self::fused_act(node);
            let has_bias = node.input.len() > 2 && !node.input[2].is_empty();
            let bias_ptr = if has_bias {
                let eb = Self::get_tensor(tensors, &self.weights, &node.input[2])?;
                Some(eb.buf().cu_deviceptr())
            } else {
                None
            };

            for b in 0..batch_n {
                for g in 0..group {
                    let in_offset = b * c_in * h_in * w_in + g * c_in_per_group * h_in * w_in;
                    let in_len = c_in_per_group * h_in * w_in;
                    let x_g = ManuallyDrop::new(unsafe {
                        DeviceBuffer::<f32>::from_raw_parts(
                            x_ptr + (in_offset * 4) as u64,
                            in_len,
                            self.ctx.clone(),
                        )
                    });

                    let w_offset = g * c_out_per_group * c_in_per_group * kh * kw;
                    let out_offset =
                        b * n_out * out_h * out_w + g * c_out_per_group * out_h * out_w;

                    let w_g = ManuallyDrop::new(unsafe {
                        DeviceBuffer::<f32>::from_raw_parts(
                            w_ptr + (w_offset * 4) as u64,
                            c_out_per_group * col_rows_g,
                            self.ctx.clone(),
                        )
                    });
                    let mut out_g = ManuallyDrop::new(unsafe {
                        DeviceBuffer::<f32>::from_raw_parts(
                            result_buf.cu_deviceptr() + (out_offset * 4) as u64,
                            c_out_per_group * col_cols,
                            self.ctx.clone(),
                        )
                    });
                    // When the conv has no bias the kernel is told not to read
                    // the operand, and gets the weight slice as a stand-in.
                    let bias_view = ManuallyDrop::new(unsafe {
                        match bias_ptr {
                            Some(ptr) => DeviceBuffer::<f32>::from_raw_parts(
                                ptr + (g * c_out_per_group * 4) as u64,
                                c_out_per_group,
                                self.ctx.clone(),
                            ),
                            None => DeviceBuffer::<f32>::from_raw_parts(
                                w_ptr + (w_offset * 4) as u64,
                                c_out_per_group,
                                self.ctx.clone(),
                            ),
                        }
                    });

                    unsafe {
                        self.module.conv2d_implicit_gemm(
                            &self.stream,
                            Self::sgemm_reg_cfg(c_out_per_group, col_cols),
                            c_out_per_group as u32,
                            col_cols as u32,
                            col_rows_g as u32,
                            &w_g,
                            &x_g,
                            &bias_view,
                            u32::from(has_bias),
                            c_in_per_group as u32,
                            h_in as u32,
                            w_in as u32,
                            kh as u32,
                            kw as u32,
                            pad_h as u32,
                            pad_w as u32,
                            stride_h as u32,
                            stride_w as u32,
                            dil_h as u32,
                            dil_w as u32,
                            out_w as u32,
                            act,
                            act_lo,
                            act_hi,
                            &mut out_g,
                        )
                    }
                    .map_err(|e| anyhow!("conv implicit gemm g={}: {:?}", g, e))?;
                }
            }

            // The epilogue already applied bias and activation.
            tensors.insert(&out_name, result_buf, vec![batch_n, n_out, out_h, out_w]);
            return Ok(());
        } else {
            // ── im2col + SGEMM (default) ────────────────────────────────────
            // Bias and activation ride in the GEMM epilogue, so the only
            // passes over the output are the GEMM's own stores.
            let (act, act_lo, act_hi) = Self::fused_act(node);
            let has_bias = node.input.len() > 2 && !node.input[2].is_empty();
            let bias_ptr = if has_bias {
                let eb = Self::get_tensor(tensors, &self.weights, &node.input[2])?;
                Some(eb.buf().cu_deviceptr())
            } else {
                None
            };
            let mut col_dev = unsafe {
                DeviceBuffer::<f32>::uninitialized_async(&self.stream, col_rows_g * col_cols)
            }
            .map_err(|e| anyhow!("conv col alloc: {:?}", e))?;

            for b in 0..batch_n {
                for g in 0..group {
                    let in_offset = b * c_in * h_in * w_in + g * c_in_per_group * h_in * w_in;
                    let in_len = c_in_per_group * h_in * w_in;
                    // ManuallyDrop prevents Drop from freeing the borrowed pointer.
                    let x_g = ManuallyDrop::new(unsafe {
                        DeviceBuffer::<f32>::from_raw_parts(
                            x_ptr + (in_offset * 4) as u64,
                            in_len,
                            self.ctx.clone(),
                        )
                    });

                    unsafe {
                        self.module.im2col(
                            &self.stream,
                            LaunchConfig::for_num_elems((col_rows_g * col_cols) as u32),
                            &*x_g,
                            c_in_per_group as u32,
                            h_in as u32,
                            w_in as u32,
                            kh as u32,
                            kw as u32,
                            pad_h as u32,
                            pad_w as u32,
                            stride_h as u32,
                            stride_w as u32,
                            dil_h as u32,
                            dil_w as u32,
                            out_h as u32,
                            out_w as u32,
                            &mut col_dev,
                        )
                    }
                    .map_err(|e| anyhow!("conv im2col g={}: {:?}", g, e))?;

                    let w_offset = g * c_out_per_group * c_in_per_group * kh * kw;
                    let out_offset =
                        b * n_out * out_h * out_w + g * c_out_per_group * out_h * out_w;

                    let w_g = ManuallyDrop::new(unsafe {
                        DeviceBuffer::<f32>::from_raw_parts(
                            w_ptr + (w_offset * 4) as u64,
                            c_out_per_group * col_rows_g,
                            self.ctx.clone(),
                        )
                    });
                    let mut out_g = ManuallyDrop::new(unsafe {
                        DeviceBuffer::<f32>::from_raw_parts(
                            result_buf.cu_deviceptr() + (out_offset * 4) as u64,
                            c_out_per_group * col_cols,
                            self.ctx.clone(),
                        )
                    });

                    // The bias slice covers this group's output channels;
                    // with no bias the kernel is told not to read the operand
                    // and gets the weight slice as a stand-in.
                    let bias_view = ManuallyDrop::new(unsafe {
                        match bias_ptr {
                            Some(ptr) => DeviceBuffer::<f32>::from_raw_parts(
                                ptr + (g * c_out_per_group * 4) as u64,
                                c_out_per_group,
                                self.ctx.clone(),
                            ),
                            None => DeviceBuffer::<f32>::from_raw_parts(
                                w_ptr + (w_offset * 4) as u64,
                                c_out_per_group,
                                self.ctx.clone(),
                            ),
                        }
                    });

                    self.dispatch_sgemm_epilogue(
                        c_out_per_group,
                        col_cols,
                        col_rows_g,
                        1.0,
                        &w_g,
                        &col_dev,
                        has_bias.then_some(&*bias_view),
                        act,
                        act_lo,
                        act_hi,
                        // Conv's `a` is the weight tensor: constant, cacheable.
                        // Its `b` is the im2col scratch, which is not.
                        true,
                        false,
                        &mut out_g,
                    )
                    .map_err(|e| anyhow!("conv sgemm g={}: {}", g, e))?;
                }
            }

            // The SGEMM reads col_dev asynchronously; keep it alive until the
            // run's final sync (DeviceBuffer::drop is an immediate cuMemFree).
            tensors.push_scratch(col_dev);

            // Bias and activation were applied in the GEMM epilogue.
            tensors.insert(&out_name, result_buf, vec![batch_n, n_out, out_h, out_w]);
            return Ok(());
        } // end !is_depthwise

        // Epilogue: optional bias (3rd input) and the activation the graph
        // rewrite folded in, in a single pass over the output.
        let (act, act_lo, act_hi) = Self::fused_act(node);
        let has_bias_ep = node.input.len() > 2 && !node.input[2].is_empty();
        // A residual Add folded in by the graph rewrite arrives as a fourth
        // input and is applied in this pass, which already exists, rather than
        // costing its own kernel and a round trip of the activation.
        let has_res_ep = attr_i(node, "oxide_residual", 0) != 0 && node.input.len() > 3;
        if has_bias_ep || has_res_ep {
            let stub = ManuallyDrop::new(unsafe {
                DeviceBuffer::<f32>::from_raw_parts(w_ptr, 1, self.ctx.clone())
            });
            let bias_ep = if has_bias_ep {
                Some(Self::get_tensor(tensors, &self.weights, &node.input[2])?)
            } else {
                None
            };
            let res_ep = if has_res_ep {
                Some(Self::get_tensor(tensors, &self.weights, &node.input[3])?)
            } else {
                None
            };
            unsafe {
                self.module.bias_act(
                    &self.stream,
                    LaunchConfig::for_num_elems(out_numel as u32),
                    &mut result_buf,
                    bias_ep.as_ref().map_or(&*stub, |e| e.buf()),
                    (out_h * out_w) as u32,
                    n_out as u32,
                    u32::from(has_bias_ep),
                    res_ep.as_ref().map_or(&*stub, |e| e.buf()),
                    u32::from(has_res_ep),
                    act,
                    act_lo,
                    act_hi,
                )
            }
            .map_err(|e| anyhow!("conv bias: {:?}", e))?;
        } else if act != graph_opt::ACT_NONE as u32 {
            self.apply_act_inplace(&mut result_buf, out_numel, act, act_lo, act_hi)?;
        }

        tensors.insert(&out_name, result_buf, vec![batch_n, n_out, out_h, out_w]);
        Ok(())
    }

    /// Device copy of a small constant metadata array — shapes, strides,
    /// permutations — cached by contents for the executor's lifetime.
    ///
    /// The obvious `DeviceBuffer::from_host` per use is correct but ruinous for
    /// latency: it allocates with `cuMemAlloc`, and the matching `cuMemFree`
    /// when the buffer drops at the end of the op **synchronizes the entire
    /// context**, draining the pipeline the executor just spent effort
    /// filling. ViT issues 49 Transposes, each building four such buffers, so
    /// nearly 200 device-wide stalls per inference — measured at ~350 µs per
    /// Transpose node, 17 ms of a 28 ms inference.
    ///
    /// These arrays are a handful of elements and repeat across inferences, so
    /// caching them removes the allocation, the copy and the stall together.
    /// The returned view must not be dropped as an owning buffer; callers get
    /// a `ManuallyDrop` alias of the cached allocation.
    fn meta_buf(&self, data: &[f32]) -> Result<ManuallyDrop<DeviceBuffer<f32>>> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        data.len().hash(&mut hasher);
        for v in data {
            v.to_bits().hash(&mut hasher);
        }
        let key = hasher.finish();

        let mut cache = self.meta_cache.borrow_mut();
        let entry = match cache.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let buf = DeviceBuffer::from_host(&self.stream, data)
                    .map_err(|e| anyhow!("metadata h2d: {:?}", e))?;
                e.insert(buf)
            }
        };
        // SAFETY: the alias is only read by kernels launched on this stream
        // while the cache entry (and therefore the allocation) is alive, and
        // ManuallyDrop keeps the alias from freeing it.
        Ok(ManuallyDrop::new(unsafe {
            DeviceBuffer::<f32>::from_raw_parts(entry.cu_deviceptr(), entry.len(), self.ctx.clone())
        }))
    }

    /// An intermediate buffer of `len` elements, recycled across inferences.
    ///
    /// Every node allocates its output and the whole graph is freed at the end
    /// of the run — for BERT that is ~450 `cuMemFreeAsync` calls enqueued on
    /// the stream, which the *next* inference's first synchronisation has to
    /// wait for the GPU to work through. Measured at 6.8 ms of GPT-2's 19.5 ms,
    /// entirely in the setup phase, before any kernel of that inference runs.
    ///
    /// Recycling avoids both the free and the allocation. Reuse is safe on a
    /// single stream: a kernel reading a recycled buffer is enqueued after
    /// every kernel that used it previously, so the stream order is the
    /// dependency order.
    fn alloc_buf(&self, len: usize) -> Result<DeviceBuffer<f32>> {
        if let Some(bucket) = self.buf_pool.borrow_mut().get_mut(&len)
            && let Some(buf) = bucket.pop()
        {
            return Ok(buf);
        }
        unsafe { DeviceBuffer::<f32>::uninitialized_async(&self.stream, len) }
            .map_err(|e| anyhow!("alloc {len} elems: {:?}", e))
    }

    /// Return a run's intermediates to the pool instead of freeing them.
    ///
    /// Called once the outputs have been copied back, which synchronises the
    /// stream, so nothing recycled here is still in flight.
    fn recycle(&self, tensors: TensorMap) {
        let mut pool = self.buf_pool.borrow_mut();
        let TensorMap { map, scratch, .. } = tensors;
        for (_, (buf, _)) in map {
            pool.entry(buf.len()).or_default().push(buf);
        }
        for buf in scratch {
            pool.entry(buf.len()).or_default().push(buf);
        }
    }

    /// An f16-packed copy of a *constant* weight matrix, built once and kept.
    ///
    /// Only valid for tensors that live for the executor's lifetime: the key
    /// includes the device pointer, and intermediate buffers are recycled, so
    /// caching a transient tensor would hand back another tensor's data. Conv
    /// weights qualify; activations do not, which is why the caller states it.
    ///
    /// Pre-packing is what makes the tensor-core path worth using. Reading the
    /// operand as ready-made half pairs removes both the per-element software
    /// conversion and half the bytes: the same kernel measured 1.88 ms over f32
    /// weights and 1.30 ms over packed ones, against 1.99 ms for the f32
    /// split-K path.
    fn packed_weights(
        &self,
        a: &DeviceBuffer<f32>,
        m: usize,
        k: usize,
    ) -> Result<ManuallyDrop<DeviceBuffer<u32>>> {
        let key = (a.cu_deviceptr(), m, k);
        let kpairs = k.div_ceil(2);
        let mut cache = self.f16_weights.borrow_mut();
        let entry = match cache.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let mut packed = DeviceBuffer::<u32>::zeroed(&self.stream, m * kpairs)
                    .map_err(|err| anyhow!("f16 weight alloc: {:?}", err))?;
                unsafe {
                    self.module.pack_f16_rows(
                        &self.stream,
                        LaunchConfig::for_num_elems((m * kpairs) as u32),
                        a,
                        k as u32,
                        kpairs as u32,
                        &mut packed,
                    )
                }
                .map_err(|err| anyhow!("f16 weight pack: {:?}", err))?;
                e.insert(packed)
            }
        };
        // SAFETY: aliases a cache entry that outlives the launch; ManuallyDrop
        // keeps the alias from freeing it.
        Ok(ManuallyDrop::new(unsafe {
            DeviceBuffer::<u32>::from_raw_parts(entry.cu_deviceptr(), entry.len(), self.ctx.clone())
        }))
    }

    /// An f16-packed copy of a constant weight matrix used as the *B* operand,
    /// laid out `[n][ceil(k/2)]`.
    ///
    /// Same lifetime rule as [`Self::packed_weights`]: only for tensors that
    /// outlive the run, since the key is the device pointer and intermediates
    /// are recycled. `Gemm` and `MatMul` put the weights here rather than in A,
    /// which is what this exists for.
    fn packed_weights_cols(
        &self,
        b: &DeviceBuffer<f32>,
        k: usize,
        n: usize,
    ) -> Result<ManuallyDrop<DeviceBuffer<u32>>> {
        let key = (b.cu_deviceptr(), k, n);
        let kpairs = k.div_ceil(2);
        let mut cache = self.f16_weights_b.borrow_mut();
        let entry = match cache.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let mut packed = DeviceBuffer::<u32>::zeroed(&self.stream, n * kpairs)
                    .map_err(|err| anyhow!("f16 B alloc: {:?}", err))?;
                unsafe {
                    self.module.pack_f16_cols(
                        &self.stream,
                        LaunchConfig::for_num_elems((n * kpairs) as u32),
                        b,
                        k as u32,
                        n as u32,
                        kpairs as u32,
                        &mut packed,
                    )
                }
                .map_err(|err| anyhow!("f16 B pack: {:?}", err))?;
                e.insert(packed)
            }
        };
        // SAFETY: aliases a cache entry that outlives the launch.
        Ok(ManuallyDrop::new(unsafe {
            DeviceBuffer::<u32>::from_raw_parts(entry.cu_deviceptr(), entry.len(), self.ctx.clone())
        }))
    }

    /// Launch geometry for the channel-wise elementwise kernels: one grid row
    /// per (batch, channel) pair, `spatial` elements wide.
    ///
    /// `blockDim.y` must stay 1 — the kernels read the row index straight out
    /// of `blockIdx.y` to avoid deriving the channel by integer division.
    fn nc_cfg(spatial: usize, rows: usize) -> LaunchConfig {
        const BLOCK_X: u32 = 256;
        LaunchConfig {
            grid_dim: ((spatial as u32).div_ceil(BLOCK_X).max(1), rows as u32, 1),
            block_dim: (BLOCK_X, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Read the activation the load-time graph rewrite fused into this node.
    ///
    /// Returns `(act_code, lo, hi)` with the bounds only meaningful for
    /// `ACT_CLIP`.
    fn fused_act(node: &NodeProto) -> (u32, f32, f32) {
        let act = attr_i(node, "oxide_act", graph_opt::ACT_NONE) as u32;
        let lo = attr_f(node, "oxide_act_lo", f32::NEG_INFINITY);
        let hi = attr_f(node, "oxide_act_hi", f32::INFINITY);
        (act, lo, hi)
    }

    /// Apply a fused activation in place, for producers that have no bias pass
    /// to piggyback on.
    fn apply_act_inplace(
        &self,
        buf: &mut DeviceBuffer<f32>,
        numel: usize,
        act: u32,
        lo: f32,
        hi: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        if act == graph_opt::ACT_RELU as u32 {
            unsafe { self.module.relu(&self.stream, cfg, buf) }
                .map_err(|e| anyhow!("fused relu: {:?}", e))?;
        } else if act == graph_opt::ACT_CLIP as u32 {
            unsafe { self.module.clip(&self.stream, cfg, buf, lo, hi) }
                .map_err(|e| anyhow!("fused clip: {:?}", e))?;
        }
        Ok(())
    }

    // =========================================================================
    // BatchNormalization
    // =========================================================================
    fn op_batchnorm(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let eps = attr_f(node, "epsilon", 1e-5);
        let (act, act_lo, act_hi) = Self::fused_act(node);

        let x_shape = {
            let ex = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
            ex.shape().clone()
        };
        let n = x_shape[0];
        let c = x_shape[1];
        let hw: usize = x_shape.iter().skip(2).product::<usize>().max(1);
        let numel: usize = x_shape.iter().product();

        let mut out = self
            .alloc_buf(numel)
            .map_err(|e| anyhow!("batchnorm alloc: {}", e))?;

        // Fast path: the load-time rewrite collapsed the five parameters into
        // (scale, shift), so the kernel is a per-channel affine and nothing more.
        if node.input.len() == 3 {
            let ex = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
            let escale = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
            let eshift = Self::get_tensor(tensors, &self.weights, &node.input[2])?;
            unsafe {
                self.module.batch_norm_act(
                    &self.stream,
                    LaunchConfig::for_num_elems(numel as u32),
                    ex.buf(),
                    escale.buf(),
                    eshift.buf(),
                    hw as u32,
                    c as u32,
                    act,
                    act_lo,
                    act_hi,
                    &mut out,
                )
            }
            .map_err(|e| anyhow!("batchnorm launch: {:?}", e))?;
        } else {
            // Fallback: parameters were not load-time constants, so normalize
            // on device from the original five inputs.
            let ex = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
            let egamma = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
            let ebeta = Self::get_tensor(tensors, &self.weights, &node.input[2])?;
            let emean = Self::get_tensor(tensors, &self.weights, &node.input[3])?;
            let evar = Self::get_tensor(tensors, &self.weights, &node.input[4])?;
            unsafe {
                self.module.batch_norm_inference(
                    &self.stream,
                    LaunchConfig::for_num_elems(numel as u32),
                    ex.buf(),
                    egamma.buf(),
                    ebeta.buf(),
                    emean.buf(),
                    evar.buf(),
                    eps,
                    n as u32,
                    c as u32,
                    hw as u32,
                    &mut out,
                )
            }
            .map_err(|e| anyhow!("batchnorm launch: {:?}", e))?;
            if act != graph_opt::ACT_NONE as u32 {
                self.apply_act_inplace(&mut out, numel, act, act_lo, act_hi)?;
            }
        }

        tensors.insert(&out_name, out, x_shape);
        Ok(())
    }

    // =========================================================================
    // MaxPool2D
    // =========================================================================
    fn op_maxpool(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let (kh, kw) = parse_kernel_shape(node);
        let (pad_h, pad_w) = parse_pads(node);
        let (stride_h, stride_w) = parse_strides(node);

        let ex = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let x_shape = ex.shape().clone();
        let (n, c, in_h, in_w) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
        let (out_h, out_w) =
            maxpool_output_shape(in_h, in_w, kh, kw, pad_h, pad_w, stride_h, stride_w);
        let out_numel = n * c * out_h * out_w;

        let mut out = self
            .alloc_buf(out_numel)
            .map_err(|e| anyhow!("maxpool alloc: {}", e))?;

        unsafe {
            self.module.maxpool2d(
                &self.stream,
                LaunchConfig::for_num_elems(out_numel as u32),
                ex.buf(),
                c as u32,
                in_h as u32,
                in_w as u32,
                kh as u32,
                kw as u32,
                pad_h as u32,
                pad_w as u32,
                stride_h as u32,
                stride_w as u32,
                out_h as u32,
                out_w as u32,
                &mut out,
            )
        }
        .map_err(|e| anyhow!("maxpool launch: {:?}", e))?;

        tensors.insert(&out_name, out, vec![n, c, out_h, out_w]);
        Ok(())
    }

    // =========================================================================
    // GlobalAveragePool
    // =========================================================================
    fn op_global_avg_pool(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();

        let ex = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let x_shape = ex.shape().clone();
        let (n, c) = (x_shape[0], x_shape[1]);
        let hw = if x_shape.len() >= 4 {
            x_shape[2] * x_shape[3]
        } else {
            1
        };

        let mut out = self
            .alloc_buf(n * c)
            .map_err(|e| anyhow!("gavgpool alloc: {}", e))?;

        unsafe {
            self.module.global_avg_pool(
                &self.stream,
                LaunchConfig::for_num_elems((n * c) as u32),
                ex.buf(),
                c as u32,
                hw as u32,
                &mut out,
            )
        }
        .map_err(|e| anyhow!("gavgpool launch: {:?}", e))?;

        tensors.insert(&out_name, out, vec![n, c, 1, 1]);
        Ok(())
    }

    // =========================================================================
    // Gemm: Y = alpha * op(A) * op(B) + beta * C
    //
    // A and B stay on GPU; the tiled SGEMM kernels handle the transB flag.
    // Bias C (vector [n]) is added with the bias_add kernel (no D2H).
    // =========================================================================
    fn op_gemm(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let alpha = attr_f(node, "alpha", 1.0);
        let beta_val = attr_f(node, "beta", 1.0);
        let trans_a = attr_i(node, "transA", 0) != 0;
        let trans_b = attr_i(node, "transB", 0) != 0;

        if trans_a {
            return Err(anyhow!("Gemm transA=1 not supported by tiled SGEMM"));
        }

        let ea = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let eb = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
        let a_shape = ea.shape().clone();
        let b_shape = eb.shape().clone();

        let (m, k_a) = (a_shape[0], a_shape[1]);
        let (k_b, n) = if trans_b {
            (b_shape[1], b_shape[0])
        } else {
            (b_shape[0], b_shape[1])
        };
        if k_a != k_b {
            return Err(anyhow!("Gemm k mismatch: k_a={} k_b={}", k_a, k_b));
        }
        if m == 0 || n == 0 || k_a == 0 {
            return Err(anyhow!("Gemm degenerate dims m={} n={} k={}", m, n, k_a));
        }

        let out_numel = m * n;
        let mut out_dev = self
            .alloc_buf(out_numel)
            .map_err(|e| anyhow!("gemm alloc: {}", e))?;

        // C = alpha * A * op(B)  (beta=0, bias added separately below)
        let cfg = Self::sgemm_cfg(m, n);
        if trans_b {
            unsafe {
                self.module.sgemm_transb_tiled(
                    &self.stream,
                    cfg,
                    m as u32,
                    n as u32,
                    k_a as u32,
                    alpha,
                    ea.buf(),
                    eb.buf(),
                    0.0,
                    &mut out_dev,
                )
            }
            .map_err(|e| anyhow!("gemm sgemm_transb: {:?}", e))?;
        } else {
            let b_static = self.weights.contains_key(&node.input[1]);
            self.dispatch_sgemm(
                m,
                n,
                k_a,
                alpha,
                ea.buf(),
                eb.buf(),
                0.0,
                b_static,
                &mut out_dev,
            )
            .map_err(|e| anyhow!("gemm sgemm: {}", e))?;
        }
        drop(ea);
        drop(eb);

        // Add optional bias C: out[row, col] += beta * C[col]
        if node.input.len() > 2 && !node.input[2].is_empty() {
            let eb = Self::get_tensor(tensors, &self.weights, &node.input[2])?;
            let c_len = eb.buf().len();
            if c_len > 0 && beta_val != 0.0 {
                if (beta_val - 1.0).abs() < 1e-6 {
                    // bias_add: out[i] += bias[i % n] (spatial=1, features=n)
                    unsafe {
                        self.module.bias_add(
                            &self.stream,
                            LaunchConfig::for_num_elems(out_numel as u32),
                            &mut out_dev,
                            eb.buf(),
                            1u32,
                            n as u32,
                        )
                    }
                    .map_err(|e| anyhow!("gemm bias_add: {:?}", e))?;
                } else {
                    // Non-unit beta: scale bias on host (C is small, e.g. 1000 floats)
                    let c_host = eb
                        .buf()
                        .to_host_vec(&self.stream)
                        .map_err(|e| anyhow!("gemm c d2h: {:?}", e))?;
                    let scaled: Vec<f32> = c_host.iter().map(|&v| v * beta_val).collect();
                    let scaled_dev = DeviceBuffer::from_host(&self.stream, &scaled)
                        .map_err(|e| anyhow!("gemm scaled c h2d: {:?}", e))?;
                    unsafe {
                        self.module.bias_add(
                            &self.stream,
                            LaunchConfig::for_num_elems(out_numel as u32),
                            &mut out_dev,
                            &scaled_dev,
                            1u32,
                            n as u32,
                        )
                    }
                    .map_err(|e| anyhow!("gemm scaled bias_add: {:?}", e))?;
                    tensors.push_scratch(scaled_dev);
                }
            }
        }

        tensors.insert(&out_name, out_dev, vec![m, n]);
        Ok(())
    }

    // =========================================================================
    // MatMul — tiled SGEMM, batched over leading dims.
    //   A: [...batch, m, k]   B: [...batch, k, n]   →  C: [...batch, m, n]
    //   Each batch slice is a ManuallyDrop GPU sub-view (no PCIe traffic).
    // =========================================================================
    fn op_matmul(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();

        let (a_ptr, a_shape) = {
            let ea = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
            (ea.buf().cu_deviceptr(), ea.shape().clone())
        };
        let (b_ptr, b_shape) = {
            let eb = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
            (eb.buf().cu_deviceptr(), eb.shape().clone())
        };

        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];
        let batch: usize = a_shape[..a_shape.len() - 2].iter().product();
        let b_batch: usize = b_shape[..b_shape.len() - 2].iter().product();
        if b_batch != batch {
            return Err(anyhow!(
                "MatMul batch mismatch: a_batch={} b_batch={} (a={:?} b={:?})",
                batch,
                b_batch,
                a_shape,
                b_shape
            ));
        }

        let mut out_shape = a_shape[..a_shape.len() - 2].to_vec();
        out_shape.extend([m, n]);

        let mut out = self
            .alloc_buf(batch * m * n)
            .map_err(|e| anyhow!("matmul alloc: {}", e))?;
        let out_ptr = out.cu_deviceptr();

        // The weights of a linear layer live in B; caching their packed f16
        // form is only sound for tensors that outlive the run.
        let b_static = self.weights.contains_key(&node.input[1]);
        for bi in 0..batch {
            let a_g = ManuallyDrop::new(unsafe {
                DeviceBuffer::<f32>::from_raw_parts(
                    a_ptr + (bi * m * k * 4) as u64,
                    m * k,
                    self.ctx.clone(),
                )
            });
            let b_g = ManuallyDrop::new(unsafe {
                DeviceBuffer::<f32>::from_raw_parts(
                    b_ptr + (bi * k * n * 4) as u64,
                    k * n,
                    self.ctx.clone(),
                )
            });
            let mut c_g = ManuallyDrop::new(unsafe {
                DeviceBuffer::<f32>::from_raw_parts(
                    out_ptr + (bi * m * n * 4) as u64,
                    m * n,
                    self.ctx.clone(),
                )
            });
            self.dispatch_sgemm(m, n, k, 1.0, &a_g, &b_g, 0.0, b_static, &mut c_g)
                .map_err(|e| anyhow!("matmul sgemm bi={}: {}", bi, e))?;
        }

        tensors.insert(&out_name, out, out_shape);
        Ok(())
    }

    // =========================================================================
    // Softmax
    // =========================================================================
    fn op_softmax(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let axis = attr_i(node, "axis", -1);

        let ex = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let x_shape = ex.shape().clone();
        let numel = ex.buf().len();
        let ndim = x_shape.len() as i64;
        let ax = if axis < 0 {
            (ndim + axis) as usize
        } else {
            axis as usize
        };
        let cols = x_shape[ax];
        let rows = numel / cols;

        let mut out = self
            .alloc_buf(numel)
            .map_err(|e| anyhow!("softmax alloc: {}", e))?;

        unsafe {
            self.module.softmax_block(
                &self.stream,
                Self::row_block_cfg(rows),
                ex.buf(),
                rows as u32,
                cols as u32,
                &mut out,
            )
        }
        .map_err(|e| anyhow!("softmax launch: {:?}", e))?;

        tensors.insert(&out_name, out, x_shape);
        Ok(())
    }

    // =========================================================================
    // LayerNormalization — normalize over the last `axis` dims (here: last 1)
    // =========================================================================
    fn op_layernorm(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let eps = attr_f(node, "epsilon", 1e-5);
        let axis = attr_i(node, "axis", -1);

        let ex = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let egamma = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
        let ebeta = Self::get_tensor(tensors, &self.weights, &node.input[2])?;

        let x_shape = ex.shape().clone();
        let numel = ex.buf().len();
        let ndim = x_shape.len() as i64;
        let ax = if axis < 0 {
            (ndim + axis) as usize
        } else {
            axis as usize
        };
        // Normalize over the product of dims [ax..]; rows = everything before.
        let cols: usize = x_shape[ax..].iter().product();
        let rows = numel / cols;

        let mut out = self
            .alloc_buf(numel)
            .map_err(|e| anyhow!("layernorm alloc: {}", e))?;

        unsafe {
            self.module.layernorm_block(
                &self.stream,
                Self::row_block_cfg(rows),
                ex.buf(),
                egamma.buf(),
                ebeta.buf(),
                rows as u32,
                cols as u32,
                eps,
                &mut out,
            )
        }
        .map_err(|e| anyhow!("layernorm launch: {:?}", e))?;

        tensors.insert(&out_name, out, x_shape);
        Ok(())
    }

    // =========================================================================
    // Erf — element-wise error function (component of exact GELU)
    // =========================================================================
    fn op_erf(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let ea = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let numel = ea.buf().len();
        let shape = ea.shape().clone();
        let mut out = self
            .alloc_buf(numel)
            .map_err(|e| anyhow!("erf alloc: {}", e))?;
        unsafe {
            self.module.erf_fwd(
                &self.stream,
                LaunchConfig::for_num_elems(numel as u32),
                ea.buf(),
                &mut out,
            )
        }
        .map_err(|e| anyhow!("erf launch: {:?}", e))?;
        drop(ea);
        tensors.insert(&out_name, out, shape);
        Ok(())
    }

    // =========================================================================
    // Tanh — element-wise (BERT pooler)
    // =========================================================================
    fn op_tanh(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let ea = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let numel = ea.buf().len();
        let shape = ea.shape().clone();
        let mut out = self
            .alloc_buf(numel)
            .map_err(|e| anyhow!("tanh alloc: {}", e))?;
        unsafe {
            self.module.tanh_fwd(
                &self.stream,
                LaunchConfig::for_num_elems(numel as u32),
                ea.buf(),
                &mut out,
            )
        }
        .map_err(|e| anyhow!("tanh launch: {:?}", e))?;
        drop(ea);
        tensors.insert(&out_name, out, shape);
        Ok(())
    }

    // =========================================================================
    // Sub — supports scalar − tensor (BERT 1 − attention_mask) and
    //       tensor − scalar; equal-shape elementwise via add of negation.
    // =========================================================================
    fn op_sub(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let ea = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let eb = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
        let na = ea.buf().len();
        let nb = eb.buf().len();
        if na == 1 && nb > 1 {
            let s = self.host_vals(tensors, &node.input[0])?[0];
            let shape = eb.shape().clone();
            let mut out = self
                .alloc_buf(nb)
                .map_err(|e| anyhow!("sub alloc: {}", e))?;
            unsafe {
                self.module.sub_scalar_lhs(
                    &self.stream,
                    LaunchConfig::for_num_elems(nb as u32),
                    eb.buf(),
                    s,
                    &mut out,
                )
            }
            .map_err(|e| anyhow!("sub_scalar_lhs launch: {:?}", e))?;
            tensors.insert(&out_name, out, shape);
            return Ok(());
        }
        if nb == 1 && na > 1 {
            let s = self.host_vals(tensors, &node.input[1])?[0];
            let shape = ea.shape().clone();
            let mut out = self
                .alloc_buf(na)
                .map_err(|e| anyhow!("sub alloc: {}", e))?;
            unsafe {
                self.module.add_scalar(
                    &self.stream,
                    LaunchConfig::for_num_elems(na as u32),
                    ea.buf(),
                    -s,
                    &mut out,
                )
            }
            .map_err(|e| anyhow!("sub add_scalar launch: {:?}", e))?;
            tensors.insert(&out_name, out, shape);
            return Ok(());
        }
        Err(anyhow!(
            "Sub: only scalar±tensor supported (na={} nb={})",
            na,
            nb
        ))
    }

    // =========================================================================
    // Pow — tensor ^ scalar integer exponent (GELU x³)
    // =========================================================================
    fn op_pow(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let p = self.host_vals(tensors, &node.input[1])?[0];
        let ea = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let numel = ea.buf().len();
        let shape = ea.shape().clone();
        let mut out = self
            .alloc_buf(numel)
            .map_err(|e| anyhow!("pow alloc: {}", e))?;
        unsafe {
            self.module.pow_scalar(
                &self.stream,
                LaunchConfig::for_num_elems(numel as u32),
                ea.buf(),
                p,
                &mut out,
            )
        }
        .map_err(|e| anyhow!("pow launch: {:?}", e))?;
        drop(ea);
        tensors.insert(&out_name, out, shape);
        Ok(())
    }

    // =========================================================================
    // Where — c = cond ? X : Y, with NumPy broadcasting (causal mask)
    // =========================================================================
    fn op_where(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let ec = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let ex = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
        let ey = Self::get_tensor(tensors, &self.weights, &node.input[2])?;
        let cs = ec.shape().clone();
        let xs = ex.shape().clone();
        let ys = ey.shape().clone();

        let out_shape = Self::bcast_shape(&[&cs, &xs, &ys])?;
        let out_numel: usize = out_shape.iter().product();
        let ost = Self::row_major(&out_shape);
        let c_str = Self::bcast_strides(&out_shape, &cs);
        let x_str = Self::bcast_strides(&out_shape, &xs);
        let y_str = Self::bcast_strides(&out_shape, &ys);
        let ndim = out_shape.len();

        let osh = self.meta_buf(&Self::to_f32(&out_shape))?;
        let osb = self.meta_buf(&Self::to_f32(&ost))?;
        let csb = self.meta_buf(&Self::to_f32(&c_str))?;
        let xsb = self.meta_buf(&Self::to_f32(&x_str))?;
        let ysb = self.meta_buf(&Self::to_f32(&y_str))?;
        let mut out = self
            .alloc_buf(out_numel)
            .map_err(|e| anyhow!("where alloc: {}", e))?;
        unsafe {
            self.module.where_bcast(
                &self.stream,
                LaunchConfig::for_num_elems(out_numel as u32),
                ec.buf(),
                ex.buf(),
                ey.buf(),
                &osh,
                &osb,
                &csb,
                &xsb,
                &ysb,
                ndim as u32,
                &mut out,
            )
        }
        .map_err(|e| anyhow!("where launch: {:?}", e))?;
        drop(ec);
        drop(ex);
        drop(ey);
        tensors.insert(&out_name, out, out_shape);
        Ok(())
    }

    // =========================================================================
    // Div — currently supports tensor ÷ scalar (GELU, attention scaling)
    // =========================================================================
    fn op_div(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let ea = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let eb = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
        let na = ea.buf().len();
        let nb = eb.buf().len();
        if nb == 1 {
            let shape = ea.shape().clone();
            let s = self.host_vals(tensors, &node.input[1])?[0];
            let mut out = self
                .alloc_buf(na)
                .map_err(|e| anyhow!("div alloc: {}", e))?;
            unsafe {
                self.module.mul_scalar(
                    &self.stream,
                    LaunchConfig::for_num_elems(na as u32),
                    ea.buf(),
                    1.0 / s,
                    &mut out,
                )
            }
            .map_err(|e| anyhow!("div mul_scalar launch: {:?}", e))?;
            tensors.insert(&out_name, out, shape);
            return Ok(());
        }
        Err(anyhow!(
            "Div: only tensor÷scalar supported (na={} nb={})",
            na,
            nb
        ))
    }

    // =========================================================================
    // Split — slice a tensor into N parts along `axis`, fully on GPU.
    // =========================================================================
    fn op_split(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let ex = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let x_shape = ex.shape().clone();
        let in_ptr = ex.buf().cu_deviceptr();
        drop(ex);

        let ndim = x_shape.len() as i64;
        let axis_i = attr_i(node, "axis", 0);
        let axis = if axis_i < 0 {
            (ndim + axis_i) as usize
        } else {
            axis_i as usize
        };
        let axis_len = x_shape[axis];
        let n_out = node.output.len();

        // Split sizes: opset ≥13 from input[1] (small tensor), else equal parts.
        let sizes: Vec<usize> = if node.input.len() > 1 && !node.input[1].is_empty() {
            self.host_vals(tensors, &node.input[1])?
                .into_iter()
                .map(|v| v as usize)
                .collect()
        } else {
            vec![axis_len / n_out; n_out]
        };

        let outer: usize = x_shape[..axis].iter().product();
        let inner: usize = x_shape[axis + 1..].iter().product();

        // Non-owning GPU view of the whole input (pointer offset, no-op Drop).
        let in_view = ManuallyDrop::new(unsafe {
            DeviceBuffer::<f32>::from_raw_parts(in_ptr, outer * axis_len * inner, self.ctx.clone())
        });

        let mut start = 0usize;
        for (oi, &sz) in sizes.iter().enumerate() {
            let piece_numel = outer * sz * inner;
            let mut piece = self
                .alloc_buf(piece_numel)
                .map_err(|e| anyhow!("split alloc: {}", e))?;
            unsafe {
                self.module.slice_axis(
                    &self.stream,
                    LaunchConfig::for_num_elems(piece_numel as u32),
                    &*in_view,
                    axis_len as u32,
                    inner as u32,
                    start as u32,
                    sz as u32,
                    &mut piece,
                )
            }
            .map_err(|e| anyhow!("split slice_axis oi={}: {:?}", oi, e))?;
            let mut out_shape = x_shape.clone();
            out_shape[axis] = sz;
            tensors.insert(&node.output[oi], piece, out_shape);
            start += sz;
        }
        Ok(())
    }

    // =========================================================================
    // Shape operations (reshape / flatten / passthrough / unsqueeze / squeeze)
    // =========================================================================

    fn op_passthrough(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        if node.input.is_empty() || node.output.is_empty() {
            return Ok(());
        }
        let out_name = node.output[0].clone();
        let shape = Self::get_tensor(tensors, &self.weights, &node.input[0])?
            .shape()
            .clone();
        self.alias_or_copy(tensors, &out_name, &node.input[0], shape)
    }

    fn op_flatten(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let axis = attr_i(node, "axis", 1) as usize;
        let x_shape = Self::get_tensor(tensors, &self.weights, &node.input[0])?
            .shape()
            .clone();
        let outer: usize = x_shape[..axis].iter().product();
        let inner: usize = x_shape[axis..].iter().product();
        self.alias_or_copy(tensors, &out_name, &node.input[0], vec![outer, inner])
    }

    fn op_reshape(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let (numel, x_shape) = {
            let eb = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
            (eb.buf().len(), eb.shape().clone())
        };

        let allow_zero = attr_i(node, "allowzero", 0) != 0;

        let new_shape_raw: Vec<i64> = if node.input.len() > 1 && !node.input[1].is_empty() {
            self.host_vals(tensors, &node.input[1])?
                .into_iter()
                .map(|v| v as i64)
                .collect()
        } else {
            vec![]
        };

        let new_shape: Vec<usize> = if new_shape_raw.is_empty() {
            vec![numel]
        } else {
            // ONNX Reshape semantics (allowzero=0, the default):
            //   0 → copy dimension from input at the same index
            //  -1 → infer from remaining dimensions
            // other → use as-is
            let resolved: Vec<i64> = new_shape_raw
                .iter()
                .enumerate()
                .map(|(i, &d)| {
                    if d == 0 && !allow_zero {
                        x_shape.get(i).map(|&v| v as i64).unwrap_or(0)
                    } else {
                        d
                    }
                })
                .collect();
            let known: i64 = resolved.iter().filter(|&&d| d != -1).product();
            let inferred = if known == 0 {
                numel as i64
            } else {
                numel as i64 / known
            };
            resolved
                .iter()
                .map(|&d| {
                    if d == -1 {
                        inferred as usize
                    } else {
                        d as usize
                    }
                })
                .collect()
        };
        self.alias_or_copy(tensors, &out_name, &node.input[0], new_shape)
    }

    fn op_shape(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let shape_vals: Vec<f32> = {
            let eb = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
            eb.shape().iter().map(|&d| d as f32).collect()
        };
        let n = shape_vals.len();
        let mut buf = self
            .alloc_buf(n)
            .map_err(|e| anyhow!("shape alloc: {}", e))?;
        buf.copy_from_host(&self.stream, &shape_vals)
            .map_err(|e| anyhow!("shape h2d: {:?}", e))?;
        tensors.insert(&out_name, buf, vec![n]);
        Ok(())
    }

    fn op_unsqueeze(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let x_shape = Self::get_tensor(tensors, &self.weights, &node.input[0])?
            .shape()
            .clone();
        // Opset ≤12: axes attribute. Opset ≥13: axes is input[1].
        let mut axes = attr_ints(node, "axes");
        if axes.is_empty() && node.input.len() > 1 && !node.input[1].is_empty() {
            axes = self
                .host_vals(tensors, &node.input[1])?
                .into_iter()
                .map(|v| v as i64)
                .collect();
        }
        let ndim_new = x_shape.len() + axes.len();
        let norm_axes: Vec<i64> = axes
            .iter()
            .map(|&a| if a < 0 { a + ndim_new as i64 } else { a })
            .collect();
        let mut result_shape = vec![0usize; ndim_new];
        let mut src_idx = 0;
        for i in 0..ndim_new {
            if norm_axes.contains(&(i as i64)) {
                result_shape[i] = 1;
            } else {
                result_shape[i] = x_shape[src_idx];
                src_idx += 1;
            }
        }
        self.alias_or_copy(tensors, &out_name, &node.input[0], result_shape)
    }

    fn op_squeeze(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let x_shape = Self::get_tensor(tensors, &self.weights, &node.input[0])?
            .shape()
            .clone();
        // Opset ≤12: axes is an attribute. Opset ≥13: axes is input[1].
        let ndim = x_shape.len() as i64;
        let mut axes = attr_ints(node, "axes");
        if axes.is_empty() && node.input.len() > 1 && !node.input[1].is_empty() {
            if let Ok(vals) = self.host_vals(tensors, &node.input[1]) {
                axes = vals
                    .into_iter()
                    .map(|v| {
                        let a = v as i64;
                        if a < 0 { a + ndim } else { a }
                    })
                    .collect();
            }
        }
        let new_shape: Vec<usize> = x_shape
            .iter()
            .enumerate()
            .filter(|(i, d)| **d != 1 || (!axes.is_empty() && !axes.contains(&(*i as i64))))
            .map(|(_, d)| *d)
            .collect();
        self.alias_or_copy(tensors, &out_name, &node.input[0], new_shape)
    }

    fn op_concat(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let raw_axis = attr_i(node, "axis", 0);

        // Collect input shapes (no data download).
        let shapes: Vec<Vec<usize>> = node
            .input
            .iter()
            .map(|nm| Self::get_tensor(tensors, &self.weights, nm).map(|e| e.shape().clone()))
            .collect::<Result<_>>()?;

        let ndim = shapes[0].len() as i64;
        let axis = if raw_axis < 0 {
            (ndim + raw_axis) as usize
        } else {
            raw_axis as usize
        };

        let mut out_shape = shapes[0].clone();
        for s in &shapes[1..] {
            out_shape[axis] += s[axis];
        }
        let out_axis_len = out_shape[axis];
        let outer: usize = out_shape[..axis].iter().product();
        let inner: usize = out_shape[axis + 1..].iter().product();
        let out_numel: usize = out_shape.iter().product();

        // GPU scatter: each input is copied straight into its axis slot.
        // No D2H/CPU/H2D and no per-input stream sync (the old path stalled
        // the whole pipeline on every Concat — heavy in ViT/BERT/GPT-2).
        let mut out = self
            .alloc_buf(out_numel)
            .map_err(|e| anyhow!("concat alloc: {}", e))?;
        let mut start = 0usize;
        for (inp_name, s) in node.input.iter().zip(shapes.iter()) {
            let sz = s[axis];
            let piece_numel = outer * sz * inner;
            if piece_numel > 0 {
                let ein = Self::get_tensor(tensors, &self.weights, inp_name)?;
                unsafe {
                    self.module.concat_axis(
                        &self.stream,
                        LaunchConfig::for_num_elems(piece_numel as u32),
                        ein.buf(),
                        out_axis_len as u32,
                        inner as u32,
                        start as u32,
                        sz as u32,
                        outer as u32,
                        &mut out,
                    )
                }
                .map_err(|e| anyhow!("concat_axis '{}': {:?}", inp_name, e))?;
            }
            start += sz;
        }
        tensors.insert(&out_name, out, out_shape);
        Ok(())
    }

    fn op_transpose(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let perm_attr = attr_ints(node, "perm");

        let ex = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let x_shape = ex.shape().clone();
        let numel = ex.buf().len();
        let ndim = x_shape.len();

        let perm: Vec<usize> = if perm_attr.is_empty() {
            (0..ndim).rev().collect()
        } else {
            perm_attr.iter().map(|&p| p as usize).collect()
        };
        let out_shape: Vec<usize> = perm.iter().map(|&p| x_shape[p]).collect();

        // Row-major strides for input and output.
        let strides = |s: &[usize]| -> Vec<usize> {
            let mut st = vec![1usize; s.len()];
            for i in (0..s.len().saturating_sub(1)).rev() {
                st[i] = st[i + 1] * s[i + 1];
            }
            st
        };
        let in_strides = strides(&x_shape);
        let out_strides = strides(&out_shape);

        let to_f32 = |v: &[usize]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        let os_buf = self.meta_buf(&to_f32(&out_shape))?;
        let ostr_buf = self.meta_buf(&to_f32(&out_strides))?;
        let istr_buf = self.meta_buf(&to_f32(&in_strides))?;
        let perm_buf = self.meta_buf(&to_f32(&perm))?;

        let mut out = self
            .alloc_buf(numel)
            .map_err(|e| anyhow!("transpose alloc: {}", e))?;
        unsafe {
            self.module.transpose_nd(
                &self.stream,
                LaunchConfig::for_num_elems(numel as u32),
                ex.buf(),
                &os_buf,
                &ostr_buf,
                &istr_buf,
                &perm_buf,
                ndim as u32,
                &mut out,
            )
        }
        .map_err(|e| anyhow!("transpose launch: {:?}", e))?;
        drop(ex);

        // transpose_nd reads these async; keep alive until the run's final sync.
        tensors.insert(&out_name, out, out_shape);
        Ok(())
    }

    // =========================================================================
    // Constant — inline constant tensor stored in graph attributes
    // =========================================================================
    fn op_constant(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        use crate::model::tensor_to_f32;
        let out_name = node.output[0].clone();
        for attr in &node.attribute {
            if attr.name == "value" {
                if let Some(t) = &attr.t {
                    let data = tensor_to_f32(t).map_err(|e| anyhow!("Constant tensor: {}", e))?;
                    let shape: Vec<usize> = if t.dims.is_empty() {
                        vec![data.len().max(1)]
                    } else {
                        t.dims.iter().map(|&d| d as usize).collect()
                    };
                    let buf = DeviceBuffer::from_host(&self.stream, &data)
                        .map_err(|e| anyhow!("Constant h2d: {:?}", e))?;
                    tensors.insert(&out_name, buf, shape);
                    return Ok(());
                }
            }
            if attr.name == "value_int" {
                let buf = DeviceBuffer::from_host(&self.stream, &[attr.i as f32])
                    .map_err(|e| anyhow!("Constant int h2d: {:?}", e))?;
                tensors.insert(&out_name, buf, vec![1]);
                return Ok(());
            }
            if attr.name == "value_float" {
                let buf = DeviceBuffer::from_host(&self.stream, &[attr.f])
                    .map_err(|e| anyhow!("Constant float h2d: {:?}", e))?;
                tensors.insert(&out_name, buf, vec![1]);
                return Ok(());
            }
        }
        Err(anyhow!(
            "Constant op '{}': no recognized value attribute",
            out_name
        ))
    }

    // =========================================================================
    // Gather — gather elements along an axis using integer indices
    //   data[i0, ..., i_{axis-1}, indices[j0,...], i_{axis+1}, ...]
    // =========================================================================
    fn op_gather(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let axis = attr_i(node, "axis", 0) as usize;

        let ed = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let data_shape = ed.shape().clone();
        let ind_shape = Self::get_tensor(tensors, &self.weights, &node.input[1])?
            .shape()
            .clone();

        let axis_len = data_shape.get(axis).copied().unwrap_or(1);
        let outer: usize = data_shape[..axis].iter().product();
        let inner: usize = data_shape[axis + 1..].iter().product();

        let n_idx: usize = if ind_shape.is_empty() {
            1
        } else {
            ind_shape.iter().product()
        };

        // The indices are already on the device — for BERT and GPT-2 they are
        // the input tokens themselves. Reading them back to normalise negative
        // values cost a D2H copy and a full synchronisation per Gather, 6.1 ms
        // of BERT's 8.3 ms of host dispatch; the kernel does it instead.
        let out_numel = outer * n_idx * inner;
        let mut out = self
            .alloc_buf(out_numel)
            .map_err(|e| anyhow!("gather alloc: {}", e))?;

        let ei = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
        unsafe {
            self.module.gather_axis(
                &self.stream,
                LaunchConfig::for_num_elems(out_numel as u32),
                ed.buf(),
                ei.buf(),
                axis_len as u32,
                inner as u32,
                n_idx as u32,
                &mut out,
            )
        }
        .map_err(|e| anyhow!("gather launch: {:?}", e))?;
        drop(ei);
        drop(ed);

        // Output shape: data_shape[:axis] + ind_shape + data_shape[axis+1:]
        let mut out_shape: Vec<usize> = data_shape[..axis].to_vec();
        if !ind_shape.is_empty() {
            out_shape.extend_from_slice(&ind_shape);
        }
        out_shape.extend_from_slice(&data_shape[axis + 1..]);
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        // gather_axis reads idx_buf async; keep alive until the run's final sync.
        tensors.insert(&out_name, out, out_shape);
        Ok(())
    }
}

// ============================================================================
// EitherBuf: separates the two borrow sources without merging lifetimes.
// ============================================================================

enum EitherBuf<'t, 'w> {
    FromTensors(&'t DeviceBuffer<f32>, &'t Vec<usize>),
    FromWeights(&'w DeviceBuffer<f32>, &'w Vec<usize>),
}

impl<'t, 'w> EitherBuf<'t, 'w> {
    fn buf(&self) -> &DeviceBuffer<f32> {
        match self {
            EitherBuf::FromTensors(b, _) => b,
            EitherBuf::FromWeights(b, _) => b,
        }
    }
    fn shape(&self) -> &Vec<usize> {
        match self {
            EitherBuf::FromTensors(_, s) => s,
            EitherBuf::FromWeights(_, s) => s,
        }
    }
}

// ============================================================================
// CPU helpers (concat, transpose)
// ============================================================================

#[allow(dead_code)]
fn transpose2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

#[allow(dead_code)]
fn cpu_concat(parts: &[(Vec<f32>, Vec<usize>)], axis: usize, out_shape: &[usize]) -> Vec<f32> {
    let out_numel: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; out_numel];
    if out_shape.len() == 2 {
        if axis == 0 {
            let mut offset = 0;
            for (data, _) in parts {
                out[offset..offset + data.len()].copy_from_slice(data);
                offset += data.len();
            }
        } else {
            let n_cols_out = out_shape[1];
            let n_rows = out_shape[0];
            for row in 0..n_rows {
                let mut col_offset = row * n_cols_out;
                for (data, shape) in parts {
                    let part_cols = shape[1];
                    out[col_offset..col_offset + part_cols]
                        .copy_from_slice(&data[row * part_cols..(row + 1) * part_cols]);
                    col_offset += part_cols;
                }
            }
        }
    } else {
        let mut offset = 0;
        for (data, _) in parts {
            out[offset..offset + data.len()].copy_from_slice(data);
            offset += data.len();
        }
    }
    out
}
