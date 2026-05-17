/*!
 * OnnxExecutor: loads an ONNX model and dispatches each graph node to the
 * corresponding CUDA kernel(s).
 *
 * Conv2D is implemented via im2col + GEMM.
 * All other ops map to a single kernel or a trivial shape manipulation.
 */

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};

use std::mem::ManuallyDrop;

use crate::kernels::gpu;
use crate::model::{
    GraphProto, NodeProto,
    attr_f, attr_i, attr_ints,
    conv2d_output_shape, maxpool_output_shape,
    load_initializers, topological_sort,
    parse_pads, parse_strides, parse_dilations, parse_kernel_shape,
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
}

impl OnnxExecutor {
    /// Create an executor from an already-parsed GraphProto.
    pub fn from_graph(graph: &GraphProto, ctx: Arc<CudaContext>) -> Result<Self> {
        let stream = ctx.default_stream();
        let module = gpu::load(&ctx)
            .map_err(|e| anyhow!("Failed to load CUDA module: {:?}", e))?;

        let host_weights = load_initializers(graph)?;
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

        let nodes = topological_sort(graph)?;

        let input_names = graph.input.iter()
            .filter(|vi| !weights.contains_key(&vi.name))
            .map(|vi| vi.name.clone())
            .collect();

        let output_names = graph.output.iter()
            .map(|vi| vi.name.clone())
            .collect();

        Ok(Self { ctx, stream, module, weights, consts, nodes, input_names, output_names })
    }

    /// Run inference on a map of input tensors.
    pub fn run(
        &self,
        inputs: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>> {
        let mut tensors = TensorMap::new();

        for (name, (data, shape)) in inputs {
            let buf = DeviceBuffer::from_host(&self.stream, data)
                .map_err(|e| anyhow!("H2D input '{}': {:?}", name, e))?;
            tensors.insert(name, buf, shape.clone());
        }

        for node in &self.nodes {
            self.dispatch_node(node, &mut tensors)
                .map_err(|e| anyhow!("op {} (inputs={:?}): {}", node.op_type, node.input, e))?;
        }

        let mut outputs = HashMap::new();
        for name in &self.output_names {
            // Resolves alias chains (e.g. a final Reshape) to the real buffer.
            if let Ok(eb) = Self::get_tensor(&tensors, &self.weights, name) {
                let data = eb.buf().to_host_vec(&self.stream)
                    .map_err(|e| anyhow!("D2H output '{}': {:?}", name, e))?;
                outputs.insert(name.clone(), (data, eb.shape().clone()));
            }
        }
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
        let out = DeviceBuffer::<f32>::zeroed(&self.stream, numel)
            .map_err(|e| anyhow!("alias_or_copy alloc: {:?}", e))?;
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
                eprintln!("[oxide_onnx] WARNING: unsupported op '{}' node '{}' — passthrough", other, node.name);
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
        eb.buf().to_host_vec(&self.stream)
            .map_err(|e| anyhow!("host_vals '{}' d2h: {:?}", name, e))
    }

    fn dtod_copy(&self, dst: &DeviceBuffer<f32>, src_ptr: u64, num_bytes: usize) -> Result<()> {
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                dst.cu_deviceptr(), src_ptr,
                num_bytes, self.stream.cu_stream(),
            ).map_err(|e| anyhow!("dtod copy: {:?}", e))
        }
    }

    fn sync(&self) -> Result<()> {
        self.stream.synchronize().map_err(|e| anyhow!("stream sync: {:?}", e))
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
    fn sgemm_mma_cfg(m: usize, n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((n as u32).div_ceil(8), (m as u32).div_ceil(16), 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        }
    }



    // =========================================================================
    // Relu
    // =========================================================================
    fn op_relu(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let (numel, src_ptr, shape) = {
            let eb = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
            (eb.buf().len(), eb.buf().cu_deviceptr(), eb.shape().clone())
        };
        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, numel)
            .map_err(|e| anyhow!("relu alloc: {:?}", e))?;
        self.dtod_copy(&out, src_ptr, numel * 4)?;
        self.module.relu(&self.stream, LaunchConfig::for_num_elems(numel as u32), &mut out)
            .map_err(|e| anyhow!("relu launch: {:?}", e))?;
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
        if node.input.len() > 1 && !node.input[1].is_empty() {
            if let Ok(t) = Self::get_tensor(tensors, &self.weights, &node.input[1]) {
                let v = t.buf().to_host_vec(&self.stream)
                    .map_err(|e| anyhow!("clip min d2h: {:?}", e))?;
                if !v.is_empty() { lo = v[0]; }
            }
        }
        if node.input.len() > 2 && !node.input[2].is_empty() {
            if let Ok(t) = Self::get_tensor(tensors, &self.weights, &node.input[2]) {
                let v = t.buf().to_host_vec(&self.stream)
                    .map_err(|e| anyhow!("clip max d2h: {:?}", e))?;
                if !v.is_empty() { hi = v[0]; }
            }
        }

        let (numel, src_ptr, shape) = {
            let eb = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
            (eb.buf().len(), eb.buf().cu_deviceptr(), eb.shape().clone())
        };
        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, numel)
            .map_err(|e| anyhow!("clip alloc: {:?}", e))?;
        self.dtod_copy(&out, src_ptr, numel * 4)?;
        self.module.clip(&self.stream, LaunchConfig::for_num_elems(numel as u32), &mut out, lo, hi)
            .map_err(|e| anyhow!("clip launch: {:?}", e))?;
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
            let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, numel)
                .map_err(|e| anyhow!("add alloc: {:?}", e))?;
            self.module.add_scalar(
                &self.stream, LaunchConfig::for_num_elems(numel as u32),
                tensor, s, &mut out,
            ).map_err(|e| anyhow!("add_scalar launch: {:?}", e))?;
            tensors.insert(&out_name, out, shape);
            return Ok(());
        }
        let a_shape = ea.shape().clone();
        let b_shape = eb.shape().clone();
        // Equal element count AND identical shape → fast elementwise path.
        if na == nb && a_shape == b_shape {
            let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, na)
                .map_err(|e| anyhow!("add alloc: {:?}", e))?;
            self.module.add_elementwise(
                &self.stream, LaunchConfig::for_num_elems(na as u32),
                ea.buf(), eb.buf(), &mut out,
            ).map_err(|e| anyhow!("add launch: {:?}", e))?;
            tensors.insert(&out_name, out, a_shape);
            return Ok(());
        }
        // General NumPy broadcasting (e.g. attention scores + mask).
        let (out_shape, out_str, a_str, b_str, ndim) =
            Self::broadcast_meta(&a_shape, &b_shape)?;
        let out_numel: usize = out_shape.iter().product();
        let osh = DeviceBuffer::from_host(&self.stream, &Self::to_f32(&out_shape))
            .map_err(|e| anyhow!("add bcast osh: {:?}", e))?;
        let ost = DeviceBuffer::from_host(&self.stream, &Self::to_f32(&out_str))
            .map_err(|e| anyhow!("add bcast ost: {:?}", e))?;
        let ast = DeviceBuffer::from_host(&self.stream, &Self::to_f32(&a_str))
            .map_err(|e| anyhow!("add bcast ast: {:?}", e))?;
        let bst = DeviceBuffer::from_host(&self.stream, &Self::to_f32(&b_str))
            .map_err(|e| anyhow!("add bcast bst: {:?}", e))?;
        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, out_numel)
            .map_err(|e| anyhow!("add bcast alloc: {:?}", e))?;
        self.module.add_bcast(
            &self.stream, LaunchConfig::for_num_elems(out_numel as u32),
            ea.buf(), eb.buf(), &osh, &ost, &ast, &bst, ndim as u32, &mut out,
        ).map_err(|e| anyhow!("add_bcast launch: {:?}", e))?;
        drop(ea);
        drop(eb);
        // add_bcast reads these async — keep alive until the run's final sync.
        tensors.push_scratch(osh);
        tensors.push_scratch(ost);
        tensors.push_scratch(ast);
        tensors.push_scratch(bst);
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
        (0..ndim).map(|i| if padded[i] == 1 { 0 } else { full[i] }).collect()
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
            let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, numel)
                .map_err(|e| anyhow!("mul alloc: {:?}", e))?;
            self.module.mul_scalar(
                &self.stream, LaunchConfig::for_num_elems(numel as u32),
                tensor, s, &mut out,
            ).map_err(|e| anyhow!("mul_scalar launch: {:?}", e))?;
            tensors.insert(&out_name, out, shape);
            return Ok(());
        }
        let numel = na;
        let shape = ea.shape().clone();
        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, numel)
            .map_err(|e| anyhow!("mul alloc: {:?}", e))?;
        self.module.mul_elementwise(
            &self.stream, LaunchConfig::for_num_elems(numel as u32),
            ea.buf(), eb.buf(), &mut out,
        ).map_err(|e| anyhow!("mul launch: {:?}", e))?;
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
        let (out_h, out_w) = conv2d_output_shape(h_in, w_in, kh, kw, pad_h, pad_w, stride_h, stride_w, dil_h, dil_w);

        let col_rows_g = c_in_per_group * kh * kw;
        let col_cols   = out_h * out_w;
        let out_numel  = batch_n * n_out * out_h * out_w;

        // Pre-allocate output on GPU (zeroed, written in-place per group).
        let mut result_buf = DeviceBuffer::<f32>::zeroed(&self.stream, out_numel)
            .map_err(|e| anyhow!("conv result alloc: {:?}", e))?;

        // ── Fast path: depthwise conv (group == c_in, 1 input chan/group) ──
        // Collapses the O(group) im2col+GEMM launches into one fused kernel.
        let is_depthwise = c_in_per_group == 1 && group == c_in && group > 1;
        if is_depthwise {
            let x_full = ManuallyDrop::new(unsafe {
                DeviceBuffer::<f32>::from_raw_parts(
                    x_ptr, batch_n * c_in * h_in * w_in, self.ctx.clone(),
                )
            });
            let w_full = ManuallyDrop::new(unsafe {
                DeviceBuffer::<f32>::from_raw_parts(
                    w_ptr, n_out * c_in_per_group * kh * kw, self.ctx.clone(),
                )
            });
            self.module.depthwise_conv2d(
                &self.stream, LaunchConfig::for_num_elems(out_numel as u32),
                &*x_full, &*w_full,
                c_in as u32, h_in as u32, w_in as u32,
                n_out as u32, c_out_per_group as u32,
                kh as u32, kw as u32,
                pad_h as u32, pad_w as u32,
                stride_h as u32, stride_w as u32,
                dil_h as u32, dil_w as u32,
                out_h as u32, out_w as u32,
                &mut result_buf,
            ).map_err(|e| anyhow!("conv depthwise: {:?}", e))?;
        } else {

        // Single col buffer reused across group iterations.
        let mut col_dev = DeviceBuffer::<f32>::zeroed(&self.stream, col_rows_g * col_cols)
            .map_err(|e| anyhow!("conv col alloc: {:?}", e))?;

        for b in 0..batch_n {
            for g in 0..group {
                // ── im2col on GPU sub-view of x (no PCIe transfer) ─────────
                let in_offset = b * c_in * h_in * w_in + g * c_in_per_group * h_in * w_in;
                let in_len    = c_in_per_group * h_in * w_in;
                // ManuallyDrop prevents Drop from freeing the borrowed pointer.
                let x_g = ManuallyDrop::new(unsafe {
                    DeviceBuffer::<f32>::from_raw_parts(
                        x_ptr + (in_offset * 4) as u64, in_len, self.ctx.clone(),
                    )
                });

                self.module.im2col(
                    &self.stream, LaunchConfig::for_num_elems((col_rows_g * col_cols) as u32),
                    &*x_g,
                    c_in_per_group as u32, h_in as u32, w_in as u32,
                    kh as u32, kw as u32,
                    pad_h as u32, pad_w as u32,
                    stride_h as u32, stride_w as u32,
                    dil_h as u32, dil_w as u32,
                    out_h as u32, out_w as u32,
                    &mut col_dev,
                ).map_err(|e| anyhow!("conv im2col g={}: {:?}", g, e))?;

                // ── Tiled SGEMM: W_g [c_out_g × col_rows_g] × col → out_g ──
                let w_offset  = g * c_out_per_group * c_in_per_group * kh * kw;
                let out_offset = b * n_out * out_h * out_w + g * c_out_per_group * out_h * out_w;

                // GPU sub-views of the weight slice and output slice (no PCIe).
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

                self.module.sgemm_mma(
                    &self.stream,
                    Self::sgemm_mma_cfg(c_out_per_group, col_cols),
                    c_out_per_group as u32, col_cols as u32, col_rows_g as u32,
                    1.0,
                    &*w_g, &col_dev,
                    0.0,
                    &mut *out_g,
                ).map_err(|e| anyhow!("conv sgemm g={}: {:?}", g, e))?;
            }
        }

        // sgemm_tiled reads col_dev asynchronously; keep it alive until the
        // run's final sync (DeviceBuffer::drop is an immediate cuMemFree).
        tensors.push_scratch(col_dev);

        } // end !is_depthwise

        // Optional bias (3rd input): x[n,c,h,w] += bias[c]
        if node.input.len() > 2 && !node.input[2].is_empty() {
            let eb = Self::get_tensor(tensors, &self.weights, &node.input[2])?;
            self.module.bias_add(
                &self.stream, LaunchConfig::for_num_elems(out_numel as u32),
                &mut result_buf, eb.buf(), (out_h * out_w) as u32, n_out as u32,
            ).map_err(|e| anyhow!("conv bias: {:?}", e))?;
        }

        tensors.insert(&out_name, result_buf, vec![batch_n, n_out, out_h, out_w]);
        Ok(())
    }

    // =========================================================================
    // BatchNormalization
    // =========================================================================
    fn op_batchnorm(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let eps = attr_f(node, "epsilon", 1e-5);

        // All five inputs are live during the kernel call; NLL ends borrows after.
        let ex     = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let egamma = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
        let ebeta  = Self::get_tensor(tensors, &self.weights, &node.input[2])?;
        let emean  = Self::get_tensor(tensors, &self.weights, &node.input[3])?;
        let evar   = Self::get_tensor(tensors, &self.weights, &node.input[4])?;

        let x_shape = ex.shape().clone();
        let numel = ex.buf().len();
        let n = x_shape[0];
        let c = x_shape[1];
        let hw = if x_shape.len() >= 4 { x_shape[2] * x_shape[3] } else { 1 };

        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, numel)
            .map_err(|e| anyhow!("batchnorm alloc: {:?}", e))?;

        self.module.batch_norm_inference(
            &self.stream, LaunchConfig::for_num_elems(numel as u32),
            ex.buf(), egamma.buf(), ebeta.buf(), emean.buf(), evar.buf(),
            eps, n as u32, c as u32, hw as u32,
            &mut out,
        ).map_err(|e| anyhow!("batchnorm launch: {:?}", e))?;
        // NLL: borrows on ex, egamma, ... (from tensors/weights) end here

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
        let (out_h, out_w) = maxpool_output_shape(in_h, in_w, kh, kw, pad_h, pad_w, stride_h, stride_w);
        let out_numel = n * c * out_h * out_w;

        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, out_numel)
            .map_err(|e| anyhow!("maxpool alloc: {:?}", e))?;

        self.module.maxpool2d(
            &self.stream, LaunchConfig::for_num_elems(out_numel as u32),
            ex.buf(),
            c as u32, in_h as u32, in_w as u32,
            kh as u32, kw as u32,
            pad_h as u32, pad_w as u32,
            stride_h as u32, stride_w as u32,
            out_h as u32, out_w as u32,
            &mut out,
        ).map_err(|e| anyhow!("maxpool launch: {:?}", e))?;

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
        let hw = if x_shape.len() >= 4 { x_shape[2] * x_shape[3] } else { 1 };

        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, n * c)
            .map_err(|e| anyhow!("gavgpool alloc: {:?}", e))?;

        self.module.global_avg_pool(
            &self.stream, LaunchConfig::for_num_elems((n * c) as u32),
            ex.buf(), c as u32, hw as u32, &mut out,
        ).map_err(|e| anyhow!("gavgpool launch: {:?}", e))?;

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
        let alpha    = attr_f(node, "alpha", 1.0);
        let beta_val = attr_f(node, "beta",  1.0);
        let trans_a  = attr_i(node, "transA", 0) != 0;
        let trans_b  = attr_i(node, "transB", 0) != 0;

        if trans_a {
            return Err(anyhow!("Gemm transA=1 not supported by tiled SGEMM"));
        }

        let ea = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let eb = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
        let a_shape = ea.shape().clone();
        let b_shape = eb.shape().clone();

        let (m, k_a) = (a_shape[0], a_shape[1]);
        let (k_b, n) = if trans_b { (b_shape[1], b_shape[0]) } else { (b_shape[0], b_shape[1]) };
        if k_a != k_b { return Err(anyhow!("Gemm k mismatch: k_a={} k_b={}", k_a, k_b)); }
        if m == 0 || n == 0 || k_a == 0 {
            return Err(anyhow!("Gemm degenerate dims m={} n={} k={}", m, n, k_a));
        }

        let out_numel = m * n;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&self.stream, out_numel)
            .map_err(|e| anyhow!("gemm alloc: {:?}", e))?;

        // C = alpha * A * op(B)  (beta=0, bias added separately below)
        let cfg = Self::sgemm_cfg(m, n);
        if trans_b {
            self.module.sgemm_transb_tiled(
                &self.stream, cfg,
                m as u32, n as u32, k_a as u32,
                alpha, ea.buf(), eb.buf(), 0.0, &mut out_dev,
            ).map_err(|e| anyhow!("gemm sgemm_transb: {:?}", e))?;
        } else {
            self.module.sgemm_mma(
                &self.stream, Self::sgemm_mma_cfg(m, n),
                m as u32, n as u32, k_a as u32,
                alpha, ea.buf(), eb.buf(), 0.0, &mut out_dev,
            ).map_err(|e| anyhow!("gemm sgemm: {:?}", e))?;
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
                    self.module.bias_add(
                        &self.stream, LaunchConfig::for_num_elems(out_numel as u32),
                        &mut out_dev, eb.buf(), 1u32, n as u32,
                    ).map_err(|e| anyhow!("gemm bias_add: {:?}", e))?;
                } else {
                    // Non-unit beta: scale bias on host (C is small, e.g. 1000 floats)
                    let c_host = eb.buf().to_host_vec(&self.stream)
                        .map_err(|e| anyhow!("gemm c d2h: {:?}", e))?;
                    let scaled: Vec<f32> = c_host.iter().map(|&v| v * beta_val).collect();
                    let scaled_dev = DeviceBuffer::from_host(&self.stream, &scaled)
                        .map_err(|e| anyhow!("gemm scaled c h2d: {:?}", e))?;
                    self.module.bias_add(
                        &self.stream, LaunchConfig::for_num_elems(out_numel as u32),
                        &mut out_dev, &scaled_dev, 1u32, n as u32,
                    ).map_err(|e| anyhow!("gemm scaled bias_add: {:?}", e))?;
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
                batch, b_batch, a_shape, b_shape
            ));
        }

        let mut out_shape = a_shape[..a_shape.len() - 2].to_vec();
        out_shape.extend([m, n]);

        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, batch * m * n)
            .map_err(|e| anyhow!("matmul alloc: {:?}", e))?;
        let out_ptr = out.cu_deviceptr();

        for bi in 0..batch {
            let a_g = ManuallyDrop::new(unsafe {
                DeviceBuffer::<f32>::from_raw_parts(
                    a_ptr + (bi * m * k * 4) as u64, m * k, self.ctx.clone(),
                )
            });
            let b_g = ManuallyDrop::new(unsafe {
                DeviceBuffer::<f32>::from_raw_parts(
                    b_ptr + (bi * k * n * 4) as u64, k * n, self.ctx.clone(),
                )
            });
            let mut c_g = ManuallyDrop::new(unsafe {
                DeviceBuffer::<f32>::from_raw_parts(
                    out_ptr + (bi * m * n * 4) as u64, m * n, self.ctx.clone(),
                )
            });
            self.module.sgemm_mma(
                &self.stream, Self::sgemm_mma_cfg(m, n),
                m as u32, n as u32, k as u32,
                1.0, &*a_g, &*b_g, 0.0, &mut *c_g,
            ).map_err(|e| anyhow!("matmul sgemm bi={}: {:?}", bi, e))?;
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
        let ax = if axis < 0 { (ndim + axis) as usize } else { axis as usize };
        let cols = x_shape[ax];
        let rows = numel / cols;

        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, numel)
            .map_err(|e| anyhow!("softmax alloc: {:?}", e))?;

        self.module.softmax_row(
            &self.stream, LaunchConfig::for_num_elems(rows as u32),
            ex.buf(), rows as u32, cols as u32, &mut out,
        ).map_err(|e| anyhow!("softmax launch: {:?}", e))?;

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

        let ex     = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
        let egamma = Self::get_tensor(tensors, &self.weights, &node.input[1])?;
        let ebeta  = Self::get_tensor(tensors, &self.weights, &node.input[2])?;

        let x_shape = ex.shape().clone();
        let numel = ex.buf().len();
        let ndim = x_shape.len() as i64;
        let ax = if axis < 0 { (ndim + axis) as usize } else { axis as usize };
        // Normalize over the product of dims [ax..]; rows = everything before.
        let cols: usize = x_shape[ax..].iter().product();
        let rows = numel / cols;

        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, numel)
            .map_err(|e| anyhow!("layernorm alloc: {:?}", e))?;

        self.module.layernorm(
            &self.stream, LaunchConfig::for_num_elems(rows as u32),
            ex.buf(), egamma.buf(), ebeta.buf(),
            rows as u32, cols as u32, eps,
            &mut out,
        ).map_err(|e| anyhow!("layernorm launch: {:?}", e))?;

        tensors.insert(&out_name, out, x_shape);
        Ok(())
    }

    // =========================================================================
    // Erf — element-wise error function (component of exact GELU)
    // =========================================================================
    fn op_erf(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let (numel, src_ptr, shape) = {
            let eb = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
            (eb.buf().len(), eb.buf().cu_deviceptr(), eb.shape().clone())
        };
        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, numel)
            .map_err(|e| anyhow!("erf alloc: {:?}", e))?;
        self.dtod_copy(&out, src_ptr, numel * 4)?;
        self.module.erf_inplace(
            &self.stream, LaunchConfig::for_num_elems(numel as u32), &mut out,
        ).map_err(|e| anyhow!("erf launch: {:?}", e))?;
        tensors.insert(&out_name, out, shape);
        Ok(())
    }

    // =========================================================================
    // Tanh — element-wise (BERT pooler)
    // =========================================================================
    fn op_tanh(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let (numel, src_ptr, shape) = {
            let eb = Self::get_tensor(tensors, &self.weights, &node.input[0])?;
            (eb.buf().len(), eb.buf().cu_deviceptr(), eb.shape().clone())
        };
        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, numel)
            .map_err(|e| anyhow!("tanh alloc: {:?}", e))?;
        self.dtod_copy(&out, src_ptr, numel * 4)?;
        self.module.tanh_inplace(
            &self.stream, LaunchConfig::for_num_elems(numel as u32), &mut out,
        ).map_err(|e| anyhow!("tanh launch: {:?}", e))?;
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
            let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, nb)
                .map_err(|e| anyhow!("sub alloc: {:?}", e))?;
            self.module.sub_scalar_lhs(
                &self.stream, LaunchConfig::for_num_elems(nb as u32),
                eb.buf(), s, &mut out,
            ).map_err(|e| anyhow!("sub_scalar_lhs launch: {:?}", e))?;
            tensors.insert(&out_name, out, shape);
            return Ok(());
        }
        if nb == 1 && na > 1 {
            let s = self.host_vals(tensors, &node.input[1])?[0];
            let shape = ea.shape().clone();
            let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, na)
                .map_err(|e| anyhow!("sub alloc: {:?}", e))?;
            self.module.add_scalar(
                &self.stream, LaunchConfig::for_num_elems(na as u32),
                ea.buf(), -s, &mut out,
            ).map_err(|e| anyhow!("sub add_scalar launch: {:?}", e))?;
            tensors.insert(&out_name, out, shape);
            return Ok(());
        }
        Err(anyhow!("Sub: only scalar±tensor supported (na={} nb={})", na, nb))
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
        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, numel)
            .map_err(|e| anyhow!("pow alloc: {:?}", e))?;
        self.module.pow_scalar(
            &self.stream, LaunchConfig::for_num_elems(numel as u32),
            ea.buf(), p, &mut out,
        ).map_err(|e| anyhow!("pow launch: {:?}", e))?;
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

        let osh = DeviceBuffer::from_host(&self.stream, &Self::to_f32(&out_shape))
            .map_err(|e| anyhow!("where osh: {:?}", e))?;
        let osb = DeviceBuffer::from_host(&self.stream, &Self::to_f32(&ost))
            .map_err(|e| anyhow!("where ost: {:?}", e))?;
        let csb = DeviceBuffer::from_host(&self.stream, &Self::to_f32(&c_str))
            .map_err(|e| anyhow!("where cst: {:?}", e))?;
        let xsb = DeviceBuffer::from_host(&self.stream, &Self::to_f32(&x_str))
            .map_err(|e| anyhow!("where xst: {:?}", e))?;
        let ysb = DeviceBuffer::from_host(&self.stream, &Self::to_f32(&y_str))
            .map_err(|e| anyhow!("where yst: {:?}", e))?;
        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, out_numel)
            .map_err(|e| anyhow!("where alloc: {:?}", e))?;
        self.module.where_bcast(
            &self.stream, LaunchConfig::for_num_elems(out_numel as u32),
            ec.buf(), ex.buf(), ey.buf(),
            &osh, &osb, &csb, &xsb, &ysb, ndim as u32, &mut out,
        ).map_err(|e| anyhow!("where launch: {:?}", e))?;
        drop(ec);
        drop(ex);
        drop(ey);
        tensors.push_scratch(osh);
        tensors.push_scratch(osb);
        tensors.push_scratch(csb);
        tensors.push_scratch(xsb);
        tensors.push_scratch(ysb);
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
            let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, na)
                .map_err(|e| anyhow!("div alloc: {:?}", e))?;
            self.module.mul_scalar(
                &self.stream, LaunchConfig::for_num_elems(na as u32),
                ea.buf(), 1.0 / s, &mut out,
            ).map_err(|e| anyhow!("div mul_scalar launch: {:?}", e))?;
            tensors.insert(&out_name, out, shape);
            return Ok(());
        }
        Err(anyhow!("Div: only tensor÷scalar supported (na={} nb={})", na, nb))
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
        let axis = if axis_i < 0 { (ndim + axis_i) as usize } else { axis_i as usize };
        let axis_len = x_shape[axis];
        let n_out = node.output.len();

        // Split sizes: opset ≥13 from input[1] (small tensor), else equal parts.
        let sizes: Vec<usize> = if node.input.len() > 1 && !node.input[1].is_empty() {
            self.host_vals(tensors, &node.input[1])?
                .into_iter().map(|v| v as usize).collect()
        } else {
            vec![axis_len / n_out; n_out]
        };

        let outer: usize = x_shape[..axis].iter().product();
        let inner: usize = x_shape[axis + 1..].iter().product();

        // Non-owning GPU view of the whole input (pointer offset, no-op Drop).
        let in_view = ManuallyDrop::new(unsafe {
            DeviceBuffer::<f32>::from_raw_parts(
                in_ptr, outer * axis_len * inner, self.ctx.clone(),
            )
        });

        let mut start = 0usize;
        for (oi, &sz) in sizes.iter().enumerate() {
            let piece_numel = outer * sz * inner;
            let mut piece = DeviceBuffer::<f32>::zeroed(&self.stream, piece_numel)
                .map_err(|e| anyhow!("split alloc: {:?}", e))?;
            self.module.slice_axis(
                &self.stream, LaunchConfig::for_num_elems(piece_numel as u32),
                &*in_view,
                axis_len as u32, inner as u32, start as u32, sz as u32,
                &mut piece,
            ).map_err(|e| anyhow!("split slice_axis oi={}: {:?}", oi, e))?;
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
        if node.input.is_empty() || node.output.is_empty() { return Ok(()); }
        let out_name = node.output[0].clone();
        let shape = Self::get_tensor(tensors, &self.weights, &node.input[0])?
            .shape().clone();
        self.alias_or_copy(tensors, &out_name, &node.input[0], shape)
    }

    fn op_flatten(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let axis = attr_i(node, "axis", 1) as usize;
        let x_shape = Self::get_tensor(tensors, &self.weights, &node.input[0])?
            .shape().clone();
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
                .into_iter().map(|v| v as i64).collect()
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
            let resolved: Vec<i64> = new_shape_raw.iter().enumerate().map(|(i, &d)| {
                if d == 0 && !allow_zero {
                    x_shape.get(i).map(|&v| v as i64).unwrap_or(0)
                } else {
                    d
                }
            }).collect();
            let known: i64 = resolved.iter().filter(|&&d| d != -1).product();
            let inferred = if known == 0 { numel as i64 } else { numel as i64 / known };
            resolved.iter().map(|&d| {
                if d == -1 { inferred as usize } else { d as usize }
            }).collect()
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
        let buf = DeviceBuffer::from_host(&self.stream, &shape_vals)
            .map_err(|e| anyhow!("shape h2d: {:?}", e))?;
        tensors.insert(&out_name, buf, vec![n]);
        Ok(())
    }

    fn op_unsqueeze(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let x_shape = Self::get_tensor(tensors, &self.weights, &node.input[0])?
            .shape().clone();
        // Opset ≤12: axes attribute. Opset ≥13: axes is input[1].
        let mut axes = attr_ints(node, "axes");
        if axes.is_empty() && node.input.len() > 1 && !node.input[1].is_empty() {
            axes = self.host_vals(tensors, &node.input[1])?
                .into_iter().map(|v| v as i64).collect();
        }
        let ndim_new = x_shape.len() + axes.len();
        let norm_axes: Vec<i64> = axes.iter()
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
            .shape().clone();
        // Opset ≤12: axes is an attribute. Opset ≥13: axes is input[1].
        let ndim = x_shape.len() as i64;
        let mut axes = attr_ints(node, "axes");
        if axes.is_empty() && node.input.len() > 1 && !node.input[1].is_empty() {
            if let Ok(vals) = self.host_vals(tensors, &node.input[1]) {
                axes = vals.into_iter()
                    .map(|v| { let a = v as i64; if a < 0 { a + ndim } else { a } })
                    .collect();
            }
        }
        let new_shape: Vec<usize> = x_shape.iter().enumerate()
            .filter(|(i, d)| **d != 1 || (!axes.is_empty() && !axes.contains(&(*i as i64))))
            .map(|(_, d)| *d)
            .collect();
        self.alias_or_copy(tensors, &out_name, &node.input[0], new_shape)
    }

    fn op_concat(&self, node: &NodeProto, tensors: &mut TensorMap) -> Result<()> {
        let out_name = node.output[0].clone();
        let axis = attr_i(node, "axis", 0) as usize;

        let mut parts: Vec<(Vec<f32>, Vec<usize>)> = Vec::new();
        for inp_name in &node.input {
            let (data, shape) = {
                let eb = Self::get_tensor(tensors, &self.weights, inp_name)?;
                let d = eb.buf().to_host_vec(&self.stream)
                    .map_err(|e| anyhow!("concat d2h '{}': {:?}", inp_name, e))?;
                (d, eb.shape().clone())
            };
            parts.push((data, shape));
        }

        let mut out_shape = parts[0].1.clone();
        for (_, s) in &parts[1..] { out_shape[axis] += s[axis]; }

        let result = cpu_concat(&parts, axis, &out_shape);
        let buf = DeviceBuffer::from_host(&self.stream, &result)
            .map_err(|e| anyhow!("concat h2d: {:?}", e))?;
        tensors.insert(&out_name, buf, out_shape);
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
            for i in (0..s.len().saturating_sub(1)).rev() { st[i] = st[i + 1] * s[i + 1]; }
            st
        };
        let in_strides = strides(&x_shape);
        let out_strides = strides(&out_shape);

        let to_f32 = |v: &[usize]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        let os_buf = DeviceBuffer::from_host(&self.stream, &to_f32(&out_shape))
            .map_err(|e| anyhow!("transpose os h2d: {:?}", e))?;
        let ostr_buf = DeviceBuffer::from_host(&self.stream, &to_f32(&out_strides))
            .map_err(|e| anyhow!("transpose ostr h2d: {:?}", e))?;
        let istr_buf = DeviceBuffer::from_host(&self.stream, &to_f32(&in_strides))
            .map_err(|e| anyhow!("transpose istr h2d: {:?}", e))?;
        let perm_buf = DeviceBuffer::from_host(&self.stream, &to_f32(&perm))
            .map_err(|e| anyhow!("transpose perm h2d: {:?}", e))?;

        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, numel)
            .map_err(|e| anyhow!("transpose alloc: {:?}", e))?;
        self.module.transpose_nd(
            &self.stream, LaunchConfig::for_num_elems(numel as u32),
            ex.buf(), &os_buf, &ostr_buf, &istr_buf, &perm_buf,
            ndim as u32, &mut out,
        ).map_err(|e| anyhow!("transpose launch: {:?}", e))?;
        drop(ex);

        // transpose_nd reads these async; keep alive until the run's final sync.
        tensors.push_scratch(os_buf);
        tensors.push_scratch(ostr_buf);
        tensors.push_scratch(istr_buf);
        tensors.push_scratch(perm_buf);
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
        Err(anyhow!("Constant op '{}': no recognized value attribute", out_name))
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
            .shape().clone();

        let axis_len = data_shape.get(axis).copied().unwrap_or(1);
        let outer: usize = data_shape[..axis].iter().product();
        let inner: usize = data_shape[axis + 1..].iter().product();

        // Indices are constant in practice (no D2H sync via the const cache).
        let indices: Vec<f32> = self.host_vals(tensors, &node.input[1])?
            .iter()
            .map(|&v| {
                let i = v as i64;
                if i < 0 { (axis_len as i64 + i) as f32 } else { i as f32 }
            })
            .collect();
        let n_idx = if ind_shape.is_empty() { 1 } else { ind_shape.iter().product() };

        let idx_buf = DeviceBuffer::from_host(&self.stream, &indices)
            .map_err(|e| anyhow!("gather idx h2d: {:?}", e))?;
        let out_numel = outer * n_idx * inner;
        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, out_numel)
            .map_err(|e| anyhow!("gather alloc: {:?}", e))?;

        self.module.gather_axis(
            &self.stream, LaunchConfig::for_num_elems(out_numel as u32),
            ed.buf(), &idx_buf,
            axis_len as u32, inner as u32, n_idx as u32,
            &mut out,
        ).map_err(|e| anyhow!("gather launch: {:?}", e))?;
        drop(ed);

        // Output shape: data_shape[:axis] + ind_shape + data_shape[axis+1:]
        let mut out_shape: Vec<usize> = data_shape[..axis].to_vec();
        if !ind_shape.is_empty() { out_shape.extend_from_slice(&ind_shape); }
        out_shape.extend_from_slice(&data_shape[axis + 1..]);
        if out_shape.is_empty() { out_shape.push(1); }

        // gather_axis reads idx_buf async; keep alive until the run's final sync.
        tensors.push_scratch(idx_buf);
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

fn transpose2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

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
