/*!
 * Tensor abstraction that can live on host (Vec<f32>) or device (DeviceBuffer<f32>).
 */

use anyhow::Result;
use cuda_core::{CudaStream, DeviceBuffer};

// ============================================================================
// Tensor
// ============================================================================

#[derive(Clone)]
pub struct Tensor {
    pub data: TensorData,
    pub shape: Vec<usize>,
}

#[derive(Clone)]
pub enum TensorData {
    Host(Vec<f32>),
    Device(std::sync::Arc<DeviceBuffer<f32>>),
}

impl Tensor {
    pub fn from_host(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data: TensorData::Host(data),
            shape,
        }
    }

    pub fn from_device(buf: DeviceBuffer<f32>, shape: Vec<usize>) -> Self {
        Self {
            data: TensorData::Device(std::sync::Arc::new(buf)),
            shape,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Upload to device if not already there; returns a reference to the DeviceBuffer.
    pub fn to_device_buf(&self, stream: &CudaStream) -> Result<DeviceBuffer<f32>> {
        match &self.data {
            TensorData::Host(v) => Ok(DeviceBuffer::from_host(stream, v)
                .map_err(|e| anyhow::anyhow!("H2D failed: {:?}", e))?),
            TensorData::Device(arc) => {
                // Clone the device buffer by copying D2D
                let n = arc.len();
                let new_buf = DeviceBuffer::<f32>::zeroed(stream, n)
                    .map_err(|e| anyhow::anyhow!("alloc failed: {:?}", e))?;
                unsafe {
                    cuda_core::memory::memcpy_dtod_async(
                        new_buf.cu_deviceptr(),
                        arc.cu_deviceptr(),
                        n * std::mem::size_of::<f32>(),
                        stream.cu_stream(),
                    )
                    .map_err(|e| anyhow::anyhow!("D2D copy failed: {:?}", e))?;
                }
                Ok(new_buf)
            }
        }
    }

    /// Download from device to host Vec<f32>.
    pub fn to_host_vec(&self, stream: &CudaStream) -> Result<Vec<f32>> {
        match &self.data {
            TensorData::Host(v) => Ok(v.clone()),
            TensorData::Device(arc) => arc
                .to_host_vec(stream)
                .map_err(|e| anyhow::anyhow!("D2H failed: {:?}", e)),
        }
    }

    pub fn shape_str(&self) -> String {
        let parts: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        format!("[{}]", parts.join(", "))
    }
}

// ============================================================================
// TensorMap — name → DeviceBuffer + shape, used during graph execution
// ============================================================================

pub struct TensorMap {
    pub map: std::collections::HashMap<String, (DeviceBuffer<f32>, Vec<usize>)>,
    /// Zero-copy views: alias name → (target name, view shape). Metadata ops
    /// (Reshape/Squeeze/Flatten/Unsqueeze/Dropout/Cast) only change shape, not
    /// data, so they record an alias instead of allocating + copying a buffer.
    /// ONNX is SSA (each name assigned once, never freed mid-run), so the
    /// target buffer outlives every alias to it.
    pub alias: std::collections::HashMap<String, (String, Vec<usize>)>,
    /// Transient buffers (e.g. conv im2col scratch, small index/stride uploads)
    /// that an *async* kernel consumes. `DeviceBuffer::drop` is a synchronous
    /// `cuMemFree`, so dropping such a buffer at op-return is a use-after-free
    /// if its kernel is still queued. Parking them here keeps them alive until
    /// run() finishes (the final output D2H drains the stream first).
    pub scratch: Vec<DeviceBuffer<f32>>,
}

impl TensorMap {
    pub fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
            alias: std::collections::HashMap::new(),
            scratch: Vec::new(),
        }
    }

    /// Keep a transient kernel-consumed buffer alive until run() ends.
    pub fn push_scratch(&mut self, buf: DeviceBuffer<f32>) {
        self.scratch.push(buf);
    }

    pub fn insert(&mut self, name: &str, buf: DeviceBuffer<f32>, shape: Vec<usize>) {
        self.map.insert(name.to_string(), (buf, shape));
    }

    /// Record a zero-copy view of `target` (which may itself be an alias).
    pub fn insert_alias(&mut self, name: &str, target: &str, shape: Vec<usize>) {
        self.alias
            .insert(name.to_string(), (target.to_string(), shape));
    }

    /// Follow an alias chain to the real owning name; returns (owner, view shape).
    /// The view shape is the alias entry for the *queried* name.
    pub fn resolve_alias(&self, name: &str) -> Option<(&str, &Vec<usize>)> {
        let first = self.alias.get(name)?;
        let shape = &first.1;
        let mut owner: &str = first.0.as_str();
        while let Some(next) = self.alias.get(owner) {
            owner = next.0.as_str();
        }
        Some((owner, shape))
    }

    pub fn get(&self, name: &str) -> Option<(&DeviceBuffer<f32>, &Vec<usize>)> {
        self.map.get(name).map(|(b, s)| (b, s))
    }

    pub fn get_shape(&self, name: &str) -> Option<&Vec<usize>> {
        self.map.get(name).map(|(_, s)| s)
    }

    pub fn contains(&self, name: &str) -> bool {
        !name.is_empty() && (self.map.contains_key(name) || self.alias.contains_key(name))
    }
}
