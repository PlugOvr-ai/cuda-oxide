/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA graphs: record a stream's work once, replay it with one host call.
//!
//! A launch costs the host a few microseconds of driver work regardless of how
//! small the kernel is. A model that dispatches hundreds of short kernels per
//! step can therefore leave the GPU idle waiting for the host to decide what
//! runs next — the gap is invisible to per-kernel timing, which only measures
//! the kernels that did run.
//!
//! Stream capture records that whole sequence, dependencies included, into a
//! [`CudaGraph`]. Instantiating it yields a [`CudaGraphExec`] that replays
//! every node from a single `cuGraphLaunch`.
//!
//! # What capture requires
//!
//! Capture records *work*, not *decisions*: the graph replays exactly the
//! launches that were captured, with exactly the arguments they were captured
//! with. So a capturable step must have
//!
//! - **stable buffers** — device pointers are baked into the graph, so
//!   anything reallocated between steps invalidates it;
//! - **stable launch geometry** — grid and block dimensions are baked in too;
//! - **no synchronisation** — a `cuStreamSynchronize` or a device-to-host copy
//!   inside the captured region fails the capture rather than being recorded;
//! - **no host-visible branching on device data** — a value the host reads
//!   back to decide the next launch cannot be captured. Values that vary per
//!   step have to live in device memory and be read by the kernels themselves.
//!
//! Capture uses [`StreamCaptureMode::Relaxed`] here rather than the driver's
//! default `Global`, because `Global` also fails the capture if *any other*
//! thread in the process performs an unsafe CUDA action during the window.
//!
//! # Example
//!
//! ```rust,ignore
//! let graph = CudaGraph::capture(&stream, || {
//!     for node in &nodes { dispatch(node)?; }
//!     Ok(())
//! })?;
//! let exec = graph.instantiate()?;
//! for _ in 0..steps {
//!     exec.launch(&stream)?;   // one host call, all N kernels
//! }
//! stream.synchronize()?;
//! ```

use crate::context::CudaContext;
use crate::error::{DriverError, IntoResult};
use crate::stream::CudaStream;
use std::mem::MaybeUninit;
use std::sync::Arc;

/// How strictly the driver polices actions during an active capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamCaptureMode {
    /// Any potentially unsafe CUDA action anywhere in the process invalidates
    /// the capture.
    Global,
    /// Unsafe actions on other threads are tolerated; this thread must still
    /// stay within the capture rules.
    ThreadLocal,
    /// No cross-thread policing. The right choice when the captured region is
    /// known to be self-contained.
    Relaxed,
}

impl StreamCaptureMode {
    fn raw(self) -> cuda_bindings::CUstreamCaptureMode {
        match self {
            Self::Global => cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_GLOBAL,
            Self::ThreadLocal => {
                cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
            }
            Self::Relaxed => cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED,
        }
    }
}

/// Whether a stream is currently capturing, and whether that capture is still
/// valid.
///
/// Worth querying node by node while bringing a new region under capture: the
/// driver usually reports an invalidated capture at `cuStreamEndCapture`, long
/// after the offending call, so polling this is the practical way to find
/// which operation broke it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureStatus {
    /// Not capturing.
    None,
    /// Capturing normally.
    Active,
    /// Capture was invalidated and will fail at end.
    Invalidated,
}

/// Returns whether `stream` is capturing, and whether the capture is intact.
pub fn capture_status(stream: &CudaStream) -> Result<CaptureStatus, DriverError> {
    stream.context().bind_to_thread()?;
    let mut status = 0;
    unsafe { cuda_bindings::cuStreamIsCapturing(stream.cu_stream(), &mut status) }.result()?;
    Ok(
        if status == cuda_bindings::CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_ACTIVE {
            CaptureStatus::Active
        } else if status
            == cuda_bindings::CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_INVALIDATED
        {
            CaptureStatus::Invalidated
        } else {
            CaptureStatus::None
        },
    )
}

/// A captured, not-yet-executable graph of stream work.
///
/// Destroyed via `cuGraphDestroy` on [`Drop`]. Instantiating consumes nothing:
/// one graph can produce several [`CudaGraphExec`]s.
#[derive(Debug)]
pub struct CudaGraph {
    cu_graph: cuda_bindings::CUgraph,
    ctx: Arc<CudaContext>,
}

/// # Safety
///
/// `CUgraph` handles are not thread-local; the driver permits use from any
/// thread with the owning context bound.
unsafe impl Send for CudaGraph {}
/// See [`Send`] impl.
unsafe impl Sync for CudaGraph {}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        self.ctx
            .record_err(unsafe { cuda_bindings::cuGraphDestroy(self.cu_graph).result() });
    }
}

impl CudaGraph {
    /// Captures everything `body` enqueues on `stream` into a graph.
    ///
    /// `body` must only *enqueue* work. Synchronising, copying to the host, or
    /// otherwise reading device results inside it fails the capture — see the
    /// module docs.
    ///
    /// If `body` returns an error the capture is still ended before returning,
    /// so the stream is never left stuck in capture mode.
    pub fn capture<F, E>(stream: &CudaStream, body: F) -> Result<Self, E>
    where
        F: FnOnce() -> Result<(), E>,
        E: From<DriverError>,
    {
        Self::capture_with_mode(stream, StreamCaptureMode::Relaxed, body)
    }

    /// [`capture`](Self::capture) with an explicit capture mode.
    pub fn capture_with_mode<F, E>(
        stream: &CudaStream,
        mode: StreamCaptureMode,
        body: F,
    ) -> Result<Self, E>
    where
        F: FnOnce() -> Result<(), E>,
        E: From<DriverError>,
    {
        let ctx = stream.context().clone();
        ctx.bind_to_thread().map_err(E::from)?;
        unsafe { cuda_bindings::cuStreamBeginCapture_v2(stream.cu_stream(), mode.raw()) }
            .result()
            .map_err(E::from)?;

        let body_result = body();

        // End the capture unconditionally. Leaving a stream in capture mode
        // after a failure would turn one error into every subsequent launch on
        // that stream failing, which is far harder to diagnose than the
        // original cause.
        let mut cu_graph = MaybeUninit::uninit();
        let end = unsafe {
            cuda_bindings::cuStreamEndCapture(stream.cu_stream(), cu_graph.as_mut_ptr()).result()
        };

        body_result?;
        end.map_err(E::from)?;

        // SAFETY: cuStreamEndCapture wrote a valid handle on success.
        let cu_graph = unsafe { cu_graph.assume_init() };
        Ok(Self { cu_graph, ctx })
    }

    /// Returns the raw `CUgraph` handle.
    pub fn cu_graph(&self) -> cuda_bindings::CUgraph {
        self.cu_graph
    }

    /// Number of nodes in the graph — the launch count a replay collapses into
    /// one host call.
    pub fn num_nodes(&self) -> Result<usize, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut n: usize = 0;
        unsafe {
            cuda_bindings::cuGraphGetNodes(self.cu_graph, std::ptr::null_mut(), &mut n).result()?;
        }
        Ok(n)
    }

    /// Compiles the graph into a replayable executable.
    ///
    /// This is the expensive step — the driver resolves the dependency DAG and
    /// prepares the launch sequence — so it belongs outside any hot loop.
    pub fn instantiate(&self) -> Result<CudaGraphExec, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut exec = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), self.cu_graph, 0)
                .result()?;
        }
        // SAFETY: cuGraphInstantiateWithFlags wrote a valid handle on success.
        let cu_graph_exec = unsafe { exec.assume_init() };
        Ok(CudaGraphExec {
            cu_graph_exec,
            ctx: self.ctx.clone(),
        })
    }
}

/// An instantiated graph, replayable with a single host call.
#[derive(Debug)]
pub struct CudaGraphExec {
    cu_graph_exec: cuda_bindings::CUgraphExec,
    ctx: Arc<CudaContext>,
}

/// # Safety
///
/// See [`CudaGraph`]'s `Send` impl.
unsafe impl Send for CudaGraphExec {}
/// See [`Send`] impl.
unsafe impl Sync for CudaGraphExec {}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        self.ctx
            .record_err(unsafe { cuda_bindings::cuGraphExecDestroy(self.cu_graph_exec).result() });
    }
}

impl CudaGraphExec {
    /// Enqueues the whole graph on `stream` as one operation.
    ///
    /// Asynchronous, like any other stream submission.
    pub fn launch(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { cuda_bindings::cuGraphLaunch(self.cu_graph_exec, stream.cu_stream()) }.result()
    }

    /// Returns the raw `CUgraphExec` handle.
    pub fn cu_graph_exec(&self) -> cuda_bindings::CUgraphExec {
        self.cu_graph_exec
    }
}
