/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Owning device memory buffer with ergonomic host-device transfer methods.
//!
//! [`DeviceBuffer<T>`] is analogous to `Vec<T>` on the host: it owns a
//! contiguous allocation of `len` elements on the device and frees it on
//! drop. Unlike cudarc's `CudaSlice`, the buffer carries no stream reference
//! and no hidden event tracking -- the stream is an explicit parameter on
//! every transfer operation, making data-flow and synchronization transparent.
//!
//! # Quick start
//!
//! ```ignore
//! let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
//! let c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
//! // ... kernel launch ...
//! let c_host = c_dev.to_host_vec(&stream)?;
//! ```

use std::marker::PhantomData;
use std::sync::Arc;

use cuda_bindings::CUdeviceptr;

use crate::context::CudaContext;
use crate::error::DriverError;
use crate::stream::CudaStream;

/// Marker trait for values that can be safely copied between host and device
/// memory as raw bytes.
///
/// Types implementing `DeviceCopy` must not contain Rust-owned allocations,
/// references, or other values whose validity depends on host-side ownership or
/// drop semantics. This is the device-memory equivalent of a plain-old-data
/// contract.
///
/// # Safety
///
/// Implementors must be safe to duplicate with a byte-for-byte copy. Values
/// copied back from device memory must have a bit pattern that is valid for
/// `Self`, and the all-zero bit pattern must also be valid because
/// [`DeviceBuffer::zeroed`] initializes memory with zero bytes.
///
/// `Copy` alone is not enough: types such as `bool`, `char`, and
/// `NonZeroU32` are `Copy`, but not every byte pattern is a valid value of
/// those types. `DeviceCopy` is the stronger promise required when
/// `DeviceBuffer` turns raw device bytes back into initialized Rust values.
pub unsafe trait DeviceCopy: Copy {}

macro_rules! impl_device_copy {
    ($($ty:ty),+ $(,)?) => {
        $(
            unsafe impl DeviceCopy for $ty {}
        )+
    };
}

impl_device_copy!(
    (),
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    f16,
    f32,
    f64
);

unsafe impl<T: DeviceCopy, const N: usize> DeviceCopy for [T; N] {}
unsafe impl<T: ?Sized> DeviceCopy for *const T {}
unsafe impl<T: ?Sized> DeviceCopy for *mut T {}

macro_rules! impl_device_copy_tuple {
    ($($name:ident),+ $(,)?) => {
        unsafe impl<$($name: DeviceCopy),+> DeviceCopy for ($($name,)+) {}
    };
}

impl_device_copy_tuple!(A);
impl_device_copy_tuple!(A, B);
impl_device_copy_tuple!(A, B, C);
impl_device_copy_tuple!(A, B, C, D);
impl_device_copy_tuple!(A, B, C, D, E);
impl_device_copy_tuple!(A, B, C, D, E, F);
impl_device_copy_tuple!(A, B, C, D, E, F, G);
impl_device_copy_tuple!(A, B, C, D, E, F, G, H);

unsafe impl DeviceCopy for half::bf16 {}
unsafe impl DeviceCopy for half::f16 {}

/// Owning handle to a contiguous device allocation of `T` elements.
///
/// Holds a raw device pointer, element count, and a reference-counted
/// context that keeps the CUDA context alive. Dropping the buffer calls
/// `cuMemFree` (synchronous); for async-sensitive workloads, use
/// `cuda_async::DeviceBox` which frees via a deallocator stream.
///
/// Device buffers may only transfer plain device-copyable values. Owning host
/// types such as [`String`] are rejected because copying their bytes to and
/// from device memory would not preserve Rust ownership invariants.
///
/// ```compile_fail
/// # use cuda_core::{CudaStream, DeviceBuffer};
/// # fn rejects_non_device_copy(stream: &CudaStream) {
/// let _ = DeviceBuffer::<String>::zeroed(stream, 1);
/// # }
/// ```
pub struct DeviceBuffer<T> {
    ptr: CUdeviceptr,
    len: usize,
    ctx: Arc<CudaContext>,
    _marker: PhantomData<T>,
}

// SAFETY: CUdeviceptr is a u64 handle valid across threads when the owning
// context is bound. The PhantomData<T> is Send if T is Send.
unsafe impl<T: Send> Send for DeviceBuffer<T> {}
// SAFETY: &DeviceBuffer only exposes cu_deviceptr() and len(), both of which
// return Copy values. No interior mutability.
unsafe impl<T: Send + Sync> Sync for DeviceBuffer<T> {}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let _ = self.ctx.bind_to_thread();
            self.ctx
                .record_err(unsafe { crate::memory::free_sync(self.ptr) });
        }
    }
}

impl<T> DeviceBuffer<T> {
    /// Returns the raw `CUdeviceptr` for use in kernel argument lists.
    #[inline]
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Number of `T` elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total size in bytes (`len * size_of::<T>()`).
    #[inline]
    pub fn num_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Returns a reference to the owning context.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Constructs a `DeviceBuffer` from pre-existing raw parts.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been allocated via `cuMemAlloc*` with at least
    ///   `len * size_of::<T>()` bytes.
    /// - `ptr` must belong to the same CUDA context as `ctx`.
    /// - The caller transfers ownership -- `ptr` will be freed on drop.
    pub unsafe fn from_raw_parts(ptr: CUdeviceptr, len: usize, ctx: Arc<CudaContext>) -> Self {
        Self {
            ptr,
            len,
            ctx,
            _marker: PhantomData,
        }
    }

    /// Consumes the buffer and returns the raw parts without freeing.
    ///
    /// The caller is responsible for eventually freeing `ptr`.
    pub fn into_raw_parts(self) -> (CUdeviceptr, usize, Arc<CudaContext>) {
        let parts = (self.ptr, self.len, self.ctx.clone());
        std::mem::forget(self);
        parts
    }
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    /// Allocates device memory and copies `data` from the host, enqueued on
    /// `stream`.
    ///
    /// The host slice must remain valid until the copy completes (i.e. until
    /// the next synchronization point on `stream`). For pageable host memory
    /// the driver may internally synchronize; use pinned memory for true
    /// async overlap.
    pub fn from_host(stream: &CudaStream, data: &[T]) -> Result<Self, DriverError> {
        let ctx = stream.context().clone();
        let len = data.len();
        let num_bytes = std::mem::size_of_val(data);

        let ptr = unsafe { crate::memory::malloc_async(stream.cu_stream(), num_bytes)? };
        unsafe {
            crate::memory::memcpy_htod_async(ptr, data.as_ptr(), num_bytes, stream.cu_stream())?;
        }
        Ok(Self {
            ptr,
            len,
            ctx,
            _marker: PhantomData,
        })
    }

    /// Allocates zero-initialized device memory of `len` elements, enqueued
    /// on `stream`.
    pub fn zeroed(stream: &CudaStream, len: usize) -> Result<Self, DriverError> {
        let ctx = stream.context().clone();
        let num_bytes = len * std::mem::size_of::<T>();

        let ptr = unsafe { crate::memory::malloc_async(stream.cu_stream(), num_bytes)? };
        if num_bytes > 0 {
            unsafe {
                crate::memory::memset_d8_async(ptr, 0, num_bytes, stream.cu_stream())?;
            }
        }
        Ok(Self {
            ptr,
            len,
            ctx,
            _marker: PhantomData,
        })
    }

    /// Allocates device memory of `len` elements **without zeroing it**,
    /// enqueued on `stream`.
    ///
    /// Use this when the caller writes every element before any read (e.g. a
    /// kernel output that fully covers its buffer). Skips the full-buffer
    /// `cuMemsetD8` that [`zeroed`](Self::zeroed) performs, which is pure
    /// wasted DRAM bandwidth when the buffer is immediately overwritten.
    ///
    /// # Safety contract
    ///
    /// The contents are indeterminate until written. Reading before writing
    /// observes arbitrary bytes (not a memory-safety hazard for `DeviceCopy`
    /// `T`, but a correctness one).
    pub fn uninit(stream: &CudaStream, len: usize) -> Result<Self, DriverError> {
        let ctx = stream.context().clone();
        let num_bytes = len * std::mem::size_of::<T>();
        let ptr = unsafe { crate::memory::malloc_async(stream.cu_stream(), num_bytes)? };
        Ok(Self {
            ptr,
            len,
            ctx,
            _marker: PhantomData,
        })
    }

    /// Copies the entire buffer back to the host, returning a `Vec<T>`.
    ///
    /// Synchronizes on `stream` before returning so the host vector is safe
    /// to read immediately.
    pub fn to_host_vec(&self, stream: &CudaStream) -> Result<Vec<T>, DriverError> {
        let mut host = Vec::with_capacity(self.len);
        unsafe {
            crate::memory::memcpy_dtoh_async(
                host.as_mut_ptr(),
                self.ptr,
                self.num_bytes(),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()?;
        unsafe { host.set_len(self.len) };
        Ok(host)
    }

    /// Copies the buffer contents into an existing host slice.
    ///
    /// Synchronizes on `stream` before returning. Panics if
    /// `dst.len() < self.len()`.
    pub fn copy_to_host(&self, stream: &CudaStream, dst: &mut [T]) -> Result<(), DriverError> {
        assert!(
            dst.len() >= self.len,
            "destination slice too small: {} < {}",
            dst.len(),
            self.len
        );
        unsafe {
            crate::memory::memcpy_dtoh_async(
                dst.as_mut_ptr(),
                self.ptr,
                self.num_bytes(),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()
    }
}
