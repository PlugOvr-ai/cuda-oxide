/*
 * CUDA kernels for the oxide_onnx ONNX inference engine.
 *
 * Module structure:
 *   crate::kernels       (this file, loaded by main.rs)
 *   crate::kernels::gpu  (#[cuda_module] block, compiled to PTX)
 *
 * Call site: use crate::kernels::gpu;
 *            let module = gpu::load(&ctx)?;
 *            module.relu(&stream, cfg, &mut buf)?;
 */

#![allow(clippy::too_many_arguments)]

use cuda_device::wgmma::mma_sync_m16n8k8_f32_tf32;
use cuda_device::{kernel, thread, DisjointSlice, SharedArray};
use cuda_host::cuda_module;

// These helpers avoid core::intrinsics::sqrtf32 / expf32, which the oxide
// mir-lower pipeline maps to __nv_sqrtf / __nv_expf (libdevice). Libdevice
// calls trigger NVVM IR mode and skip PTX embedding. Instead we use only
// standard LLVM instructions: bitcast, integer arithmetic, fmul, fadd.

/// Compute 1/sqrt(x) using the Carmack bit-hack followed by two Newton steps.
/// Relative error < 0.001%.
#[inline(always)]
fn gpu_rsqrt(x: f32) -> f32 {
    let bits = x.to_bits();
    let guess = f32::from_bits(0x5f375a86u32.wrapping_sub(bits >> 1));
    let step = |y: f32| y * (1.5f32 - 0.5f32 * x * y * y);
    step(step(guess))
}

/// Compute e^x via range reduction + degree-5 polynomial + 2^n scaling.
/// No libdevice; uses only bitcast + integer ops + fused float arithmetic.
/// Max relative error ≈ 2e-7 (matches single-precision needs for softmax).
#[inline(always)]
fn gpu_expf(x: f32) -> f32 {
    // Clamp to avoid u32 wrap in the exponent scaling step.
    let x = if x > 88.0f32 {
        88.0f32
    } else if x < -88.0f32 {
        -88.0f32
    } else {
        x
    };
    // Decompose: x = n*ln2 + r,  |r| <= ln2/2
    let t = x * 1.442695040888963f32; // x * log2(e)
    let n = t as i32; // floor (truncate toward zero, good enough for |r|<=0.5*ln2)
    let r = x - (n as f32) * 0.693147180559945f32;
    // Minimax polynomial for e^r on [-0.347, 0.347]
    let p = 1.0f32
        + r * (1.0f32
            + r * (0.5f32 + r * (0.16666667f32 + r * (0.041666668f32 + r * 0.008333334f32))));
    // Multiply by 2^n via IEEE-754 exponent field: bits = (n+127) << 23
    let exp2n_bits = ((n + 127) as u32) << 23;
    p * f32::from_bits(exp2n_bits)
}

/// erf(x) via Abramowitz & Stegun 7.1.26 (max abs error ≈ 1.5e-7).
/// Uses only gpu_expf + integer/float ops — no libdevice.
#[inline(always)]
fn gpu_erf(x: f32) -> f32 {
    let sign = if x < 0.0f32 { -1.0f32 } else { 1.0f32 };
    let ax = if x < 0.0f32 { -x } else { x };
    let t = 1.0f32 / (1.0f32 + 0.3275911f32 * ax);
    // poly = a1·t + a2·t² + a3·t³ + a4·t⁴ + a5·t⁵  (Horner in t)
    let poly = t
        * (0.254829592f32
            + t * (-0.284496736f32
                + t * (1.421413741f32 + t * (-1.453152027f32 + t * 1.061405429f32))));
    sign * (1.0f32 - poly * gpu_expf(-ax * ax))
}

/// tanh(x) via gpu_expf; numerically stable for large |x| (expf clamps).
#[inline(always)]
fn gpu_tanh(x: f32) -> f32 {
    let sign = if x < 0.0f32 { -1.0f32 } else { 1.0f32 };
    let ax = if x < 0.0f32 { -x } else { x };
    let e = gpu_expf(-2.0f32 * ax); // in (0, 1]
    sign * (1.0f32 - e) / (1.0f32 + e)
}

#[cuda_module]
pub mod gpu {
    use super::*;

    // =========================================================================
    // ReLU — in-place element-wise max(0, x)
    // =========================================================================
    #[kernel]
    pub fn relu(mut x: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(v) = x.get_mut(idx) {
            let val = *v;
            *v = if val > 0.0f32 { val } else { 0.0f32 };
        }
        let _ = i; // silence warning
    }

    // =========================================================================
    // ReLU (out-of-place) — c[i] = max(0, a[i])
    //   Fused read→write: avoids a separate full-tensor D2D copy + launch.
    // =========================================================================
    #[kernel]
    pub fn relu_fwd(a: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = c.get_mut(idx) {
            let v = a[i];
            *o = if v > 0.0f32 { v } else { 0.0f32 };
        }
    }

    // =========================================================================
    // Clip (out-of-place) — c[i] = clamp(a[i], lo, hi)
    // =========================================================================
    #[kernel]
    pub fn clip_fwd(a: &[f32], lo: f32, hi: f32, mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = c.get_mut(idx) {
            let v = a[i];
            *o = if v < lo {
                lo
            } else if v > hi {
                hi
            } else {
                v
            };
        }
    }

    // =========================================================================
    // Clip — in-place clamp to [lo, hi]
    // =========================================================================
    #[kernel]
    pub fn clip(mut x: DisjointSlice<f32>, lo: f32, hi: f32) {
        let idx = thread::index_1d();
        if let Some(v) = x.get_mut(idx) {
            let val = *v;
            *v = if val < lo {
                lo
            } else if val > hi {
                hi
            } else {
                val
            };
        }
    }

    // =========================================================================
    // Add element-wise — c[i] = a[i] + b[i]
    // =========================================================================
    #[kernel]
    pub fn add_elementwise(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = c.get_mut(idx) {
            *out = a[i] + b[i];
        }
    }

    // =========================================================================
    // Mul element-wise — c[i] = a[i] * b[i]
    // =========================================================================
    #[kernel]
    pub fn mul_elementwise(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = c.get_mut(idx) {
            *out = a[i] * b[i];
        }
    }

    // =========================================================================
    // GEMM naive — C = alpha * A * B + beta * C  (row-major)
    //   A: m×k,  B: k×n,  C: m×n.  One thread per C element.
    // =========================================================================
    #[kernel]
    pub fn sgemm_naive(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n as usize) } {
            if row < m as usize {
                let n_sz = n as usize;
                let k_sz = k as usize;
                let mut sum = 0.0f32;
                let mut i = 0usize;
                while i < k_sz {
                    sum += a[row * k_sz + i] * b[i * n_sz + col];
                    i += 1;
                }
                if let Some(elem) = c.get_mut(c_idx) {
                    *elem = alpha * sum + beta * (*elem);
                }
            }
        }
    }

    // =========================================================================
    // GEMM tiled — C = alpha * A * B + beta * C  (row-major, shared memory)
    //   A: m×k,  B: k×n,  C: m×n.  16×16 tile, 256 threads/block.
    //   Each block computes a 16×16 tile of C; A/B tiles are staged in shared
    //   memory so each global element is read ~16× fewer times than naive.
    // =========================================================================
    #[kernel]
    pub fn sgemm_tiled(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * 16 + ty;
        let col = thread::blockIdx_x() as usize * 16 + tx;

        let m_sz = m as usize;
        let n_sz = n as usize;
        let k_sz = k as usize;

        let num_tiles = k_sz.div_ceil(16);
        let smem_idx = ty * 16 + tx;
        let mut sum = 0.0f32;

        let mut t = 0usize;
        while t < num_tiles {
            let tile_start = t * 16;
            unsafe {
                let a_col = tile_start + tx;
                TILE_A[smem_idx] = if row < m_sz && a_col < k_sz {
                    a[row * k_sz + a_col]
                } else {
                    0.0f32
                };
                let b_row = tile_start + ty;
                TILE_B[smem_idx] = if b_row < k_sz && col < n_sz {
                    b[b_row * n_sz + col]
                } else {
                    0.0f32
                };
            }
            thread::sync_threads();
            unsafe {
                let mut i = 0usize;
                while i < 16 {
                    sum += TILE_A[ty * 16 + i] * TILE_B[i * 16 + tx];
                    i += 1;
                }
            }
            thread::sync_threads();
            t += 1;
        }

        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n_sz) } {
            if row < m_sz {
                if let Some(c_elem) = c.get_mut(c_idx) {
                    // Output buffer may be uninitialized (skip-memset alloc):
                    // never read/scale it when beta == 0.
                    *c_elem = if beta != 0.0f32 {
                        alpha * sum + beta * (*c_elem)
                    } else {
                        alpha * sum
                    };
                }
            }
        }
    }

    // =========================================================================
    // GEMM via Ampere tensor cores — C = alpha·A·B + beta·C  (row-major).
    //   One warp computes a 16×8 output tile; K looped in steps of 8 via
    //   mma.sync.m16n8k8 (tf32 inputs, f32 accumulate). f32 reinterpreted as
    //   tf32 (HW uses the top 19 bits). Launch: grid=(⌈n/8⌉,⌈m/16⌉,1), block=32.
    // =========================================================================
    #[kernel]
    pub fn sgemm_mma(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32>,
    ) {
        let lane = thread::threadIdx_x() as usize;
        let gid = lane >> 2;
        let tig = lane & 3;
        let tile_m = thread::blockIdx_y() as usize * 16;
        let tile_n = thread::blockIdx_x() as usize * 8;

        let m_sz = m as usize;
        let n_sz = n as usize;
        let k_sz = k as usize;

        let mut acc = [0.0f32; 4];
        let mut k0 = 0usize;
        while k0 < k_sz {
            let ar0 = tile_m + gid;
            let ar1 = tile_m + gid + 8;
            let ac0 = k0 + tig;
            let ac1 = k0 + tig + 4;
            let a0 = if ar0 < m_sz && ac0 < k_sz {
                a[ar0 * k_sz + ac0].to_bits()
            } else {
                0u32
            };
            let a1 = if ar1 < m_sz && ac0 < k_sz {
                a[ar1 * k_sz + ac0].to_bits()
            } else {
                0u32
            };
            let a2 = if ar0 < m_sz && ac1 < k_sz {
                a[ar0 * k_sz + ac1].to_bits()
            } else {
                0u32
            };
            let a3 = if ar1 < m_sz && ac1 < k_sz {
                a[ar1 * k_sz + ac1].to_bits()
            } else {
                0u32
            };

            let bc = tile_n + gid;
            let br0 = k0 + tig;
            let br1 = k0 + tig + 4;
            let b0 = if br0 < k_sz && bc < n_sz {
                b[br0 * n_sz + bc].to_bits()
            } else {
                0u32
            };
            let b1 = if br1 < k_sz && bc < n_sz {
                b[br1 * n_sz + bc].to_bits()
            } else {
                0u32
            };

            unsafe {
                mma_sync_m16n8k8_f32_tf32(&mut acc, a0, a1, a2, a3, b0, b1);
            }
            k0 += 8;
        }

        let dr0 = tile_m + gid;
        let dr1 = tile_m + gid + 8;
        let dc0 = tile_n + 2 * tig;
        let dc1 = tile_n + 2 * tig + 1;
        if dr0 < m_sz && dc0 < n_sz {
            let i = dr0 * n_sz + dc0;
            let cell = unsafe { c.get_unchecked_mut(i) };
            *cell = alpha * acc[0] + beta * (*cell);
        }
        if dr0 < m_sz && dc1 < n_sz {
            let i = dr0 * n_sz + dc1;
            let cell = unsafe { c.get_unchecked_mut(i) };
            *cell = alpha * acc[1] + beta * (*cell);
        }
        if dr1 < m_sz && dc0 < n_sz {
            let i = dr1 * n_sz + dc0;
            let cell = unsafe { c.get_unchecked_mut(i) };
            *cell = alpha * acc[2] + beta * (*cell);
        }
        if dr1 < m_sz && dc1 < n_sz {
            let i = dr1 * n_sz + dc1;
            let cell = unsafe { c.get_unchecked_mut(i) };
            *cell = alpha * acc[3] + beta * (*cell);
        }
    }

    // =========================================================================
    // GEMM register-blocked — C = alpha·A·B + beta·C  (row-major).
    //
    //   A: m×k,  B: k×n,  C: m×n.  Block computes a 64×64 C tile; 256 threads
    //   (16×16), each thread a 4×4 micro-tile held in registers. A/B are staged
    //   in shared memory in BK=8 slabs, so every global element is reused 64×
    //   and each thread does 16 FMAs per 8 shared loads (high arithmetic
    //   intensity — the naive 1-elem/thread tiled and the no-reuse mma kernels
    //   are both memory-bound; this is not).
    //
    //   Launch: grid=(⌈n/64⌉, ⌈m/64⌉, 1), block=(16,16,1).
    // =========================================================================
    #[kernel]
    pub fn sgemm_rb(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32>,
    ) {
        // AS: [BM=64][BK=8] = 512.  BS: [BK=8][BN=64] = 512.
        static mut AS: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut BS: SharedArray<f32, 512> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let tid = ty * 16 + tx;
        let row0 = thread::blockIdx_y() as usize * 64;
        let col0 = thread::blockIdx_x() as usize * 64;

        let m_sz = m as usize;
        let n_sz = n as usize;
        let k_sz = k as usize;

        // 16 *scalar* accumulators. This rustc→PTX backend spills
        // array-indexed per-thread locals to local memory (killing the whole
        // point of register blocking), so the micro-tile must be plain
        // scalars to stay in registers.
        let mut c00 = 0.0f32;
        let mut c01 = 0.0f32;
        let mut c02 = 0.0f32;
        let mut c03 = 0.0f32;
        let mut c10 = 0.0f32;
        let mut c11 = 0.0f32;
        let mut c12 = 0.0f32;
        let mut c13 = 0.0f32;
        let mut c20 = 0.0f32;
        let mut c21 = 0.0f32;
        let mut c22 = 0.0f32;
        let mut c23 = 0.0f32;
        let mut c30 = 0.0f32;
        let mut c31 = 0.0f32;
        let mut c32 = 0.0f32;
        let mut c33 = 0.0f32;

        let a_base = ty * 4;
        let b_base = tx * 4;

        let mut k0 = 0usize;
        while k0 < k_sz {
            // Stage A (64×8) and B (8×64) into shared memory: 512 elems each,
            // 256 threads → 2 elements per thread (tid and tid+256).
            let mut s = 0usize;
            while s < 2 {
                let e = tid + s * 256;
                let ar = e >> 3; // 0..64
                let ac = e & 7; //  0..8
                let gar = row0 + ar;
                let gac = k0 + ac;
                unsafe {
                    AS[e] = if gar < m_sz && gac < k_sz {
                        a[gar * k_sz + gac]
                    } else {
                        0.0f32
                    };
                }
                let br = e >> 6; // 0..8
                let bc = e & 63; // 0..64
                let gbr = k0 + br;
                let gbc = col0 + bc;
                unsafe {
                    BS[e] = if gbr < k_sz && gbc < n_sz {
                        b[gbr * n_sz + gbc]
                    } else {
                        0.0f32
                    };
                }
                s += 1;
            }
            thread::sync_threads();

            let mut kk = 0usize;
            while kk < 8 {
                let a0 = unsafe { AS[a_base * 8 + kk] };
                let a1 = unsafe { AS[(a_base + 1) * 8 + kk] };
                let a2 = unsafe { AS[(a_base + 2) * 8 + kk] };
                let a3 = unsafe { AS[(a_base + 3) * 8 + kk] };
                let bk = kk * 64 + b_base;
                let b0 = unsafe { BS[bk] };
                let b1 = unsafe { BS[bk + 1] };
                let b2 = unsafe { BS[bk + 2] };
                let b3 = unsafe { BS[bk + 3] };
                c00 += a0 * b0;
                c01 += a0 * b1;
                c02 += a0 * b2;
                c03 += a0 * b3;
                c10 += a1 * b0;
                c11 += a1 * b1;
                c12 += a1 * b2;
                c13 += a1 * b3;
                c20 += a2 * b0;
                c21 += a2 * b1;
                c22 += a2 * b2;
                c23 += a2 * b3;
                c30 += a3 * b0;
                c31 += a3 * b1;
                c32 += a3 * b2;
                c33 += a3 * b3;
                kk += 1;
            }
            thread::sync_threads();
            k0 += 8;
        }

        // Fully-unrolled 4×4 store (no closures/arrays — keep it register-only).
        // Output may be uninitialized (skip-memset alloc), so never read/scale
        // a cell when beta == 0.
        let gr0 = row0 + a_base;
        let gc0 = col0 + b_base;
        let has_beta = beta != 0.0f32;

        if gr0 < m_sz {
            if gc0 < n_sz {
                let p = gr0 * n_sz + gc0;
                let cell = unsafe { c.get_unchecked_mut(p) };
                *cell = if has_beta { alpha * c00 + beta * (*cell) } else { alpha * c00 };
            }
            if gc0 + 1 < n_sz {
                let p = gr0 * n_sz + gc0 + 1;
                let cell = unsafe { c.get_unchecked_mut(p) };
                *cell = if has_beta { alpha * c01 + beta * (*cell) } else { alpha * c01 };
            }
            if gc0 + 2 < n_sz {
                let p = gr0 * n_sz + gc0 + 2;
                let cell = unsafe { c.get_unchecked_mut(p) };
                *cell = if has_beta { alpha * c02 + beta * (*cell) } else { alpha * c02 };
            }
            if gc0 + 3 < n_sz {
                let p = gr0 * n_sz + gc0 + 3;
                let cell = unsafe { c.get_unchecked_mut(p) };
                *cell = if has_beta { alpha * c03 + beta * (*cell) } else { alpha * c03 };
            }
        }
        if gr0 + 1 < m_sz {
            let r = gr0 + 1;
            if gc0 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0) };
                *cell = if has_beta { alpha * c10 + beta * (*cell) } else { alpha * c10 };
            }
            if gc0 + 1 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 1) };
                *cell = if has_beta { alpha * c11 + beta * (*cell) } else { alpha * c11 };
            }
            if gc0 + 2 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 2) };
                *cell = if has_beta { alpha * c12 + beta * (*cell) } else { alpha * c12 };
            }
            if gc0 + 3 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 3) };
                *cell = if has_beta { alpha * c13 + beta * (*cell) } else { alpha * c13 };
            }
        }
        if gr0 + 2 < m_sz {
            let r = gr0 + 2;
            if gc0 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0) };
                *cell = if has_beta { alpha * c20 + beta * (*cell) } else { alpha * c20 };
            }
            if gc0 + 1 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 1) };
                *cell = if has_beta { alpha * c21 + beta * (*cell) } else { alpha * c21 };
            }
            if gc0 + 2 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 2) };
                *cell = if has_beta { alpha * c22 + beta * (*cell) } else { alpha * c22 };
            }
            if gc0 + 3 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 3) };
                *cell = if has_beta { alpha * c23 + beta * (*cell) } else { alpha * c23 };
            }
        }
        if gr0 + 3 < m_sz {
            let r = gr0 + 3;
            if gc0 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0) };
                *cell = if has_beta { alpha * c30 + beta * (*cell) } else { alpha * c30 };
            }
            if gc0 + 1 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 1) };
                *cell = if has_beta { alpha * c31 + beta * (*cell) } else { alpha * c31 };
            }
            if gc0 + 2 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 2) };
                *cell = if has_beta { alpha * c32 + beta * (*cell) } else { alpha * c32 };
            }
            if gc0 + 3 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 3) };
                *cell = if has_beta { alpha * c33 + beta * (*cell) } else { alpha * c33 };
            }
        }
    }

    // =========================================================================
    // GEMM tiled, B transposed — C = alpha * A * Bᵀ + beta * C  (row-major)
    //   A: m×k,  B: n×k (stored row-major; logically Bᵀ is k×n),  C: m×n.
    //   Used for ONNX Gemm with transB=1 (final classifier layer).
    // =========================================================================
    #[kernel]
    pub fn sgemm_transb_tiled(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32, thread::Runtime2DIndex>,
    ) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let row = thread::blockIdx_y() as usize * 16 + ty;
        let col = thread::blockIdx_x() as usize * 16 + tx;

        let m_sz = m as usize;
        let n_sz = n as usize;
        let k_sz = k as usize;

        let num_tiles = k_sz.div_ceil(16);
        let smem_idx = ty * 16 + tx;
        let mut sum = 0.0f32;

        let mut t = 0usize;
        while t < num_tiles {
            let tile_start = t * 16;
            unsafe {
                let a_col = tile_start + tx;
                TILE_A[smem_idx] = if row < m_sz && a_col < k_sz {
                    a[row * k_sz + a_col]
                } else {
                    0.0f32
                };
                // Bᵀ[tile_start+ty, col] = B[col, tile_start+ty]
                let b_k = tile_start + ty;
                TILE_B[smem_idx] = if col < n_sz && b_k < k_sz {
                    b[col * k_sz + b_k]
                } else {
                    0.0f32
                };
            }
            thread::sync_threads();
            unsafe {
                let mut i = 0usize;
                while i < 16 {
                    sum += TILE_A[ty * 16 + i] * TILE_B[i * 16 + tx];
                    i += 1;
                }
            }
            thread::sync_threads();
            t += 1;
        }

        if let Some(c_idx) = unsafe { thread::index_2d_runtime(n_sz) } {
            if row < m_sz {
                if let Some(c_elem) = c.get_mut(c_idx) {
                    // Output buffer may be uninitialized (skip-memset alloc):
                    // never read/scale it when beta == 0.
                    *c_elem = if beta != 0.0f32 {
                        alpha * sum + beta * (*c_elem)
                    } else {
                        alpha * sum
                    };
                }
            }
        }
    }

    // =========================================================================
    // Bias add — x[n, f, ...] += bias[f]
    //   x: flat [batch * features * spatial], one thread per element.
    //   spatial: number of elements per channel (1 for FC layers, h*w for conv).
    // =========================================================================
    #[kernel]
    pub fn bias_add(mut x: DisjointSlice<f32>, bias: &[f32], spatial: u32, features: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(v) = x.get_mut(idx) {
            *v += bias[(i / spatial as usize) % features as usize];
        }
    }

    // =========================================================================
    // Batch normalization — inference mode
    //   y[n,c,hw] = gamma[c] * (x[n,c,hw] - mean[c]) / sqrt(var[c]+eps) + beta[c]
    //   n: batch, c: channels, hw: H*W spatial.  One thread per element.
    // =========================================================================
    #[kernel]
    pub fn batch_norm_inference(
        x: &[f32],
        gamma: &[f32],
        beta_bn: &[f32],
        mean: &[f32],
        var: &[f32],
        eps: f32,
        _n: u32,
        c: u32,
        hw: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = y.get_mut(idx) {
            let c_sz = c as usize;
            let hw_sz = hw as usize;
            let chan = (i / hw_sz) % c_sz;
            let xi = x[i];
            let inv_std = gpu_rsqrt(var[chan] + eps);
            *out = gamma[chan] * (xi - mean[chan]) * inv_std + beta_bn[chan];
        }
    }

    // =========================================================================
    // Depthwise Conv2D — fused, one thread per output element.
    //   input:  [N, C_in, H_in, W_in]
    //   weight: [C_out, 1, Kh, Kw]   (depthwise: c_in_per_group == 1)
    //   output: [N, C_out, out_H, out_W]
    //   Input channel for output channel oc = oc / cout_per_group.
    //   Replaces the per-group im2col+GEMM loop for MobileNetV2 depthwise
    //   layers: O(group) kernel launches collapse to a single launch.
    // =========================================================================
    #[kernel]
    pub fn depthwise_conv2d(
        input: &[f32],
        weight: &[f32],
        c_in: u32,
        h_in: u32,
        w_in: u32,
        c_out: u32,
        cout_per_group: u32,
        kh: u32,
        kw: u32,
        pad_h: u32,
        pad_w: u32,
        stride_h: u32,
        stride_w: u32,
        dil_h: u32,
        dil_w: u32,
        out_h: u32,
        out_w: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = y.get_mut(idx) {
            let c_out_sz = c_out as usize;
            let out_hw = (out_h * out_w) as usize;
            let in_hw = h_in as usize * w_in as usize;

            let batch_i = i / (c_out_sz * out_hw);
            let local_i = i % (c_out_sz * out_hw);
            let oc = local_i / out_hw;
            let spatial = local_i % out_hw;
            let oh = spatial / out_w as usize;
            let ow = spatial % out_w as usize;

            let in_ch = oc / cout_per_group as usize;

            let kh_sz = kh as usize;
            let kw_sz = kw as usize;
            let w_base = oc * kh_sz * kw_sz;
            let in_base = batch_i * c_in as usize * in_hw + in_ch * in_hw;

            let mut sum = 0.0f32;
            let mut kr = 0usize;
            while kr < kh_sz {
                let mut kc = 0usize;
                while kc < kw_sz {
                    let ih = oh * stride_h as usize + kr * dil_h as usize;
                    let iw = ow * stride_w as usize + kc * dil_w as usize;
                    let ih_unpad = ih as isize - pad_h as isize;
                    let iw_unpad = iw as isize - pad_w as isize;
                    if ih_unpad >= 0
                        && ih_unpad < h_in as isize
                        && iw_unpad >= 0
                        && iw_unpad < w_in as isize
                    {
                        let in_off =
                            in_base + ih_unpad as usize * w_in as usize + iw_unpad as usize;
                        sum += weight[w_base + kr * kw_sz + kc] * input[in_off];
                    }
                    kc += 1;
                }
                kr += 1;
            }
            *out = sum;
        }
    }

    // =========================================================================
    // im2col — expand padded input patches into column matrix for Conv2D.
    //   input:  [N, C_in, H_in, W_in]
    //   output: [N * C_in*Kh*Kw, out_H*out_W]   (one patch per element)
    //   One thread per output element.
    // =========================================================================
    #[kernel]
    pub fn im2col(
        input: &[f32],
        c_in: u32,
        h_in: u32,
        w_in: u32,
        kh: u32,
        kw: u32,
        pad_h: u32,
        pad_w: u32,
        stride_h: u32,
        stride_w: u32,
        dil_h: u32,
        dil_w: u32,
        out_h: u32,
        out_w: u32,
        mut col: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = col.get_mut(idx) {
            let col_rows = (c_in * kh * kw) as usize;
            let out_spatial = (out_h * out_w) as usize;
            let col_per_batch = col_rows * out_spatial;

            let batch_i = i / col_per_batch;
            let local_i = i % col_per_batch;
            let kk = local_i / out_spatial;
            let spatial = local_i % out_spatial;
            let oh = spatial / out_w as usize;
            let ow = spatial % out_w as usize;

            let kh_sz = kh as usize;
            let kw_sz = kw as usize;
            let c_sz = c_in as usize;

            let ki = kk / (kh_sz * kw_sz);
            let kr = (kk % (kh_sz * kw_sz)) / kw_sz;
            let kc = kk % kw_sz;

            let ih = oh * stride_h as usize + kr * dil_h as usize;
            let iw = ow * stride_w as usize + kc * dil_w as usize;
            let ih_unpad = ih as isize - pad_h as isize;
            let iw_unpad = iw as isize - pad_w as isize;

            let val = if ih_unpad < 0
                || ih_unpad >= h_in as isize
                || iw_unpad < 0
                || iw_unpad >= w_in as isize
            {
                0.0f32
            } else {
                let offset = batch_i * c_sz * h_in as usize * w_in as usize
                    + ki * h_in as usize * w_in as usize
                    + ih_unpad as usize * w_in as usize
                    + iw_unpad as usize;
                input[offset]
            };
            *out = val;
        }
    }

    // =========================================================================
    // MaxPool2D — sliding-window maximum.
    //   x:  [N, C, in_H, in_W]   y:  [N, C, out_H, out_W]
    //   One thread per output element.
    // =========================================================================
    #[kernel]
    pub fn maxpool2d(
        x: &[f32],
        c: u32,
        in_h: u32,
        in_w: u32,
        kh: u32,
        kw: u32,
        pad_h: u32,
        pad_w: u32,
        stride_h: u32,
        stride_w: u32,
        out_h: u32,
        out_w: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = y.get_mut(idx) {
            let c_sz = c as usize;
            let out_hw = (out_h * out_w) as usize;
            let in_hw = in_h as usize * in_w as usize;
            let batch_i = i / (c_sz * out_hw);
            let local_i = i % (c_sz * out_hw);
            let ch = local_i / out_hw;
            let spatial = local_i % out_hw;
            let oh = spatial / out_w as usize;
            let ow = spatial % out_w as usize;

            let mut max_val = f32::NEG_INFINITY;
            let mut ki = 0usize;
            while ki < kh as usize {
                let mut kj = 0usize;
                while kj < kw as usize {
                    let ih = oh * stride_h as usize + ki;
                    let iw = ow * stride_w as usize + kj;
                    let ih_unpad = ih as isize - pad_h as isize;
                    let iw_unpad = iw as isize - pad_w as isize;
                    if ih_unpad >= 0
                        && ih_unpad < in_h as isize
                        && iw_unpad >= 0
                        && iw_unpad < in_w as isize
                    {
                        let offset = batch_i * c_sz * in_hw
                            + ch * in_hw
                            + ih_unpad as usize * in_w as usize
                            + iw_unpad as usize;
                        let v = x[offset];
                        if v > max_val {
                            max_val = v;
                        }
                    }
                    kj += 1;
                }
                ki += 1;
            }
            *out = max_val;
        }
    }

    // =========================================================================
    // Global Average Pool — y[n,c] = mean(x[n,c,:,:])
    //   x: [N,C,H,W]  y: [N,C].  One thread per [n,c] pair.
    // =========================================================================
    #[kernel]
    pub fn global_avg_pool(x: &[f32], c: u32, hw: u32, mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = y.get_mut(idx) {
            let c_sz = c as usize;
            let hw_sz = hw as usize;
            let batch_i = i / c_sz;
            let ch = i % c_sz;
            let base = batch_i * c_sz * hw_sz + ch * hw_sz;
            let mut sum = 0.0f32;
            let mut j = 0usize;
            while j < hw_sz {
                sum += x[base + j];
                j += 1;
            }
            *out = sum / hw_sz as f32;
        }
    }

    // =========================================================================
    // Softmax — numerically stable, row-wise.
    //   x: [rows, cols]  y: [rows, cols].
    //   One thread per row; sequential over cols.
    //   For ResNet50: [1, 1000] → 1 thread over 1000 values.
    // =========================================================================
    #[kernel]
    pub fn softmax_row(x: &[f32], rows: u32, cols: u32, mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let row = idx.get();
        if row >= rows as usize {
            return;
        }
        let cols_sz = cols as usize;
        let base = row * cols_sz;

        // find max for numerical stability
        let mut max_val = x[base];
        let mut j = 1usize;
        while j < cols_sz {
            let v = x[base + j];
            if v > max_val {
                max_val = v;
            }
            j += 1;
        }

        // exp(x - max) and accumulate
        let mut sum = 0.0f32;
        let mut j = 0usize;
        while j < cols_sz {
            let e = gpu_expf(x[base + j] - max_val);
            unsafe {
                *y.get_unchecked_mut(base + j) = e;
            }
            sum += e;
            j += 1;
        }

        // normalize
        let inv = 1.0f32 / sum;
        let mut j = 0usize;
        while j < cols_sz {
            unsafe {
                *y.get_unchecked_mut(base + j) *= inv;
            }
            j += 1;
        }
    }

    // =========================================================================
    // Softmax — one 256-thread BLOCK per row (parallel reduction).
    //   Replaces the 1-thread-per-row softmax_row, which left ~all GPU lanes
    //   idle for transformer shapes (few rows, wide cols). Launch:
    //   grid=(rows,1,1), block=(256,1,1).
    // =========================================================================
    #[kernel]
    pub fn softmax_block(x: &[f32], _rows: u32, cols: u32, mut y: DisjointSlice<f32>) {
        static mut SM: SharedArray<f32, 256> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize;
        let c = cols as usize;
        let base = row * c;

        // Phase 1: row max (strided local, then shared tree reduce).
        let mut lmax = f32::NEG_INFINITY;
        let mut j = tid;
        while j < c {
            let v = x[base + j];
            if v > lmax {
                lmax = v;
            }
            j += 256;
        }
        unsafe { SM[tid] = lmax };
        thread::sync_threads();
        let mut s = 128usize;
        while s > 0 {
            if tid < s {
                let a = unsafe { SM[tid] };
                let b = unsafe { SM[tid + s] };
                unsafe { SM[tid] = if a > b { a } else { b } };
            }
            thread::sync_threads();
            s >>= 1;
        }
        let rmax = unsafe { SM[0] };
        thread::sync_threads();

        // Phase 2: write exp(x-rmax) and accumulate the row sum.
        let mut lsum = 0.0f32;
        let mut j = tid;
        while j < c {
            let e = gpu_expf(x[base + j] - rmax);
            unsafe { *y.get_unchecked_mut(base + j) = e };
            lsum += e;
            j += 256;
        }
        unsafe { SM[tid] = lsum };
        thread::sync_threads();
        let mut s = 128usize;
        while s > 0 {
            if tid < s {
                let a = unsafe { SM[tid] };
                let b = unsafe { SM[tid + s] };
                unsafe { SM[tid] = a + b };
            }
            thread::sync_threads();
            s >>= 1;
        }
        let inv = 1.0f32 / unsafe { SM[0] };
        thread::sync_threads();

        // Phase 3: normalize.
        let mut j = tid;
        while j < c {
            unsafe { *y.get_unchecked_mut(base + j) *= inv };
            j += 256;
        }
    }

    // =========================================================================
    // LayerNormalization — one 256-thread BLOCK per row (parallel reduction).
    //   Launch: grid=(rows,1,1), block=(256,1,1).
    // =========================================================================
    #[kernel]
    pub fn layernorm_block(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        _rows: u32,
        cols: u32,
        eps: f32,
        mut y: DisjointSlice<f32>,
    ) {
        static mut SM: SharedArray<f32, 256> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize;
        let c = cols as usize;
        let base = row * c;
        let inv_c = 1.0f32 / c as f32;

        // Phase 1: mean.
        let mut lsum = 0.0f32;
        let mut j = tid;
        while j < c {
            lsum += x[base + j];
            j += 256;
        }
        unsafe { SM[tid] = lsum };
        thread::sync_threads();
        let mut s = 128usize;
        while s > 0 {
            if tid < s {
                let a = unsafe { SM[tid] };
                let b = unsafe { SM[tid + s] };
                unsafe { SM[tid] = a + b };
            }
            thread::sync_threads();
            s >>= 1;
        }
        let mean = unsafe { SM[0] } * inv_c;
        thread::sync_threads();

        // Phase 2: variance.
        let mut lss = 0.0f32;
        let mut j = tid;
        while j < c {
            let d = x[base + j] - mean;
            lss += d * d;
            j += 256;
        }
        unsafe { SM[tid] = lss };
        thread::sync_threads();
        let mut s = 128usize;
        while s > 0 {
            if tid < s {
                let a = unsafe { SM[tid] };
                let b = unsafe { SM[tid + s] };
                unsafe { SM[tid] = a + b };
            }
            thread::sync_threads();
            s >>= 1;
        }
        let inv = gpu_rsqrt(unsafe { SM[0] } * inv_c + eps);
        thread::sync_threads();

        // Phase 3: normalize + affine.
        let mut j = tid;
        while j < c {
            let nrm = (x[base + j] - mean) * inv;
            unsafe { *y.get_unchecked_mut(base + j) = nrm * gamma[j] + beta[j] };
            j += 256;
        }
    }

    // =========================================================================
    // LayerNormalization — normalize over the last axis (size `cols`).
    //   y = (x - mean) / sqrt(var + eps) * gamma + beta
    //   One thread per row; `rows` = numel / cols.
    // =========================================================================
    #[kernel]
    pub fn layernorm(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        rows: u32,
        cols: u32,
        eps: f32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let row = idx.get();
        if row >= rows as usize {
            return;
        }
        let c = cols as usize;
        let base = row * c;

        let mut mean = 0.0f32;
        let mut j = 0usize;
        while j < c {
            mean += x[base + j];
            j += 1;
        }
        mean /= c as f32;

        let mut var = 0.0f32;
        let mut j = 0usize;
        while j < c {
            let d = x[base + j] - mean;
            var += d * d;
            j += 1;
        }
        var /= c as f32;
        let inv = gpu_rsqrt(var + eps);

        let mut j = 0usize;
        while j < c {
            let nrm = (x[base + j] - mean) * inv;
            unsafe {
                *y.get_unchecked_mut(base + j) = nrm * gamma[j] + beta[j];
            }
            j += 1;
        }
    }

    // =========================================================================
    // Erf — in-place element-wise error function (for exact GELU).
    // =========================================================================
    #[kernel]
    pub fn erf_inplace(mut x: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(v) = x.get_mut(idx) {
            *v = gpu_erf(*v);
        }
    }

    // Erf (out-of-place) — c[i] = erf(a[i]). Fused read→write.
    #[kernel]
    pub fn erf_fwd(a: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = c.get_mut(idx) {
            *o = gpu_erf(a[i]);
        }
    }

    // Tanh (out-of-place) — c[i] = tanh(a[i]). Fused read→write.
    #[kernel]
    pub fn tanh_fwd(a: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = c.get_mut(idx) {
            *o = gpu_tanh(a[i]);
        }
    }

    // =========================================================================
    // Tensor ⊙ scalar — c[i] = a[i] + s   /   c[i] = a[i] * s
    //   Used for scalar-broadcast Add/Mul/Div (e.g. GELU constants).
    // =========================================================================
    #[kernel]
    pub fn add_scalar(a: &[f32], s: f32, mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = c.get_mut(idx) {
            *o = a[i] + s;
        }
    }

    #[kernel]
    pub fn mul_scalar(a: &[f32], s: f32, mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = c.get_mut(idx) {
            *o = a[i] * s;
        }
    }

    // =========================================================================
    // Generic N-D Transpose — one thread per output element, fully on GPU.
    //   For output linear index i, decode coords via out_strides/out_shape,
    //   then gather from input at Σ coord[d] · in_strides[perm[d]].
    //   Shape/stride/perm vectors are passed as f32 (values < 2^24, exact).
    // =========================================================================
    #[kernel]
    pub fn transpose_nd(
        input: &[f32],
        out_shape: &[f32],
        out_strides: &[f32],
        in_strides: &[f32],
        perm: &[f32],
        ndim: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            let nd = ndim as usize;
            let mut in_off = 0usize;
            let mut d = 0usize;
            while d < nd {
                let os = out_strides[d] as usize;
                let osz = out_shape[d] as usize;
                let coord = (i / os) % osz;
                let src_dim = perm[d] as usize;
                in_off += coord * (in_strides[src_dim] as usize);
                d += 1;
            }
            *o = input[in_off];
        }
    }

    // =========================================================================
    // Concat one input into the output along `axis`, fully on GPU (no host
    // round-trip / stream sync). Output is [outer, out_axis_len, inner]; this
    // input is [outer, sz, inner] placed at axis offset `start`.
    //   One thread per *input* element; scatters into the big output.
    // =========================================================================
    #[kernel]
    pub fn concat_axis(
        input: &[f32],
        out_axis_len: u32,
        inner: u32,
        start: u32,
        sz: u32,
        outer: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let inn = inner as usize;
        let sz_inn = sz as usize * inn;
        let total = outer as usize * sz_inn;
        if i < total {
            let oo = i / sz_inn;
            let rem = i % sz_inn;
            let j = rem / inn;
            let k = rem % inn;
            let dst = oo * (out_axis_len as usize) * inn + (start as usize + j) * inn + k;
            unsafe {
                *y.get_unchecked_mut(dst) = input[i];
            }
        }
    }

    // =========================================================================
    // Slice along one axis — y = input[..., start:start+sz, ...]  (GPU Split).
    //   input is [outer, axis_len, inner]; output is [outer, sz, inner].
    // =========================================================================
    #[kernel]
    pub fn slice_axis(
        input: &[f32],
        axis_len: u32,
        inner: u32,
        start: u32,
        sz: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            let inn = inner as usize;
            let sz_inn = sz as usize * inn;
            let oo = i / sz_inn;
            let rem = i % sz_inn;
            let j = rem / inn;
            let k = rem % inn;
            let src = oo * (axis_len as usize) * inn + (start as usize + j) * inn + k;
            *o = input[src];
        }
    }

    // =========================================================================
    // Gather along one axis — y[o, ii, k] = data[o, indices[ii], k]  (GPU).
    //   data is [outer, axis_len, inner]; indices are pre-resolved (≥0).
    // =========================================================================
    #[kernel]
    pub fn gather_axis(
        data: &[f32],
        indices: &[f32],
        axis_len: u32,
        inner: u32,
        n_idx: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            let inn = inner as usize;
            let ni_inn = n_idx as usize * inn;
            let oo = i / ni_inn;
            let rem = i % ni_inn;
            let ii = rem / inn;
            let k = rem % inn;
            let g = indices[ii] as usize;
            let src = oo * (axis_len as usize) * inn + g * inn + k;
            *o = data[src];
        }
    }

    // =========================================================================
    // Tanh — in-place element-wise (BERT pooler).
    // =========================================================================
    #[kernel]
    pub fn tanh_inplace(mut x: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(v) = x.get_mut(idx) {
            *v = gpu_tanh(*v);
        }
    }

    // =========================================================================
    // Scalar − tensor:  c[i] = s − a[i]   (BERT 1.0 − attention_mask).
    // =========================================================================
    #[kernel]
    pub fn sub_scalar_lhs(a: &[f32], s: f32, mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = c.get_mut(idx) {
            *o = s - a[i];
        }
    }

    // =========================================================================
    // Broadcasting Add — c = a + b with NumPy broadcasting.
    //   Shapes/strides are left-padded to `ndim` and passed as f32 (exact,
    //   values < 2^24); a broadcast dim has stride 0. One thread per output.
    // =========================================================================
    #[kernel]
    pub fn add_bcast(
        a: &[f32],
        b: &[f32],
        out_shape: &[f32],
        out_strides: &[f32],
        a_strides: &[f32],
        b_strides: &[f32],
        ndim: u32,
        mut c: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = c.get_mut(idx) {
            let nd = ndim as usize;
            let mut ao = 0usize;
            let mut bo = 0usize;
            let mut d = 0usize;
            while d < nd {
                let coord = (i / out_strides[d] as usize) % out_shape[d] as usize;
                ao += coord * a_strides[d] as usize;
                bo += coord * b_strides[d] as usize;
                d += 1;
            }
            *o = a[ao] + b[bo];
        }
    }

    // =========================================================================
    // Pow with a small non-negative integer exponent (GELU x³, etc.).
    //   Repeated multiply — correct for negative bases (unlike exp/log).
    // =========================================================================
    #[kernel]
    pub fn pow_scalar(a: &[f32], exp: f32, mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = c.get_mut(idx) {
            let n = exp as i32;
            let x = a[i];
            let mut acc = 1.0f32;
            let mut k = 0i32;
            while k < n {
                acc *= x;
                k += 1;
            }
            *o = acc;
        }
    }

    // =========================================================================
    // Where (broadcasting) — c = cond ? x : y.  cond is f32 (1.0 = true).
    //   Shapes/strides left-padded to ndim, passed as f32; stride 0 broadcasts.
    // =========================================================================
    #[kernel]
    pub fn where_bcast(
        cond: &[f32],
        x: &[f32],
        y: &[f32],
        out_shape: &[f32],
        out_strides: &[f32],
        c_strides: &[f32],
        x_strides: &[f32],
        y_strides: &[f32],
        ndim: u32,
        mut o: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = o.get_mut(idx) {
            let nd = ndim as usize;
            let mut co = 0usize;
            let mut xo = 0usize;
            let mut yo = 0usize;
            let mut d = 0usize;
            while d < nd {
                let coord = (i / out_strides[d] as usize) % out_shape[d] as usize;
                co += coord * c_strides[d] as usize;
                xo += coord * x_strides[d] as usize;
                yo += coord * y_strides[d] as usize;
                d += 1;
            }
            *out = if cond[co] != 0.0f32 { x[xo] } else { y[yo] };
        }
    }
}
