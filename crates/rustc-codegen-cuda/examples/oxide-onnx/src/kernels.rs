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

use cuda_device::convert::cvt_f16x2_f32;
use cuda_device::wgmma::{mma_sync_m16n8k8_f32_tf32, mma_sync_m16n8k16_f32_f16};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

// These helpers avoid core::intrinsics::sqrtf32 / expf32, which the oxide
// mir-lower pipeline maps to __nv_sqrtf / __nv_expf (libdevice). Libdevice
// calls trigger NVVM IR mode and skip PTX embedding. Instead we use only
// standard LLVM instructions: bitcast, integer arithmetic, fmul, fadd.

/// Logistic sigmoid, built from `gpu_exp` for the same reason as `gpu_rsqrt`:
/// calling libdevice would switch the backend to NVVM IR mode.
#[inline(always)]
fn gpu_sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + gpu_expf(-x))
}

/// Compute 1/sqrt(x) using the Carmack bit-hack followed by two Newton steps.
/// Relative error < 0.001%.
#[inline(always)]
fn gpu_rsqrt(x: f32) -> f32 {
    let bits = x.to_bits();
    let guess = f32::from_bits(0x5f375a86u32.wrapping_sub(bits >> 1));
    let step = |y: f32| y * (1.5f32 - 0.5f32 * x * y * y);
    step(step(guess))
}

/// Apply a fused epilogue activation.
///
/// `act` matches the `graph_opt::ACT_*` codes the load-time graph rewrite
/// stamps onto a node: 0 = none, 1 = relu, 2 = clip to `[lo, hi]`. Branching on
/// a kernel-uniform argument costs nothing — every thread in the grid takes the
/// same path.
#[inline(always)]
fn apply_act(v: f32, act: u32, lo: f32, hi: f32) -> f32 {
    if act == 1u32 {
        if v > 0.0f32 { v } else { 0.0f32 }
    } else if act == 2u32 {
        if v < lo {
            lo
        } else if v > hi {
            hi
        } else {
            v
        }
    } else if act == 4u32 {
        // tanh-approximate GELU, the form GPT-2 exports:
        //   x/2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
        // Eight graph nodes, each a full pass over the activation, for this.
        let inner = 0.7978845608f32 * (v + 0.044715f32 * v * v * v);
        0.5f32 * v * (1.0f32 + gpu_tanh(inner))
    } else if act == 3u32 {
        // Exact GELU: x/2 * (1 + erf(x/sqrt(2))). A transformer's MLP spells
        // this as five separate nodes, each a full round trip of the 197x3072
        // activation through DRAM; here it costs a few instructions on a value
        // the epilogue is already holding.
        0.5f32 * v * (1.0f32 + gpu_erf(v * core::f32::consts::FRAC_1_SQRT_2))
    } else {
        v
    }
}

/// IEEE-754 binary32 → binary16 bit pattern, round-to-nearest-even.
///
/// Superseded on device by `cvt.rn.f16x2.f32` (see `pack_f16x2`); kept because
/// it documents the rounding the hardware instruction performs and needs no
/// device to run.
#[allow(dead_code)]
///
/// Integer ops only, for the same reason as `gpu_rsqrt`: float intrinsics map
/// to libdevice, which pulls the kernel into NVVM IR mode and skips PTX
/// embedding. Infinities and NaNs saturate to infinity; subnormal results
/// flush to zero, which is what a GEMM operand path wants.
#[inline(always)]
fn f32_to_f16_bits(x: f32) -> u32 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        return sign | 0x7c00;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        return sign;
    }
    let round_bias = 0x0fff + ((mant >> 13) & 1);
    let mant_rounded = mant + round_bias;
    let exp_adjusted = new_exp as u32 + (mant_rounded >> 23);
    if exp_adjusted >= 0x1f {
        return sign | 0x7c00;
    }
    sign | (exp_adjusted << 10) | ((mant_rounded >> 13) & 0x03ff)
}

/// Pack two f32 values as two f16 halves in one register, `lo` in the low 16
/// bits — the order an `.f16x2` operand of `mma.sync` expects.
///
/// `cvt.rn.f16x2.f32` does this in one instruction with the same
/// round-to-nearest-even semantics as the integer emulation below. The
/// emulation exists because float intrinsics that route through libdevice
/// would push the kernel into NVVM IR mode; this one is a generated PTX
/// intrinsic, so it does not. It matters most in the convolution, which packs
/// activations on the fly rather than at load time: the software form is about
/// fifteen integer ops per value, four values per thread per K step, and the
/// kernel's dominant stall is execution dependency.
#[inline(always)]
fn pack_f16x2(lo: f32, hi: f32) -> u32 {
    cvt_f16x2_f32(lo, hi)
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
    // GEMM 16×16 tiled, **u32-indexed** — C = alpha·A·B + beta·C (row-major).
    //
    //   Same algorithm as sgemm_tiled, but every index/loop var is u32 and the
    //   output is a flat slice (no Runtime2DIndex struct). On this backend
    //   `usize` math is widened to 64-bit, and the b64 register bloat caps
    //   occupancy (sgemm_tiled spills ~178 regs/thread). Staying in 32-bit
    //   roughly halves the register file footprint → far higher occupancy.
    //   All index products here are < 2^31 for the benchmarked models.
    //
    //   Launch: grid=(⌈n/16⌉, ⌈m/16⌉, 1), block=(16,16,1).
    // =========================================================================
    #[kernel]
    pub fn sgemm_u32(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32>,
    ) {
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();
        let row = thread::blockIdx_y() * 16 + ty;
        let col = thread::blockIdx_x() * 16 + tx;
        let smem = (ty * 16 + tx) as usize;

        let num_tiles = k.div_ceil(16);
        let mut sum = 0.0f32;

        let mut t = 0u32;
        while t < num_tiles {
            let tile_start = t * 16;
            unsafe {
                let a_col = tile_start + tx;
                TILE_A[smem] = if row < m && a_col < k {
                    a[(row * k + a_col) as usize]
                } else {
                    0.0f32
                };
                let b_row = tile_start + ty;
                TILE_B[smem] = if b_row < k && col < n {
                    b[(b_row * n + col) as usize]
                } else {
                    0.0f32
                };
            }
            thread::sync_threads();
            unsafe {
                let arow = (ty * 16) as usize;
                let mut i = 0usize;
                while i < 16 {
                    sum += TILE_A[arow + i] * TILE_B[i * 16 + tx as usize];
                    i += 1;
                }
            }
            thread::sync_threads();
            t += 1;
        }

        if row < m && col < n {
            let idx = (row * n + col) as usize;
            let cell = unsafe { c.get_unchecked_mut(idx) };
            *cell = if beta != 0.0f32 {
                alpha * sum + beta * (*cell)
            } else {
                alpha * sum
            };
        }
    }

    // =========================================================================
    // GEMM with fused bias + activation epilogue.
    //
    //   Identical inner loop to `sgemm_tiled` — which measured fastest on every
    //   batch-1 shape in these models — but the per-row bias and the activation
    //   the graph rewrite folded in are applied while the result is still in a
    //   register. For convolution that removes an entire extra pass over the
    //   output tensor per layer (53 of them in ResNet50).
    //
    //   `bias` is indexed by output row (= output channel for im2col conv);
    //   when `has_bias` is 0 the operand is not read and may be any slice.
    //
    //   Launch: same as `sgemm_tiled` — grid=(⌈n/16⌉, ⌈m/16⌉, 1), block=(16,16,1).
    // =========================================================================
    #[kernel]
    pub fn sgemm_bias_act(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        bias: &[f32],
        has_bias: u32,
        act: u32,
        lo: f32,
        hi: f32,
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
                let b_val = if has_bias != 0u32 && row < bias.len() {
                    bias[row]
                } else {
                    0.0f32
                };
                if let Some(c_elem) = c.get_mut(c_idx) {
                    *c_elem = apply_act(alpha * sum + b_val, act, lo, hi);
                }
            }
        }
    }

    // =========================================================================
    // GEMM, N-register-tiled — C = alpha·A·B + beta·C  (row-major).
    //
    //   The naive 16×16 tiled kernel does 1 MAC per 2 shared loads inside a
    //   non-unrolled K-loop; this backend also can't fuse mul+add, so that is
    //   ~4 instructions per useful MAC. Here each thread owns ONE C row and
    //   EIGHT C columns (scalar accumulators), so each staged A value feeds 8
    //   MACs: arithmetic intensity ~2× higher and loop overhead amortised 8×.
    //   Block tile 16(M)×128(N), BK=8. Row-major M-parallel mapping keeps the
    //   block count high for the small-M conv-im2col GEMM shapes (where the
    //   square 64×64 register-blocked variant starves occupancy).
    //
    //   Launch: grid=(⌈n/128⌉, ⌈m/16⌉, 1), block=(16,16,1).
    // =========================================================================
    #[kernel]
    pub fn sgemm_fast(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        beta: f32,
        mut c: DisjointSlice<f32>,
    ) {
        // AS: [16][8] = 128.  BS: [8][128] = 1024.
        static mut AS: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut BS: SharedArray<f32, 1024> = SharedArray::UNINIT;

        // All index math is u32 (this backend widens usize→64-bit, and the
        // b64 register bloat caps occupancy). Cast to usize only at the
        // slice/shared-array access. Index products < 2^31 for these models.
        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();
        let tid = ty * 16 + tx;
        let row = thread::blockIdx_y() * 16 + ty;
        let brow0 = thread::blockIdx_y() * 16;
        let col0 = thread::blockIdx_x() * 128;

        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let mut s4 = 0.0f32;
        let mut s5 = 0.0f32;
        let mut s6 = 0.0f32;
        let mut s7 = 0.0f32;

        let mut k0 = 0u32;
        while k0 < k {
            // Stage A (16×8 = 128): threads [0,128) load one element each.
            if tid < 128 {
                let r = tid >> 3; // 0..16
                let cc = tid & 7; // 0..8
                let gr = brow0 + r;
                let gc = k0 + cc;
                unsafe {
                    AS[tid as usize] = if gr < m && gc < k {
                        a[(gr * k + gc) as usize]
                    } else {
                        0.0f32
                    };
                }
            }
            // Stage B (8×128 = 1024): 256 threads, 4 elements each.
            let mut q = 0u32;
            while q < 4 {
                let e = tid + q * 256;
                let r = e >> 7; // 0..8
                let cc = e & 127; // 0..128
                let gr = k0 + r;
                let gc = col0 + cc;
                unsafe {
                    BS[e as usize] = if gr < k && gc < n {
                        b[(gr * n + gc) as usize]
                    } else {
                        0.0f32
                    };
                }
                q += 1;
            }
            thread::sync_threads();

            let arow = (ty * 8) as usize;
            let bcol = tx as usize;
            let mut kk = 0usize;
            while kk < 8 {
                let av = unsafe { AS[arow + kk] };
                let bo = kk * 128 + bcol;
                s0 += av * unsafe { BS[bo] };
                s1 += av * unsafe { BS[bo + 16] };
                s2 += av * unsafe { BS[bo + 32] };
                s3 += av * unsafe { BS[bo + 48] };
                s4 += av * unsafe { BS[bo + 64] };
                s5 += av * unsafe { BS[bo + 80] };
                s6 += av * unsafe { BS[bo + 96] };
                s7 += av * unsafe { BS[bo + 112] };
                kk += 1;
            }
            thread::sync_threads();
            k0 += 8;
        }

        if row < m {
            let has_beta = beta != 0.0f32;
            let base = row * n;
            let mut t = 0u32;
            while t < 8 {
                let gc = col0 + tx + 16 * t;
                if gc < n {
                    let sv = if t == 0 {
                        s0
                    } else if t == 1 {
                        s1
                    } else if t == 2 {
                        s2
                    } else if t == 3 {
                        s3
                    } else if t == 4 {
                        s4
                    } else if t == 5 {
                        s5
                    } else if t == 6 {
                        s6
                    } else {
                        s7
                    };
                    let cell = unsafe { c.get_unchecked_mut((base + gc) as usize) };
                    *cell = if has_beta {
                        alpha * sv + beta * (*cell)
                    } else {
                        alpha * sv
                    };
                }
                t += 1;
            }
        }
    }

    // =========================================================================
    // GEMM, 2-D register-tiled — C = alpha·A·B + beta·C  (row-major).
    //
    //   Arithmetic intensity is what separates a toy GEMM from a fast one:
    //
    //     sgemm_tiled  1 output/thread   1 MAC  per 2 shared loads  (0.5)
    //     sgemm_fast   1×8 outputs       8 MACs per 9 shared loads  (0.9)
    //     this kernel  4×4 outputs      16 MACs per 8 shared loads  (2.0)
    //
    //   Each thread holds a 4×4 accumulator block in registers, so one staged
    //   A value feeds 4 MACs and one staged B value feeds 4 more. The
    //   local-memory spilling that made the earlier register-blocked attempt
    //   slower than the naive kernel is gone (no `.local` in the emitted PTX).
    //
    //   The tile is 64×64 rather than the textbook 128×128 because these models
    //   run at batch 1: a 128×128 tile leaves a late ResNet stage (M=512, N=49)
    //   with 4 blocks for 82 SMs and measured 4× *slower* than the naive
    //   kernel. Arithmetic intensity is worth nothing on an idle GPU.
    //
    //   Rows and columns are assigned to threads with stride 16 rather than in
    //   contiguous runs of 8: for a fixed sub-index the 16 threads of a row
    //   then read 16 consecutive floats out of shared memory, which is
    //   conflict-free, where contiguous blocking would make it 4-way banked.
    //
    //   Block tile 64(M)×64(N), BK=8.
    //   Launch: grid=(⌈n/64⌉, ⌈m/64⌉, 1), block=(16,16,1).
    // =========================================================================
    #[kernel]
    pub fn sgemm_reg(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        bias: &[f32],
        has_bias: u32,
        act: u32,
        lo: f32,
        hi: f32,
        mut c: DisjointSlice<f32>,
    ) {
        // AS/BS are [BK=8][64] tiles: 2 KB each, 4 KB per block.
        static mut AS: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut BS: SharedArray<f32, 512> = SharedArray::UNINIT;

        // u32 index math throughout: the backend widens usize to 64-bit, and
        // the extra b64 registers cap occupancy. All products stay < 2^31.
        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();
        let tid = ty * 16 + tx;
        let row0 = thread::blockIdx_y() * 64;
        let col0 = thread::blockIdx_x() * 64;

        let mut acc = [[0.0f32; 4]; 4];

        let mut k0 = 0u32;
        while k0 < k {
            // Stage A (64 rows x 8 k) and B (8 k x 64 cols): 512 elements
            // each, 256 threads, 2 elements per thread.
            let mut q = 0u32;
            #[unroll]
            while q < 2 {
                let e = tid + q * 256;

                // A: element e is (row r, k-offset cc), stored transposed as
                // AS[cc][r] so the inner loop reads along r.
                let ar = e >> 3; // 0..64
                let ak = e & 7; // 0..8
                let agr = row0 + ar;
                let agc = k0 + ak;
                unsafe {
                    AS[(ak * 64 + ar) as usize] = if agr < m && agc < k {
                        a[(agr * k + agc) as usize]
                    } else {
                        0.0f32
                    };
                }

                // B: element e is (k-offset r, column cc), stored as BS[r][cc].
                let br = e >> 6; // 0..8
                let bc = e & 63; // 0..64
                let bgr = k0 + br;
                let bgc = col0 + bc;
                unsafe {
                    BS[e as usize] = if bgr < k && bgc < n {
                        b[(bgr * n + bgc) as usize]
                    } else {
                        0.0f32
                    };
                }
                q += 1;
            }
            thread::sync_threads();

            let mut kk = 0u32;
            #[unroll]
            while kk < 8 {
                let arow_base = (kk * 64 + ty) as usize;
                let bcol_base = (kk * 64 + tx) as usize;

                let a_frag = [
                    unsafe { AS[arow_base] },
                    unsafe { AS[arow_base + 16] },
                    unsafe { AS[arow_base + 32] },
                    unsafe { AS[arow_base + 48] },
                ];
                let b_frag = [
                    unsafe { BS[bcol_base] },
                    unsafe { BS[bcol_base + 16] },
                    unsafe { BS[bcol_base + 32] },
                    unsafe { BS[bcol_base + 48] },
                ];

                let mut i = 0usize;
                #[unroll]
                while i < 4 {
                    let av = a_frag[i];
                    let mut j = 0usize;
                    #[unroll]
                    while j < 4 {
                        acc[i][j] += av * b_frag[j];
                        j += 1;
                    }
                    i += 1;
                }
                kk += 1;
            }
            thread::sync_threads();
            k0 += 8;
        }

        let bias_len = bias.len();
        let mut i = 0u32;
        #[unroll]
        while i < 4 {
            let gr = row0 + ty + 16 * i;
            if gr < m {
                let b_val = if has_bias != 0u32 && (gr as usize) < bias_len {
                    bias[gr as usize]
                } else {
                    0.0f32
                };
                let base = gr * n;
                let mut j = 0u32;
                #[unroll]
                while j < 4 {
                    let gc = col0 + tx + 16 * j;
                    if gc < n {
                        let v = alpha * acc[i as usize][j as usize] + b_val;
                        unsafe {
                            *c.get_unchecked_mut((base + gc) as usize) = apply_act(v, act, lo, hi);
                        }
                    }
                    j += 1;
                }
            }
            i += 1;
        }
    }

    // =========================================================================
    // f16 tensor-core GEMM — C = act(alpha·A·B + bias), row-major.
    //
    //   On GA10x the f16 tensor cores with f32 accumulate run at roughly twice
    //   the FP32 CUDA-core rate, while tf32 runs at parity with it — which is
    //   why the tf32 mma.sync path never paid for itself here and this one can.
    //
    //   128 threads (4 warps) per 64x64 output tile, K stepped 16 at a time.
    //   Warp w owns rows [16w, 16w+16) across all 64 columns, i.e. eight
    //   m16n8k16 tiles, so one A fragment (4 registers) feeds all eight MMAs
    //   while only B changes: 512 MACs per lane per K-step against 20 shared
    //   loads. Accumulators are 8x4 f32 in registers.
    //
    //   Shared memory holds halves already packed two-per-register in the
    //   layout the fragments want — A packed along K row-major, B packed along
    //   K column-major — so the f32→f16 conversion happens once per staged
    //   element rather than once per fragment read.
    //
    //   Launch: grid=(ceil(n/64), ceil(m/64), 1), block=(128,1,1).
    // =========================================================================
    #[kernel]
    pub fn sgemm_f16_tc(
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        a: &[f32],
        b: &[f32],
        bias: &[f32],
        has_bias: u32,
        act: u32,
        lo: f32,
        hi: f32,
        mut c: DisjointSlice<f32>,
    ) {
        // 64 rows x 16 halves = 512 u32 each; 2 KB per tile, 4 KB per block.
        static mut AS: SharedArray<u32, 512> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, 512> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let warp = tid >> 5;
        let lane = tid & 31;
        let gid = lane >> 2; // groupID 0..7
        let tig = lane & 3; // threadID_in_group 0..3

        let row0 = thread::blockIdx_y() * 64;
        let col0 = thread::blockIdx_x() * 64;

        let mut acc = [[0.0f32; 4]; 8];

        let mut k0 = 0u32;
        while k0 < k {
            // Stage A as AS[row][kpair]: 512 registers, 4 per thread.
            let mut q = 0u32;
            #[unroll]
            while q < 4 {
                let e = tid + q * 128;
                let r = e >> 3;
                let kk = (e & 7) * 2;
                let gr = row0 + r;
                let g0 = k0 + kk;
                let v0 = if gr < m && g0 < k {
                    a[(gr * k + g0) as usize]
                } else {
                    0.0f32
                };
                let v1 = if gr < m && g0 + 1 < k {
                    a[(gr * k + g0 + 1) as usize]
                } else {
                    0.0f32
                };
                unsafe {
                    AS[e as usize] = pack_f16x2(v0, v1);
                }
                q += 1;
            }
            // Stage B as BS[col][kpair] — transposed relative to A, because the
            // B fragment wants a column's two K neighbours in one register.
            let mut q2 = 0u32;
            #[unroll]
            while q2 < 4 {
                let e = tid + q2 * 128;
                let cc = e >> 3;
                let kk = (e & 7) * 2;
                let gc = col0 + cc;
                let g0 = k0 + kk;
                let v0 = if gc < n && g0 < k {
                    b[(g0 * n + gc) as usize]
                } else {
                    0.0f32
                };
                let v1 = if gc < n && g0 + 1 < k {
                    b[((g0 + 1) * n + gc) as usize]
                } else {
                    0.0f32
                };
                unsafe {
                    BS[e as usize] = pack_f16x2(v0, v1);
                }
                q2 += 1;
            }
            thread::sync_threads();

            // One A fragment for this warp's 16 rows, reused by all 8 tiles.
            let arow = warp * 16;
            let a0 = unsafe { AS[((arow + gid) * 8 + tig) as usize] };
            let a1 = unsafe { AS[((arow + gid + 8) * 8 + tig) as usize] };
            let a2 = unsafe { AS[((arow + gid) * 8 + tig + 4) as usize] };
            let a3 = unsafe { AS[((arow + gid + 8) * 8 + tig + 4) as usize] };

            let mut t = 0usize;
            #[unroll]
            while t < 8 {
                let ncol = t as u32 * 8 + gid;
                let b0 = unsafe { BS[(ncol * 8 + tig) as usize] };
                let b1 = unsafe { BS[(ncol * 8 + tig + 4) as usize] };
                unsafe {
                    mma_sync_m16n8k16_f32_f16(&mut acc[t], a0, a1, a2, a3, b0, b1);
                }
                t += 1;
            }
            thread::sync_threads();
            k0 += 16;
        }

        // Epilogue: rows gid and gid+8 of this warp's strip, columns
        // 2*tig and 2*tig+1 of each 8-wide tile.
        let bias_len = bias.len();
        let mut t = 0usize;
        #[unroll]
        while t < 8 {
            let gc = col0 + t as u32 * 8 + 2 * tig;
            let mut half = 0u32;
            #[unroll]
            while half < 2 {
                let gr = row0 + warp * 16 + gid + half * 8;
                if gr < m {
                    let b_val = if has_bias != 0u32 && (gr as usize) < bias_len {
                        bias[gr as usize]
                    } else {
                        0.0f32
                    };
                    let base = gr * n;
                    let v0 = alpha * acc[t][(half * 2) as usize] + b_val;
                    let v1 = alpha * acc[t][(half * 2 + 1) as usize] + b_val;
                    if gc < n {
                        unsafe {
                            *c.get_unchecked_mut((base + gc) as usize) = apply_act(v0, act, lo, hi);
                        }
                    }
                    if gc + 1 < n {
                        unsafe {
                            *c.get_unchecked_mut((base + gc + 1) as usize) =
                                apply_act(v1, act, lo, hi);
                        }
                    }
                }
                half += 1;
            }
            t += 1;
        }
    }

    #[kernel]
    pub fn sgemm_f16_tc_splitk(
        m: u32,
        n: u32,
        k: u32,
        k_per_split: u32,
        a: &[f32],
        b: &[f32],
        mut partials: DisjointSlice<f32>,
    ) {
        // 64 rows x 16 halves = 512 u32 each; 2 KB per tile, 4 KB per block.
        static mut AS: SharedArray<u32, 512> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, 512> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let warp = tid >> 5;
        let lane = tid & 31;
        let gid = lane >> 2; // groupID 0..7
        let tig = lane & 3; // threadID_in_group 0..3

        let row0 = thread::blockIdx_y() * 64;
        let col0 = thread::blockIdx_x() * 64;
        let split = thread::blockIdx_z();

        let k_begin = split * k_per_split;
        let k_stop = if k_begin + k_per_split < k {
            k_begin + k_per_split
        } else {
            k
        };

        let mut acc = [[0.0f32; 4]; 8];

        let mut k0 = k_begin;
        while k0 < k_stop {
            // Stage A as AS[row][kpair]: 512 registers, 4 per thread.
            let mut q = 0u32;
            #[unroll]
            while q < 4 {
                let e = tid + q * 128;
                let r = e >> 3;
                let kk = (e & 7) * 2;
                let gr = row0 + r;
                let g0 = k0 + kk;
                let v0 = if gr < m && g0 < k_stop {
                    a[(gr * k + g0) as usize]
                } else {
                    0.0f32
                };
                let v1 = if gr < m && g0 + 1 < k_stop {
                    a[(gr * k + g0 + 1) as usize]
                } else {
                    0.0f32
                };
                unsafe {
                    AS[e as usize] = pack_f16x2(v0, v1);
                }
                q += 1;
            }
            // Stage B as BS[col][kpair] — transposed relative to A, because the
            // B fragment wants a column's two K neighbours in one register.
            let mut q2 = 0u32;
            #[unroll]
            while q2 < 4 {
                let e = tid + q2 * 128;
                let cc = e >> 3;
                let kk = (e & 7) * 2;
                let gc = col0 + cc;
                let g0 = k0 + kk;
                let v0 = if gc < n && g0 < k_stop {
                    b[(g0 * n + gc) as usize]
                } else {
                    0.0f32
                };
                let v1 = if gc < n && g0 + 1 < k_stop {
                    b[((g0 + 1) * n + gc) as usize]
                } else {
                    0.0f32
                };
                unsafe {
                    BS[e as usize] = pack_f16x2(v0, v1);
                }
                q2 += 1;
            }
            thread::sync_threads();

            // One A fragment for this warp's 16 rows, reused by all 8 tiles.
            let arow = warp * 16;
            let a0 = unsafe { AS[((arow + gid) * 8 + tig) as usize] };
            let a1 = unsafe { AS[((arow + gid + 8) * 8 + tig) as usize] };
            let a2 = unsafe { AS[((arow + gid) * 8 + tig + 4) as usize] };
            let a3 = unsafe { AS[((arow + gid + 8) * 8 + tig + 4) as usize] };

            let mut t = 0usize;
            #[unroll]
            while t < 8 {
                let ncol = t as u32 * 8 + gid;
                let b0 = unsafe { BS[(ncol * 8 + tig) as usize] };
                let b1 = unsafe { BS[(ncol * 8 + tig + 4) as usize] };
                unsafe {
                    mma_sync_m16n8k16_f32_f16(&mut acc[t], a0, a1, a2, a3, b0, b1);
                }
                t += 1;
            }
            thread::sync_threads();
            k0 += 16;
        }

        // Partial store; alpha, bias and activation belong to `reduce_splits`.
        let plane = split * m * n;
        let mut t = 0usize;
        #[unroll]
        while t < 8 {
            let gc = col0 + t as u32 * 8 + 2 * tig;
            let mut half = 0u32;
            #[unroll]
            while half < 2 {
                let gr = row0 + warp * 16 + gid + half * 8;
                if gr < m {
                    let base = plane + gr * n;
                    if gc < n {
                        unsafe {
                            *partials.get_unchecked_mut((base + gc) as usize) =
                                acc[t][(half * 2) as usize];
                        }
                    }
                    if gc + 1 < n {
                        unsafe {
                            *partials.get_unchecked_mut((base + gc + 1) as usize) =
                                acc[t][(half * 2 + 1) as usize];
                        }
                    }
                }
                half += 1;
            }
            t += 1;
        }
    }

    #[kernel]
    pub fn sgemm_f16_tc_splitk_wpacked(
        m: u32,
        n: u32,
        k: u32,
        k_per_split: u32,
        a_packed: &[u32],
        kpairs: u32,
        b: &[f32],
        mut partials: DisjointSlice<f32>,
    ) {
        // 64 rows x 16 halves = 512 u32 each; 2 KB per tile, 4 KB per block.
        static mut AS: SharedArray<u32, 512> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, 512> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let warp = tid >> 5;
        let lane = tid & 31;
        let gid = lane >> 2; // groupID 0..7
        let tig = lane & 3; // threadID_in_group 0..3

        let row0 = thread::blockIdx_y() * 64;
        let col0 = thread::blockIdx_x() * 64;
        let split = thread::blockIdx_z();

        let k_begin = split * k_per_split;
        let k_stop = if k_begin + k_per_split < k {
            k_begin + k_per_split
        } else {
            k
        };

        let mut acc = [[0.0f32; 4]; 8];

        let mut k0 = k_begin;
        while k0 < k_stop {
            // Stage A as AS[row][kpair]: 512 registers, 4 per thread.
            let mut q = 0u32;
            #[unroll]
            while q < 4 {
                let e = tid + q * 128;
                let r = e >> 3;
                let kk = (e & 7) * 2;
                let gr = row0 + r;
                let g0 = k0 + kk;
                // Already half-packed at load time: one 32-bit read, no
                // conversion, half the bytes of the f32 form.
                unsafe {
                    AS[e as usize] = if gr < m && g0 < k_stop {
                        a_packed[(gr * kpairs + g0 / 2) as usize]
                    } else {
                        0u32
                    };
                }
                q += 1;
            }
            // Stage B as BS[col][kpair] — transposed relative to A, because the
            // B fragment wants a column's two K neighbours in one register.
            let mut q2 = 0u32;
            #[unroll]
            while q2 < 4 {
                let e = tid + q2 * 128;
                let cc = e >> 3;
                let kk = (e & 7) * 2;
                let gc = col0 + cc;
                let g0 = k0 + kk;
                let v0 = if gc < n && g0 < k_stop {
                    b[(g0 * n + gc) as usize]
                } else {
                    0.0f32
                };
                let v1 = if gc < n && g0 + 1 < k_stop {
                    b[((g0 + 1) * n + gc) as usize]
                } else {
                    0.0f32
                };
                unsafe {
                    BS[e as usize] = pack_f16x2(v0, v1);
                }
                q2 += 1;
            }
            thread::sync_threads();

            // One A fragment for this warp's 16 rows, reused by all 8 tiles.
            let arow = warp * 16;
            let a0 = unsafe { AS[((arow + gid) * 8 + tig) as usize] };
            let a1 = unsafe { AS[((arow + gid + 8) * 8 + tig) as usize] };
            let a2 = unsafe { AS[((arow + gid) * 8 + tig + 4) as usize] };
            let a3 = unsafe { AS[((arow + gid + 8) * 8 + tig + 4) as usize] };

            let mut t = 0usize;
            #[unroll]
            while t < 8 {
                let ncol = t as u32 * 8 + gid;
                let b0 = unsafe { BS[(ncol * 8 + tig) as usize] };
                let b1 = unsafe { BS[(ncol * 8 + tig + 4) as usize] };
                unsafe {
                    mma_sync_m16n8k16_f32_f16(&mut acc[t], a0, a1, a2, a3, b0, b1);
                }
                t += 1;
            }
            thread::sync_threads();
            k0 += 16;
        }

        // Partial store; alpha, bias and activation belong to `reduce_splits`.
        let plane = split * m * n;
        let mut t = 0usize;
        #[unroll]
        while t < 8 {
            let gc = col0 + t as u32 * 8 + 2 * tig;
            let mut half = 0u32;
            #[unroll]
            while half < 2 {
                let gr = row0 + warp * 16 + gid + half * 8;
                if gr < m {
                    let base = plane + gr * n;
                    if gc < n {
                        unsafe {
                            *partials.get_unchecked_mut((base + gc) as usize) =
                                acc[t][(half * 2) as usize];
                        }
                    }
                    if gc + 1 < n {
                        unsafe {
                            *partials.get_unchecked_mut((base + gc + 1) as usize) =
                                acc[t][(half * 2 + 1) as usize];
                        }
                    }
                }
                half += 1;
            }
            t += 1;
        }
    }

    #[kernel]
    pub fn sgemm_f16_tc_splitk_bpacked(
        m: u32,
        n: u32,
        k: u32,
        k_per_split: u32,
        a: &[f32],
        b_packed: &[u32],
        kpairs: u32,
        mut partials: DisjointSlice<f32>,
    ) {
        // 64 rows x 16 halves = 512 u32 each; 2 KB per tile, 4 KB per block.
        static mut AS: SharedArray<u32, 512> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, 512> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let warp = tid >> 5;
        let lane = tid & 31;
        let gid = lane >> 2; // groupID 0..7
        let tig = lane & 3; // threadID_in_group 0..3

        let row0 = thread::blockIdx_y() * 64;
        let col0 = thread::blockIdx_x() * 64;
        let split = thread::blockIdx_z();

        let k_begin = split * k_per_split;
        let k_stop = if k_begin + k_per_split < k {
            k_begin + k_per_split
        } else {
            k
        };

        let mut acc = [[0.0f32; 4]; 8];

        let mut k0 = k_begin;
        while k0 < k_stop {
            // Stage A as AS[row][kpair]: 512 registers, 4 per thread.
            let mut q = 0u32;
            #[unroll]
            while q < 4 {
                let e = tid + q * 128;
                let r = e >> 3;
                let kk = (e & 7) * 2;
                let gr = row0 + r;
                let g0 = k0 + kk;
                let v0 = if gr < m && g0 < k_stop {
                    a[(gr * k + g0) as usize]
                } else {
                    0.0f32
                };
                let v1 = if gr < m && g0 + 1 < k_stop {
                    a[(gr * k + g0 + 1) as usize]
                } else {
                    0.0f32
                };
                unsafe {
                    AS[e as usize] = pack_f16x2(v0, v1);
                }
                q += 1;
            }
            // Stage B as BS[col][kpair] — transposed relative to A, because the
            // B fragment wants a column's two K neighbours in one register.
            let mut q2 = 0u32;
            #[unroll]
            while q2 < 4 {
                let e = tid + q2 * 128;
                let cc = e >> 3;
                let kk = (e & 7) * 2;
                let gc = col0 + cc;
                let g0 = k0 + kk;
                // Pre-packed at load time as [n][ceil(k/2)]: one 32-bit read
                // per staged register, no conversion, half the bytes.
                unsafe {
                    BS[e as usize] = if gc < n && g0 < k_stop {
                        b_packed[(gc * kpairs + g0 / 2) as usize]
                    } else {
                        0u32
                    };
                }
                q2 += 1;
            }
            thread::sync_threads();

            // One A fragment for this warp's 16 rows, reused by all 8 tiles.
            let arow = warp * 16;
            let a0 = unsafe { AS[((arow + gid) * 8 + tig) as usize] };
            let a1 = unsafe { AS[((arow + gid + 8) * 8 + tig) as usize] };
            let a2 = unsafe { AS[((arow + gid) * 8 + tig + 4) as usize] };
            let a3 = unsafe { AS[((arow + gid + 8) * 8 + tig + 4) as usize] };

            let mut t = 0usize;
            #[unroll]
            while t < 8 {
                let ncol = t as u32 * 8 + gid;
                let b0 = unsafe { BS[(ncol * 8 + tig) as usize] };
                let b1 = unsafe { BS[(ncol * 8 + tig + 4) as usize] };
                unsafe {
                    mma_sync_m16n8k16_f32_f16(&mut acc[t], a0, a1, a2, a3, b0, b1);
                }
                t += 1;
            }
            thread::sync_threads();
            k0 += 16;
        }

        // Partial store; alpha, bias and activation belong to `reduce_splits`.
        let plane = split * m * n;
        let mut t = 0usize;
        #[unroll]
        while t < 8 {
            let gc = col0 + t as u32 * 8 + 2 * tig;
            let mut half = 0u32;
            #[unroll]
            while half < 2 {
                let gr = row0 + warp * 16 + gid + half * 8;
                if gr < m {
                    let base = plane + gr * n;
                    if gc < n {
                        unsafe {
                            *partials.get_unchecked_mut((base + gc) as usize) =
                                acc[t][(half * 2) as usize];
                        }
                    }
                    if gc + 1 < n {
                        unsafe {
                            *partials.get_unchecked_mut((base + gc + 1) as usize) =
                                acc[t][(half * 2 + 1) as usize];
                        }
                    }
                }
                half += 1;
            }
            t += 1;
        }
    }

    // =========================================================================
    // Pack an f32 matrix into f16 pairs for the tensor-core A operand.
    //   src: [rows][k] row-major f32.  dst: [rows][ceil(k/2)] u32, each
    //   holding k and k+1 as two halves. Odd k pads the final half with zero,
    //   which the GEMM masks off by its K bound anyway.
    // =========================================================================
    #[kernel]
    pub fn pack_f16_rows(src: &[f32], k: u32, kpairs: u32, mut dst: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let Some(o) = dst.get_mut(idx) {
            let row = i / kpairs;
            let pair = i % kpairs;
            let k0 = pair * 2;
            let lo = src[(row * k + k0) as usize];
            let hi = if k0 + 1 < k {
                src[(row * k + k0 + 1) as usize]
            } else {
                0.0f32
            };
            *o = pack_f16x2(lo, hi);
        }
    }

    // =========================================================================
    // f16 tensor-core GEMM, both operands packed, double-buffered staging.
    //
    //   The single-buffered kernel stalls: it issues a K-step's global loads,
    //   waits on them at the barrier, then does the math, so memory latency is
    //   never hidden behind the MMAs. Here the loads for step i+1 are issued
    //   into registers *before* the math for step i and land in the other
    //   shared buffer afterwards, so the two overlap.
    //
    //   It also halves the barriers. Reading S[b] and writing S[b^1] cannot
    //   conflict, so one sync per K-step suffices where the single-buffered
    //   version needed two: the barrier ending step i guarantees every warp has
    //   finished reading S[b] before any warp overwrites it in step i+1.
    //
    //   Shared: 2 x (64 x 8) u32 per operand = 8 KB per block.
    //   Launch: grid=(ceil(n/64), ceil(m/64), splits), block=(128,1,1).
    // =========================================================================
    #[kernel]
    pub fn sgemm_f16_tc_splitk_ab_db(
        m: u32,
        n: u32,
        k: u32,
        k_per_split: u32,
        a_packed: &[u32],
        b_packed: &[u32],
        kpairs: u32,
        mut partials: DisjointSlice<f32>,
    ) {
        static mut AS: SharedArray<u32, 1024> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, 1024> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let warp = tid >> 5;
        let lane = tid & 31;
        let gid = lane >> 2;
        let tig = lane & 3;

        let row0 = thread::blockIdx_y() * 64;
        let col0 = thread::blockIdx_x() * 64;
        let split = thread::blockIdx_z();

        let k_begin = split * k_per_split;
        let k_stop = if k_begin + k_per_split < k {
            k_begin + k_per_split
        } else {
            k
        };

        let mut acc = [[0.0f32; 4]; 8];
        let mut areg = [0u32; 4];
        let mut breg = [0u32; 4];

        // Both operands are [rows-or-cols][kpairs], 64 of the former and 8 of
        // the latter per tile, so one index decomposition serves both.
        let mut q = 0u32;
        #[unroll]
        while q < 4 {
            let e = tid + q * 128;
            let r = e >> 3;
            let kk = (e & 7) * 2;
            let g0 = k_begin + kk;
            let gr = row0 + r;
            let gc = col0 + r;
            areg[q as usize] = if gr < m && g0 < k_stop {
                a_packed[(gr * kpairs + g0 / 2) as usize]
            } else {
                0u32
            };
            breg[q as usize] = if gc < n && g0 < k_stop {
                b_packed[(gc * kpairs + g0 / 2) as usize]
            } else {
                0u32
            };
            q += 1;
        }
        let mut q2 = 0u32;
        #[unroll]
        while q2 < 4 {
            let e = tid + q2 * 128;
            unsafe {
                AS[e as usize] = areg[q2 as usize];
                BS[e as usize] = breg[q2 as usize];
            }
            q2 += 1;
        }
        thread::sync_threads();

        let mut buf = 0u32;
        let mut k0 = k_begin;
        while k0 < k_stop {
            let k_next = k0 + 16;

            // Issue the next tile's global loads before the math so they are in
            // flight while the MMAs run.
            if k_next < k_stop {
                let mut qn = 0u32;
                #[unroll]
                while qn < 4 {
                    let e = tid + qn * 128;
                    let r = e >> 3;
                    let kk = (e & 7) * 2;
                    let g0 = k_next + kk;
                    let gr = row0 + r;
                    let gc = col0 + r;
                    areg[qn as usize] = if gr < m && g0 < k_stop {
                        a_packed[(gr * kpairs + g0 / 2) as usize]
                    } else {
                        0u32
                    };
                    breg[qn as usize] = if gc < n && g0 < k_stop {
                        b_packed[(gc * kpairs + g0 / 2) as usize]
                    } else {
                        0u32
                    };
                    qn += 1;
                }
            }

            let base = buf * 512;
            let arow = warp * 16;
            let a0 = unsafe { AS[(base + (arow + gid) * 8 + tig) as usize] };
            let a1 = unsafe { AS[(base + (arow + gid + 8) * 8 + tig) as usize] };
            let a2 = unsafe { AS[(base + (arow + gid) * 8 + tig + 4) as usize] };
            let a3 = unsafe { AS[(base + (arow + gid + 8) * 8 + tig + 4) as usize] };

            let mut t = 0usize;
            #[unroll]
            while t < 8 {
                let ncol = t as u32 * 8 + gid;
                let b0 = unsafe { BS[(base + ncol * 8 + tig) as usize] };
                let b1 = unsafe { BS[(base + ncol * 8 + tig + 4) as usize] };
                unsafe {
                    mma_sync_m16n8k16_f32_f16(&mut acc[t], a0, a1, a2, a3, b0, b1);
                }
                t += 1;
            }

            // Land the prefetched tile in the other buffer; the reads above were
            // from `buf`, so one barrier closes the step.
            if k_next < k_stop {
                let other = (buf ^ 1) * 512;
                let mut qs = 0u32;
                #[unroll]
                while qs < 4 {
                    let e = tid + qs * 128;
                    unsafe {
                        AS[(other + e) as usize] = areg[qs as usize];
                        BS[(other + e) as usize] = breg[qs as usize];
                    }
                    qs += 1;
                }
            }
            thread::sync_threads();
            buf ^= 1;
            k0 = k_next;
        }

        let plane = split * m * n;
        let mut t = 0usize;
        #[unroll]
        while t < 8 {
            let gc = col0 + t as u32 * 8 + 2 * tig;
            let mut half = 0u32;
            #[unroll]
            while half < 2 {
                let gr = row0 + warp * 16 + gid + half * 8;
                if gr < m {
                    let base_o = plane + gr * n;
                    if gc < n {
                        unsafe {
                            *partials.get_unchecked_mut((base_o + gc) as usize) =
                                acc[t][(half * 2) as usize];
                        }
                    }
                    if gc + 1 < n {
                        unsafe {
                            *partials.get_unchecked_mut((base_o + gc + 1) as usize) =
                                acc[t][(half * 2 + 1) as usize];
                        }
                    }
                }
                half += 1;
            }
            t += 1;
        }
    }

    #[kernel]
    pub fn sgemm_f16_tc_splitk_ab_w8(
        m: u32,
        n: u32,
        k: u32,
        k_per_split: u32,
        a_packed: &[u32],
        b_packed: &[u32],
        kpairs: u32,
        // When the split factor is one there is nothing to reduce, and the
        // reduction pass degenerates into a full extra read and write of the
        // result to apply an epilogue. `direct` makes this kernel apply it
        // instead and write the finished value.
        alpha: f32,
        bias: &[f32],
        has_bias: u32,
        act: u32,
        act_lo: f32,
        act_hi: f32,
        direct: u32,
        mut partials: DisjointSlice<f32>,
    ) {
        static mut AS: SharedArray<u32, 1024> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, 1024> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let warp = tid >> 5;
        let lane = tid & 31;
        let gid = lane >> 2;
        let tig = lane & 3;

        let row0 = thread::blockIdx_y() * 64;
        let col0 = thread::blockIdx_x() * 64;
        let split = thread::blockIdx_z();

        let k_begin = split * k_per_split;
        let k_stop = if k_begin + k_per_split < k {
            k_begin + k_per_split
        } else {
            k
        };

        let mut acc = [[0.0f32; 4]; 4];
        let mut areg = [0u32; 2];
        let mut breg = [0u32; 2];

        // Both operands are [rows-or-cols][kpairs], 64 of the former and 8 of
        // the latter per tile, so one index decomposition serves both.
        let mut q = 0u32;
        #[unroll]
        while q < 2 {
            let e = tid + q * 256;
            let r = e >> 3;
            let kk = (e & 7) * 2;
            let g0 = k_begin + kk;
            let gr = row0 + r;
            let gc = col0 + r;
            areg[q as usize] = if gr < m && g0 < k_stop {
                a_packed[(gr * kpairs + g0 / 2) as usize]
            } else {
                0u32
            };
            breg[q as usize] = if gc < n && g0 < k_stop {
                b_packed[(gc * kpairs + g0 / 2) as usize]
            } else {
                0u32
            };
            q += 1;
        }
        let mut q2 = 0u32;
        #[unroll]
        while q2 < 2 {
            let e = tid + q2 * 256;
            unsafe {
                AS[e as usize] = areg[q2 as usize];
                BS[e as usize] = breg[q2 as usize];
            }
            q2 += 1;
        }
        thread::sync_threads();

        let mut buf = 0u32;
        let mut k0 = k_begin;
        while k0 < k_stop {
            let k_next = k0 + 16;

            // Issue the next tile's global loads before the math so they are in
            // flight while the MMAs run.
            if k_next < k_stop {
                let mut qn = 0u32;
                #[unroll]
                while qn < 2 {
                    let e = tid + qn * 256;
                    let r = e >> 3;
                    let kk = (e & 7) * 2;
                    let g0 = k_next + kk;
                    let gr = row0 + r;
                    let gc = col0 + r;
                    areg[qn as usize] = if gr < m && g0 < k_stop {
                        a_packed[(gr * kpairs + g0 / 2) as usize]
                    } else {
                        0u32
                    };
                    breg[qn as usize] = if gc < n && g0 < k_stop {
                        b_packed[(gc * kpairs + g0 / 2) as usize]
                    } else {
                        0u32
                    };
                    qn += 1;
                }
            }

            let base = buf * 512;
            let arow = (warp & 3) * 16;
            let ncol_base = (warp >> 2) * 32;
            let a0 = unsafe { AS[(base + (arow + gid) * 8 + tig) as usize] };
            let a1 = unsafe { AS[(base + (arow + gid + 8) * 8 + tig) as usize] };
            let a2 = unsafe { AS[(base + (arow + gid) * 8 + tig + 4) as usize] };
            let a3 = unsafe { AS[(base + (arow + gid + 8) * 8 + tig + 4) as usize] };

            let mut t = 0usize;
            #[unroll]
            while t < 4 {
                let ncol = ncol_base + t as u32 * 8 + gid;
                let b0 = unsafe { BS[(base + ncol * 8 + tig) as usize] };
                let b1 = unsafe { BS[(base + ncol * 8 + tig + 4) as usize] };
                unsafe {
                    mma_sync_m16n8k16_f32_f16(&mut acc[t], a0, a1, a2, a3, b0, b1);
                }
                t += 1;
            }

            // Land the prefetched tile in the other buffer; the reads above were
            // from `buf`, so one barrier closes the step.
            if k_next < k_stop {
                let other = (buf ^ 1) * 512;
                let mut qs = 0u32;
                #[unroll]
                while qs < 2 {
                    let e = tid + qs * 256;
                    unsafe {
                        AS[(other + e) as usize] = areg[qs as usize];
                        BS[(other + e) as usize] = breg[qs as usize];
                    }
                    qs += 1;
                }
            }
            thread::sync_threads();
            buf ^= 1;
            k0 = k_next;
        }

        let plane = split * m * n;
        let mut t = 0usize;
        #[unroll]
        while t < 4 {
            let gc = col0 + (warp >> 2) * 32 + t as u32 * 8 + 2 * tig;
            let mut half = 0u32;
            #[unroll]
            while half < 2 {
                let gr = row0 + (warp & 3) * 16 + gid + half * 8;
                if gr < m {
                    let base_o = plane + gr * n;
                    if gc < n {
                        let v = acc[t][(half * 2) as usize];
                        let v = if direct != 0 {
                            let b = if has_bias != 0 {
                                bias[gc as usize]
                            } else {
                                0.0f32
                            };
                            apply_act(alpha * v + b, act, act_lo, act_hi)
                        } else {
                            v
                        };
                        unsafe {
                            *partials.get_unchecked_mut((base_o + gc) as usize) = v;
                        }
                    }
                    if gc + 1 < n {
                        let v = acc[t][(half * 2 + 1) as usize];
                        let v = if direct != 0 {
                            let b = if has_bias != 0 {
                                bias[(gc + 1) as usize]
                            } else {
                                0.0f32
                            };
                            apply_act(alpha * v + b, act, act_lo, act_hi)
                        } else {
                            v
                        };
                        unsafe {
                            *partials.get_unchecked_mut((base_o + gc + 1) as usize) = v;
                        }
                    }
                }
                half += 1;
            }
            t += 1;
        }
    }

    #[kernel]
    pub fn sgemm_f16_tc_splitk_abpacked(
        m: u32,
        n: u32,
        k: u32,
        k_per_split: u32,
        a_packed: &[u32],
        b_packed: &[u32],
        kpairs: u32,
        mut partials: DisjointSlice<f32>,
    ) {
        // 64 rows x 16 halves = 512 u32 each; 2 KB per tile, 4 KB per block.
        static mut AS: SharedArray<u32, 512> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, 512> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let warp = tid >> 5;
        let lane = tid & 31;
        let gid = lane >> 2; // groupID 0..7
        let tig = lane & 3; // threadID_in_group 0..3

        let row0 = thread::blockIdx_y() * 64;
        let col0 = thread::blockIdx_x() * 64;
        let split = thread::blockIdx_z();

        let k_begin = split * k_per_split;
        let k_stop = if k_begin + k_per_split < k {
            k_begin + k_per_split
        } else {
            k
        };

        let mut acc = [[0.0f32; 4]; 8];

        let mut k0 = k_begin;
        while k0 < k_stop {
            // Stage A as AS[row][kpair]: 512 registers, 4 per thread.
            let mut q = 0u32;
            #[unroll]
            while q < 4 {
                let e = tid + q * 128;
                let r = e >> 3;
                let kk = (e & 7) * 2;
                let gr = row0 + r;
                let g0 = k0 + kk;
                unsafe {
                    AS[e as usize] = if gr < m && g0 < k_stop {
                        a_packed[(gr * kpairs + g0 / 2) as usize]
                    } else {
                        0u32
                    };
                }
                q += 1;
            }
            // Stage B as BS[col][kpair] — transposed relative to A, because the
            // B fragment wants a column's two K neighbours in one register.
            let mut q2 = 0u32;
            #[unroll]
            while q2 < 4 {
                let e = tid + q2 * 128;
                let cc = e >> 3;
                let kk = (e & 7) * 2;
                let gc = col0 + cc;
                let g0 = k0 + kk;
                // Pre-packed at load time as [n][ceil(k/2)]: one 32-bit read
                // per staged register, no conversion, half the bytes.
                unsafe {
                    BS[e as usize] = if gc < n && g0 < k_stop {
                        b_packed[(gc * kpairs + g0 / 2) as usize]
                    } else {
                        0u32
                    };
                }
                q2 += 1;
            }
            thread::sync_threads();

            // One A fragment for this warp's 16 rows, reused by all 8 tiles.
            let arow = warp * 16;
            let a0 = unsafe { AS[((arow + gid) * 8 + tig) as usize] };
            let a1 = unsafe { AS[((arow + gid + 8) * 8 + tig) as usize] };
            let a2 = unsafe { AS[((arow + gid) * 8 + tig + 4) as usize] };
            let a3 = unsafe { AS[((arow + gid + 8) * 8 + tig + 4) as usize] };

            let mut t = 0usize;
            #[unroll]
            while t < 8 {
                let ncol = t as u32 * 8 + gid;
                let b0 = unsafe { BS[(ncol * 8 + tig) as usize] };
                let b1 = unsafe { BS[(ncol * 8 + tig + 4) as usize] };
                unsafe {
                    mma_sync_m16n8k16_f32_f16(&mut acc[t], a0, a1, a2, a3, b0, b1);
                }
                t += 1;
            }
            thread::sync_threads();
            k0 += 16;
        }

        // Partial store; alpha, bias and activation belong to `reduce_splits`.
        let plane = split * m * n;
        let mut t = 0usize;
        #[unroll]
        while t < 8 {
            let gc = col0 + t as u32 * 8 + 2 * tig;
            let mut half = 0u32;
            #[unroll]
            while half < 2 {
                let gr = row0 + warp * 16 + gid + half * 8;
                if gr < m {
                    let base = plane + gr * n;
                    if gc < n {
                        unsafe {
                            *partials.get_unchecked_mut((base + gc) as usize) =
                                acc[t][(half * 2) as usize];
                        }
                    }
                    if gc + 1 < n {
                        unsafe {
                            *partials.get_unchecked_mut((base + gc + 1) as usize) =
                                acc[t][(half * 2 + 1) as usize];
                        }
                    }
                }
                half += 1;
            }
            t += 1;
        }
    }

    // =========================================================================
    // One LSTM timestep, ONNX gate order (i, o, f, c).
    //
    //   Every other model here is feed-forward: each node runs once. A
    //   recurrent layer applies the same weights T times with a serial
    //   dependency between steps, so the step is the kernel and the loop over
    //   time stays on the host. One thread per hidden unit computes all four
    //   of that unit's gates, which keeps the four dot products that share the
    //   same h_prev reads in one thread.
    //
    //   W is [4H, input] and R is [4H, H], both gate-major, so gate g of unit
    //   j is row g*H + j. Bias is [8H]: input-side gates then recurrence-side.
    // =========================================================================
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn lstm_step(
        x_t: &[f32],
        w: &[f32],
        r: &[f32],
        b: &[f32],
        h_prev: &[f32],
        c_prev: &[f32],
        input_size: u32,
        hidden: u32,
        has_bias: u32,
        mut h_out: DisjointSlice<f32>,
        mut c_out: DisjointSlice<f32>,
    ) {
        // Four partial sums per thread, reduced across the block.
        static mut RED: SharedArray<f32, 1024> = SharedArray::UNINIT;

        let j = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if j >= hidden {
            return;
        }

        // One block per hidden unit, so the whole block sweeps that unit's
        // weight rows together and consecutive threads read consecutive
        // weights. One thread per unit instead leaves a single block resident
        // — one SM of 82 — and reads each row with a stride.
        let mut acc = [0.0f32; 4];
        let mut g = 0usize;
        while g < 4 {
            let row = g as u32 * hidden + j;
            let wbase = row * input_size;
            let mut k = tid;
            let mut a = 0.0f32;
            while k < input_size {
                a += w[(wbase + k) as usize] * x_t[k as usize];
                k += 256;
            }
            let rbase = row * hidden;
            let mut m = tid;
            while m < hidden {
                a += r[(rbase + m) as usize] * h_prev[m as usize];
                m += 256;
            }
            acc[g] = a;
            g += 1;
        }

        let mut g2 = 0usize;
        while g2 < 4 {
            unsafe {
                RED[(g2 as u32 * 256 + tid) as usize] = acc[g2];
            }
            g2 += 1;
        }
        thread::sync_threads();
        let mut step = 128u32;
        while step > 0 {
            if tid < step {
                let mut g3 = 0u32;
                while g3 < 4 {
                    unsafe {
                        RED[(g3 * 256 + tid) as usize] += RED[(g3 * 256 + tid + step) as usize];
                    }
                    g3 += 1;
                }
            }
            thread::sync_threads();
            step >>= 1;
        }

        if tid == 0 {
            let mut pre = [0.0f32; 4];
            let mut g4 = 0usize;
            while g4 < 4 {
                let row = g4 as u32 * hidden + j;
                let mut v = unsafe { RED[(g4 as u32 * 256) as usize] };
                if has_bias != 0 {
                    v += b[row as usize] + b[(4 * hidden + row) as usize];
                }
                pre[g4] = v;
                g4 += 1;
            }
            // ONNX orders the gates i, o, f, c.
            let i_g = gpu_sigmoid(pre[0]);
            let o_g = gpu_sigmoid(pre[1]);
            let f_g = gpu_sigmoid(pre[2]);
            let c_g = gpu_tanh(pre[3]);
            let c_new = f_g * c_prev[j as usize] + i_g * c_g;
            let h_new = o_g * gpu_tanh(c_new);
            unsafe {
                *c_out.get_unchecked_mut(j as usize) = c_new;
                *h_out.get_unchecked_mut(j as usize) = h_new;
            }
        }
    }

    // =========================================================================
    // Broadcasting elementwise binary op.
    //
    //   `op`: 0 add, 1 sub, 2 mul, 3 div. Only Add had a broadcasting path,
    //   which meant a graph dividing by a broadcast tensor — style transfer
    //   normalising by a computed size — simply failed. One kernel covers all
    //   four rather than four near-identical ones.
    // =========================================================================
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn binary_bcast(
        a: &[f32],
        b: &[f32],
        out_shape: &[f32],
        out_strides: &[f32],
        a_strides: &[f32],
        b_strides: &[f32],
        ndim: u32,
        op: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let lin = idx.get();
        if let Some(o) = y.get_mut(idx) {
            let nd = ndim as usize;
            let mut ao = 0usize;
            let mut bo = 0usize;
            let mut d = 0usize;
            while d < nd {
                let os = out_strides[d] as usize;
                let osz = out_shape[d] as usize;
                let coord = (lin / os) % osz;
                ao += coord * (a_strides[d] as usize);
                bo += coord * (b_strides[d] as usize);
                d += 1;
            }
            let va = a[ao];
            let vb = b[bo];
            *o = if op == 0 {
                va + vb
            } else if op == 1 {
                va - vb
            } else if op == 2 {
                va * vb
            } else {
                va / vb
            };
        }
    }

    // =========================================================================
    // Strided N-D slice: gather `y` from `x` given per-axis start and step.
    // =========================================================================
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn slice_nd(
        x: &[f32],
        out_shape: &[f32],
        out_strides: &[f32],
        in_strides: &[f32],
        starts: &[f32],
        steps: &[f32],
        ndim: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let lin = idx.get();
        if let Some(o) = y.get_mut(idx) {
            let nd = ndim as usize;
            let mut off = 0i64;
            let mut d = 0usize;
            while d < nd {
                let os = out_strides[d] as usize;
                let osz = out_shape[d] as usize;
                let coord = ((lin / os) % osz) as i64;
                off += (starts[d] as i64 + coord * (steps[d] as i64)) * (in_strides[d] as i64);
                d += 1;
            }
            *o = x[off as usize];
        }
    }

    // =========================================================================
    // LeakyRelu: y = x for x >= 0, alpha*x otherwise.
    //   Detection backbones (Tiny-YOLOv2, the Darknet family) use this in
    //   place of Relu throughout.
    // =========================================================================
    #[kernel]
    pub fn leaky_relu(a: &[f32], alpha: f32, mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = c.get_mut(idx) {
            let v = a[i];
            *o = if v >= 0.0f32 { v } else { alpha * v };
        }
    }

    // =========================================================================
    // Floor.
    // =========================================================================
    #[kernel]
    pub fn floor_fwd(a: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = c.get_mut(idx) {
            let v = a[i];
            // Truncation rounds toward zero, so negatives need a nudge; this
            // avoids pulling in libdevice, which would switch the backend to
            // NVVM IR mode and skip PTX embedding.
            let t = v as i64 as f32;
            *o = if t > v { t - 1.0f32 } else { t };
        }
    }

    // =========================================================================
    // Mean over a contiguous run of `inner` elements, for each of `outer`.
    //   ReduceMean over trailing axes reduces to exactly this after the shape
    //   bookkeeping is done on the host.
    // =========================================================================
    #[kernel]
    pub fn reduce_mean_inner(x: &[f32], inner: u32, mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let lin = idx.get() as u32;
        if let Some(o) = y.get_mut(idx) {
            let base = lin * inner;
            let mut acc = 0.0f32;
            let mut i = 0u32;
            while i < inner {
                acc += x[(base + i) as usize];
                i += 1;
            }
            *o = acc / inner as f32;
        }
    }

    // =========================================================================
    // InstanceNormalization: normalise each (n, c) plane by its own mean and
    // variance, then apply the per-channel affine.
    //
    //   Style-transfer networks use this instead of BatchNorm, and unlike
    //   BatchNorm it cannot be folded into the convolution weights, because
    //   the statistics depend on the image rather than on the training set.
    //   One block per plane, reduced in shared memory.
    // =========================================================================
    #[kernel]
    pub fn instance_norm(
        x: &[f32],
        scale: &[f32],
        bias: &[f32],
        channels: u32,
        hw: u32,
        eps: f32,
        mut y: DisjointSlice<f32>,
    ) {
        static mut RED: SharedArray<f32, 256> = SharedArray::UNINIT;

        let plane = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        let base = plane * hw;

        let mut sum = 0.0f32;
        let mut i = tid;
        while i < hw {
            sum += x[(base + i) as usize];
            i += 256;
        }
        unsafe {
            RED[tid as usize] = sum;
        }
        thread::sync_threads();
        let mut step = 128u32;
        while step > 0 {
            if tid < step {
                unsafe {
                    RED[tid as usize] += RED[(tid + step) as usize];
                }
            }
            thread::sync_threads();
            step >>= 1;
        }
        let mean = unsafe { RED[0] } / hw as f32;
        thread::sync_threads();

        let mut vs = 0.0f32;
        let mut j = tid;
        while j < hw {
            let d = x[(base + j) as usize] - mean;
            vs += d * d;
            j += 256;
        }
        unsafe {
            RED[tid as usize] = vs;
        }
        thread::sync_threads();
        let mut step2 = 128u32;
        while step2 > 0 {
            if tid < step2 {
                unsafe {
                    RED[tid as usize] += RED[(tid + step2) as usize];
                }
            }
            thread::sync_threads();
            step2 >>= 1;
        }
        let var = unsafe { RED[0] } / hw as f32;
        let inv = gpu_rsqrt(var + eps);

        let c = plane % channels;
        let sc = scale[c as usize] * inv;
        let sh = bias[c as usize] - mean * sc;
        let mut t = tid;
        while t < hw {
            unsafe {
                *y.get_unchecked_mut((base + t) as usize) = x[(base + t) as usize] * sc + sh;
            }
            t += 256;
        }
    }

    // =========================================================================
    // Nearest-neighbour and bilinear resize of an NCHW tensor.
    //   `mode`: 0 = nearest, 1 = bilinear. Covers both Resize and the older
    //   Upsample, which differ only in how the scales reach the node.
    // =========================================================================
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn resize2d(
        x: &[f32],
        in_h: u32,
        in_w: u32,
        out_h: u32,
        out_w: u32,
        mode: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let lin = idx.get() as u32;
        if let Some(o) = y.get_mut(idx) {
            let i = lin;
            let ow = i % out_w;
            let oh = (i / out_w) % out_h;
            let plane = i / (out_w * out_h);
            let base = plane * in_h * in_w;

            let sh = in_h as f32 / out_h as f32;
            let sw = in_w as f32 / out_w as f32;

            if mode == 0 {
                let ih = {
                    let v = (oh as f32 * sh) as u32;
                    if v >= in_h { in_h - 1 } else { v }
                };
                let iw = {
                    let v = (ow as f32 * sw) as u32;
                    if v >= in_w { in_w - 1 } else { v }
                };
                *o = x[(base + ih * in_w + iw) as usize];
            } else {
                // half-pixel centres, which is what both ORT and PyTorch use
                // for align_corners=false.
                let fy = (oh as f32 + 0.5f32) * sh - 0.5f32;
                let fx = (ow as f32 + 0.5f32) * sw - 0.5f32;
                let fy = if fy < 0.0f32 { 0.0f32 } else { fy };
                let fx = if fx < 0.0f32 { 0.0f32 } else { fx };
                let y0 = fy as u32;
                let x0 = fx as u32;
                let y1 = if y0 + 1 < in_h { y0 + 1 } else { in_h - 1 };
                let x1 = if x0 + 1 < in_w { x0 + 1 } else { in_w - 1 };
                let wy = fy - y0 as f32;
                let wx = fx - x0 as f32;
                let p00 = x[(base + y0 * in_w + x0) as usize];
                let p01 = x[(base + y0 * in_w + x1) as usize];
                let p10 = x[(base + y1 * in_w + x0) as usize];
                let p11 = x[(base + y1 * in_w + x1) as usize];
                let top = p00 + (p01 - p00) * wx;
                let bot = p10 + (p11 - p10) * wx;
                *o = top + (bot - top) * wy;
            }
        }
    }

    // =========================================================================
    // Spatial padding of an NCHW tensor with a constant value.
    //   `mode`: 0 = constant, 1 = reflect. Style-transfer graphs pad by
    //   reflection before every convolution to avoid border artefacts.
    // =========================================================================
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn pad2d(
        x: &[f32],
        in_h: u32,
        in_w: u32,
        out_h: u32,
        out_w: u32,
        pad_top: u32,
        pad_left: u32,
        mode: u32,
        value: f32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let lin = idx.get() as u32;
        if let Some(o) = y.get_mut(idx) {
            let i = lin;
            let ow = i % out_w;
            let oh = (i / out_w) % out_h;
            let plane = i / (out_w * out_h);
            let base = plane * in_h * in_w;

            let sy = oh as i32 - pad_top as i32;
            let sx = ow as i32 - pad_left as i32;

            if mode == 0 {
                if sy < 0 || sx < 0 || sy >= in_h as i32 || sx >= in_w as i32 {
                    *o = value;
                } else {
                    *o = x[(base + sy as u32 * in_w + sx as u32) as usize];
                }
            } else {
                // Reflect without repeating the edge pixel.
                let refl = |v: i32, n: i32| -> u32 {
                    let mut t = v;
                    if t < 0 {
                        t = -t;
                    }
                    if t >= n {
                        t = 2 * (n - 1) - t;
                    }
                    if t < 0 { 0u32 } else { t as u32 }
                };
                let ry = refl(sy, in_h as i32);
                let rx = refl(sx, in_w as i32);
                *o = x[(base + ry * in_w + rx) as usize];
            }
        }
    }

    // =========================================================================
    // Pack a row-major [k][n] matrix into f16 pairs laid out as [n][ceil(k/2)].
    //   Each register holds rows k and k+1 of one column, which is what the
    //   tensor-core B fragment reads. Odd k pads the final half with zero.
    // =========================================================================
    #[kernel]
    pub fn pack_f16_cols(src: &[f32], k: u32, n: u32, kpairs: u32, mut dst: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let Some(o) = dst.get_mut(idx) {
            let col = i / kpairs;
            let pair = i % kpairs;
            let k0 = pair * 2;
            let lo = src[(k0 * n + col) as usize];
            let hi = if k0 + 1 < k {
                src[((k0 + 1) * n + col) as usize]
            } else {
                0.0f32
            };
            *o = pack_f16x2(lo, hi);
        }
    }

    // =========================================================================
    // Pack a batch of row-major [k][n] matrices into f16 pairs as
    // [batch][n][ceil(k/2)] — the column-packed form the B fragment wants,
    // with the batch stride folded into the index so one launch covers every
    // attention head.
    // =========================================================================
    #[kernel]
    pub fn pack_f16_cols_batched(
        src: &[f32],
        k: u32,
        n: u32,
        kpairs: u32,
        mut dst: DisjointSlice<u32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let Some(o) = dst.get_mut(idx) {
            let per = n * kpairs;
            let b = i / per;
            let rem = i % per;
            let col = rem / kpairs;
            let k0 = (rem % kpairs) * 2;
            let base = b * k * n;
            let lo = src[(base + k0 * n + col) as usize];
            let hi = if k0 + 1 < k {
                src[(base + (k0 + 1) * n + col) as usize]
            } else {
                0.0f32
            };
            *o = pack_f16x2(lo, hi);
        }
    }

    // =========================================================================
    // Batched f16 tensor-core GEMM: C[b] = A[b] * B[b] for every b in one
    // launch, blockIdx.z selecting the batch.
    //
    //   Attention matmuls are small — 197x64x197 per head — so running them
    //   one head per launch leaves the grid nearly empty and pays a launch
    //   plus a pack per head. Folding the batch into gridDim.z gives a grid
    //   twelve times larger and one launch for the lot.
    //
    //   K is short here, so there is no split; the result is written straight
    //   out with no partials and no reduction pass.
    // =========================================================================
    #[kernel]
    pub fn sgemm_f16_tc_batched_w8(
        m: u32,
        n: u32,
        k: u32,
        a_packed: &[u32],
        b_packed: &[u32],
        kpairs: u32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut AS: SharedArray<u32, 1024> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, 1024> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let warp = tid >> 5;
        let lane = tid & 31;
        let gid = lane >> 2;
        let tig = lane & 3;

        let row0 = thread::blockIdx_y() * 64;
        let col0 = thread::blockIdx_x() * 64;
        let bat = thread::blockIdx_z();

        let a_base = bat * m * kpairs;
        let b_base = bat * n * kpairs;

        let mut acc = [[0.0f32; 4]; 4];
        let mut areg = [0u32; 2];
        let mut breg = [0u32; 2];

        let mut q = 0u32;
        #[unroll]
        while q < 2 {
            let e = tid + q * 256;
            let r = e >> 3;
            let kk = (e & 7) * 2;
            let gr = row0 + r;
            let gc = col0 + r;
            areg[q as usize] = if gr < m && kk < k {
                a_packed[(a_base + gr * kpairs + kk / 2) as usize]
            } else {
                0u32
            };
            breg[q as usize] = if gc < n && kk < k {
                b_packed[(b_base + gc * kpairs + kk / 2) as usize]
            } else {
                0u32
            };
            q += 1;
        }
        let mut q2 = 0u32;
        #[unroll]
        while q2 < 2 {
            let e = tid + q2 * 256;
            unsafe {
                AS[e as usize] = areg[q2 as usize];
                BS[e as usize] = breg[q2 as usize];
            }
            q2 += 1;
        }
        thread::sync_threads();

        let mut buf = 0u32;
        let mut k0 = 0u32;
        while k0 < k {
            let k_next = k0 + 16;

            if k_next < k {
                let mut qn = 0u32;
                #[unroll]
                while qn < 2 {
                    let e = tid + qn * 256;
                    let r = e >> 3;
                    let kk = (e & 7) * 2;
                    let g0 = k_next + kk;
                    let gr = row0 + r;
                    let gc = col0 + r;
                    areg[qn as usize] = if gr < m && g0 < k {
                        a_packed[(a_base + gr * kpairs + g0 / 2) as usize]
                    } else {
                        0u32
                    };
                    breg[qn as usize] = if gc < n && g0 < k {
                        b_packed[(b_base + gc * kpairs + g0 / 2) as usize]
                    } else {
                        0u32
                    };
                    qn += 1;
                }
            }

            let base = buf * 512;
            let arow = (warp & 3) * 16;
            let ncol_base = (warp >> 2) * 32;
            let a0 = unsafe { AS[(base + (arow + gid) * 8 + tig) as usize] };
            let a1 = unsafe { AS[(base + (arow + gid + 8) * 8 + tig) as usize] };
            let a2 = unsafe { AS[(base + (arow + gid) * 8 + tig + 4) as usize] };
            let a3 = unsafe { AS[(base + (arow + gid + 8) * 8 + tig + 4) as usize] };

            let mut t = 0usize;
            #[unroll]
            while t < 4 {
                let ncol = ncol_base + t as u32 * 8 + gid;
                let b0 = unsafe { BS[(base + ncol * 8 + tig) as usize] };
                let b1 = unsafe { BS[(base + ncol * 8 + tig + 4) as usize] };
                unsafe {
                    mma_sync_m16n8k16_f32_f16(&mut acc[t], a0, a1, a2, a3, b0, b1);
                }
                t += 1;
            }

            if k_next < k {
                let other = (buf ^ 1) * 512;
                let mut qs = 0u32;
                #[unroll]
                while qs < 2 {
                    let e = tid + qs * 256;
                    unsafe {
                        AS[(other + e) as usize] = areg[qs as usize];
                        BS[(other + e) as usize] = breg[qs as usize];
                    }
                    qs += 1;
                }
            }
            thread::sync_threads();
            buf ^= 1;
            k0 = k_next;
        }

        let plane = bat * m * n;
        let mut t = 0usize;
        #[unroll]
        while t < 4 {
            let gc = col0 + (warp >> 2) * 32 + t as u32 * 8 + 2 * tig;
            let mut half = 0u32;
            #[unroll]
            while half < 2 {
                let gr = row0 + (warp & 3) * 16 + gid + half * 8;
                if gr < m {
                    let base_o = plane + gr * n;
                    if gc < n {
                        unsafe {
                            *out.get_unchecked_mut((base_o + gc) as usize) =
                                acc[t][(half * 2) as usize];
                        }
                    }
                    if gc + 1 < n {
                        unsafe {
                            *out.get_unchecked_mut((base_o + gc + 1) as usize) =
                                acc[t][(half * 2 + 1) as usize];
                        }
                    }
                }
                half += 1;
            }
            t += 1;
        }
    }

    // =========================================================================
    // Batched f16 tensor-core GEMM reading f32 operands in place.
    //
    //   The attention pipeline spends most of its time moving data, not
    //   multiplying it: two pack kernels per batched MatMul, fed by transposes
    //   whose work largely cancels. K is transposed from [S, H, D] to
    //   [H, D, S] and then read back column-wise by the packer — which is the
    //   layout it started in.
    //
    //   So this kernel takes strides rather than a layout: element (batch,
    //   row, kk) of A sits at a_batch*batch + a_row*row + kk, and (batch, col,
    //   kk) of B at b_batch*batch + b_col*col + b_k*kk. Q, K and V can then be
    //   read straight out of the fused QKV projection, converted to f16 on the
    //   way into shared memory, with no transpose and no pack.
    //
    //   `b_k_contig` picks which of B's axes the staging threads walk so the
    //   global reads stay coalesced either way. Shared rows of B are padded to
    //   9 u32, which makes both walks bank-conflict-free.
    // =========================================================================
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_f16_tc_bmm_strided(
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
        a_batch: u32,
        a_row: u32,
        b_batch: u32,
        b_col: u32,
        b_k: u32,
        b_k_contig: u32,
        alpha: f32,
        mut out: DisjointSlice<f32>,
    ) {
        static mut AS: SharedArray<u32, 1024> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, 1152> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let warp = tid >> 5;
        let lane = tid & 31;
        let gid = lane >> 2;
        let tig = lane & 3;

        let row0 = thread::blockIdx_y() * 64;
        let col0 = thread::blockIdx_x() * 64;
        let bat = thread::blockIdx_z();

        let abase = bat * a_batch;
        let bbase = bat * b_batch;

        // A always has kk contiguous: eight threads cover one row's 16 halves.
        let ar = tid >> 3;
        let akp = tid & 7;
        let ar2 = (tid + 256) >> 3;
        let akp2 = (tid + 256) & 7;

        // B walks whichever of its axes is contiguous in memory.
        let e2 = tid + 256;
        let bc = if b_k_contig != 0 { tid >> 3 } else { tid & 63 };
        let bkp = if b_k_contig != 0 { tid & 7 } else { tid >> 6 };
        let bc2 = if b_k_contig != 0 { e2 >> 3 } else { e2 & 63 };
        let bkp2 = if b_k_contig != 0 { e2 & 7 } else { e2 >> 6 };

        let mut acc = [[0.0f32; 4]; 4];

        // The next K step's global reads are issued before the current step's
        // MMAs and packed only afterwards, so the load latency is spent on
        // tensor-core work. Single-buffered, this kernel stalls on
        // long_scoreboard at 5.6 warps per issue-active cycle with the tensor
        // pipe at 3%.
        let mut alo = [0.0f32; 2];
        let mut ahi = [0.0f32; 2];
        let mut blo = [0.0f32; 2];
        let mut bhi = [0.0f32; 2];

        let mut q = 0u32;
        #[unroll]
        while q < 2 {
            let r = if q == 0 { ar } else { ar2 };
            let kp = if q == 0 { akp } else { akp2 };
            let c = if q == 0 { bc } else { bc2 };
            let ck = if q == 0 { bkp } else { bkp2 };
            let gr = row0 + r;
            let gc = col0 + c;
            let ka = kp * 2;
            let kb = ck * 2;
            alo[q as usize] = if gr < m && ka < k {
                unsafe { *a.get_unchecked((abase + gr * a_row + ka) as usize) }
            } else {
                0.0f32
            };
            ahi[q as usize] = if gr < m && ka + 1 < k {
                unsafe { *a.get_unchecked((abase + gr * a_row + ka + 1) as usize) }
            } else {
                0.0f32
            };
            blo[q as usize] = if gc < n && kb < k {
                unsafe { *b.get_unchecked((bbase + gc * b_col + kb * b_k) as usize) }
            } else {
                0.0f32
            };
            bhi[q as usize] = if gc < n && kb + 1 < k {
                unsafe { *b.get_unchecked((bbase + gc * b_col + (kb + 1) * b_k) as usize) }
            } else {
                0.0f32
            };
            q += 1;
        }
        let mut q2 = 0u32;
        #[unroll]
        while q2 < 2 {
            let r = if q2 == 0 { ar } else { ar2 };
            let kp = if q2 == 0 { akp } else { akp2 };
            let c = if q2 == 0 { bc } else { bc2 };
            let ck = if q2 == 0 { bkp } else { bkp2 };
            unsafe {
                AS[(r * 8 + kp) as usize] = pack_f16x2(alo[q2 as usize], ahi[q2 as usize]);
                BS[(c * 9 + ck) as usize] = pack_f16x2(blo[q2 as usize], bhi[q2 as usize]);
            }
            q2 += 1;
        }
        thread::sync_threads();

        let mut buf = 0u32;
        let mut k0 = 0u32;
        while k0 < k {
            let k_next = k0 + 16;

            if k_next < k {
                let mut qn = 0u32;
                #[unroll]
                while qn < 2 {
                    let r = if qn == 0 { ar } else { ar2 };
                    let kp = if qn == 0 { akp } else { akp2 };
                    let c = if qn == 0 { bc } else { bc2 };
                    let ck = if qn == 0 { bkp } else { bkp2 };
                    let gr = row0 + r;
                    let gc = col0 + c;
                    let ka = k_next + kp * 2;
                    let kb = k_next + ck * 2;
                    alo[qn as usize] = if gr < m && ka < k {
                        unsafe { *a.get_unchecked((abase + gr * a_row + ka) as usize) }
                    } else {
                        0.0f32
                    };
                    ahi[qn as usize] = if gr < m && ka + 1 < k {
                        unsafe { *a.get_unchecked((abase + gr * a_row + ka + 1) as usize) }
                    } else {
                        0.0f32
                    };
                    blo[qn as usize] = if gc < n && kb < k {
                        unsafe { *b.get_unchecked((bbase + gc * b_col + kb * b_k) as usize) }
                    } else {
                        0.0f32
                    };
                    bhi[qn as usize] = if gc < n && kb + 1 < k {
                        unsafe { *b.get_unchecked((bbase + gc * b_col + (kb + 1) * b_k) as usize) }
                    } else {
                        0.0f32
                    };
                    qn += 1;
                }
            }

            let asb = buf * 512;
            let bsb = buf * 576;
            let arow = (warp & 3) * 16;
            let ncol_base = (warp >> 2) * 32;
            let a0 = unsafe { AS[(asb + (arow + gid) * 8 + tig) as usize] };
            let a1 = unsafe { AS[(asb + (arow + gid + 8) * 8 + tig) as usize] };
            let a2 = unsafe { AS[(asb + (arow + gid) * 8 + tig + 4) as usize] };
            let a3 = unsafe { AS[(asb + (arow + gid + 8) * 8 + tig + 4) as usize] };

            let mut t = 0usize;
            #[unroll]
            while t < 4 {
                let ncol = ncol_base + t as u32 * 8 + gid;
                let b0 = unsafe { BS[(bsb + ncol * 9 + tig) as usize] };
                let b1 = unsafe { BS[(bsb + ncol * 9 + tig + 4) as usize] };
                unsafe {
                    mma_sync_m16n8k16_f32_f16(&mut acc[t], a0, a1, a2, a3, b0, b1);
                }
                t += 1;
            }

            if k_next < k {
                let ao = (buf ^ 1) * 512;
                let bo = (buf ^ 1) * 576;
                let mut qs = 0u32;
                #[unroll]
                while qs < 2 {
                    let r = if qs == 0 { ar } else { ar2 };
                    let kp = if qs == 0 { akp } else { akp2 };
                    let c = if qs == 0 { bc } else { bc2 };
                    let ck = if qs == 0 { bkp } else { bkp2 };
                    unsafe {
                        AS[(ao + r * 8 + kp) as usize] =
                            pack_f16x2(alo[qs as usize], ahi[qs as usize]);
                        BS[(bo + c * 9 + ck) as usize] =
                            pack_f16x2(blo[qs as usize], bhi[qs as usize]);
                    }
                    qs += 1;
                }
            }
            thread::sync_threads();
            buf ^= 1;
            k0 = k_next;
        }

        let plane = bat * m * n;
        let mut t = 0usize;
        #[unroll]
        while t < 4 {
            let gc = col0 + (warp >> 2) * 32 + t as u32 * 8 + 2 * tig;
            let mut half = 0u32;
            #[unroll]
            while half < 2 {
                let gr = row0 + (warp & 3) * 16 + gid + half * 8;
                if gr < m {
                    let base_o = plane + gr * n;
                    if gc < n {
                        unsafe {
                            *out.get_unchecked_mut((base_o + gc) as usize) =
                                acc[t][(half * 2) as usize] * alpha;
                        }
                    }
                    if gc + 1 < n {
                        unsafe {
                            *out.get_unchecked_mut((base_o + gc + 1) as usize) =
                                acc[t][(half * 2 + 1) as usize] * alpha;
                        }
                    }
                }
                half += 1;
            }
            t += 1;
        }
    }

    // =========================================================================
    // Winograd F(2x2, 3x3).
    //
    //   A 3x3 convolution over a 2x2 output tile needs 36 multiplies done
    //   directly and 16 in the Winograd domain — 2.25x fewer. That ratio is
    //   not the main attraction here. The implicit-GEMM convolution is limited
    //   by address arithmetic and memory latency, not by arithmetic: its
    //   tensor pipe runs at 13-20% while `wait` sits at 2.5. Winograd replaces
    //   it with three cheap elementwise passes and a plain batched GEMM, which
    //   has no im2col addressing at all.
    //
    //   Layout is transform-plane-major throughout — U is [16][K][C], V is
    //   [16][C][T], M is [16][K][T] — so each of the sixteen planes is a
    //   contiguous matrix and the whole thing is one batched GEMM with the
    //   plane as the batch index.
    //
    //   The transforms are the standard F(2,3) ones:
    //     U = G g G^T,   V = B^T d B,   Y = A^T (U .* V) A
    //   with G = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]],
    //        B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]],
    //        A^T = [[1,1,1,0],[0,1,-1,-1]].
    // =========================================================================

    /// Filter transform: `w[K][C][3][3]` -> `u[16][K][C]`.
    ///
    /// The weights are load-time constants, so this runs once per convolution
    /// and the result is cached.
    #[kernel]
    pub fn winograd_filter_transform(w: &[f32], k_out: u32, c_in: u32, mut u: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let lin = idx.get() as u32;
        let total = k_out * c_in;
        if lin >= total {
            return;
        }
        let kk = lin / c_in;
        let cc = lin % c_in;
        let base = (kk * c_in + cc) * 9;

        let mut g = [0.0f32; 9];
        let mut i = 0usize;
        while i < 9 {
            g[i] = w[(base + i as u32) as usize];
            i += 1;
        }

        // tmp = G g   (4x3)
        let mut tmp = [0.0f32; 12];
        let mut j = 0usize;
        while j < 3 {
            let g0 = g[j];
            let g1 = g[3 + j];
            let g2 = g[6 + j];
            tmp[j] = g0;
            tmp[3 + j] = 0.5f32 * (g0 + g1 + g2);
            tmp[6 + j] = 0.5f32 * (g0 - g1 + g2);
            tmp[9 + j] = g2;
            j += 1;
        }
        // U = tmp G^T  (4x4), written plane-major.
        let mut r = 0usize;
        while r < 4 {
            let t0 = tmp[r * 3];
            let t1 = tmp[r * 3 + 1];
            let t2 = tmp[r * 3 + 2];
            let v0 = t0;
            let v1 = 0.5f32 * (t0 + t1 + t2);
            let v2 = 0.5f32 * (t0 - t1 + t2);
            let v3 = t2;
            unsafe {
                *u.get_unchecked_mut(((r as u32 * 4) * total + lin) as usize) = v0;
                *u.get_unchecked_mut(((r as u32 * 4 + 1) * total + lin) as usize) = v1;
                *u.get_unchecked_mut(((r as u32 * 4 + 2) * total + lin) as usize) = v2;
                *u.get_unchecked_mut(((r as u32 * 4 + 3) * total + lin) as usize) = v3;
            }
            r += 1;
        }
    }

    /// Input transform: `x[C][H][W]` -> `v[16][C][T]`, T tiles of 2x2 output.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn winograd_input_transform(
        x: &[f32],
        c_in: u32,
        h_in: u32,
        w_in: u32,
        pad_h: u32,
        pad_w: u32,
        tiles_h: u32,
        tiles_w: u32,
        mut v: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let lin = idx.get() as u32;
        let t_total = tiles_h * tiles_w;
        let total = c_in * t_total;
        if lin >= total {
            return;
        }
        let cc = lin / t_total;
        let t = lin % t_total;
        let th = t / tiles_w;
        let tw = t % tiles_w;

        let ih0 = (th * 2) as i32 - pad_h as i32;
        let iw0 = (tw * 2) as i32 - pad_w as i32;
        let plane = cc * h_in * w_in;

        let mut d = [0.0f32; 16];
        let mut r = 0i32;
        while r < 4 {
            let ih = ih0 + r;
            let mut c = 0i32;
            while c < 4 {
                let iw = iw0 + c;
                if ih >= 0 && ih < h_in as i32 && iw >= 0 && iw < w_in as i32 {
                    d[(r * 4 + c) as usize] =
                        x[(plane + (ih as u32) * w_in + (iw as u32)) as usize];
                }
                c += 1;
            }
            r += 1;
        }

        // B^T d, column by column.
        let mut tmp = [0.0f32; 16];
        let mut j = 0usize;
        while j < 4 {
            let d0 = d[j];
            let d1 = d[4 + j];
            let d2 = d[8 + j];
            let d3 = d[12 + j];
            tmp[j] = d0 - d2;
            tmp[4 + j] = d1 + d2;
            tmp[8 + j] = d2 - d1;
            tmp[12 + j] = d1 - d3;
            j += 1;
        }
        // (B^T d) B, row by row.
        let mut i = 0usize;
        while i < 4 {
            let t0 = tmp[i * 4];
            let t1 = tmp[i * 4 + 1];
            let t2 = tmp[i * 4 + 2];
            let t3 = tmp[i * 4 + 3];
            let o0 = t0 - t2;
            let o1 = t1 + t2;
            let o2 = t2 - t1;
            let o3 = t1 - t3;
            unsafe {
                *v.get_unchecked_mut(((i as u32 * 4) * total + lin) as usize) = o0;
                *v.get_unchecked_mut(((i as u32 * 4 + 1) * total + lin) as usize) = o1;
                *v.get_unchecked_mut(((i as u32 * 4 + 2) * total + lin) as usize) = o2;
                *v.get_unchecked_mut(((i as u32 * 4 + 3) * total + lin) as usize) = o3;
            }
            i += 1;
        }
    }

    /// Output transform: `m[16][K][T]` -> `y[K][out_h][out_w]`, with the same
    /// epilogue the direct path fuses (bias, residual, activation).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn winograd_output_transform(
        mm: &[f32],
        k_out: u32,
        tiles_h: u32,
        tiles_w: u32,
        out_h: u32,
        out_w: u32,
        bias: &[f32],
        has_bias: u32,
        residual: &[f32],
        has_residual: u32,
        act: u32,
        lo: f32,
        hi: f32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let lin = idx.get() as u32;
        let t_total = tiles_h * tiles_w;
        let total = k_out * t_total;
        if lin >= total {
            return;
        }
        let kk = lin / t_total;
        let t = lin % t_total;
        let th = t / tiles_w;
        let tw = t % tiles_w;

        let mut d = [0.0f32; 16];
        let mut i = 0usize;
        while i < 16 {
            d[i] = mm[((i as u32) * total + lin) as usize];
            i += 1;
        }

        // A^T m, column by column -> 2x4.
        let mut tmp = [0.0f32; 8];
        let mut j = 0usize;
        while j < 4 {
            let m0 = d[j];
            let m1 = d[4 + j];
            let m2 = d[8 + j];
            let m3 = d[12 + j];
            tmp[j] = m0 + m1 + m2;
            tmp[4 + j] = m1 - m2 - m3;
            j += 1;
        }

        let b_val = if has_bias != 0 {
            bias[kk as usize]
        } else {
            0.0f32
        };
        let plane = kk * out_h * out_w;
        let mut r = 0u32;
        while r < 2 {
            let t0 = tmp[(r * 4) as usize];
            let t1 = tmp[(r * 4 + 1) as usize];
            let t2 = tmp[(r * 4 + 2) as usize];
            let t3 = tmp[(r * 4 + 3) as usize];
            let oh = th * 2 + r;
            if oh < out_h {
                let mut c = 0u32;
                while c < 2 {
                    let ow = tw * 2 + c;
                    if ow < out_w {
                        let val = if c == 0 { t0 + t1 + t2 } else { t1 - t2 - t3 };
                        let off = plane + oh * out_w + ow;
                        let res = if has_residual != 0 {
                            residual[off as usize]
                        } else {
                            0.0f32
                        };
                        unsafe {
                            *y.get_unchecked_mut(off as usize) =
                                apply_act(val + b_val + res, act, lo, hi);
                        }
                    }
                    c += 1;
                }
            }
            r += 1;
        }
    }

    #[kernel]
    pub fn conv2d_f16_tc_splitk(
        m: u32,
        n: u32,
        k: u32,
        k_per_split: u32,
        a_packed: &[u32],
        kpairs: u32,
        x: &[f32],
        _c_in: u32,
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
        out_w: u32,
        mut partials: DisjointSlice<f32>,
    ) {
        // 64 rows x 16 halves = 512 u32 each; 2 KB per tile, 4 KB per block.
        static mut AS: SharedArray<u32, 512> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, 512> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let warp = tid >> 5;
        let lane = tid & 31;
        let gid = lane >> 2; // groupID 0..7
        let tig = lane & 3; // threadID_in_group 0..3

        let row0 = thread::blockIdx_y() * 64;
        let col0 = thread::blockIdx_x() * 64;
        let split = thread::blockIdx_z();

        let k_begin = split * k_per_split;
        let k_stop = if k_begin + k_per_split < k {
            k_begin + k_per_split
        } else {
            k
        };

        let mut acc = [[0.0f32; 4]; 8];

        let mut k0 = k_begin;
        while k0 < k_stop {
            // Stage A as AS[row][kpair]: 512 registers, 4 per thread.
            let mut q = 0u32;
            #[unroll]
            while q < 4 {
                let e = tid + q * 128;
                let r = e >> 3;
                let kk = (e & 7) * 2;
                let gr = row0 + r;
                let g0 = k0 + kk;
                // Already half-packed at load time: one 32-bit read, no
                // conversion, half the bytes of the f32 form.
                unsafe {
                    AS[e as usize] = if gr < m && g0 < k_stop {
                        a_packed[(gr * kpairs + g0 / 2) as usize]
                    } else {
                        0u32
                    };
                }
                q += 1;
            }
            // Stage B directly from the input tensor: the column matrix is
            // never materialised. For a 3x3 convolution im2col inflates the
            // input ninefold, so reading X here saves both that write and the
            // inflated read the GEMM would otherwise do.
            let khw = kh * kw;
            let plane = h_in * w_in;
            let mut q2 = 0u32;
            #[unroll]
            while q2 < 4 {
                let e = tid + q2 * 128;
                let cc = e >> 3;
                let kk = (e & 7) * 2;
                let gc = col0 + cc;
                let g0 = k0 + kk;
                // One register holds the two K-neighbours of a column, which
                // for a convolution are two adjacent taps of the same patch.
                let oh = gc / out_w;
                let ow = gc % out_w;
                let mut lo = 0.0f32;
                let mut hi = 0.0f32;
                if gc < n {
                    if g0 < k_stop {
                        let ci = g0 / khw;
                        let krc = g0 % khw;
                        let ih = (oh * stride_h + (krc / kw) * dil_h) as i32 - pad_h as i32;
                        let iw = (ow * stride_w + (krc % kw) * dil_w) as i32 - pad_w as i32;
                        if ih >= 0 && ih < h_in as i32 && iw >= 0 && iw < w_in as i32 {
                            lo = x[(ci * plane + (ih as u32) * w_in + (iw as u32)) as usize];
                        }
                    }
                    let g1 = g0 + 1;
                    if g1 < k_stop {
                        let ci = g1 / khw;
                        let krc = g1 % khw;
                        let ih = (oh * stride_h + (krc / kw) * dil_h) as i32 - pad_h as i32;
                        let iw = (ow * stride_w + (krc % kw) * dil_w) as i32 - pad_w as i32;
                        if ih >= 0 && ih < h_in as i32 && iw >= 0 && iw < w_in as i32 {
                            hi = x[(ci * plane + (ih as u32) * w_in + (iw as u32)) as usize];
                        }
                    }
                }
                unsafe {
                    BS[e as usize] = pack_f16x2(lo, hi);
                }
                q2 += 1;
            }
            thread::sync_threads();

            // One A fragment for this warp's 16 rows, reused by all 8 tiles.
            let arow = warp * 16;
            let a0 = unsafe { AS[((arow + gid) * 8 + tig) as usize] };
            let a1 = unsafe { AS[((arow + gid + 8) * 8 + tig) as usize] };
            let a2 = unsafe { AS[((arow + gid) * 8 + tig + 4) as usize] };
            let a3 = unsafe { AS[((arow + gid + 8) * 8 + tig + 4) as usize] };

            let mut t = 0usize;
            #[unroll]
            while t < 8 {
                let ncol = t as u32 * 8 + gid;
                let b0 = unsafe { BS[(ncol * 8 + tig) as usize] };
                let b1 = unsafe { BS[(ncol * 8 + tig + 4) as usize] };
                unsafe {
                    mma_sync_m16n8k16_f32_f16(&mut acc[t], a0, a1, a2, a3, b0, b1);
                }
                t += 1;
            }
            thread::sync_threads();
            k0 += 16;
        }

        // Partial store; alpha, bias and activation belong to `reduce_splits`.
        let plane = split * m * n;
        let mut t = 0usize;
        #[unroll]
        while t < 8 {
            let gc = col0 + t as u32 * 8 + 2 * tig;
            let mut half = 0u32;
            #[unroll]
            while half < 2 {
                let gr = row0 + warp * 16 + gid + half * 8;
                if gr < m {
                    let base = plane + gr * n;
                    if gc < n {
                        unsafe {
                            *partials.get_unchecked_mut((base + gc) as usize) =
                                acc[t][(half * 2) as usize];
                        }
                    }
                    if gc + 1 < n {
                        unsafe {
                            *partials.get_unchecked_mut((base + gc + 1) as usize) =
                                acc[t][(half * 2 + 1) as usize];
                        }
                    }
                }
                half += 1;
            }
            t += 1;
        }
    }

    #[kernel]
    pub fn conv2d_f16_tc_w8(
        m: u32,
        n: u32,
        k: u32,
        k_per_split: u32,
        a_packed: &[u32],
        kpairs: u32,
        x: &[f32],
        _c_in: u32,
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
        out_w: u32,
        mut partials: DisjointSlice<f32>,
    ) {
        // 64 rows x 16 halves = 512 u32 per tile, double-buffered.
        static mut AS: SharedArray<u32, 1024> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, 1024> = SharedArray::UNINIT;
        static mut KTAB: SharedArray<u32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let warp = tid >> 5; // 0..8
        let lane = tid & 31;
        let gid = lane >> 2; // groupID 0..7
        let tig = lane & 3; // threadID_in_group 0..3

        let row0 = thread::blockIdx_y() * 64;
        let col0 = thread::blockIdx_x() * 64;
        let split = thread::blockIdx_z();

        let k_begin = split * k_per_split;
        let k_stop = if k_begin + k_per_split < k {
            k_begin + k_per_split
        } else {
            k
        };

        let mut acc = [[0.0f32; 4]; 4];

        // Loop-invariant per-thread geometry: which output pixel each staged
        // column belongs to, and which K offset this thread carries. 256 is a
        // multiple of 8, so both staged elements share the same K offset.
        let khw = kh * kw;
        let plane = h_in * w_in;
        let kk_t = (tid & 7) * 2;
        // Tap lookup: krc -> (kr, kc), packed. Filled once, so the two
        // divisions that decompose a tap index become one shared read.
        let mut tap = 0u32;
        while tap * 256 + tid < khw {
            let idx = tap * 256 + tid;
            unsafe {
                KTAB[idx as usize] = ((idx / kw) << 16) | (idx % kw);
            }
            tap += 1;
        }
        thread::sync_threads();

        let g_start = k_begin + kk_t;
        let mut ci_c = g_start / khw;
        let mut krc_c = g_start % khw;
        let gc_a = col0 + (tid >> 3);
        let gc_b = col0 + ((tid + 256) >> 3);
        let oh_a = gc_a / out_w;
        let ow_a = gc_a % out_w;
        let oh_b = gc_b / out_w;
        let ow_b = gc_b % out_w;

        // Global reads for the next K step are issued before the current
        // step's MMAs and stored to shared only afterwards, so their latency
        // is spent on tensor-core work. Single-buffered, this kernel stalled on
        // long_scoreboard at 6.3 warps per issue-active cycle with the tensor
        // pipe under 18% — the same shape the batched GEMM showed before it
        // was given a prefetch.
        let mut areg = [0u32; 2];
        let mut blo = [0.0f32; 2];
        let mut bhi = [0.0f32; 2];

        // ---- prologue: stage the first tile -------------------------------
        let mut q = 0u32;
        #[unroll]
        while q < 2 {
            let e = tid + q * 256;
            let r = e >> 3;
            let kk = (e & 7) * 2;
            let gr = row0 + r;
            let g0 = k_begin + kk;
            areg[q as usize] = if gr < m && g0 < k_stop {
                a_packed[(gr * kpairs + g0 / 2) as usize]
            } else {
                0u32
            };
            q += 1;
        }
        let g0p = k_begin + kk_t;
        let g1p = g0p + 1;
        let plo_p = unsafe { KTAB[krc_c as usize] };
        let (ci_lo_p, kr_lo_p, kc_lo_p) = (ci_c, plo_p >> 16, plo_p & 0xffff);
        let (ci_hi_p, krc_hi_p) = if krc_c + 1 >= khw {
            (ci_c + 1, 0u32)
        } else {
            (ci_c, krc_c + 1)
        };
        let phi_p = unsafe { KTAB[krc_hi_p as usize] };
        let (kr_hi_p, kc_hi_p) = (phi_p >> 16, phi_p & 0xffff);
        let mut qp = 0u32;
        #[unroll]
        while qp < 2 {
            let gc = if qp == 0 { gc_a } else { gc_b };
            let oh = if qp == 0 { oh_a } else { oh_b };
            let ow = if qp == 0 { ow_a } else { ow_b };
            let mut lo = 0.0f32;
            let mut hi = 0.0f32;
            if gc < n {
                if g0p < k_stop {
                    let ih = (oh * stride_h + kr_lo_p * dil_h) as i32 - pad_h as i32;
                    let iw = (ow * stride_w + kc_lo_p * dil_w) as i32 - pad_w as i32;
                    if ih >= 0 && ih < h_in as i32 && iw >= 0 && iw < w_in as i32 {
                        lo = x[(ci_lo_p * plane + (ih as u32) * w_in + (iw as u32)) as usize];
                    }
                }
                if g1p < k_stop {
                    let ih = (oh * stride_h + kr_hi_p * dil_h) as i32 - pad_h as i32;
                    let iw = (ow * stride_w + kc_hi_p * dil_w) as i32 - pad_w as i32;
                    if ih >= 0 && ih < h_in as i32 && iw >= 0 && iw < w_in as i32 {
                        hi = x[(ci_hi_p * plane + (ih as u32) * w_in + (iw as u32)) as usize];
                    }
                }
            }
            blo[qp as usize] = lo;
            bhi[qp as usize] = hi;
            qp += 1;
        }
        let mut qs0 = 0u32;
        #[unroll]
        while qs0 < 2 {
            let e = tid + qs0 * 256;
            unsafe {
                AS[e as usize] = areg[qs0 as usize];
                BS[e as usize] = pack_f16x2(blo[qs0 as usize], bhi[qs0 as usize]);
            }
            qs0 += 1;
        }
        thread::sync_threads();
        krc_c += 16;
        while krc_c >= khw {
            krc_c -= khw;
            ci_c += 1;
        }

        let mut buf = 0u32;
        let mut k0 = k_begin;
        while k0 < k_stop {
            let k_next = k0 + 16;

            if k_next < k_stop {
                let mut qa = 0u32;
                #[unroll]
                while qa < 2 {
                    let e = tid + qa * 256;
                    let r = e >> 3;
                    let kk = (e & 7) * 2;
                    let gr = row0 + r;
                    let g = k_next + kk;
                    areg[qa as usize] = if gr < m && g < k_stop {
                        a_packed[(gr * kpairs + g / 2) as usize]
                    } else {
                        0u32
                    };
                    qa += 1;
                }
                let g0 = k_next + kk_t;
                let g1 = g0 + 1;
                let plo = unsafe { KTAB[krc_c as usize] };
                let (ci_lo, kr_lo, kc_lo) = (ci_c, plo >> 16, plo & 0xffff);
                let (ci_hi, krc_hi) = if krc_c + 1 >= khw {
                    (ci_c + 1, 0u32)
                } else {
                    (ci_c, krc_c + 1)
                };
                let phi = unsafe { KTAB[krc_hi as usize] };
                let (kr_hi, kc_hi) = (phi >> 16, phi & 0xffff);
                let mut qb = 0u32;
                #[unroll]
                while qb < 2 {
                    let gc = if qb == 0 { gc_a } else { gc_b };
                    let oh = if qb == 0 { oh_a } else { oh_b };
                    let ow = if qb == 0 { ow_a } else { ow_b };
                    let mut lo = 0.0f32;
                    let mut hi = 0.0f32;
                    if gc < n {
                        if g0 < k_stop {
                            let ih = (oh * stride_h + kr_lo * dil_h) as i32 - pad_h as i32;
                            let iw = (ow * stride_w + kc_lo * dil_w) as i32 - pad_w as i32;
                            if ih >= 0 && ih < h_in as i32 && iw >= 0 && iw < w_in as i32 {
                                lo = x[(ci_lo * plane + (ih as u32) * w_in + (iw as u32)) as usize];
                            }
                        }
                        if g1 < k_stop {
                            let ih = (oh * stride_h + kr_hi * dil_h) as i32 - pad_h as i32;
                            let iw = (ow * stride_w + kc_hi * dil_w) as i32 - pad_w as i32;
                            if ih >= 0 && ih < h_in as i32 && iw >= 0 && iw < w_in as i32 {
                                hi = x[(ci_hi * plane + (ih as u32) * w_in + (iw as u32)) as usize];
                            }
                        }
                    }
                    blo[qb as usize] = lo;
                    bhi[qb as usize] = hi;
                    qb += 1;
                }
            }

            let base = buf * 512;
            let arow = (warp & 3) * 16;
            let ncol_base = (warp >> 2) * 32;
            let a0 = unsafe { AS[(base + (arow + gid) * 8 + tig) as usize] };
            let a1 = unsafe { AS[(base + (arow + gid + 8) * 8 + tig) as usize] };
            let a2 = unsafe { AS[(base + (arow + gid) * 8 + tig + 4) as usize] };
            let a3 = unsafe { AS[(base + (arow + gid + 8) * 8 + tig + 4) as usize] };

            let mut t = 0usize;
            #[unroll]
            while t < 4 {
                let ncol = ncol_base + t as u32 * 8 + gid;
                let b0 = unsafe { BS[(base + ncol * 8 + tig) as usize] };
                let b1 = unsafe { BS[(base + ncol * 8 + tig + 4) as usize] };
                unsafe {
                    mma_sync_m16n8k16_f32_f16(&mut acc[t], a0, a1, a2, a3, b0, b1);
                }
                t += 1;
            }

            if k_next < k_stop {
                let other = (buf ^ 1) * 512;
                let mut qs = 0u32;
                #[unroll]
                while qs < 2 {
                    let e = tid + qs * 256;
                    unsafe {
                        AS[(other + e) as usize] = areg[qs as usize];
                        BS[(other + e) as usize] = pack_f16x2(blo[qs as usize], bhi[qs as usize]);
                    }
                    qs += 1;
                }
                krc_c += 16;
                while krc_c >= khw {
                    krc_c -= khw;
                    ci_c += 1;
                }
            }
            thread::sync_threads();
            buf ^= 1;
            k0 = k_next;
        }

        let plane_o = split * m * n;
        let mut t = 0usize;
        #[unroll]
        while t < 4 {
            let gc = col0 + (warp >> 2) * 32 + t as u32 * 8 + 2 * tig;
            let mut half = 0u32;
            #[unroll]
            while half < 2 {
                let gr = row0 + (warp & 3) * 16 + gid + half * 8;
                if gr < m {
                    let base_o = plane_o + gr * n;
                    if gc < n {
                        unsafe {
                            *partials.get_unchecked_mut((base_o + gc) as usize) =
                                acc[t][(half * 2) as usize];
                        }
                    }
                    if gc + 1 < n {
                        unsafe {
                            *partials.get_unchecked_mut((base_o + gc + 1) as usize) =
                                acc[t][(half * 2 + 1) as usize];
                        }
                    }
                }
                half += 1;
            }
            t += 1;
        }
    }

    #[kernel]
    pub fn sgemm_reg_splitk(
        m: u32,
        n: u32,
        k: u32,
        k_per_split: u32,
        a: &[f32],
        b: &[f32],
        mut partials: DisjointSlice<f32>,
    ) {
        static mut AS: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut BS: SharedArray<f32, 512> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();
        let tid = ty * 16 + tx;
        let row0 = thread::blockIdx_y() * 64;
        let col0 = thread::blockIdx_x() * 64;
        let split = thread::blockIdx_z();

        let k_begin = split * k_per_split;
        let k_stop = if k_begin + k_per_split < k {
            k_begin + k_per_split
        } else {
            k
        };

        let mut acc = [[0.0f32; 4]; 4];

        let mut k0 = k_begin;
        while k0 < k_stop {
            let mut q = 0u32;
            #[unroll]
            while q < 2 {
                let e = tid + q * 256;

                let ar = e >> 3;
                let ak = e & 7;
                let agr = row0 + ar;
                let agc = k0 + ak;
                unsafe {
                    AS[(ak * 64 + ar) as usize] = if agr < m && agc < k_stop {
                        a[(agr * k + agc) as usize]
                    } else {
                        0.0f32
                    };
                }

                let br = e >> 6;
                let bc = e & 63;
                let bgr = k0 + br;
                let bgc = col0 + bc;
                unsafe {
                    BS[e as usize] = if bgr < k_stop && bgc < n {
                        b[(bgr * n + bgc) as usize]
                    } else {
                        0.0f32
                    };
                }
                q += 1;
            }
            thread::sync_threads();

            let mut kk = 0u32;
            #[unroll]
            while kk < 8 {
                let arow_base = (kk * 64 + ty) as usize;
                let bcol_base = (kk * 64 + tx) as usize;
                let a_frag = [
                    unsafe { AS[arow_base] },
                    unsafe { AS[arow_base + 16] },
                    unsafe { AS[arow_base + 32] },
                    unsafe { AS[arow_base + 48] },
                ];
                let b_frag = [
                    unsafe { BS[bcol_base] },
                    unsafe { BS[bcol_base + 16] },
                    unsafe { BS[bcol_base + 32] },
                    unsafe { BS[bcol_base + 48] },
                ];
                let mut i = 0usize;
                #[unroll]
                while i < 4 {
                    let av = a_frag[i];
                    let mut j = 0usize;
                    #[unroll]
                    while j < 4 {
                        acc[i][j] += av * b_frag[j];
                        j += 1;
                    }
                    i += 1;
                }
                kk += 1;
            }
            thread::sync_threads();
            k0 += 8;
        }

        let plane = split * m * n;
        let mut i = 0u32;
        #[unroll]
        while i < 4 {
            let gr = row0 + ty + 16 * i;
            if gr < m {
                let base = plane + gr * n;
                let mut j = 0u32;
                #[unroll]
                while j < 4 {
                    let gc = col0 + tx + 16 * j;
                    if gc < n {
                        unsafe {
                            *partials.get_unchecked_mut((base + gc) as usize) =
                                acc[i as usize][j as usize];
                        }
                    }
                    j += 1;
                }
            }
            i += 1;
        }
    }

    // =========================================================================
    // Split-K GEMM, 128x128 tile with an 8x8 register block.
    //
    //   Twice the arithmetic intensity of the 64x64 variant — 64 MACs per 16
    //   shared loads against 16 per 8 — at the cost of four times fewer blocks
    //   per output. That trade sank the plain register-tiled kernel, but with
    //   K split the block count is no longer bounded by the output, so the
    //   intensity is available again. Which of the two wins is per-shape and
    //   measured, not assumed.
    //
    //   Launch: grid=(ceil(n/128), ceil(m/128), splits), block=(16,16,1).
    // =========================================================================
    #[kernel]
    pub fn sgemm_reg8_splitk(
        m: u32,
        n: u32,
        k: u32,
        k_per_split: u32,
        a: &[f32],
        b: &[f32],
        mut partials: DisjointSlice<f32>,
    ) {
        static mut AS: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut BS: SharedArray<f32, 1024> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();
        let tid = ty * 16 + tx;
        let row0 = thread::blockIdx_y() * 128;
        let col0 = thread::blockIdx_x() * 128;
        let split = thread::blockIdx_z();

        let k_begin = split * k_per_split;
        let k_stop = if k_begin + k_per_split < k {
            k_begin + k_per_split
        } else {
            k
        };

        let mut acc = [[0.0f32; 8]; 8];

        let mut k0 = k_begin;
        while k0 < k_stop {
            let mut q = 0u32;
            #[unroll]
            while q < 4 {
                let e = tid + q * 256;

                let ar = e >> 3;
                let ak = e & 7;
                let agr = row0 + ar;
                let agc = k0 + ak;
                unsafe {
                    AS[(ak * 128 + ar) as usize] = if agr < m && agc < k_stop {
                        a[(agr * k + agc) as usize]
                    } else {
                        0.0f32
                    };
                }

                let br = e >> 7;
                let bc = e & 127;
                let bgr = k0 + br;
                let bgc = col0 + bc;
                unsafe {
                    BS[e as usize] = if bgr < k_stop && bgc < n {
                        b[(bgr * n + bgc) as usize]
                    } else {
                        0.0f32
                    };
                }
                q += 1;
            }
            thread::sync_threads();

            let mut kk = 0u32;
            #[unroll]
            while kk < 8 {
                let arow_base = (kk * 128 + ty) as usize;
                let bcol_base = (kk * 128 + tx) as usize;
                let a_frag = [
                    unsafe { AS[arow_base] },
                    unsafe { AS[arow_base + 16] },
                    unsafe { AS[arow_base + 32] },
                    unsafe { AS[arow_base + 48] },
                    unsafe { AS[arow_base + 64] },
                    unsafe { AS[arow_base + 80] },
                    unsafe { AS[arow_base + 96] },
                    unsafe { AS[arow_base + 112] },
                ];
                let b_frag = [
                    unsafe { BS[bcol_base] },
                    unsafe { BS[bcol_base + 16] },
                    unsafe { BS[bcol_base + 32] },
                    unsafe { BS[bcol_base + 48] },
                    unsafe { BS[bcol_base + 64] },
                    unsafe { BS[bcol_base + 80] },
                    unsafe { BS[bcol_base + 96] },
                    unsafe { BS[bcol_base + 112] },
                ];
                let mut i = 0usize;
                #[unroll]
                while i < 8 {
                    let av = a_frag[i];
                    let mut j = 0usize;
                    #[unroll]
                    while j < 8 {
                        acc[i][j] += av * b_frag[j];
                        j += 1;
                    }
                    i += 1;
                }
                kk += 1;
            }
            thread::sync_threads();
            k0 += 8;
        }

        let plane = split * m * n;
        let mut i = 0u32;
        #[unroll]
        while i < 8 {
            let gr = row0 + ty + 16 * i;
            if gr < m {
                let base = plane + gr * n;
                let mut j = 0u32;
                #[unroll]
                while j < 8 {
                    let gc = col0 + tx + 16 * j;
                    if gc < n {
                        unsafe {
                            *partials.get_unchecked_mut((base + gc) as usize) =
                                acc[i as usize][j as usize];
                        }
                    }
                    j += 1;
                }
            }
            i += 1;
        }
    }

    // =========================================================================
    // Reduce the split-K partials, applying alpha, the per-row bias and the
    // fused activation in the same pass.
    //   partials: [splits][mn]   out: [mn], row = i / n
    // =========================================================================
    #[kernel]
    pub fn reduce_splits(
        partials: &[f32],
        splits: u32,
        mn: u32,
        n: u32,
        alpha: f32,
        bias: &[f32],
        has_bias: u32,
        // Which axis the bias runs along. Convolution's bias is per output
        // channel over a [channels][spatial] result, so it is indexed by row;
        // a Gemm's is per output feature over [rows][features], so by column.
        // Reading a column bias by row is silent and catastrophic — it was
        // wrong on BERT by a relative half.
        bias_per_col: u32,
        // Residual tensor added before the activation, for the skip connection
        // a ResNet block would otherwise spend a whole kernel and a full
        // round trip of the activation on.
        residual: &[f32],
        has_residual: u32,
        act: u32,
        lo: f32,
        hi: f32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let Some(o) = out.get_mut(idx) {
            let mut sum = 0.0f32;
            let mut s = 0u32;
            while s < splits {
                sum += partials[(s * mn + i) as usize];
                s += 1;
            }
            let b_val = if has_bias != 0u32 {
                if bias_per_col != 0u32 {
                    bias[(i % n) as usize]
                } else {
                    bias[(i / n) as usize]
                }
            } else {
                0.0f32
            };
            let r_val = if has_residual != 0u32 {
                residual[i as usize]
            } else {
                0.0f32
            };
            *o = apply_act(alpha * sum + b_val + r_val, act, lo, hi);
        }
    }

    // =========================================================================
    // Implicit-GEMM Conv2D — the convolution as a GEMM whose B operand is
    // never materialized.
    //
    //   out[co, oh·ow] = sum_{ci, kr, kc} w[co, ci, kr, kc] · x[ci, ih, iw]
    //
    //   is the product W[M×K] · Col[K×N] with M = out channels,
    //   K = in_channels·Kh·Kw, N = out_h·out_w. The explicit path writes Col to
    //   DRAM and reads it straight back — for a 56×56×64 layer that is a 7 MB
    //   round trip per convolution, pure overhead. Here each staged tile
    //   element derives its input coordinate from (k, n) and reads the input
    //   tensor directly, so the column matrix exists only in shared memory.
    //
    //   Same 64×64 tile and 4×4 register block as `sgemm_reg`, and the bias
    //   and activation are folded into the epilogue, which removes another
    //   full-tensor pass. Padding is handled by substituting zero, exactly as
    //   im2col would.
    //
    //   Launch: grid=(⌈n/64⌉, ⌈m/64⌉, 1), block=(16,16,1).
    // =========================================================================
    #[kernel]
    pub fn conv2d_implicit_gemm(
        m: u32,
        n: u32,
        k: u32,
        w: &[f32],
        x: &[f32],
        bias: &[f32],
        has_bias: u32,
        _c_in: u32,
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
        out_w: u32,
        act: u32,
        lo: f32,
        hi: f32,
        mut c: DisjointSlice<f32>,
    ) {
        static mut AS: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut BS: SharedArray<f32, 512> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();
        let tid = ty * 16 + tx;
        let row0 = thread::blockIdx_y() * 64;
        let col0 = thread::blockIdx_x() * 64;

        let khw = kh * kw;
        let in_hw = h_in * w_in;

        let mut acc = [[0.0f32; 4]; 4];

        let mut k0 = 0u32;
        while k0 < k {
            let mut q = 0u32;
            #[unroll]
            while q < 2 {
                let e = tid + q * 256;

                // Stage the weight tile, transposed to AS[k][m].
                let ar = e >> 3;
                let ak = e & 7;
                let agr = row0 + ar;
                let agc = k0 + ak;
                unsafe {
                    AS[(ak * 64 + ar) as usize] = if agr < m && agc < k {
                        w[(agr * k + agc) as usize]
                    } else {
                        0.0f32
                    };
                }

                // Stage the implicit column tile: element (k0+br, col0+bc) of
                // the column matrix is input pixel (ci, ih, iw).
                let br = e >> 6;
                let bc = e & 63;
                let bk = k0 + br;
                let bn = col0 + bc;
                unsafe {
                    BS[e as usize] = if bk < k && bn < n {
                        let ci = bk / khw;
                        let krc = bk % khw;
                        let kr = krc / kw;
                        let kc = krc % kw;
                        let oh = bn / out_w;
                        let ow = bn % out_w;
                        let ih = (oh * stride_h + kr * dil_h) as i32 - pad_h as i32;
                        let iw = (ow * stride_w + kc * dil_w) as i32 - pad_w as i32;
                        if ih < 0 || ih >= h_in as i32 || iw < 0 || iw >= w_in as i32 {
                            0.0f32
                        } else {
                            x[(ci * in_hw + (ih as u32) * w_in + (iw as u32)) as usize]
                        }
                    } else {
                        0.0f32
                    };
                }
                q += 1;
            }
            thread::sync_threads();

            let mut kk = 0u32;
            #[unroll]
            while kk < 8 {
                let arow_base = (kk * 64 + ty) as usize;
                let bcol_base = (kk * 64 + tx) as usize;

                let a_frag = [
                    unsafe { AS[arow_base] },
                    unsafe { AS[arow_base + 16] },
                    unsafe { AS[arow_base + 32] },
                    unsafe { AS[arow_base + 48] },
                ];
                let b_frag = [
                    unsafe { BS[bcol_base] },
                    unsafe { BS[bcol_base + 16] },
                    unsafe { BS[bcol_base + 32] },
                    unsafe { BS[bcol_base + 48] },
                ];

                let mut i = 0usize;
                #[unroll]
                while i < 4 {
                    let av = a_frag[i];
                    let mut j = 0usize;
                    #[unroll]
                    while j < 4 {
                        acc[i][j] += av * b_frag[j];
                        j += 1;
                    }
                    i += 1;
                }
                kk += 1;
            }
            thread::sync_threads();
            k0 += 8;
        }

        // Epilogue: bias (per output channel = per row) and activation, while
        // the results are still in registers.
        let bias_len = bias.len();
        let mut i = 0u32;
        #[unroll]
        while i < 4 {
            let gr = row0 + ty + 16 * i;
            if gr < m {
                let b_val = if has_bias != 0u32 && (gr as usize) < bias_len {
                    bias[gr as usize]
                } else {
                    0.0f32
                };
                let base = gr * n;
                let mut j = 0u32;
                #[unroll]
                while j < 4 {
                    let gc = col0 + tx + 16 * j;
                    if gc < n {
                        let v = acc[i as usize][j as usize] + b_val;
                        unsafe {
                            *c.get_unchecked_mut((base + gc) as usize) = apply_act(v, act, lo, hi);
                        }
                    }
                    j += 1;
                }
            }
            i += 1;
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
                *cell = if has_beta {
                    alpha * c00 + beta * (*cell)
                } else {
                    alpha * c00
                };
            }
            if gc0 + 1 < n_sz {
                let p = gr0 * n_sz + gc0 + 1;
                let cell = unsafe { c.get_unchecked_mut(p) };
                *cell = if has_beta {
                    alpha * c01 + beta * (*cell)
                } else {
                    alpha * c01
                };
            }
            if gc0 + 2 < n_sz {
                let p = gr0 * n_sz + gc0 + 2;
                let cell = unsafe { c.get_unchecked_mut(p) };
                *cell = if has_beta {
                    alpha * c02 + beta * (*cell)
                } else {
                    alpha * c02
                };
            }
            if gc0 + 3 < n_sz {
                let p = gr0 * n_sz + gc0 + 3;
                let cell = unsafe { c.get_unchecked_mut(p) };
                *cell = if has_beta {
                    alpha * c03 + beta * (*cell)
                } else {
                    alpha * c03
                };
            }
        }
        if gr0 + 1 < m_sz {
            let r = gr0 + 1;
            if gc0 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0) };
                *cell = if has_beta {
                    alpha * c10 + beta * (*cell)
                } else {
                    alpha * c10
                };
            }
            if gc0 + 1 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 1) };
                *cell = if has_beta {
                    alpha * c11 + beta * (*cell)
                } else {
                    alpha * c11
                };
            }
            if gc0 + 2 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 2) };
                *cell = if has_beta {
                    alpha * c12 + beta * (*cell)
                } else {
                    alpha * c12
                };
            }
            if gc0 + 3 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 3) };
                *cell = if has_beta {
                    alpha * c13 + beta * (*cell)
                } else {
                    alpha * c13
                };
            }
        }
        if gr0 + 2 < m_sz {
            let r = gr0 + 2;
            if gc0 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0) };
                *cell = if has_beta {
                    alpha * c20 + beta * (*cell)
                } else {
                    alpha * c20
                };
            }
            if gc0 + 1 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 1) };
                *cell = if has_beta {
                    alpha * c21 + beta * (*cell)
                } else {
                    alpha * c21
                };
            }
            if gc0 + 2 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 2) };
                *cell = if has_beta {
                    alpha * c22 + beta * (*cell)
                } else {
                    alpha * c22
                };
            }
            if gc0 + 3 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 3) };
                *cell = if has_beta {
                    alpha * c23 + beta * (*cell)
                } else {
                    alpha * c23
                };
            }
        }
        if gr0 + 3 < m_sz {
            let r = gr0 + 3;
            if gc0 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0) };
                *cell = if has_beta {
                    alpha * c30 + beta * (*cell)
                } else {
                    alpha * c30
                };
            }
            if gc0 + 1 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 1) };
                *cell = if has_beta {
                    alpha * c31 + beta * (*cell)
                } else {
                    alpha * c31
                };
            }
            if gc0 + 2 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 2) };
                *cell = if has_beta {
                    alpha * c32 + beta * (*cell)
                } else {
                    alpha * c32
                };
            }
            if gc0 + 3 < n_sz {
                let cell = unsafe { c.get_unchecked_mut(r * n_sz + gc0 + 3) };
                *cell = if has_beta {
                    alpha * c33 + beta * (*cell)
                } else {
                    alpha * c33
                };
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
    // Bias add + fused activation — x[n, c, ...] = act(x[n, c, ...] + bias[c])
    //   act: 0 = none, 1 = relu, 2 = clip to [lo, hi]  (see graph_opt::ACT_*)
    //
    //   The channel index is derived in 32-bit arithmetic on purpose. The
    //   natural `(i / spatial) % channels` on usize operands compiles to
    //   div.u64 + rem.u64, which the GPU emulates in ~100 instructions *per
    //   element* — enough to make a pure-bandwidth op compute-bound. The 32-bit
    //   forms are a few instructions each, and every real tensor here is far
    //   below 2^32 elements.
    // =========================================================================
    #[kernel]
    pub fn bias_act(
        mut x: DisjointSlice<f32>,
        bias: &[f32],
        spatial: u32,
        channels: u32,
        has_bias: u32,
        // Residual added before the activation: a fused skip connection then
        // costs one extra load in a pass that already runs, instead of its own
        // kernel and a full round trip of the activation.
        residual: &[f32],
        has_residual: u32,
        act: u32,
        lo: f32,
        hi: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if let Some(v) = x.get_mut(idx) {
            let chan = ((i / spatial) % channels) as usize;
            let b = if has_bias != 0u32 { bias[chan] } else { 0.0f32 };
            let r = if has_residual != 0u32 {
                residual[i as usize]
            } else {
                0.0f32
            };
            *v = apply_act(*v + b + r, act, lo, hi);
        }
    }

    // =========================================================================
    // Batch normalization — inference mode
    //   y[n,c,hw] = gamma[c] * (x[n,c,hw] - mean[c]) / sqrt(var[c]+eps) + beta[c]
    //   n: batch, c: channels, hw: H*W spatial.  One thread per element.
    //   act: fused activation applied to the normalized value (see ACT_*).
    // =========================================================================
    //   32-bit channel math as in `bias_act`. The per-channel affine
    //   (scale, shift) is precomputed on the host, so there is no rsqrt and no
    //   mean/var traffic here either — this kernel is pure bandwidth.
    #[kernel]
    pub fn batch_norm_act(
        x: &[f32],
        scale: &[f32],
        shift: &[f32],
        spatial: u32,
        channels: u32,
        act: u32,
        lo: f32,
        hi: f32,
        mut y: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = y.get_mut(idx) {
            let chan = (((i as u32) / spatial) % channels) as usize;
            *out = apply_act(x[i] * scale[chan] + shift[chan], act, lo, hi);
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
        let i = idx.get() as u32;
        if let Some(out) = y.get_mut(idx) {
            // 32-bit index unpacking: see the note in `im2col`.
            let out_hw = out_h * out_w;
            let in_hw = h_in * w_in;

            let per_batch = c_out * out_hw;
            let batch_i = i / per_batch;
            let local_i = i % per_batch;
            let oc = local_i / out_hw;
            let spatial = local_i % out_hw;
            let oh = spatial / out_w;
            let ow = spatial % out_w;

            let in_ch = oc / cout_per_group;
            let w_base = oc * kh * kw;
            let in_base = batch_i * c_in * in_hw + in_ch * in_hw;

            let mut sum = 0.0f32;
            let mut kr = 0u32;
            while kr < kh {
                let ih = (oh * stride_h + kr * dil_h) as i32 - pad_h as i32;
                let mut kc = 0u32;
                while kc < kw {
                    let iw = (ow * stride_w + kc * dil_w) as i32 - pad_w as i32;
                    if ih >= 0 && ih < h_in as i32 && iw >= 0 && iw < w_in as i32 {
                        let in_off = in_base + (ih as u32) * w_in + (iw as u32);
                        sum += weight[(w_base + kr * kw + kc) as usize] * input[in_off as usize];
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
        let i = idx.get() as u32;
        if let Some(out) = col.get_mut(idx) {
            // Unpacking one flat index costs nine divisions; in 64-bit those
            // are emulated (~100 instructions each) and dominate a kernel that
            // otherwise just moves bytes. Every quantity here fits in 32 bits.
            let col_rows = c_in * kh * kw;
            let out_spatial = out_h * out_w;
            let col_per_batch = col_rows * out_spatial;

            let batch_i = i / col_per_batch;
            let local_i = i % col_per_batch;
            let kk = local_i / out_spatial;
            let spatial = local_i % out_spatial;
            let oh = spatial / out_w;
            let ow = spatial % out_w;

            let khw = kh * kw;
            let ki = kk / khw;
            let kr = (kk % khw) / kw;
            let kc = kk % kw;

            // Signed 32-bit so the padding test sees negative coordinates.
            let ih = (oh * stride_h + kr * dil_h) as i32 - pad_h as i32;
            let iw = (ow * stride_w + kc * dil_w) as i32 - pad_w as i32;

            *out = if ih < 0 || ih >= h_in as i32 || iw < 0 || iw >= w_in as i32 {
                0.0f32
            } else {
                let plane = h_in * w_in;
                let offset = batch_i * c_in * plane + ki * plane + (ih as u32) * w_in + (iw as u32);
                input[offset as usize]
            };
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
        let i = idx.get() as u32;
        if let Some(out) = y.get_mut(idx) {
            // 32-bit index unpacking: see the note in `im2col`.
            let out_hw = out_h * out_w;
            let in_hw = in_h * in_w;
            let per_batch = c * out_hw;
            let batch_i = i / per_batch;
            let local_i = i % per_batch;
            let ch = local_i / out_hw;
            let spatial = local_i % out_hw;
            let oh = spatial / out_w;
            let ow = spatial % out_w;

            let mut max_val = f32::NEG_INFINITY;
            let mut ki = 0u32;
            while ki < kh {
                let mut kj = 0u32;
                while kj < kw {
                    let ih_unpad = (oh * stride_h + ki) as i32 - pad_h as i32;
                    let iw_unpad = (ow * stride_w + kj) as i32 - pad_w as i32;
                    if ih_unpad >= 0
                        && ih_unpad < in_h as i32
                        && iw_unpad >= 0
                        && iw_unpad < in_w as i32
                    {
                        let offset = batch_i * c * in_hw
                            + ch * in_hw
                            + (ih_unpad as u32) * in_w
                            + (iw_unpad as u32);
                        let v = x[offset as usize];
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
    // =========================================================================
    // Transpose [B, S, H, D] -> [B, H, S, D] (perm 0,2,1,3).
    //
    //   This is every attention head-split in the graph — 36 of BERT's 48
    //   transposes, 48 of GPT-2's 60. D stays innermost on both sides, so it
    //   is not really a transpose at all: it is a permutation of contiguous
    //   D-element rows, fully coalesced in both directions.
    //
    //   The general N-D kernel cannot see that. It runs a loop over the rank
    //   with two integer divisions per dimension per element — eight divisions
    //   to move one float. Here the grid carries the coordinates instead:
    //   blockIdx.z is the flattened (b, s), blockIdx.y is the head, and the
    //   only division left is once per block on a uniform value.
    // =========================================================================
    #[kernel]
    pub fn transpose_0213(sdim: u32, hdim: u32, ddim: u32, x: &[f32], mut y: DisjointSlice<f32>) {
        let bs = thread::blockIdx_z(); // b * S + s
        let hh = thread::blockIdx_y();
        let dd = thread::blockIdx_x() * 128 + thread::threadIdx_x();
        if dd >= ddim {
            return;
        }
        let b = bs / sdim;
        let ss = bs % sdim;
        let in_off = (bs * hdim + hh) * ddim + dd;
        let out_off = ((b * hdim + hh) * sdim + ss) * ddim + dd;
        unsafe {
            *y.get_unchecked_mut(out_off as usize) = *x.get_unchecked(in_off as usize);
        }
    }

    // =========================================================================
    // Transpose [B, S, H, D] -> [B, H, D, S] (perm 0,2,3,1).
    //
    //   The K operand of every QK^T. Unlike 0213 this really does exchange the
    //   innermost axis, so a direct copy would read or write with stride H*D.
    //   Staging a 32x32 tile in shared memory makes both sides coalesced; the
    //   tile is padded to 33 columns so the transposed read hits 32 distinct
    //   banks.
    // =========================================================================
    #[kernel]
    pub fn transpose_0231(sdim: u32, hdim: u32, ddim: u32, x: &[f32], mut y: DisjointSlice<f32>) {
        static mut TILE: SharedArray<f32, 1056> = SharedArray::UNINIT;

        let bh = thread::blockIdx_z();
        let b = bh / hdim;
        let hh = bh % hdim;
        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();

        let s0 = thread::blockIdx_x() * 32;
        let d0 = thread::blockIdx_y() * 32;

        // Read [s][d] rows, D contiguous.
        let ss = s0 + ty;
        let dd = d0 + tx;
        let v = if ss < sdim && dd < ddim {
            unsafe { *x.get_unchecked((((b * sdim + ss) * hdim + hh) * ddim + dd) as usize) }
        } else {
            0.0f32
        };
        unsafe {
            TILE[(ty * 33 + tx) as usize] = v;
        }
        thread::sync_threads();

        // Write [d][s] rows, S contiguous.
        let so = s0 + tx;
        let dout = d0 + ty;
        if so < sdim && dout < ddim {
            let val = unsafe { TILE[(tx * 33 + ty) as usize] };
            unsafe {
                *y.get_unchecked_mut((((b * hdim + hh) * ddim + dout) * sdim + so) as usize) = val;
            }
        }
    }

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
        let i = idx.get() as u32;
        if let Some(o) = y.get_mut(idx) {
            // 32-bit index unpacking: see the note in `im2col`.
            let ni_inn = n_idx * inner;
            let oo = i / ni_inn;
            let rem = i % ni_inn;
            let ii = rem / inner;
            let k = rem % inner;
            // Negative indices count from the end, as ONNX specifies. Doing it
            // here rather than on the host is what lets the indices stay on the
            // device: for BERT they are the input tokens, so normalising them
            // host-side meant a device-to-host copy and a full synchronisation
            // per Gather (measured at 3 ms each).
            let raw = indices[ii as usize];
            let g = if raw < 0.0f32 {
                (raw + axis_len as f32) as u32
            } else {
                raw as u32
            };
            let src = oo * axis_len * inner + g * inner + k;
            *o = data[src as usize];
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
