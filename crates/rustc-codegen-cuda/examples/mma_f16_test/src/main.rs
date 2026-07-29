/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single-tile correctness for `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
//!
//! One warp computes D[16x8] = A[16x16] · B[16x8] (C=0) using the PTX-ISA
//! fragment↔lane mapping, and we compare against a CPU reference. Values are
//! small integers, exactly representable in f16, so a mismatch means the
//! layout or the two-halves-per-register packing is wrong rather than that
//! the rounding is imprecise.
//!
//! Build/run:  cargo oxide run mma_f16_test

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::wgmma::mma_sync_m16n8k16_f32_f16;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// IEEE-754 binary32 → binary16 bit pattern, round-to-nearest-even.
///
/// Integer ops only: float intrinsics map to libdevice, which pulls the kernel
/// into NVVM IR mode and skips PTX embedding. Infinities and NaNs saturate to
/// infinity and subnormal results flush to zero, which is what a GEMM operand
/// path wants anyway.
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
    // Round to nearest, ties to even, over the 13 discarded mantissa bits.
    let round_bias = 0x0fff + ((mant >> 13) & 1);
    let mant_rounded = mant + round_bias;
    // A carry out of the mantissa bumps the exponent.
    let exp_adjusted = new_exp as u32 + (mant_rounded >> 23);
    if exp_adjusted >= 0x1f {
        return sign | 0x7c00;
    }
    sign | (exp_adjusted << 10) | ((mant_rounded >> 13) & 0x03ff)
}

/// Pack two f32 values as two f16 halves in one register, `lo` in the low 16
/// bits — the order an `.f16x2` operand of `mma.sync` expects.
#[inline(always)]
fn pack_f16x2(lo: f32, hi: f32) -> u32 {
    f32_to_f16_bits(lo) | (f32_to_f16_bits(hi) << 16)
}

#[cuda_module]
mod kernels {
    use super::*;

    /// `a`: 16x16 row-major (M x K). `b`: 16x8 row-major (K x N).
    /// `d`: 16x8 row-major. One warp (32 lanes).
    #[kernel]
    pub fn mma_tile_f16(a: &[f32], b: &[f32], mut d: DisjointSlice<f32>) {
        let lane = thread::threadIdx_x() as usize;
        let gid = lane >> 2; // groupID 0..7
        let tig = lane & 3; // threadID_in_group 0..3

        // A (16x16 row-major), 4 registers of two halves:
        //   a0: row gid,   k = 2*tig, 2*tig+1
        //   a1: row gid+8, k = 2*tig, 2*tig+1
        //   a2: row gid,   k = 2*tig+8, 2*tig+9
        //   a3: row gid+8, k = 2*tig+8, 2*tig+9
        let k0 = 2 * tig;
        let k8 = 2 * tig + 8;
        let a0 = pack_f16x2(a[gid * 16 + k0], a[gid * 16 + k0 + 1]);
        let a1 = pack_f16x2(a[(gid + 8) * 16 + k0], a[(gid + 8) * 16 + k0 + 1]);
        let a2 = pack_f16x2(a[gid * 16 + k8], a[gid * 16 + k8 + 1]);
        let a3 = pack_f16x2(a[(gid + 8) * 16 + k8], a[(gid + 8) * 16 + k8 + 1]);

        // B (16x8 row-major = K x N), 2 registers of two halves, column gid:
        //   b0: k = 2*tig, 2*tig+1
        //   b1: k = 2*tig+8, 2*tig+9
        let b0 = pack_f16x2(b[k0 * 8 + gid], b[(k0 + 1) * 8 + gid]);
        let b1 = pack_f16x2(b[k8 * 8 + gid], b[(k8 + 1) * 8 + gid]);

        let mut acc = [0.0f32; 4];
        unsafe {
            mma_sync_m16n8k16_f32_f16(&mut acc, a0, a1, a2, a3, b0, b1);
        }

        // C/D (16x8), 4 f32 regs — same mapping as the tf32 variant:
        //   c0:(gid, 2*tig)    c1:(gid, 2*tig+1)
        //   c2:(gid+8, 2*tig)  c3:(gid+8, 2*tig+1)
        unsafe {
            *d.get_unchecked_mut(gid * 8 + 2 * tig) = acc[0];
            *d.get_unchecked_mut(gid * 8 + 2 * tig + 1) = acc[1];
            *d.get_unchecked_mut((gid + 8) * 8 + 2 * tig) = acc[2];
            *d.get_unchecked_mut((gid + 8) * 8 + 2 * tig + 1) = acc[3];
        }
    }
}

fn main() {
    println!("=== mma.sync.m16n8k16 f16 single-tile GEMM verification ===");
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    // Small integers: exact in f16, so this isolates layout from rounding.
    let a: Vec<f32> = (0..16 * 16).map(|i| ((i * 7 + 1) % 5) as f32).collect();
    let b: Vec<f32> = (0..16 * 8).map(|i| ((i * 3 + 2) % 4) as f32).collect();

    // CPU reference: D[16x8] = A[16x16] · B[16x8].
    let mut expected = vec![0.0f32; 16 * 8];
    for m in 0..16 {
        for n in 0..8 {
            let mut s = 0.0f32;
            for k in 0..16 {
                s += a[m * 16 + k] * b[k * 8 + n];
            }
            expected[m * 8 + n] = s;
        }
    }

    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
    let mut d_dev = DeviceBuffer::<f32>::zeroed(&stream, 16 * 8).unwrap();

    let module = ctx
        .load_module_from_file("mma_f16_test.ptx")
        .expect("load PTX");
    let module = kernels::from_module(module).expect("typed module");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { module.mma_tile_f16(stream.as_ref(), cfg, &a_dev, &b_dev, &mut d_dev) }
        .expect("launch");
    stream.synchronize().unwrap();

    let got = d_dev.to_host_vec(&stream).unwrap();
    let max_err = got
        .iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0f32, f32::max);
    let n_ok = got
        .iter()
        .zip(expected.iter())
        .filter(|(g, e)| (**g - **e).abs() < 1e-3)
        .count();

    println!("got[0..8]      = {:?}", &got[..8]);
    println!("expected[0..8] = {:?}", &expected[..8]);
    println!("{}/128 elements match; max abs err = {:.3e}", n_ok, max_err);

    if n_ok == 128 {
        println!("\n✓ SUCCESS: mma.sync.m16n8k16 f16 tile matches CPU GEMM (layout correct)");
    } else {
        println!("\n✗ FAILED: fragment layout or half packing is wrong");
        std::process::exit(1);
    }
}
