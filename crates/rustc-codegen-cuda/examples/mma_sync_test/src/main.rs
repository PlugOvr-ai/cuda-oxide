/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single-tile correctness for `mma.sync.aligned.m16n8k8.row.col.f32.tf32`.
//!
//! One warp computes D[16x8] = A[16x8] · B[8x8] (C=0) using the PTX-ISA
//! fragment↔lane mapping, and we compare against a CPU reference. Values
//! are small integers (tf32-exact) so this isolates the LAYOUT.
//!
//! Build/run:  cargo oxide run mma_sync_test

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::wgmma::mma_sync_m16n8k8_f32_tf32;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// `a`: 16x8 row-major. `b`: 8x8 row-major (K x N). `d`: 16x8 row-major.
    /// One warp (32 lanes).  tf32 reg value = the f32 bit pattern.
    #[kernel]
    pub fn mma_tile(a: &[f32], b: &[f32], mut d: DisjointSlice<f32>) {
        let lane = thread::threadIdx_x() as usize;
        let gid = lane >> 2; // groupID 0..7
        let tig = lane & 3; // threadID_in_group 0..3

        // A (16x8 row-major), 4 tf32 regs:
        //  a0:(gid,    tig)    a1:(gid+8, tig)
        //  a2:(gid,    tig+4)  a3:(gid+8, tig+4)
        let a0 = a[gid * 8 + tig].to_bits();
        let a1 = a[(gid + 8) * 8 + tig].to_bits();
        let a2 = a[gid * 8 + tig + 4].to_bits();
        let a3 = a[(gid + 8) * 8 + tig + 4].to_bits();

        // B (8x8 row-major = K x N), 2 tf32 regs:
        //  b0:(k=tig,   n=gid)   b1:(k=tig+4, n=gid)
        let b0 = b[tig * 8 + gid].to_bits();
        let b1 = (tig + 4) * 8 + gid;
        let b1 = b[b1].to_bits();

        let mut acc = [0.0f32; 4];
        unsafe {
            mma_sync_m16n8k8_f32_tf32(&mut acc, a0, a1, a2, a3, b0, b1);
        }

        // C/D (16x8), 4 f32 regs:
        //  c0:(gid,   2*tig)    c1:(gid,   2*tig+1)
        //  c2:(gid+8, 2*tig)    c3:(gid+8, 2*tig+1)
        unsafe {
            *d.get_unchecked_mut(gid * 8 + 2 * tig) = acc[0];
            *d.get_unchecked_mut(gid * 8 + 2 * tig + 1) = acc[1];
            *d.get_unchecked_mut((gid + 8) * 8 + 2 * tig) = acc[2];
            *d.get_unchecked_mut((gid + 8) * 8 + 2 * tig + 1) = acc[3];
        }
    }
}

fn main() {
    println!("=== mma.sync.m16n8k8 tf32 single-tile GEMM verification ===");
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    // Deterministic small integers (tf32-exact).
    let a: Vec<f32> = (0..16 * 8).map(|i| ((i * 7 + 1) % 5) as f32).collect();
    let b: Vec<f32> = (0..8 * 8).map(|i| ((i * 3 + 2) % 4) as f32).collect();

    // CPU reference: D[16x8] = A[16x8] · B[8x8].
    let mut expected = vec![0.0f32; 16 * 8];
    for m in 0..16 {
        for n in 0..8 {
            let mut s = 0.0f32;
            for k in 0..8 {
                s += a[m * 8 + k] * b[k * 8 + n];
            }
            expected[m * 8 + n] = s;
        }
    }

    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
    let mut d_dev = DeviceBuffer::<f32>::zeroed(&stream, 16 * 8).unwrap();

    let module = ctx
        .load_module_from_file("mma_sync_test.ptx")
        .expect("load PTX");
    let module = kernels::from_module(module).expect("typed module");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    module
        .mma_tile(stream.as_ref(), cfg, &a_dev, &b_dev, &mut d_dev)
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
        println!("\n✓ SUCCESS: mma.sync tile matches CPU GEMM (layout correct)");
    } else {
        println!("\n✗ FAILED: fragment layout is wrong");
        std::process::exit(1);
    }
}
