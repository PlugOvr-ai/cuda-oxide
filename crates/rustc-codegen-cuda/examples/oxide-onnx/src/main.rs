/*
 * oxide_onnx — CUDA ONNX inference engine built on cuda-oxide.
 *
 * Sections:
 *   1. Proto include (prost-generated ONNX types)
 *   2. Module declarations
 *   3. Unit tests for individual CUDA kernels (vs CPU reference)
 *   4. End-to-end model test (ResNet50/MobileNetV2 if present)
 *   5. Speed benchmark: oxide vs Candle (if CUDA Candle available)
 *
 * Build and run:
 *   cargo oxide run oxide_onnx
 */

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

// ---------------------------------------------------------------------------
// 1. Include prost-generated ONNX types.
// ---------------------------------------------------------------------------
pub mod proto {
    pub mod onnx {
        include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
    }
}

// ---------------------------------------------------------------------------
// 2. Module declarations
// ---------------------------------------------------------------------------
pub mod cpu_ref;
pub mod executor;
pub mod graph_opt;
pub mod kernels;
pub mod model;
pub mod tensor;

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

use crate::executor::OnnxExecutor;
use crate::kernels::gpu;
use crate::model::load_model;

const RESNET50_PATH: &str = "models/resnet50-v2-7.onnx";
const MOBILENET_PATH: &str = "models/mobilenetv2-10.onnx";
const VIT_PATH: &str = "models/vit-base-patch16-224.onnx";
const BERT_PATH: &str = "models/bert-base-uncased.onnx";
const GPT2_PATH: &str = "models/gpt2-lmhead.onnx";
// Architectures beyond classification and transformers, used to keep the
// engine honest about ops it would otherwise never see.
const SUPERRES_PATH: &str = "models/super-resolution-10.onnx";
const SHUFFLENET_PATH: &str = "models/shufflenet-v2-10.onnx";
const YOLO_PATH: &str = "models/tinyyolov2-8.onnx";
const FCN_PATH: &str = "models/fcn-resnet50-11.onnx";
const STYLE_PATH: &str = "models/mosaic-9.onnx";

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║         oxide_onnx — CUDA ONNX inference engine       ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();

    println!("═══ Section 1: Unit kernel tests ═══");
    unit_tests()?;
    println!();

    // OXIDE_BENCH_KERNELS=1 runs the per-kernel microbenchmark instead of the
    // models. Whole-model timings cannot attribute a regression to a kernel,
    // and the per-node profile inflates cheap ops by synchronising after each
    // one; this measures kernels the way they actually run.
    if std::env::var("OXIDE_BENCH_KERNELS").is_ok() {
        bench_kernels()?;
        return Ok(());
    }

    let resnet_present = std::path::Path::new(RESNET50_PATH).exists();
    let mobilenet_present = std::path::Path::new(MOBILENET_PATH).exists();
    let vit_present = std::path::Path::new(VIT_PATH).exists();
    let bert_present = std::path::Path::new(BERT_PATH).exists();
    let gpt2_present = std::path::Path::new(GPT2_PATH).exists();

    if resnet_present || mobilenet_present || vit_present || bert_present || gpt2_present {
        println!("═══ Section 2: End-to-end model inference ═══");
        if resnet_present {
            run_model(RESNET50_PATH, "ResNet50-v2")?;
        }
        if mobilenet_present {
            run_model(MOBILENET_PATH, "MobileNetV2")?;
        }
        if vit_present {
            run_model(VIT_PATH, "ViT-B/16")?;
        }
        if bert_present {
            run_bert(BERT_PATH, "BERT-base")?;
        }
        if gpt2_present {
            run_gpt2(GPT2_PATH, "GPT-2-LMHead")?;
        }

        // Same engine, architectures it was not built around.
        for (path, name, shape) in [
            (SHUFFLENET_PATH, "ShuffleNet-v2", vec![1usize, 3, 224, 224]),
            (
                SUPERRES_PATH,
                "Sub-pixel CNN (super-res)",
                vec![1, 1, 224, 224],
            ),
            (YOLO_PATH, "Tiny-YOLOv2 (detection)", vec![1, 3, 416, 416]),
            (
                FCN_PATH,
                "FCN-ResNet50 (segmentation)",
                vec![1, 3, 224, 224],
            ),
            (STYLE_PATH, "Mosaic (style transfer)", vec![1, 3, 224, 224]),
        ] {
            if std::path::Path::new(path).exists() {
                if let Err(e) = run_generic_model(path, name, &shape) {
                    println!("  {} FAILED: {}", name, e);
                }
            }
        }
        println!();

        println!("═══ Section 3: Throughput benchmark ═══");
        if resnet_present {
            run_benchmarks(RESNET50_PATH, "ResNet50-v2")?;
        }
        if vit_present {
            run_benchmarks(VIT_PATH, "ViT-B/16")?;
        }
    } else {
        println!(
            "No ONNX models found. Run scripts/download_models.sh to download ResNet50 and MobileNetV2."
        );
        println!();
        println!("══════════════════════════════════════════");
        println!("Unit tests passed. Build is correct.");
        println!("══════════════════════════════════════════");
    }

    Ok(())
}

// ===========================================================================
// Section 1: Unit tests — GPU kernel vs CPU reference
// ===========================================================================

// ===========================================================================
// Kernel microbenchmark
// ===========================================================================

/// Time one kernel launch in steady state.
///
/// Launches `iters` times back to back and synchronises once, so the result is
/// the kernel's own throughput rather than launch latency, and takes the
/// minimum over several rounds because this GPU is shared — the minimum is the
/// only statistic that survives another process taking the SMs.
fn time_kernel<F>(stream: &cuda_core::CudaStream, iters: usize, mut launch: F) -> Result<f64>
where
    F: FnMut() -> Result<()>,
{
    const ROUNDS: usize = 5;
    // Warm up: first launch pays PTX JIT and cache population.
    for _ in 0..8 {
        launch()?;
    }
    stream
        .synchronize()
        .map_err(|e| anyhow::anyhow!("sync: {:?}", e))?;

    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let t0 = Instant::now();
        for _ in 0..iters {
            launch()?;
        }
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync: {:?}", e))?;
        let per_iter = t0.elapsed().as_secs_f64() / iters as f64;
        best = best.min(per_iter);
    }
    Ok(best)
}

/// Per-kernel throughput on the shapes these models actually run.
fn bench_kernels() -> Result<()> {
    let ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CUDA context: {:?}", e))?;
    let stream = ctx.default_stream();
    let module = gpu::load(&ctx).map_err(|e| anyhow::anyhow!("load module: {:?}", e))?;

    println!("═══ Kernel microbenchmark ═══");
    println!("  (min of 5 rounds × 50 launches; 3090 peak ≈ 936 GB/s)");
    println!();

    // The BatchNorm shapes ResNet50-v2 leaves after Conv→BN folding: the
    // residual-stream normalisations, one per block.
    let bn_shapes: [(usize, usize, usize); 4] = [
        (256, 3136, 3), // stage 1: 256×56×56
        (512, 784, 4),  // stage 2: 512×28×28
        (1024, 196, 6), // stage 3: 1024×14×14
        (2048, 49, 3),  // stage 4: 2048×7×7
    ];

    println!("  batch_norm_act");
    println!(
        "    {:>6} {:>8} {:>10} {:>10} {:>9} {:>7}",
        "chan", "spatial", "elems", "µs", "GB/s", "n×"
    );
    let mut bn_total_us = 0.0;
    for (channels, spatial, count) in bn_shapes {
        let numel = channels * spatial;
        let x = DeviceBuffer::<f32>::zeroed(&stream, numel)
            .map_err(|e| anyhow::anyhow!("alloc x: {:?}", e))?;
        let scale = DeviceBuffer::<f32>::zeroed(&stream, channels)
            .map_err(|e| anyhow::anyhow!("alloc scale: {:?}", e))?;
        let shift = DeviceBuffer::<f32>::zeroed(&stream, channels)
            .map_err(|e| anyhow::anyhow!("alloc shift: {:?}", e))?;
        let mut y = DeviceBuffer::<f32>::zeroed(&stream, numel)
            .map_err(|e| anyhow::anyhow!("alloc y: {:?}", e))?;

        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let secs = time_kernel(&stream, 50, || {
            unsafe {
                module.batch_norm_act(
                    &stream,
                    cfg,
                    &x,
                    &scale,
                    &shift,
                    spatial as u32,
                    channels as u32,
                    1,
                    0.0,
                    0.0,
                    &mut y,
                )
            }
            .map_err(|e| anyhow::anyhow!("batch_norm_act: {:?}", e))
        })?;

        // One read of x plus one write of y; the per-channel parameters are
        // negligible and cached.
        let bytes = (numel * 4 * 2) as f64;
        let gbps = bytes / secs / 1e9;
        bn_total_us += secs * 1e6 * count as f64;
        println!(
            "    {:>6} {:>8} {:>10} {:>10.1} {:>9.1} {:>7}",
            channels,
            spatial,
            numel,
            secs * 1e6,
            gbps,
            count
        );
    }
    println!(
        "    ResNet50 total for these {} launches: {:.2} ms",
        bn_shapes.iter().map(|s| s.2).sum::<usize>(),
        bn_total_us / 1000.0
    );
    println!();

    // Every GEMM ResNet50-v2 issues at batch 1, with how many times each
    // occurs, so the column sums account for the whole model rather than a
    // sample of it. 3×3 convolutions go through im2col; 1×1 convolutions are
    // fed to the GEMM directly.
    let gemm_shapes: [(usize, usize, usize, usize, &str); 13] = [
        (64, 12544, 147, 1, "stem 7×7"),
        (64, 3136, 576, 3, "s1 3×3"),
        (64, 3136, 256, 3, "s1 1×1 red"),
        (256, 3136, 64, 3, "s1 1×1 exp"),
        (128, 784, 1152, 4, "s2 3×3"),
        (128, 784, 512, 4, "s2 1×1 red"),
        (512, 784, 128, 4, "s2 1×1 exp"),
        (256, 196, 2304, 6, "s3 3×3"),
        (256, 196, 1024, 6, "s3 1×1 red"),
        (1024, 196, 256, 6, "s3 1×1 exp"),
        (512, 49, 4608, 3, "s4 3×3"),
        (512, 49, 2048, 3, "s4 1×1 red"),
        (2048, 49, 512, 3, "s4 1×1 exp"),
    ];

    println!("  sgemm_tiled — every GEMM in ResNet50-v2, batch 1");
    println!(
        "    {:>11} {:>5} {:>6} {:>5} {:>3} {:>9} {:>9} {:>9}",
        "layer", "M", "N", "K", "n×", "µs", "GFLOP/s", "total ms"
    );
    let mut gemm_total_ms = 0.0;
    for (m, n, k, count, name) in gemm_shapes {
        let a = DeviceBuffer::<f32>::zeroed(&stream, m * k)
            .map_err(|e| anyhow::anyhow!("alloc a: {:?}", e))?;
        let b = DeviceBuffer::<f32>::zeroed(&stream, k * n)
            .map_err(|e| anyhow::anyhow!("alloc b: {:?}", e))?;
        let mut c = DeviceBuffer::<f32>::zeroed(&stream, m * n)
            .map_err(|e| anyhow::anyhow!("alloc c: {:?}", e))?;

        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let secs = time_kernel(&stream, 50, || {
            unsafe {
                module.sgemm_tiled(
                    &stream, cfg, m as u32, n as u32, k as u32, 1.0, &a, &b, 0.0, &mut c,
                )
            }
            .map_err(|e| anyhow::anyhow!("sgemm_tiled: {:?}", e))
        })?;
        let gflops = (2.0 * m as f64 * n as f64 * k as f64) / secs / 1e9;
        let total_ms = secs * count as f64 * 1e3;
        gemm_total_ms += total_ms;
        println!(
            "    {:>11} {:>5} {:>6} {:>5} {:>3} {:>9.1} {:>9.0} {:>9.2}",
            name,
            m,
            n,
            k,
            count,
            secs * 1e6,
            gflops,
            total_ms
        );
    }
    println!("    {:>50} {:>9.2}", "GEMM total:", gemm_total_ms);
    println!();

    // Same shapes through the split-K path: 64×64 tile, 4×4 per thread, with
    // the K dimension partitioned so every shape gets enough blocks to fill
    // the GPU. Includes the reduction pass in the timing, since that is part
    // of the cost of the method.
    println!("  sgemm_reg_splitk (+ reduction)");
    println!(
        "    {:>11} {:>5} {:>6} {:>5} {:>3} {:>6} {:>9} {:>9} {:>9}",
        "layer", "M", "N", "K", "n×", "split", "µs", "GFLOP/s", "total ms"
    );
    println!("    {:>62} {:>8} {:>7}", "", "reduce", "of it");
    let mut splitk_total_ms = 0.0;
    let mut splitk_reduce_ms = 0.0;
    for (m, n, k, count, name) in gemm_shapes {
        let a = DeviceBuffer::<f32>::zeroed(&stream, m * k)
            .map_err(|e| anyhow::anyhow!("alloc a: {:?}", e))?;
        let b = DeviceBuffer::<f32>::zeroed(&stream, k * n)
            .map_err(|e| anyhow::anyhow!("alloc b: {:?}", e))?;
        let mut c = DeviceBuffer::<f32>::zeroed(&stream, m * n)
            .map_err(|e| anyhow::anyhow!("alloc c: {:?}", e))?;
        let bias = DeviceBuffer::<f32>::zeroed(&stream, m)
            .map_err(|e| anyhow::anyhow!("alloc bias: {:?}", e))?;

        let splits = OnnxExecutor::split_factor(m, n, k);
        let k_per_split = k.div_ceil(splits).next_multiple_of(8);
        let mut partials = DeviceBuffer::<f32>::zeroed(&stream, splits * m * n)
            .map_err(|e| anyhow::anyhow!("alloc partials: {:?}", e))?;

        let gemm_cfg = LaunchConfig {
            grid_dim: (
                (n as u32).div_ceil(64).max(1),
                (m as u32).div_ceil(64).max(1),
                splits as u32,
            ),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let red_cfg = LaunchConfig::for_num_elems((m * n) as u32);

        let secs = time_kernel(&stream, 50, || {
            unsafe {
                module.sgemm_reg_splitk(
                    &stream,
                    gemm_cfg,
                    m as u32,
                    n as u32,
                    k as u32,
                    k_per_split as u32,
                    &a,
                    &b,
                    &mut partials,
                )
            }
            .map_err(|e| anyhow::anyhow!("splitk: {:?}", e))?;
            unsafe {
                module.reduce_splits(
                    &stream,
                    red_cfg,
                    &partials,
                    splits as u32,
                    (m * n) as u32,
                    n as u32,
                    1.0,
                    &bias,
                    0,
                    &bias,
                    0,
                    0,
                    0.0,
                    0.0,
                    &mut c,
                )
            }
            .map_err(|e| anyhow::anyhow!("reduce: {:?}", e))
        })?;
        // Time the reduction alone, to see what fraction of the split-K path
        // is the extra pass over the output rather than the multiply.
        let red_secs = time_kernel(&stream, 50, || {
            unsafe {
                module.reduce_splits(
                    &stream,
                    red_cfg,
                    &partials,
                    splits as u32,
                    (m * n) as u32,
                    n as u32,
                    1.0,
                    &bias,
                    0,
                    &bias,
                    0,
                    0,
                    0.0,
                    0.0,
                    &mut c,
                )
            }
            .map_err(|e| anyhow::anyhow!("reduce: {:?}", e))
        })?;

        let gflops = (2.0 * m as f64 * n as f64 * k as f64) / secs / 1e9;
        let total_ms = secs * count as f64 * 1e3;
        splitk_total_ms += total_ms;
        splitk_reduce_ms += red_secs * count as f64 * 1e3;
        println!(
            "    {:>11} {:>5} {:>6} {:>5} {:>3} {:>6} {:>9.1} {:>9.0} {:>9.2} {:>8.1} {:>6.0}%",
            name,
            m,
            n,
            k,
            count,
            splits,
            secs * 1e6,
            gflops,
            total_ms,
            red_secs * 1e6,
            100.0 * red_secs / secs
        );
    }
    println!("    {:>57} {:>9.2}", "split-K total:", splitk_total_ms);
    println!();

    // The 8x8 register block: twice the intensity, a quarter of the blocks.
    // Split-K supplies the blocks, so the trade may now pay.
    println!("  sgemm_reg8_splitk (128×128 tile, 8×8 per thread)");
    println!(
        "    {:>11} {:>5} {:>6} {:>5} {:>3} {:>6} {:>9} {:>9} {:>9}",
        "layer", "M", "N", "K", "n×", "split", "µs", "GFLOP/s", "total ms"
    );
    let mut reg8_total_ms = 0.0;
    for (m, n, k, count, name) in gemm_shapes {
        let a = DeviceBuffer::<f32>::zeroed(&stream, m * k)
            .map_err(|e| anyhow::anyhow!("alloc a: {:?}", e))?;
        let b = DeviceBuffer::<f32>::zeroed(&stream, k * n)
            .map_err(|e| anyhow::anyhow!("alloc b: {:?}", e))?;
        let mut c = DeviceBuffer::<f32>::zeroed(&stream, m * n)
            .map_err(|e| anyhow::anyhow!("alloc c: {:?}", e))?;
        let bias = DeviceBuffer::<f32>::zeroed(&stream, m)
            .map_err(|e| anyhow::anyhow!("alloc bias: {:?}", e))?;

        // Same block target as split_factor, against 128×128 tiles.
        let base_blocks = m.div_ceil(128) * n.div_ceil(128);
        let splits = (164usize.div_ceil(base_blocks.max(1))).clamp(1, (k / 128).max(1).min(16));
        let k_per_split = k.div_ceil(splits).next_multiple_of(8);
        let mut partials = DeviceBuffer::<f32>::zeroed(&stream, splits * m * n)
            .map_err(|e| anyhow::anyhow!("alloc partials: {:?}", e))?;

        let gemm_cfg = LaunchConfig {
            grid_dim: (
                (n as u32).div_ceil(128).max(1),
                (m as u32).div_ceil(128).max(1),
                splits as u32,
            ),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let red_cfg = LaunchConfig::for_num_elems((m * n) as u32);

        let secs = time_kernel(&stream, 50, || {
            unsafe {
                module.sgemm_reg8_splitk(
                    &stream,
                    gemm_cfg,
                    m as u32,
                    n as u32,
                    k as u32,
                    k_per_split as u32,
                    &a,
                    &b,
                    &mut partials,
                )
            }
            .map_err(|e| anyhow::anyhow!("reg8: {:?}", e))?;
            unsafe {
                module.reduce_splits(
                    &stream,
                    red_cfg,
                    &partials,
                    splits as u32,
                    (m * n) as u32,
                    n as u32,
                    1.0,
                    &bias,
                    0,
                    &bias,
                    0,
                    0,
                    0.0,
                    0.0,
                    &mut c,
                )
            }
            .map_err(|e| anyhow::anyhow!("reduce: {:?}", e))
        })?;
        let gflops = (2.0 * m as f64 * n as f64 * k as f64) / secs / 1e9;
        let total_ms = secs * count as f64 * 1e3;
        reg8_total_ms += total_ms;
        println!(
            "    {:>11} {:>5} {:>6} {:>5} {:>3} {:>6} {:>9.1} {:>9.0} {:>9.2}",
            name,
            m,
            n,
            k,
            count,
            splits,
            secs * 1e6,
            gflops,
            total_ms
        );
    }
    println!("    {:>57} {:>9.2}", "8×8 split-K total:", reg8_total_ms);
    println!(
        "    {:>57} {:>9.2}x",
        "vs 4×4 split-K:",
        splitk_total_ms / reg8_total_ms
    );
    println!(
        "    {:>57} {:>9.2}  ({:.0}%)",
        "of which reduction:",
        splitk_reduce_ms,
        100.0 * splitk_reduce_ms / splitk_total_ms
    );
    println!(
        "    {:>57} {:>9.2}x",
        "speed-up over sgemm_tiled:",
        gemm_total_ms / splitk_total_ms
    );
    println!();

    // f16 tensor cores: accuracy against the f32 kernel, then speed. f16 has a
    // 10-bit mantissa, so the question is not whether it differs but whether
    // the difference is small relative to the values involved.
    println!("  sgemm_f16_tc_splitk_wpacked (f16 tensor cores, pre-packed weights)");
    println!(
        "    {:>11} {:>5} {:>6} {:>5} {:>3} {:>9} {:>9} {:>9} {:>10}",
        "layer", "M", "N", "K", "sp", "µs", "GFLOP/s", "total ms", "max rel err"
    );
    let mut f16_total_ms = 0.0;
    for (m, n, k, count, name) in gemm_shapes {
        // Values in [-1, 1), as activations and weights are after training.
        let host_a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 37 % 1000) as f32 / 500.0) - 1.0)
            .collect();
        let host_b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 53 % 1000) as f32 / 500.0) - 1.0)
            .collect();
        let a = DeviceBuffer::from_host(&stream, &host_a)
            .map_err(|e| anyhow::anyhow!("alloc a: {:?}", e))?;
        let b = DeviceBuffer::from_host(&stream, &host_b)
            .map_err(|e| anyhow::anyhow!("alloc b: {:?}", e))?;
        let mut c_f16 = DeviceBuffer::<f32>::zeroed(&stream, m * n)
            .map_err(|e| anyhow::anyhow!("alloc c: {:?}", e))?;
        let mut c_f32 = DeviceBuffer::<f32>::zeroed(&stream, m * n)
            .map_err(|e| anyhow::anyhow!("alloc c: {:?}", e))?;
        let bias = DeviceBuffer::<f32>::zeroed(&stream, m)
            .map_err(|e| anyhow::anyhow!("alloc bias: {:?}", e))?;

        // Same split policy as the f32 path: without it the small-N shapes
        // starve the GPU exactly as they did there.
        let splits = OnnxExecutor::split_factor(m, n, k);
        let k_per_split = k.div_ceil(splits).next_multiple_of(16).max(16);
        let mut partials = DeviceBuffer::<f32>::zeroed(&stream, splits * m * n)
            .map_err(|e| anyhow::anyhow!("alloc partials: {:?}", e))?;
        let tc_cfg = LaunchConfig {
            grid_dim: (
                (n as u32).div_ceil(64).max(1),
                (m as u32).div_ceil(64).max(1),
                splits as u32,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let red_cfg2 = LaunchConfig::for_num_elems((m * n) as u32);
        // Weights pre-packed once, as the executor would do at load time.
        let kpairs = k.div_ceil(2);
        let mut a_packed = DeviceBuffer::<u32>::zeroed(&stream, m * kpairs)
            .map_err(|e| anyhow::anyhow!("alloc packed: {:?}", e))?;
        unsafe {
            module.pack_f16_rows(
                &stream,
                LaunchConfig::for_num_elems((m * kpairs) as u32),
                &a,
                k as u32,
                kpairs as u32,
                &mut a_packed,
            )
        }
        .map_err(|e| anyhow::anyhow!("pack: {:?}", e))?;

        let secs = time_kernel(&stream, 50, || {
            unsafe {
                module.sgemm_f16_tc_splitk_wpacked(
                    &stream,
                    tc_cfg,
                    m as u32,
                    n as u32,
                    k as u32,
                    k_per_split as u32,
                    &a_packed,
                    kpairs as u32,
                    &b,
                    &mut partials,
                )
            }
            .map_err(|e| anyhow::anyhow!("f16 tc: {:?}", e))?;
            unsafe {
                module.reduce_splits(
                    &stream,
                    red_cfg2,
                    &partials,
                    splits as u32,
                    (m * n) as u32,
                    n as u32,
                    1.0,
                    &bias,
                    0,
                    &bias,
                    0,
                    0,
                    0.0,
                    0.0,
                    &mut c_f16,
                )
            }
            .map_err(|e| anyhow::anyhow!("reduce: {:?}", e))
        })?;

        // f32 reference through the existing tiled kernel.
        let ref_cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            module.sgemm_tiled(
                &stream, ref_cfg, m as u32, n as u32, k as u32, 1.0, &a, &b, 0.0, &mut c_f32,
            )
        }
        .map_err(|e| anyhow::anyhow!("ref: {:?}", e))?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync: {:?}", e))?;

        let got = c_f16
            .to_host_vec(&stream)
            .map_err(|e| anyhow::anyhow!("d2h: {:?}", e))?;
        let want = c_f32
            .to_host_vec(&stream)
            .map_err(|e| anyhow::anyhow!("d2h: {:?}", e))?;
        let scale = want
            .iter()
            .fold(0.0f32, |acc, v| acc.max(v.abs()))
            .max(1e-6);
        let max_rel = got
            .iter()
            .zip(want.iter())
            .map(|(g, w)| (g - w).abs() / scale)
            .fold(0.0f32, f32::max);

        let gflops = (2.0 * m as f64 * n as f64 * k as f64) / secs / 1e9;
        let total_ms = secs * count as f64 * 1e3;
        f16_total_ms += total_ms;
        println!(
            "    {:>11} {:>5} {:>6} {:>5} {:>3} {:>9.1} {:>9.0} {:>9.2} {:>10.2e}",
            name,
            m,
            n,
            k,
            splits,
            secs * 1e6,
            gflops,
            total_ms,
            max_rel
        );
    }
    println!("    {:>50} {:>9.2}", "f16 TC total:", f16_total_ms);
    println!(
        "    {:>50} {:>9.2}x",
        "vs split-K f32:",
        splitk_total_ms / f16_total_ms
    );
    println!();

    // Would Winograd pay off here? F(4x4,3x3) cuts multiplies 3.4x on these
    // layers, but turns each convolution into 36 batched GEMMs whose N is the
    // tile count — 196 down to 4 at batch 1. FLOPs saved only become time
    // saved if those shapes run at a decent fraction of peak, so measure them
    // before writing any transform kernels.
    //
    // Two bounds per layer: 36 separate small GEMMs (no batching), and one
    // GEMM of the same total FLOPs with N widened 36x (perfect batching, but
    // ignoring that each position has its own filter matrix). Real batched
    // Winograd sits between them.
    println!("  Winograd F(4×4,3×3) feasibility — GEMM shapes it would produce");
    println!(
        "    {:>6} {:>16} {:>4} {:>11} {:>11} {:>11}",
        "layer", "(M,N,K) per pos", "pos", "unbatched", "batched", "direct now"
    );
    for (c, tiles, positions, count, name, direct_us) in [
        (64usize, 196usize, 36usize, 3usize, "s1 3×3", 50.1f64),
        (128, 49, 36, 4, "s2 3×3", 52.4),
        (256, 16, 36, 6, "s3 3×3", 60.5),
        (512, 4, 36, 3, "s4 3×3", 71.2),
    ] {
        let bench_gemm = |m: usize, n: usize, k: usize| -> Result<f64> {
            let a = DeviceBuffer::<f32>::zeroed(&stream, m * k)
                .map_err(|e| anyhow::anyhow!("alloc: {:?}", e))?;
            let b = DeviceBuffer::<f32>::zeroed(&stream, k * n)
                .map_err(|e| anyhow::anyhow!("alloc: {:?}", e))?;
            let mut cbuf = DeviceBuffer::<f32>::zeroed(&stream, m * n)
                .map_err(|e| anyhow::anyhow!("alloc: {:?}", e))?;
            let bias = DeviceBuffer::<f32>::zeroed(&stream, m)
                .map_err(|e| anyhow::anyhow!("alloc: {:?}", e))?;
            let splits = OnnxExecutor::split_factor(m, n, k);
            let cfg = LaunchConfig {
                grid_dim: (
                    (n as u32).div_ceil(64).max(1),
                    (m as u32).div_ceil(64).max(1),
                    1,
                ),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let _ = splits;
            time_kernel(&stream, 50, || {
                unsafe {
                    module.sgemm_reg(
                        &stream, cfg, m as u32, n as u32, k as u32, 1.0, &a, &b, &bias, 0, 0, 0.0,
                        0.0, &mut cbuf,
                    )
                }
                .map_err(|e| anyhow::anyhow!("wino gemm: {:?}", e))
            })
        };

        let one = bench_gemm(c, tiles, c)?;
        let wide = bench_gemm(c, tiles * positions, c)?;
        println!(
            "    {:>6} {:>16} {:>4} {:>9.1}µs {:>9.1}µs {:>9.1}µs",
            name,
            format!("({c},{tiles},{c})"),
            positions,
            one * positions as f64 * 1e6,
            wide * 1e6,
            direct_us
        );
        let _ = count;
    }
    println!();

    // im2col for the 3×3 convolutions: the cost of materialising the column
    // matrix that the implicit-GEMM kernel would remove.
    println!("  im2col (3×3 layers)");
    let mut im2col_total_ms = 0.0;
    for (c_in, h, w, count, name) in [
        (64usize, 56usize, 56usize, 3usize, "s1"),
        (128, 28, 28, 4, "s2"),
        (256, 14, 14, 6, "s3"),
        (512, 7, 7, 3, "s4"),
    ] {
        let col_rows = c_in * 9;
        let col_cols = h * w;
        let x = DeviceBuffer::<f32>::zeroed(&stream, c_in * h * w)
            .map_err(|e| anyhow::anyhow!("alloc x: {:?}", e))?;
        let mut col = DeviceBuffer::<f32>::zeroed(&stream, col_rows * col_cols)
            .map_err(|e| anyhow::anyhow!("alloc col: {:?}", e))?;
        let cfg = LaunchConfig::for_num_elems((col_rows * col_cols) as u32);
        let secs = time_kernel(&stream, 50, || {
            unsafe {
                module.im2col(
                    &stream,
                    cfg,
                    &x,
                    c_in as u32,
                    h as u32,
                    w as u32,
                    3,
                    3,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    h as u32,
                    w as u32,
                    &mut col,
                )
            }
            .map_err(|e| anyhow::anyhow!("im2col: {:?}", e))
        })?;
        let total_ms = secs * count as f64 * 1e3;
        im2col_total_ms += total_ms;
        println!(
            "    {:>11} {:>9.1} µs  ×{}  = {:.2} ms",
            name,
            secs * 1e6,
            count,
            total_ms
        );
    }
    println!("    {:>50} {:>9.2}", "im2col total:", im2col_total_ms);
    println!();
    println!(
        "  accounted kernel time: {:.2} ms  (GEMM {:.2} + im2col {:.2} + BN {:.2})",
        gemm_total_ms + im2col_total_ms + bn_total_us / 1000.0,
        gemm_total_ms,
        im2col_total_ms,
        bn_total_us / 1000.0
    );
    println!("  (3090 FP32 peak ≈ 35 600 GFLOP/s)");
    println!();

    // Per-node host overhead. The executor allocates a fresh output buffer for
    // every node, so a 91-node graph pays this 91 times per inference; if it
    // is tens of microseconds it outweighs several of the kernels.
    println!("  host-side per-node costs");
    let alloc_secs = {
        const N: usize = 200;
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..N {
                let buf = unsafe { DeviceBuffer::<f32>::uninitialized_async(&stream, 802_816) }
                    .map_err(|e| anyhow::anyhow!("alloc: {:?}", e))?;
                drop(buf);
            }
            best = best.min(t0.elapsed().as_secs_f64() / N as f64);
        }
        best
    };
    println!(
        "    uninitialized_async + drop (3 MB): {:>8.1} µs   ×91 nodes = {:.2} ms",
        alloc_secs * 1e6,
        alloc_secs * 91.0 * 1e3
    );

    // The same allocation, but with the stream already loaded with work.
    // cuMemAllocAsync is stream-ordered: when the pool cannot satisfy a request
    // from free blocks it waits for the stream to progress far enough to reuse
    // memory, which turns an allocation into a host-side stall.
    let alloc_busy_secs = {
        const N: usize = 50;
        let m = 512usize;
        let n = 49usize;
        let k = 4608usize;
        let a = DeviceBuffer::<f32>::zeroed(&stream, m * k)
            .map_err(|e| anyhow::anyhow!("alloc a: {:?}", e))?;
        let b = DeviceBuffer::<f32>::zeroed(&stream, k * n)
            .map_err(|e| anyhow::anyhow!("alloc b: {:?}", e))?;
        let mut c = DeviceBuffer::<f32>::zeroed(&stream, m * n)
            .map_err(|e| anyhow::anyhow!("alloc c: {:?}", e))?;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let mut best = f64::INFINITY;
        for _ in 0..5 {
            // Queue roughly 5 ms of GPU work, then allocate while it runs.
            for _ in 0..20 {
                unsafe {
                    module.sgemm_tiled(
                        &stream, cfg, m as u32, n as u32, k as u32, 1.0, &a, &b, 0.0, &mut c,
                    )
                }
                .map_err(|e| anyhow::anyhow!("sgemm: {:?}", e))?;
            }
            let t0 = Instant::now();
            let mut bufs = Vec::with_capacity(N);
            for _ in 0..N {
                bufs.push(
                    unsafe { DeviceBuffer::<f32>::uninitialized_async(&stream, 802_816) }
                        .map_err(|e| anyhow::anyhow!("alloc: {:?}", e))?,
                );
            }
            best = best.min(t0.elapsed().as_secs_f64() / N as f64);
            drop(bufs);
            stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("sync: {:?}", e))?;
        }
        best
    };
    println!(
        "    same, while the stream is busy:    {:>8.1} µs   ×18 BN nodes = {:.2} ms",
        alloc_busy_secs * 1e6,
        alloc_busy_secs * 18.0 * 1e3
    );

    let launch_secs = {
        let mut dummy = DeviceBuffer::<f32>::zeroed(&stream, 1024)
            .map_err(|e| anyhow::anyhow!("alloc: {:?}", e))?;
        let cfg = LaunchConfig::for_num_elems(1024);
        time_kernel(&stream, 200, || {
            unsafe { module.relu(&stream, cfg, &mut dummy) }
                .map_err(|e| anyhow::anyhow!("relu: {:?}", e))
        })?
    };
    println!(
        "    empty-ish kernel launch:           {:>8.1} µs   ×91 nodes = {:.2} ms",
        launch_secs * 1e6,
        launch_secs * 91.0 * 1e3
    );
    Ok(())
}

fn unit_tests() -> Result<()> {
    let ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CUDA context: {:?}", e))?;
    let stream = ctx.default_stream();
    let module = gpu::load(&ctx).map_err(|e| anyhow::anyhow!("load module: {:?}", e))?;

    let mut all_pass = true;
    all_pass &= test_relu(&stream, &module)?;
    all_pass &= test_clip(&stream, &module)?;
    all_pass &= test_add(&stream, &module)?;
    all_pass &= test_sgemm(&stream, &module)?;
    all_pass &= test_bias_add(&stream, &module)?;
    all_pass &= test_batchnorm(&stream, &module)?;
    all_pass &= test_conv2d(&stream, &module)?;
    all_pass &= test_maxpool(&stream, &module)?;
    all_pass &= test_global_avg_pool(&stream, &module)?;
    all_pass &= test_softmax(&stream, &module)?;

    println!();
    if all_pass {
        println!("✓ All unit tests PASSED");
    } else {
        println!("✗ Some unit tests FAILED");
        std::process::exit(1);
    }
    Ok(())
}

fn test_relu(stream: &cuda_core::CudaStream, module: &gpu::LoadedModule) -> Result<bool> {
    let n = 1024usize;
    let x: Vec<f32> = (0..n).map(|i| i as f32 - 512.0).collect();
    let expected = cpu_ref::relu(&x);
    let mut dev = DeviceBuffer::from_host(stream, &x).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    unsafe { module.relu(stream, LaunchConfig::for_num_elems(n as u32), &mut dev) }
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let got = dev
        .to_host_vec(stream)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let err = cpu_ref::max_abs_diff(&got, &expected);
    let pass = err < 1e-6;
    println!(
        "  relu         max_err={:.2e}  {}",
        err,
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

fn test_clip(stream: &cuda_core::CudaStream, module: &gpu::LoadedModule) -> Result<bool> {
    let n = 512usize;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 256.0) * 0.5).collect();
    let (lo, hi) = (-10.0f32, 10.0f32);
    let expected = cpu_ref::clip(&x, lo, hi);
    let mut dev = DeviceBuffer::from_host(stream, &x).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    unsafe {
        module.clip(
            stream,
            LaunchConfig::for_num_elems(n as u32),
            &mut dev,
            lo,
            hi,
        )
    }
    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let got = dev
        .to_host_vec(stream)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let err = cpu_ref::max_abs_diff(&got, &expected);
    let pass = err < 1e-6;
    println!(
        "  clip         max_err={:.2e}  {}",
        err,
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

fn test_add(stream: &cuda_core::CudaStream, module: &gpu::LoadedModule) -> Result<bool> {
    let n = 1024usize;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n).map(|i| -(i as f32 * 0.05)).collect();
    let expected = cpu_ref::add(&a, &b);
    let a_dev = DeviceBuffer::from_host(stream, &a).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let b_dev = DeviceBuffer::from_host(stream, &b).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let mut c_dev =
        DeviceBuffer::<f32>::zeroed(stream, n).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    unsafe {
        module.add_elementwise(
            stream,
            LaunchConfig::for_num_elems(n as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
    }
    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let got = c_dev
        .to_host_vec(stream)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let err = cpu_ref::max_abs_diff(&got, &expected);
    let pass = err < 1e-6;
    println!(
        "  add          max_err={:.2e}  {}",
        err,
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

fn test_sgemm(stream: &cuda_core::CudaStream, module: &gpu::LoadedModule) -> Result<bool> {
    // Non-tile-aligned dims exercise the 16×16 tiled kernel's boundary guards.
    let (m, n, k) = (100usize, 70usize, 130usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.1 - 0.2).collect();
    let c_init = vec![0.0f32; m * n];
    let expected = cpu_ref::sgemm(m, n, k, 1.0, &a, &b, 0.0, &c_init);
    let a_dev = DeviceBuffer::from_host(stream, &a).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let b_dev = DeviceBuffer::from_host(stream, &b).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let mut c_dev =
        DeviceBuffer::<f32>::zeroed(stream, m * n).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let block = 16u32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(block), (m as u32).div_ceil(block), 1),
        block_dim: (block, block, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        module.sgemm_tiled(
            stream, cfg, m as u32, n as u32, k as u32, 1.0, &a_dev, &b_dev, 0.0, &mut c_dev,
        )
    }
    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let got = c_dev
        .to_host_vec(stream)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let err = cpu_ref::max_abs_diff(&got, &expected);
    let pass = err < 1e-2;
    println!(
        "  sgemm tiled  max_err={:.2e}  {}",
        err,
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

fn test_bias_add(stream: &cuda_core::CudaStream, module: &gpu::LoadedModule) -> Result<bool> {
    let (batch, feat) = (4usize, 16usize);
    let x: Vec<f32> = (0..batch * feat).map(|i| i as f32 * 0.1).collect();
    let bias: Vec<f32> = (0..feat).map(|i| i as f32 * 0.5).collect();
    let expected = cpu_ref::bias_add(&x, &bias, feat);
    let mut x_dev = DeviceBuffer::from_host(stream, &x).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let bias_dev =
        DeviceBuffer::from_host(stream, &bias).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    unsafe {
        module.bias_add(
            stream,
            LaunchConfig::for_num_elems((batch * feat) as u32),
            &mut x_dev,
            &bias_dev,
            1u32,
            feat as u32,
        )
    } // spatial=1 for [batch, feat] layout
    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let got = x_dev
        .to_host_vec(stream)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let err = cpu_ref::max_abs_diff(&got, &expected);
    let pass = err < 1e-5;
    println!(
        "  bias_add     max_err={:.2e}  {}",
        err,
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

fn test_batchnorm(stream: &cuda_core::CudaStream, module: &gpu::LoadedModule) -> Result<bool> {
    let (n, c, h, w) = (1usize, 8usize, 4usize, 4usize);
    let hw = h * w;
    let numel = n * c * hw;
    let x: Vec<f32> = (0..numel).map(|i| i as f32 * 0.1 - 1.6).collect();
    let gamma: Vec<f32> = (0..c).map(|i| 1.0 + i as f32 * 0.1).collect();
    let beta: Vec<f32> = (0..c).map(|i| i as f32 * 0.05).collect();
    let mean: Vec<f32> = (0..c).map(|i| i as f32 * 0.2).collect();
    let var: Vec<f32> = (0..c).map(|_| 1.0f32).collect();
    let eps = 1e-5f32;
    let expected = cpu_ref::batch_norm_inference(&x, &gamma, &beta, &mean, &var, eps, c, hw);
    let x_dev = DeviceBuffer::from_host(stream, &x).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let g_dev = DeviceBuffer::from_host(stream, &gamma).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let b_dev = DeviceBuffer::from_host(stream, &beta).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let m_dev = DeviceBuffer::from_host(stream, &mean).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let v_dev = DeviceBuffer::from_host(stream, &var).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let mut out_dev =
        DeviceBuffer::<f32>::zeroed(stream, numel).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    unsafe {
        module.batch_norm_inference(
            stream,
            LaunchConfig::for_num_elems(numel as u32),
            &x_dev,
            &g_dev,
            &b_dev,
            &m_dev,
            &v_dev,
            eps,
            n as u32,
            c as u32,
            hw as u32,
            &mut out_dev,
        )
    }
    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let got = out_dev
        .to_host_vec(stream)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let err = cpu_ref::max_abs_diff(&got, &expected);
    let pass = err < 1e-4;
    println!(
        "  batch_norm   max_err={:.2e}  {}",
        err,
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

fn test_conv2d(stream: &cuda_core::CudaStream, module: &gpu::LoadedModule) -> Result<bool> {
    let (bn, c_in, h_in, w_in) = (1usize, 3usize, 8usize, 8usize);
    let (n_out, kh, kw) = (4usize, 3usize, 3usize);
    let (pad_h, pad_w, stride_h, stride_w) = (0usize, 0usize, 1usize, 1usize);
    let x: Vec<f32> = (0..bn * c_in * h_in * w_in)
        .map(|i| i as f32 * 0.01)
        .collect();
    let w: Vec<f32> = (0..n_out * c_in * kh * kw)
        .map(|i| (i % 5) as f32 * 0.1 - 0.2)
        .collect();
    let (expected, out_h, out_w) = cpu_ref::conv2d(
        &x, &w, None, bn, c_in, h_in, w_in, n_out, kh, kw, pad_h, pad_w, stride_h, stride_w,
    );
    let col_rows = c_in * kh * kw;
    let col_cols = out_h * out_w;
    let x_dev = DeviceBuffer::from_host(stream, &x).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let mut col_dev = DeviceBuffer::<f32>::zeroed(stream, col_rows * col_cols)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    unsafe {
        module.im2col(
            stream,
            LaunchConfig::for_num_elems((col_rows * col_cols) as u32),
            &x_dev,
            c_in as u32,
            h_in as u32,
            w_in as u32,
            kh as u32,
            kw as u32,
            pad_h as u32,
            pad_w as u32,
            stride_h as u32,
            stride_w as u32,
            1u32,
            1u32,
            out_h as u32,
            out_w as u32,
            &mut col_dev,
        )
    }
    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let w_dev = DeviceBuffer::from_host(stream, &w).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(stream, n_out * out_h * out_w)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let block = 16u32;
    let cfg = LaunchConfig {
        grid_dim: (
            (col_cols as u32).div_ceil(block),
            (n_out as u32).div_ceil(block),
            1,
        ),
        block_dim: (block, block, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        module.sgemm_naive(
            stream,
            cfg,
            n_out as u32,
            col_cols as u32,
            col_rows as u32,
            1.0,
            &w_dev,
            &col_dev,
            0.0,
            &mut out_dev,
        )
    }
    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let got = out_dev
        .to_host_vec(stream)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let err = cpu_ref::max_abs_diff(&got, &expected);
    let pass = err < 1e-3;
    println!(
        "  conv2d 3×3   max_err={:.2e}  {}",
        err,
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

fn test_maxpool(stream: &cuda_core::CudaStream, module: &gpu::LoadedModule) -> Result<bool> {
    let (n, c, in_h, in_w) = (1usize, 4usize, 8usize, 8usize);
    let (kh, kw, pad_h, pad_w, stride_h, stride_w) =
        (2usize, 2usize, 0usize, 0usize, 2usize, 2usize);
    let x: Vec<f32> = (0..n * c * in_h * in_w)
        .map(|i| (i % 13) as f32 - 5.0)
        .collect();
    let (expected, out_h, out_w) = cpu_ref::maxpool2d(
        &x, n, c, in_h, in_w, kh, kw, pad_h, pad_w, stride_h, stride_w,
    );
    let x_dev = DeviceBuffer::from_host(stream, &x).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(stream, n * c * out_h * out_w)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    unsafe {
        module.maxpool2d(
            stream,
            LaunchConfig::for_num_elems((n * c * out_h * out_w) as u32),
            &x_dev,
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
            &mut out_dev,
        )
    }
    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let got = out_dev
        .to_host_vec(stream)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let err = cpu_ref::max_abs_diff(&got, &expected);
    let pass = err < 1e-6;
    println!(
        "  maxpool2d    max_err={:.2e}  {}",
        err,
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

fn test_global_avg_pool(
    stream: &cuda_core::CudaStream,
    module: &gpu::LoadedModule,
) -> Result<bool> {
    let (n, c, h, w) = (2usize, 8usize, 7usize, 7usize);
    let hw = h * w;
    let x: Vec<f32> = (0..n * c * hw).map(|i| i as f32 * 0.01).collect();
    let expected = cpu_ref::global_avg_pool(&x, n, c, hw);
    let x_dev = DeviceBuffer::from_host(stream, &x).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let mut out_dev =
        DeviceBuffer::<f32>::zeroed(stream, n * c).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    unsafe {
        module.global_avg_pool(
            stream,
            LaunchConfig::for_num_elems((n * c) as u32),
            &x_dev,
            c as u32,
            hw as u32,
            &mut out_dev,
        )
    }
    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let got = out_dev
        .to_host_vec(stream)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let err = cpu_ref::max_abs_diff(&got, &expected);
    let pass = err < 1e-5;
    println!(
        "  global_avg   max_err={:.2e}  {}",
        err,
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

fn test_softmax(stream: &cuda_core::CudaStream, module: &gpu::LoadedModule) -> Result<bool> {
    let (rows, cols) = (4usize, 1000usize);
    let x: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 17) as f32 * 0.1 - 0.8)
        .collect();
    let expected = cpu_ref::softmax(&x, rows, cols);
    let x_dev = DeviceBuffer::from_host(stream, &x).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let mut out_dev =
        DeviceBuffer::<f32>::zeroed(stream, rows * cols).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    unsafe {
        module.softmax_row(
            stream,
            LaunchConfig::for_num_elems(rows as u32),
            &x_dev,
            rows as u32,
            cols as u32,
            &mut out_dev,
        )
    }
    .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let got = out_dev
        .to_host_vec(stream)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let err = cpu_ref::max_abs_diff(&got, &expected);
    let pass = err < 1e-5;
    println!(
        "  softmax      max_err={:.2e}  {}",
        err,
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

// ===========================================================================
// Section 2: End-to-end model inference
// ===========================================================================

fn run_model(model_path: &str, model_name: &str) -> Result<()> {
    print!("  Loading {}... ", model_name);
    let model = load_model(model_path)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Model has no graph"))?;
    let ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CUDA: {:?}", e))?;
    let executor = OnnxExecutor::from_graph(graph, ctx)?;
    println!(
        "{} nodes, {} inputs, {} outputs",
        executor.nodes.len(),
        executor.input_names.len(),
        executor.output_names.len()
    );

    let input_shape = vec![1usize, 3, 224, 224];
    let input_numel: usize = input_shape.iter().product();
    // Deterministic pseudo-random input in [0, 1)
    let input_data: Vec<f32> = (0..input_numel)
        .map(|i| ((i * 6271 + 1337) % 1000) as f32 / 1000.0)
        .collect();

    let input_name = executor
        .input_names
        .first()
        .ok_or_else(|| anyhow::anyhow!("No input names"))?
        .clone();
    let mut inputs = HashMap::new();
    inputs.insert(input_name, (input_data.clone(), input_shape));

    // --- oxide GPU inference ---
    // When profiling, discard one run first: the first inference pays one-off
    // costs (memory-pool growth, PTX module warm-up) that otherwise land on
    // whichever node happens to allocate first and swamp its measurement.
    if std::env::var("OXIDE_PROFILE").is_ok() {
        let _ = executor.run(&inputs)?;
        eprintln!("  (profile: steady state, after one warm-up inference)");
    }
    let t0 = Instant::now();
    let outputs = executor.run(&inputs)?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut oxide_logits: Option<Vec<f32>> = None;
    for (out_name, (logits, shape)) in &outputs {
        let top5 = cpu_ref::top_k(logits, 5);
        let top5_vals: Vec<f32> = top5.iter().map(|&i| logits[i]).collect();
        println!("  [oxide] Output '{}' shape={:?}", out_name, shape);
        println!("  [oxide] Top-5 indices: {:?}", top5);
        println!(
            "  [oxide] Top-5 scores:  {:?}",
            top5_vals
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
        );
        oxide_logits = Some(logits.clone());
    }
    println!("  [oxide] Inference time: {:.1} ms", elapsed_ms);

    // --- tract CPU reference ---
    print!("  [tract] Running reference inference... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    match run_tract_inference(model_path, &input_data) {
        Ok(tract_logits) => {
            println!("done");
            let top5_tract = cpu_ref::top_k(&tract_logits, 5);
            let top5_tract_vals: Vec<f32> = top5_tract.iter().map(|&i| tract_logits[i]).collect();
            println!("  [tract] Top-5 indices: {:?}", top5_tract);
            println!(
                "  [tract] Top-5 scores:  {:?}",
                top5_tract_vals
                    .iter()
                    .map(|v| format!("{:.4}", v))
                    .collect::<Vec<_>>()
            );

            if let Some(ref ol) = oxide_logits {
                let top1_oxide = cpu_ref::top_k(ol, 1)[0];
                let top1_tract = top5_tract[0];
                let oxide_in_tract5 = top5_tract.contains(&top1_oxide);
                let max_err: f32 = ol
                    .iter()
                    .zip(tract_logits.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!("  ┌─ vs tract (CPU) ───────────────────────────────");
                println!(
                    "  │ top-1  oxide={:<4} tract={:<4} exact_match={}",
                    top1_oxide,
                    top1_tract,
                    top1_oxide == top1_tract
                );
                println!("  │ oxide top-1 in tract top-5: {}", oxide_in_tract5);
                println!("  │ max |oxide − tract|: {:.4e}", max_err);
                println!("  └────────────────────────────────────────────────");
            }
        }
        Err(e) => println!("skipped ({})", e),
    }

    // --- ORT CUDA correctness comparison ---
    print!("  [ort]    Running ORT CUDA inference... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    match run_ort_cuda_inference(model_path, &input_data) {
        Ok(ort_logits) => {
            println!("done");
            let top5_ort = cpu_ref::top_k(&ort_logits, 5);
            let top5_ort_vals: Vec<f32> = top5_ort.iter().map(|&i| ort_logits[i]).collect();
            println!("  [ort]    Top-5 indices: {:?}", top5_ort);
            println!(
                "  [ort]    Top-5 scores:  {:?}",
                top5_ort_vals
                    .iter()
                    .map(|v| format!("{:.4}", v))
                    .collect::<Vec<_>>()
            );

            if let Some(ref ol) = oxide_logits {
                let top1_oxide = cpu_ref::top_k(ol, 1)[0];
                let top1_ort = top5_ort[0];
                let oxide_in_ort5 = top5_ort.contains(&top1_oxide);
                let max_err: f32 = ol
                    .iter()
                    .zip(ort_logits.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!("  ┌─ vs ORT CUDA ────────────────────────────────────");
                println!(
                    "  │ top-1  oxide={:<4} ort={:<4}    exact_match={}",
                    top1_oxide,
                    top1_ort,
                    top1_oxide == top1_ort
                );
                println!("  │ oxide top-1 in ORT top-5: {}", oxide_in_ort5);
                println!("  │ max |oxide − ort|: {:.4e}", max_err);
                println!("  └────────────────────────────────────────────────");
            }
        }
        Err(e) => println!("skipped ({})", e),
    }

    Ok(())
}

// ===========================================================================
// Generic model runner — architecture-agnostic correctness and timing
// ===========================================================================

/// Run any single-input, single-output model and check it numerically against
/// ONNX Runtime.
///
/// The classification runner compares top-1 indices, which only means anything
/// for a classifier. Super-resolution, style transfer, segmentation and
/// detection all produce dense tensors where the right question is how far the
/// values are from the reference, so this reports absolute and relative error
/// over the whole output instead. That keeps the engine honest on any
/// architecture rather than on the two it started with.
fn run_generic_model(model_path: &str, model_name: &str, input_shape: &[usize]) -> Result<()> {
    print!("  Loading {}... ", model_name);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let model = load_model(model_path)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Model has no graph"))?;
    let ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CUDA: {:?}", e))?;
    let executor = OnnxExecutor::from_graph(graph, ctx)?;
    println!(
        "{} nodes, {} inputs, {} outputs",
        executor.nodes.len(),
        executor.input_names.len(),
        executor.output_names.len()
    );

    let numel: usize = input_shape.iter().product();
    let input_data: Vec<f32> = (0..numel)
        .map(|i| ((i * 6271 + 1337) % 1000) as f32 / 1000.0)
        .collect();
    let input_name = executor
        .input_names
        .first()
        .ok_or_else(|| anyhow::anyhow!("No input names"))?
        .clone();
    let mut inputs = HashMap::new();
    inputs.insert(input_name, (input_data.clone(), input_shape.to_vec()));

    let t0 = Instant::now();
    let outputs = executor.run(&inputs)?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let first_out = executor
        .output_names
        .first()
        .ok_or_else(|| anyhow::anyhow!("No output names"))?;
    let (oxide_out, oshape) = outputs
        .get(first_out)
        .ok_or_else(|| anyhow::anyhow!("Output '{}' missing from run", first_out))?;
    println!(
        "  [oxide] Output '{}' shape={:?}  ({:.1} ms)",
        first_out, oshape, elapsed_ms
    );

    print!("  [ort]   Running ORT CUDA reference... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    match run_ort_cuda_inference_shaped(model_path, &input_data, input_shape) {
        Ok(reference) => {
            println!("done");
            let n = oxide_out.len().min(reference.len());
            if n == 0 {
                println!("  │ empty output, nothing to compare");
                return Ok(());
            }
            let mut max_abs = 0.0f32;
            let mut sum_abs = 0.0f64;
            let mut ref_mag = 0.0f64;
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for i in 0..n {
                let (a, b) = (oxide_out[i], reference[i]);
                let d = (a - b).abs();
                if d > max_abs {
                    max_abs = d;
                }
                sum_abs += d as f64;
                ref_mag += b.abs() as f64;
                dot += (a as f64) * (b as f64);
                na += (a as f64) * (a as f64);
                nb += (b as f64) * (b as f64);
            }
            let mean_abs = sum_abs / n as f64;
            let rel = if ref_mag > 0.0 {
                sum_abs / ref_mag
            } else {
                0.0
            };
            let cos = if na > 0.0 && nb > 0.0 {
                dot / (na.sqrt() * nb.sqrt())
            } else {
                0.0
            };
            // f16 tensor cores carry about three decimal digits, so agreement
            // is judged on relative error and direction rather than on bits.
            let ok = rel < 2e-2 && cos > 0.999;
            println!("  ┌─ vs ORT CUDA ────────────────────────────────────");
            println!("  │ elements compared : {}", n);
            println!("  │ max abs diff      : {:.5}", max_abs);
            println!("  │ mean abs diff     : {:.6}", mean_abs);
            println!("  │ relative L1 error : {:.6}", rel);
            println!("  │ cosine similarity : {:.6}", cos);
            println!("  │ MATCH             : {}", ok);
            println!("  └──────────────────────────────────────────────────");
            if !ok {
                println!("  !! {} disagrees with ORT beyond tolerance", model_name);
            }
        }
        Err(e) => println!("skipped ({})", e),
    }

    Ok(())
}

// ===========================================================================
// ORT CUDA single-shot inference (correctness check)
// ===========================================================================

/// Run one forward pass via the Python bench_gpu.py in correctness mode.
/// Returns the output logits by writing them to a temp file.
fn run_ort_cuda_inference(model_path: &str, input_data: &[f32]) -> Result<Vec<f32>> {
    run_ort_cuda_inference_shaped(model_path, input_data, &[1, 3, 224, 224])
}

/// As [`run_ort_cuda_inference`], for models whose input is not a 224x224 RGB
/// image — grayscale super-resolution, 416x416 detection, and so on.
fn run_ort_cuda_inference_shaped(
    model_path: &str,
    input_data: &[f32],
    shape: &[usize],
) -> Result<Vec<f32>> {
    let script = std::path::Path::new(model_path)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("./scripts/infer_ort.py"))
        .unwrap_or_else(|| std::path::PathBuf::from("./scripts/infer_ort.py"));

    if !script.exists() {
        return Err(anyhow::anyhow!("infer_ort.py not found at {:?}", script));
    }

    // Write input as raw f32 LE bytes to a temp file
    let tmp_in = std::env::temp_dir().join("oxide_ort_input.bin");
    let tmp_out = std::env::temp_dir().join("oxide_ort_output.bin");
    {
        use std::io::Write;
        let bytes: Vec<u8> = input_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
        std::fs::File::create(&tmp_in)?.write_all(&bytes)?;
    }

    let shape_arg = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let status = std::process::Command::new("python3")
        .arg(&script)
        .arg(model_path)
        .arg(&tmp_in)
        .arg(&tmp_out)
        .arg(&shape_arg)
        .status()
        .map_err(|e| anyhow::anyhow!("python3: {}", e))?;

    if !status.success() {
        return Err(anyhow::anyhow!("infer_ort.py exited with {}", status));
    }

    let bytes = std::fs::read(&tmp_out)?;
    let logits: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Ok(logits)
}

// ===========================================================================
// Tract CPU reference inference
// ===========================================================================

fn run_tract_inference(model_path: &str, input_data: &[f32]) -> Result<Vec<f32>> {
    use tract_onnx::prelude::*;

    let model = tract_onnx::onnx()
        .model_for_path(model_path)
        .map_err(|e| anyhow::anyhow!("tract load: {}", e))?
        .into_optimized()
        .map_err(|e| anyhow::anyhow!("tract optimize: {}", e))?
        .into_runnable()
        .map_err(|e| anyhow::anyhow!("tract runnable: {}", e))?;

    let input_arr =
        tract_ndarray::Array4::<f32>::from_shape_vec((1, 3, 224, 224), input_data.to_vec())
            .map_err(|e| anyhow::anyhow!("tract input array: {}", e))?;
    let input_tensor: Tensor = input_arr.into();

    let result = model
        .run(tvec![input_tensor.into()])
        .map_err(|e| anyhow::anyhow!("tract run: {}", e))?;

    let output = result[0]
        .to_array_view::<f32>()
        .map_err(|e| anyhow::anyhow!("tract output: {}", e))?;

    Ok(output.iter().copied().collect())
}

// ===========================================================================
// BERT — two integer/float inputs, output diff vs tract (not classification)
// ===========================================================================

fn run_bert(model_path: &str, model_name: &str) -> Result<()> {
    print!("  Loading {}... ", model_name);
    let model = load_model(model_path)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Model has no graph"))?;
    let ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CUDA: {:?}", e))?;
    let executor = OnnxExecutor::from_graph(graph, ctx)?;
    println!(
        "{} nodes, {} inputs, {} outputs",
        executor.nodes.len(),
        executor.input_names.len(),
        executor.output_names.len()
    );

    let seq = 128usize;
    // Deterministic token ids; mask: first 100 real tokens, last 28 padding.
    let ids: Vec<i64> = (0..seq).map(|i| ((i * 7919 + 13) % 30522) as i64).collect();
    let mask: Vec<f32> = (0..seq).map(|i| if i < 100 { 1.0 } else { 0.0 }).collect();
    let ids_f: Vec<f32> = ids.iter().map(|&v| v as f32).collect();

    let mut inputs = HashMap::new();
    inputs.insert("input_ids".to_string(), (ids_f, vec![1, seq]));
    inputs.insert("attention_mask".to_string(), (mask.clone(), vec![1, seq]));

    let t0 = Instant::now();
    let outputs = executor.run(&inputs)?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    for name in &executor.output_names {
        if let Some((v, shape)) = outputs.get(name) {
            let sum: f64 = v.iter().map(|&x| x.abs() as f64).sum();
            println!("  [oxide] '{}' shape={:?}  |Σ|={:.4}", name, shape, sum);
        }
    }
    println!("  [oxide] Inference time: {:.1} ms", elapsed_ms);

    // ORT is the authoritative reference here: this BERT graph is already
    // ORT-graph-optimized, and tract.into_optimized() re-optimizes it into a
    // numerically divergent form (verified), so tract is NOT a valid oracle.
    // ORT is the authoritative reference: this BERT graph is already
    // ORT-graph-optimized; tract.into_optimized() diverges, so it is not used.
    print!("  [ort]    Running ORT reference + benchmark... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    match run_ort_script("infer_bert_ort.py", model_path, seq) {
        Ok((ort_blob, ort_ms)) => {
            println!("done");
            let mut off = 0usize;
            for name in &executor.output_names {
                if let Some((ov, shape)) = outputs.get(name) {
                    let n = ov.len();
                    if off + n <= ort_blob.len() {
                        let tv = &ort_blob[off..off + n];
                        let max_err = ov
                            .iter()
                            .zip(tv.iter())
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0f32, f32::max);
                        let rel = max_err / tv.iter().map(|v| v.abs()).fold(1e-9, f32::max);
                        println!("  ┌─ '{}' {:?} vs ORT ───", name, shape);
                        println!("  │ max |oxide − ort|: {:.4e}  (rel {:.2e})", max_err, rel);
                        println!("  └────────────────────");
                        off += n;
                    }
                }
            }
            let ox = bench_fn("BERT oxide", 3, 10, || {
                executor.run(&inputs).unwrap();
            });
            print_speed_row(model_name, ox, ort_ms);
        }
        Err(e) => println!("skipped ({})", e),
    }
    Ok(())
}

/// Run a model via a named ORT helper script (deterministic inputs are
/// generated inside the script). Returns (output blob, ORT mean ms).
/// The script writes its f32 output to a temp file and prints `ort_ms=<x>`.
fn run_ort_script(
    script_name: &str,
    model_path: &str,
    seq: usize,
) -> Result<(Vec<f32>, Option<f64>)> {
    let script = std::path::Path::new(model_path)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("./scripts").join(script_name))
        .unwrap_or_else(|| std::path::PathBuf::from(format!("./scripts/{}", script_name)));
    if !script.exists() {
        return Err(anyhow::anyhow!("{} not found at {:?}", script_name, script));
    }
    let tmp_out = std::env::temp_dir().join("oxide_ort_ref.bin");
    let output = std::process::Command::new("python3")
        .arg(&script)
        .arg(model_path)
        .arg(&tmp_out)
        .arg(seq.to_string())
        .output()
        .map_err(|e| anyhow::anyhow!("python3: {}", e))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("{} failed: {}", script_name, err.trim()));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let ort_ms = stdout.lines().find_map(|l| {
        l.strip_prefix("ort_ms=")
            .and_then(|v| v.trim().parse::<f64>().ok())
    });
    let bytes = std::fs::read(&tmp_out)?;
    let blob = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Ok((blob, ort_ms))
}

// ===========================================================================
// GPT-2 LM head — causal decoder; compare logits vs ORT
// ===========================================================================

fn run_gpt2(model_path: &str, model_name: &str) -> Result<()> {
    print!("  Loading {}... ", model_name);
    let model = load_model(model_path)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Model has no graph"))?;
    let ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CUDA: {:?}", e))?;
    let executor = OnnxExecutor::from_graph(graph, ctx)?;
    println!(
        "{} nodes, {} inputs, {} outputs",
        executor.nodes.len(),
        executor.input_names.len(),
        executor.output_names.len()
    );

    let seq = 128usize;
    let ids: Vec<f32> = (0..seq).map(|i| ((i * 7919 + 13) % 50257) as f32).collect();
    let mask: Vec<f32> = vec![1.0; seq];

    let mut inputs = HashMap::new();
    inputs.insert("input_ids".to_string(), (ids, vec![1, seq]));
    inputs.insert("attention_mask".to_string(), (mask, vec![1, seq]));

    let t0 = Instant::now();
    let outputs = executor.run(&inputs)?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Logits are the first graph output ([1, seq, vocab]); ignore KV cache.
    let logits_name = executor
        .output_names
        .first()
        .ok_or_else(|| anyhow::anyhow!("no GPT-2 outputs"))?;
    let (logits, lshape) = outputs
        .get(logits_name)
        .ok_or_else(|| anyhow::anyhow!("logits '{}' missing", logits_name))?;
    let vocab = *lshape.last().unwrap();
    let last = &logits[(seq - 1) * vocab..seq * vocab];
    let next_tok = last
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    println!(
        "  [oxide] logits '{}' shape={:?}  argmax(last)={}",
        logits_name, lshape, next_tok
    );
    println!("  [oxide] Inference time: {:.1} ms", elapsed_ms);

    print!("  [ort]    Running ORT reference + benchmark... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    match run_ort_script("infer_gpt2_ort.py", model_path, seq) {
        Ok((ort_logits, ort_ms)) => {
            println!("done");
            let n = logits.len().min(ort_logits.len());
            let max_err = logits[..n]
                .iter()
                .zip(ort_logits[..n].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let denom = ort_logits[..n].iter().map(|v| v.abs()).fold(1e-9, f32::max);
            let ort_next = ort_logits[(seq - 1) * vocab..seq * vocab]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            println!("  ┌─ logits vs ORT ───");
            println!(
                "  │ argmax(last) oxide={} ort={} match={}",
                next_tok,
                ort_next,
                next_tok == ort_next
            );
            println!(
                "  │ max |oxide − ort|: {:.4e}  (rel {:.2e})",
                max_err,
                max_err / denom
            );
            println!("  └────────────────────");
            let ox = bench_fn("GPT-2 oxide", 3, 10, || {
                executor.run(&inputs).unwrap();
            });
            print_speed_row(model_name, ox, ort_ms);
        }
        Err(e) => println!("skipped ({})", e),
    }
    Ok(())
}

// ===========================================================================
// Section 3: Throughput benchmark
// ===========================================================================

fn run_benchmarks(model_path: &str, model_name: &str) -> Result<()> {
    const WARMUP: usize = 3;
    const RUNS: usize = 20;

    let input_shape = vec![1usize, 3, 224, 224];
    let input_numel: usize = input_shape.iter().product();
    let input_data: Vec<f32> = (0..input_numel)
        .map(|i| i as f32 / input_numel as f32)
        .collect();

    let model = load_model(model_path)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No graph"))?;
    let ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CUDA: {:?}", e))?;
    let executor = OnnxExecutor::from_graph(graph, ctx)?;
    let input_name = executor.input_names.first().unwrap().clone();
    let mut inputs = HashMap::new();
    inputs.insert(input_name, (input_data.clone(), input_shape.clone()));

    let oxide_ms = bench_fn("oxide_onnx", WARMUP, RUNS, || {
        executor.run(&inputs).expect("oxide run failed");
    });

    let tract_ms = bench_tract(model_path, WARMUP, RUNS);
    let ort_gpu_ms = bench_ort_gpu(model_path, WARMUP, RUNS);
    let trt_ms = bench_trtexec(model_path);

    println!();
    println!("  Model: {} (batch=1, 3×224×224)", model_name);
    println!("  ──────────────────────────────────────────────────────────────────");
    println!(
        "  oxide_onnx    GPU  tiled shared-mem SGEMM: {:>8.2} ms/inference",
        oxide_ms
    );
    match &ort_gpu_ms {
        Ok(ms) => println!(
            "  ORT CUDA      GPU  cuDNN/cuBLAS          : {:>8.2} ms/inference  [{:.1}× faster]",
            ms,
            oxide_ms / ms
        ),
        Err(e) => println!("  ORT CUDA      GPU  skipped: {}", e),
    }
    match &trt_ms {
        Ok(ms) => println!(
            "  TensorRT      GPU  fused/optimised        : {:>8.2} ms/inference  [{:.1}× faster]",
            ms,
            oxide_ms / ms
        ),
        Err(e) => println!("  TensorRT      GPU  skipped: {}", e),
    }
    match &tract_ms {
        Ok(ms) => println!(
            "  tract-onnx    CPU  reference              : {:>8.2} ms/inference  [{:.1}× slower]",
            ms,
            ms / oxide_ms
        ),
        Err(e) => println!("  tract-onnx    CPU  skipped: {}", e),
    }
    println!("  ──────────────────────────────────────────────────────────────────");
    Ok(())
}

fn bench_fn<F: Fn()>(name: &str, warmup: usize, runs: usize, f: F) -> f64 {
    print!(
        "  Benchmarking {} ({} warmup + {} timed runs)... ",
        name, warmup, runs
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());
    // The GPU idles at 210 MHz against a 2130 MHz boost clock, and the ORT and
    // TensorRT subprocesses that run between our benchmarks leave it there. A
    // fixed warmup count is not enough for the short models — GPT-2's thirteen
    // iterations are 150 ms of work, and it was timing anywhere from 9 to 17 ms
    // depending on where the clocks happened to be when it started. Warm by
    // wall time instead, so every model reaches the same clock state.
    let warm_t0 = Instant::now();
    let mut warmed = 0usize;
    while warmed < warmup || (warm_t0.elapsed().as_millis() < 800 && warmed < 10_000) {
        f();
        warmed += 1;
    }
    let t0 = Instant::now();
    for _ in 0..runs {
        f();
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / runs as f64;
    println!("{:.2} ms/run", ms);
    ms
}

/// One-line oxide-vs-ORT comparison row (used by the NLP models).
fn print_speed_row(model: &str, oxide_ms: f64, ort_ms: Option<f64>) {
    println!("  ──────────────────────────────────────────────────────────────────");
    println!(
        "  {:<14} oxide GPU (tiled SGEMM) : {:>8.2} ms/inference",
        model, oxide_ms
    );
    match ort_ms {
        Some(o) => println!(
            "  {:<14} ORT  GPU (cuDNN/cuBLAS) : {:>8.2} ms/inference  [{:.1}× faster]",
            "",
            o,
            oxide_ms / o
        ),
        None => println!("  {:<14} ORT timing unavailable", ""),
    }
    println!("  ──────────────────────────────────────────────────────────────────");
}

/// Run `scripts/bench_gpu.py` which benchmarks ONNX Runtime with the CUDA
/// execution provider.  Returns the mean inference latency in milliseconds.
fn bench_ort_gpu(model_path: &str, warmup: usize, runs: usize) -> Result<f64> {
    // The script is one directory above oxide-onnx/
    let script = std::path::Path::new(model_path)
        .parent() // models/
        .and_then(|p| p.parent()) // oxide-onnx/
        .map(|p| p.join("./scripts/bench_gpu.py"))
        .unwrap_or_else(|| std::path::PathBuf::from("./scripts/bench_gpu.py"));

    if !script.exists() {
        return Err(anyhow::anyhow!("bench_gpu.py not found at {:?}", script));
    }

    print!("  Running ORT CUDA benchmark (python3 bench_gpu.py)... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());

    let output = std::process::Command::new("python3")
        .arg(&script)
        .arg(model_path)
        .arg(warmup.to_string())
        .arg(runs.to_string())
        .output()
        .map_err(|e| anyhow::anyhow!("python3 exec: {}", e))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("bench_gpu.py failed: {}", err.trim()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("ort_cuda_ms=") {
            let ms: f64 = rest
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("bad ort_cuda_ms: '{}'", rest))?;
            println!("{:.2} ms/run", ms);
            return Ok(ms);
        }
    }
    Err(anyhow::anyhow!(
        "ort_cuda_ms not found in output: {}",
        stdout.trim()
    ))
}

/// Run `trtexec` (TensorRT CLI) and return mean latency in milliseconds.
fn bench_trtexec(model_path: &str) -> Result<f64> {
    let exe = which_trtexec()?;

    print!("  Running TensorRT benchmark (trtexec)... ");
    let _ = std::io::Write::flush(&mut std::io::stdout());

    let output = std::process::Command::new(&exe)
        .args([
            &format!("--onnx={}", model_path),
            "--warmUp=2000",
            "--iterations=20",
        ])
        .output()
        .map_err(|e| anyhow::anyhow!("trtexec exec: {}", e))?;

    // trtexec writes the "Latency:" summary to stdout
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("Latency:")
            && line.contains("mean =")
            && !line.contains("H2D")
            && !line.contains("D2H")
        {
            if let Some(mean_part) = line.split("mean =").nth(1) {
                let ms_str = mean_part
                    .trim()
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .trim_end_matches(',');
                let ms: f64 = ms_str
                    .parse()
                    .map_err(|_| anyhow::anyhow!("bad trtexec mean: '{}'", ms_str))?;
                println!("{:.2} ms/run", ms);
                return Ok(ms);
            }
        }
    }
    Err(anyhow::anyhow!("trtexec Latency line not found"))
}

fn which_trtexec() -> Result<String> {
    for candidate in &["/usr/bin/trtexec", "/usr/local/bin/trtexec"] {
        if std::path::Path::new(candidate).exists() {
            return Ok(candidate.to_string());
        }
    }
    // fall back to PATH lookup
    let out = std::process::Command::new("which").arg("trtexec").output();
    if let Ok(o) = out {
        let path = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if !path.is_empty() {
            return Ok(path);
        }
    }
    Err(anyhow::anyhow!("trtexec not found"))
}

fn bench_tract(model_path: &str, warmup: usize, runs: usize) -> Result<f64> {
    use tract_onnx::prelude::*;

    let model = tract_onnx::onnx()
        .model_for_path(model_path)
        .map_err(|e| anyhow::anyhow!("tract: {}", e))?
        .into_optimized()
        .map_err(|e| anyhow::anyhow!("tract: {}", e))?
        .into_runnable()
        .map_err(|e| anyhow::anyhow!("tract: {}", e))?;

    let input_numel = 1usize * 3 * 224 * 224;
    let input_data: Vec<f32> = (0..input_numel)
        .map(|i| i as f32 / input_numel as f32)
        .collect();
    let input_arr = tract_ndarray::Array4::<f32>::from_shape_vec((1, 3, 224, 224), input_data)
        .map_err(|e| anyhow::anyhow!("tract shape: {}", e))?;
    let proto_tensor: Tensor = input_arr.into();

    let ms = bench_fn("tract CPU", warmup, runs, || {
        let t: Tensor = proto_tensor.clone();
        model.run(tvec![t.into()]).expect("tract run failed");
    });
    Ok(ms)
}
