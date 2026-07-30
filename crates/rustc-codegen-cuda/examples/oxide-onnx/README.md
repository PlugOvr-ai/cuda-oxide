# oxide-onnx

An ONNX inference engine written entirely in Rust, with every GPU kernel
compiled from Rust source by `rustc-codegen-cuda`. It links only
`libcuda.so.1` — the CUDA driver. No cuBLAS, no cuDNN, no CUTLASS, no
TensorRT: the GEMMs, convolutions, normalisations and everything else are in
[`src/kernels.rs`](src/kernels.rs).

It runs eleven models spanning CNN classification, vision transformers, NLP
encoders and decoders, detection, segmentation, super-resolution, style
transfer and a recurrent network, and checks every one of them against ONNX
Runtime on each run.

## Requirements

- An NVIDIA GPU. Ampere (sm_80) or later uses the f16 tensor-core paths;
  older cards fall back to f32 kernels automatically (see
  [Precision](#precision)).
- The CUDA driver. The CUDA toolkit is needed to build the backend, not to run.
- `python3` with `onnx`, `onnxruntime-gpu` and (for the LSTM export) `torch`,
  if you want the reference comparisons.
- `trtexec` on `PATH` for the TensorRT column. Optional; skipped if absent.

## Getting the models

```bash
bash scripts/download_models.sh
```

Fetches all eleven into `models/` (about 1.6 GB). Four of them are
pre-simplified with ONNX Runtime's BASIC optimiser, which constant-folds the
dynamic-shape machinery that timm and transformers exports carry; the script
does this for you. The LSTM is exported by `scripts/export_lstm.py` and is
skipped with a note if `torch` is missing.

## Running

From the **repository root**:

```bash
cargo run --package cargo-oxide -- run oxide-onnx
```

Note the explicit form. A `cargo oxide` shell alias shadows the binary in some
setups, and a plain `cargo build` inside this directory fails to link — the
PTX artifact anchor comes from the `cargo-oxide` driver.

A full run takes a few minutes and prints three sections:

1. **Unit kernel tests** — each kernel against a CPU reference, including the
   Winograd transforms.
2. **End-to-end inference** — every model, with its correctness check.
3. **Throughput benchmark** — oxide vs ONNX Runtime vs TensorRT vs tract (CPU).

## The comparison against TensorRT and ONNX Runtime

This is what section 3 and the per-model rows in section 2 do; there is no
separate command. Every model is timed three ways:

| engine | how |
|---|---|
| oxide | in-process, `bench_fn` |
| ONNX Runtime CUDA | `scripts/bench_gpu.py` subprocess |
| TensorRT | `trtexec` subprocess |
| tract | in-process CPU reference (ResNet50, MobileNetV2, ViT only) |

Output looks like:

```
  ShuffleNet-v2              oxide    0.91 ms   ORT    0.70 ms   [1.30x]
                                             TRT    0.67 ms   [1.36x]
```

Two things about this comparison are easy to get wrong, and the harness now
guards both:

- **Clocks.** This GPU idles at 210 MHz against a 2130 MHz boost clock, so a
  short benchmark measures whatever clock state it started in. Every harness —
  Rust, `bench_gpu.py`, the ORT scripts — warms for 800 ms of wall time, not a
  fixed iteration count. `trtexec` already did this via `--warmUp`.
- **Shapes.** Given a model with dynamic dimensions and no `--shapes`,
  `trtexec` picks its own geometry and says so only in a warning on stderr. It
  benchmarked FCN-ResNet50 at `1x3x1x1` — one pixel — which read as 3.5x
  faster than us. The harness now detects that and re-runs with the shape
  pinned.

Both were real bugs that made published numbers wrong. If you add an engine or
a model, check the same two things.

## Correctness

Every model is checked on every run; nothing is timed that has not been
verified. Classifiers must match top-1 **exactly** against both tract and ONNX
Runtime. GPT-2 must produce the same argmax token. The dense-output models —
super-resolution, detection, segmentation, style transfer, BERT — are compared
elementwise and must stay within 2e-2 relative L1 error with cosine similarity
above 0.999, since f16 tensor cores carry about three decimal digits.

A failure prints `MATCH: false` and a `!!` line. Nothing is asserted silently:
a printed number that nothing checks is not a test, and that mistake has been
made here more than once.

## Environment variables

Measurement and debugging:

| variable | effect |
|---|---|
| `OXIDE_PROFILE=event` | device time per op type, using CUDA events |
| `OXIDE_PROFILE=1` \| `host` | wall time per node, with or without a sync |
| `OXIDE_PROFILE_NODES=1` | per-node times, costliest first (needs `OXIDE_PROFILE=event`) |
| `OXIDE_SHAPES=1` | every node's output shape as it executes |
| `OXIDE_PHASES=1` | setup / submit / drain split per inference |
| `OXIDE_BENCH_KERNELS=1` | per-kernel microbenchmarks |
| `OXIDE_QUIET=1` | suppress the graph-rewrite summary |

`OXIDE_PROFILE_NODES` is the one that has found the most: a node far out of
line with its arithmetic is usually a dispatch bug, not a slow kernel.

Execution-path overrides, for A/B testing:

| variable | effect |
|---|---|
| `OXIDE_F16=0` | disable all f16 tensor-core paths (f32 fallback) |
| `OXIDE_GEMM=tiled\|reg\|splitk` | force a GEMM kernel |
| `OXIDE_SPLIT_TARGET=<n>` | target block count for the split-K heuristic |
| `OXIDE_WINOGRAD=<n>` | enable Winograd for convolutions with ≥ n tiles |
| `OXIDE_CONV_IMPLICIT=1` | force the implicit-GEMM convolution |

## Precision

The f16 tensor-core paths need `mma.sync.aligned.m16n8k16`, which is Ampere
and later. The executor reads the device's compute capability once and routes
GEMM, batched MatMul and convolution to f32 register-tiled kernels below
sm_80, saying so on stderr. `OXIDE_F16=0` forces that path on any card; all
eleven models pass either way, at roughly 1.3-2.4x the runtime.

Winograd F(2x2,3x3) is implemented and correct but **off by default**: at
batch 1 it measures neutral, because it divides the GEMM's K by nine and
materialises two intermediates, costing 8-10x the DRAM traffic. Both effects
shrink as batch grows. `OXIDE_WINOGRAD=128` turns it on.

## Adding a model

1. Drop the `.onnx` in `models/` and add a `const` path in
   [`src/main.rs`](src/main.rs).
2. For a classifier, call `run_model`; for anything with a dense output, call
   `run_generic_model`, which compares the whole tensor rather than a top-1
   index.
3. Run it. Unsupported ops print a warning and pass their input through, which
   shows up immediately as a correctness failure.
4. When shapes go wrong, `OXIDE_SHAPES=1` shows which node first disagreed —
   a rank mismatch usually surfaces several nodes downstream of its cause.

Ops live in `dispatch_node` in [`src/executor.rs`](src/executor.rs); kernels in
[`src/kernels.rs`](src/kernels.rs); load-time graph rewrites (BatchNorm
folding, activation/bias/transpose fusion) in
[`src/graph_opt.rs`](src/graph_opt.rs).

## Where it stands

Batch 1, RTX 3090, best of two runs. Lower is better; the last column is
against the faster of the two reference engines.

| Model | Kind | oxide | ORT | TensorRT | vs best |
|---|---|---:|---:|---:|---:|
| ResNet50-v2 | CNN classifier | **1.30** | 1.58 | 1.40 | **0.93x** |
| FCN-ResNet50 | segmentation | **4.36** | 5.00 | 4.56 | **0.96x** |
| BERT-base | NLP encoder | 2.48 | 2.44 | — | 1.02x |
| GPT-2-LMHead | NLP decoder | 7.33 | 6.69 | — | 1.10x |
| ViT-B/16 | vision transformer | 3.05 | 3.16 | 2.65 | 1.15x |
| Tiny-YOLOv2 | detection | 1.08 | 1.18 | 0.89 | 1.21x |
| LSTM | recurrent | 0.97 | 0.79 | 0.93 | 1.23x |
| ShuffleNet-v2 | channel-shuffle CNN | 0.91 | 0.70 | 0.60 | 1.52x |
| Mosaic | style transfer | 2.27 | 2.38 | 1.46 | 1.55x |
| Sub-pixel CNN | super-resolution | 0.99 | 0.73 | 0.56 | 1.77x |
| MobileNetV2 | depthwise CNN | 0.73 | 0.68 | 0.41 | 1.78x |

For scale, tract on CPU takes 355 ms for ResNet50 and 871 ms for ViT.

TensorRT runs these in FP32 with TF32 enabled, not FP16 — its Winograd kernel
shows zero tensor-pipe activity — so it is carrying more precision than we are
where it wins.

The models still behind share a shape: many small kernels. MobileNetV2's 1x1
convolutions tile to 3-39 blocks on an 82-SM device and reach 2-4% of peak;
the fix is a narrower tile than the current 64x64, which is measured and
understood but not yet written.
