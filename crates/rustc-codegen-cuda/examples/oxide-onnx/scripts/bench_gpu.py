#!/usr/bin/env python3
"""
GPU ONNX inference benchmark using ONNX Runtime with CUDA execution provider.

Exposes PyTorch's bundled CUDA libs on LD_LIBRARY_PATH so that onnxruntime-gpu
can find them regardless of the system CUDA version.

Output (stdout, key=value lines):
  ort_cuda_ms=<float>    — ORT CUDA mean inference ms
  provider=<string>      — execution provider that was actually used
"""
import os
import sys
import time

# Expose torch's bundled CUDA so onnxruntime-gpu can load libcublasLt etc.
try:
    import torch as _torch
    _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
    os.environ["LD_LIBRARY_PATH"] = _torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
except ImportError:
    pass

import numpy as np
import onnxruntime as ort


def bench(model_path: str, warmup: int = 3, runs: int = 20):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # suppress INFO/WARNING noise

    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    provider_used = session.get_providers()[0]

    inp_name = session.get_inputs()[0].name
    numel = 1 * 3 * 224 * 224
    # Same deterministic input as the Rust benchmark
    input_data = (np.arange(numel, dtype=np.float32) / numel).reshape(1, 3, 224, 224)

    # Warm to a fixed wall time, not a fixed count: the GPU idles at 210 MHz
    # against a 2130 MHz boost clock, so a short benchmark otherwise measures
    # whatever clock state it happened to start in. Matches the Rust harness.
    w0 = time.perf_counter()
    warmed = 0
    while warmed < warmup or (time.perf_counter() - w0 < 0.8 and warmed < 10000):
        session.run(None, {inp_name: input_data})
        warmed += 1

    t0 = time.perf_counter()
    for _ in range(runs):
        session.run(None, {inp_name: input_data})
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / runs

    return elapsed_ms, provider_used


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: bench_gpu.py <model.onnx> [warmup] [runs]", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    avg_ms, provider = bench(model_path, warmup, runs)
    print(f"ort_cuda_ms={avg_ms:.4f}")
    print(f"provider={provider}", file=sys.stderr)
