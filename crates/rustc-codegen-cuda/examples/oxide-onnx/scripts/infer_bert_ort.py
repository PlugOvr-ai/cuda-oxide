#!/usr/bin/env python3
"""
BERT single-shot inference via ONNX Runtime (authoritative reference).

Generates the SAME deterministic inputs as oxide_onnx's run_bert
(ids[i] = (i*7919+13) % 30522 ; attention_mask = 1 for i<100 else 0),
runs the model, and writes every output as raw f32-LE, concatenated in
graph-output order, to <output.bin>.

Usage: infer_bert_ort.py <model.onnx> <output.bin> [seq]
"""
import os
import sys

try:
    import torch as _torch
    _lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
    os.environ["LD_LIBRARY_PATH"] = _lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
except ImportError:
    pass

import numpy as np
import onnxruntime as ort

model_path = sys.argv[1]
output_bin = sys.argv[2]
seq = int(sys.argv[3]) if len(sys.argv) > 3 else 128

ids = np.array([[(i * 7919 + 13) % 30522 for i in range(seq)]], dtype=np.int64)
mask = np.array([[1.0 if i < 100 else 0.0 for i in range(seq)]], dtype=np.float32)

opts = ort.SessionOptions()
opts.log_severity_level = 3
available = ort.get_available_providers()
providers = (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else []) + ["CPUExecutionProvider"]

session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
feeds = {}
for inp in session.get_inputs():
    feeds[inp.name] = ids if "id" in inp.name.lower() else mask

outputs = session.run(None, feeds)
with open(output_bin, "wb") as f:
    for arr in outputs:
        f.write(np.asarray(arr).astype(np.float32).tobytes())

import time
for _ in range(3):
    session.run(None, feeds)
N = 10
t0 = time.perf_counter()
for _ in range(N):
    session.run(None, feeds)
print(f"ort_ms={(time.perf_counter() - t0) * 1000.0 / N:.4f}")
