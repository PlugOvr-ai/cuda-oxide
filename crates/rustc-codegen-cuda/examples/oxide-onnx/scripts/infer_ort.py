#!/usr/bin/env python3
"""
Single-shot ONNX inference via ONNX Runtime CUDA.
Reads input tensor from a raw f32-LE binary file, writes output to another.

Usage: infer_ort.py <model.onnx> <input.bin> <output.bin> [shape]

`shape` is a comma-separated input shape, e.g. 1,1,224,224. It defaults to
1,3,224,224 for the image classifiers. Only the first output is written; its
shape is printed so the caller can check it.
"""
import os
import sys

# Expose torch's bundled CUDA so onnxruntime-gpu can load libcublasLt etc.
try:
    import torch as _torch
    _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
    os.environ["LD_LIBRARY_PATH"] = _torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
except ImportError:
    pass

import numpy as np
import onnxruntime as ort

model_path, input_bin, output_bin = sys.argv[1], sys.argv[2], sys.argv[3]
shape = tuple(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else (1, 3, 224, 224)

opts = ort.SessionOptions()
opts.log_severity_level = 3
available = ort.get_available_providers()
providers = (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else []) + ["CPUExecutionProvider"]

session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
inp_name = session.get_inputs()[0].name

raw = np.frombuffer(open(input_bin, "rb").read(), dtype=np.float32)
input_data = raw.reshape(shape)

outputs = session.run(None, {inp_name: input_data})
result = np.asarray(outputs[0]).astype(np.float32).flatten()
print("out_shape=" + ",".join(str(d) for d in np.asarray(outputs[0]).shape))
open(output_bin, "wb").write(result.tobytes())
