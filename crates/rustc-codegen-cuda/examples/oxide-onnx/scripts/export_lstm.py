#!/usr/bin/env python3
"""Export a small LSTM sequence classifier to ONNX.

The engine's other models are feed-forward: every node runs once, in
topological order. A recurrent network is the opposite — the same weights are
applied H times with a serial dependency between steps — so it exercises a
control-flow shape nothing else here does.

Usage: export_lstm.py [out.onnx]
"""
import sys
import torch
import torch.nn as nn

OUT = sys.argv[1] if len(sys.argv) > 1 else "models/lstm-seq.onnx"
SEQ, INPUT, HIDDEN, CLASSES = 64, 128, 256, 32


class SeqClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT, HIDDEN, num_layers=2, batch_first=True)
        self.head = nn.Linear(HIDDEN, CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


torch.manual_seed(0)
model = SeqClassifier().eval()
dummy = torch.randn(1, SEQ, INPUT)
torch.onnx.export(
    model,
    dummy,
    OUT,
    input_names=["input"],
    output_names=["logits"],
    opset_version=13,
    dynamo=False,
)
print(f"wrote {OUT}")
