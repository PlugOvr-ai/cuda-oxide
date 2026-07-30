#!/usr/bin/env bash
# Download ONNX models for oxide-onnx testing.
# Run from the repository root:
#   bash scripts/download_models.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"

mkdir -p "$MODELS_DIR"

# ---------------------------------------------------------------------------
# ResNet50 v2 (opset 7) — from ONNX Model Zoo
# ---------------------------------------------------------------------------
RESNET50_FILE="$MODELS_DIR/resnet50-v2-7.onnx"
if [ ! -f "$RESNET50_FILE" ]; then
    echo "Downloading ResNet50 v2..."
    curl -fL \
        "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx" \
        -o "$RESNET50_FILE"
    echo "  Saved to $RESNET50_FILE ($(du -sh "$RESNET50_FILE" | cut -f1))"
else
    echo "ResNet50 already present: $RESNET50_FILE"
fi

# ---------------------------------------------------------------------------
# MobileNetV2 (opset 10) — lighter model for quick tests
# ---------------------------------------------------------------------------
MOBILENET_FILE="$MODELS_DIR/mobilenetv2-10.onnx"
if [ ! -f "$MOBILENET_FILE" ]; then
    echo "Downloading MobileNetV2..."
    curl -fL \
        "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx" \
        -o "$MOBILENET_FILE"
    echo "  Saved to $MOBILENET_FILE ($(du -sh "$MOBILENET_FILE" | cut -f1))"
else
    echo "MobileNetV2 already present: $MOBILENET_FILE"
fi

# ---------------------------------------------------------------------------
# ViT-B/16 (timm, opset 17) — Vision Transformer.
# The upstream timm export is a dynamic-batch graph with Shape/Slice/Where/
# Expand machinery. We constant-fold it for fixed [1,3,224,224] via ONNX
# Runtime's BASIC graph optimization, which yields a static-shape graph
# (Conv + 12× {LayerNorm, batched-MatMul attention, GELU}) the engine runs.
# ---------------------------------------------------------------------------
VIT_FILE="$MODELS_DIR/vit-base-patch16-224.onnx"
VIT_RAW="$MODELS_DIR/.vit-base-patch16-224.raw.onnx"
if [ ! -f "$VIT_FILE" ]; then
    echo "Downloading ViT-B/16 (timm, ~346 MB)..."
    curl -fL \
        "https://media.githubusercontent.com/media/onnx/models/main/Computer_Vision/skip/vit_base_patch16_224_Opset17_timm/vit_base_patch16_224_Opset17.onnx" \
        -o "$VIT_RAW"
    echo "Simplifying to static shapes (ONNX Runtime BASIC optimization)..."
    python3 - "$VIT_RAW" "$VIT_FILE" <<'PY'
import sys, onnxruntime as ort
raw, out = sys.argv[1], sys.argv[2]
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
so.optimized_model_filepath = out
ort.InferenceSession(raw, so, providers=["CPUExecutionProvider"])
print("  wrote", out)
PY
    rm -f "$VIT_RAW"
    echo "  Saved to $VIT_FILE ($(du -sh "$VIT_FILE" | cut -f1))"
else
    echo "ViT-B/16 already present: $VIT_FILE"
fi

# ---------------------------------------------------------------------------
# BERT-base (HF transformers export, opset 17). Inputs are already static
# ([1,128] input_ids + attention_mask); ORT BASIC optimization folds the
# remaining shape/mask machinery and fuses MatMul+Add -> Gemm.
# ---------------------------------------------------------------------------
BERT_FILE="$MODELS_DIR/bert-base-uncased.onnx"
BERT_RAW="$MODELS_DIR/.bert-base.raw.onnx"
if [ ! -f "$BERT_FILE" ]; then
    echo "Downloading BERT-base (HF transformers, ~438 MB)..."
    curl -fL \
        "https://media.githubusercontent.com/media/onnx/models/main/Natural_Language_Processing/bert_Opset17_transformers/bert_Opset17.onnx" \
        -o "$BERT_RAW"
    echo "Simplifying to a static graph (ONNX Runtime BASIC optimization)..."
    python3 - "$BERT_RAW" "$BERT_FILE" <<'PY'
import sys, onnxruntime as ort
raw, out = sys.argv[1], sys.argv[2]
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
so.optimized_model_filepath = out
ort.InferenceSession(raw, so, providers=["CPUExecutionProvider"])
print("  wrote", out)
PY
    rm -f "$BERT_RAW"
    echo "  Saved to $BERT_FILE ($(du -sh "$BERT_FILE" | cut -f1))"
else
    echo "BERT-base already present: $BERT_FILE"
fi

# ---------------------------------------------------------------------------
# GPT-2 LM head (HF transformers export, opset 17) — causal decoder LM.
# Static [1,128] inputs; ORT BASIC folds the causal-mask construction and
# fuses MatMul+Add -> Gemm. Output: logits [1,128,50257] (+KV cache).
# ---------------------------------------------------------------------------
GPT2_FILE="$MODELS_DIR/gpt2-lmhead.onnx"
GPT2_RAW="$MODELS_DIR/.gpt2.raw.onnx"
if [ ! -f "$GPT2_FILE" ]; then
    echo "Downloading GPT-2 LM head (HF transformers, ~652 MB)..."
    curl -fL \
        "https://media.githubusercontent.com/media/onnx/models/main/Generative_AI/skip/gpt2lmhead_Opset17_transformers/gpt2lmhead_Opset17.onnx" \
        -o "$GPT2_RAW"
    echo "Simplifying to a static graph (ONNX Runtime BASIC optimization)..."
    python3 - "$GPT2_RAW" "$GPT2_FILE" <<'PY'
import sys, onnxruntime as ort
raw, out = sys.argv[1], sys.argv[2]
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
so.optimized_model_filepath = out
ort.InferenceSession(raw, so, providers=["CPUExecutionProvider"])
print("  wrote", out)
PY
    rm -f "$GPT2_RAW"
    echo "  Saved to $GPT2_FILE ($(du -sh "$GPT2_FILE" | cut -f1))"
else
    echo "GPT-2 already present: $GPT2_FILE"
fi

# ---------------------------------------------------------------------------
# Architectures beyond classification and transformers. These need no
# pre-simplification: they are already static graphs.
# ---------------------------------------------------------------------------
ZOO="https://github.com/onnx/models/raw/main/validated"
fetch() {
    local file="$MODELS_DIR/$1"
    local url="$ZOO/$2"
    if [ ! -f "$file" ]; then
        echo "Downloading $1..."
        curl -fL "$url" -o "$file"
        echo "  Saved to $file ($(du -sh "$file" | cut -f1))"
    else
        echo "$1 already present"
    fi
}

fetch shufflenet-v2-10.onnx  "vision/classification/shufflenet/model/shufflenet-v2-10.onnx"
fetch super-resolution-10.onnx \
    "vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx"
fetch tinyyolov2-8.onnx \
    "vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx"
fetch fcn-resnet50-11.onnx \
    "vision/object_detection_segmentation/fcn/model/fcn-resnet50-11.onnx"
fetch mosaic-9.onnx "vision/style_transfer/fast_neural_style/model/mosaic-9.onnx"

# ---------------------------------------------------------------------------
# The recurrent model has no zoo equivalent worth using, so it is exported
# here. Needs torch; skipped with a note if that is missing.
# ---------------------------------------------------------------------------
LSTM_FILE="$MODELS_DIR/lstm-seq.onnx"
if [ ! -f "$LSTM_FILE" ]; then
    if python3 -c "import torch" 2>/dev/null; then
        echo "Exporting LSTM sequence classifier..."
        python3 "$SCRIPT_DIR/export_lstm.py" "$LSTM_FILE"
    else
        echo "torch not installed; skipping lstm-seq.onnx (the LSTM model will be skipped)"
    fi
else
    echo "LSTM already present: $LSTM_FILE"
fi

echo ""
echo "Models ready in $MODELS_DIR"
echo "Now run, from the repository root:"
echo "  cargo run --package cargo-oxide -- run oxide-onnx"
