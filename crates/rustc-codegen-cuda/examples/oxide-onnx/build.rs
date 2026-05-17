fn main() {
    prost_build::compile_protos(&["onnx.proto3"], &["."]).expect("Failed to compile ONNX proto3");
}
