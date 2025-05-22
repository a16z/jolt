use super::ONNXProgram;

#[test]
fn test_tracer() {
    ONNXProgram::new("onnx/perceptron.onnx").trace();
}
