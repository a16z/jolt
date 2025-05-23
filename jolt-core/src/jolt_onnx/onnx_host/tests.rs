use super::ONNXProgram;

#[test]
fn test_perceptron() {
    ONNXProgram::new("onnx/perceptron.onnx").trace();
}

#[test]
fn test_perceptron_2() {
    ONNXProgram::new("onnx/perceptron_2.onnx").trace();
}
