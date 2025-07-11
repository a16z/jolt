#[cfg(test)]
mod e2e_tests {
    use onnx_tracer::tensor::Tensor;

    use crate::host::ONNXProgram;

    #[test]
    fn test_simple_classification() {
        let text_classification = ONNXProgram {
            model_path: "../onnx-tracer/models/simple_text_classification/network.onnx".into(),
            inputs: Tensor::new(Some(&[1, 2, 3]), &[1, 3]).unwrap(), // Example input
        };
        let program_code = text_classification.decode();
    }
}
