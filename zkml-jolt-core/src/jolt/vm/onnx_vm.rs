#[cfg(test)]
mod e2e_tests {
    use crate::host::ONNXProgram;
    use onnx_tracer::{logger::init_logger, tensor::Tensor};

    // TODO(Forpee): refactor duplicate code in these tests
    #[test]
    fn test_simple_classification() {
        init_logger();
        let text_classification = ONNXProgram {
            model_path: "../onnx-tracer/models/simple_text_classification/network.onnx".into(),
            inputs: Tensor::new(Some(&[1, 2, 3]), &[1, 3]).unwrap(), // Example input
        };
        let program_code = text_classification.decode();
        println!("Program code: {program_code:#?}",);
    }

    #[test]
    fn test_medium_classification() {
        init_logger();
        let text_classification = ONNXProgram {
            model_path: "../onnx-tracer/models/medium_text_classification/network.onnx".into(),
            inputs: Tensor::new(Some(&[1, 2, 3, 4, 5]), &[1, 5]).unwrap(), // Example input
        };
        let program_code = text_classification.decode();
        println!("Program code: {program_code:#?}",);
    }
}
