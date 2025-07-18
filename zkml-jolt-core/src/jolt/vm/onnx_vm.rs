#[cfg(test)]
mod e2e_tests {
    use crate::{
        jolt::vm::{JoltProverPreprocessing, JoltSNARK},
        program::ONNXProgram,
    };
    use ark_bn254::Fr;
    use jolt_core::utils::transcript::KeccakTranscript;
    use onnx_tracer::{logger::init_logger, tensor::Tensor};
    // TODO(Forpee): refactor duplicate code in these tests
    #[test]
    fn test_simple_classification() {
        init_logger();
        let text_classification = ONNXProgram {
            model_path: "../onnx-tracer/models/simple_text_classification/network.onnx".into(),
            inputs: Tensor::new(Some(&[1, 2, 3, 4, 5]), &[1, 5]).unwrap(), // Example input
        };
        let program_bytecode = text_classification.decode();
        println!("Program code: {program_bytecode:#?}",);
        let execution_trace = text_classification.trace();
        println!("Execution trace: {execution_trace:#?}");
        let pp: JoltProverPreprocessing<Fr, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);
        let snark: JoltSNARK<Fr, KeccakTranscript> = JoltSNARK::prove(pp.clone(), execution_trace);
        snark.verify((&pp).into()).unwrap();
    }

    #[test]
    fn test_medium_classification() {
        init_logger();
        let text_classification = ONNXProgram {
            model_path: "../onnx-tracer/models/medium_text_classification/network.onnx".into(),
            inputs: Tensor::new(Some(&[1, 2, 3, 4, 5]), &[1, 5]).unwrap(), // Example input
        };
        let program_bytecode = text_classification.decode();
        println!("Program code: {program_bytecode:#?}",);
        text_classification.trace();
    }
}
