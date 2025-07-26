#[cfg(test)]
mod e2e_tests {
    use crate::{
        jolt::vm::{JoltProverPreprocessing, JoltSNARK},
        program::ONNXProgram,
    };
    use ark_bn254::Fr;
    use jolt_core::utils::transcript::KeccakTranscript;
    use onnx_tracer::{logger::init_logger, tensor::Tensor};

    #[test]
    fn test_addsubmul0() {
        // --- Preprocessing ---
        init_logger();
        let text_classification_model = ONNXProgram::new(
            "../onnx-tracer/models/addsubmul0/network.onnx".into(),
            Tensor::new(Some(&[10]), &[1]).unwrap(),
        );
        let program_bytecode = text_classification_model.decode();
        println!("{program_bytecode:#?}");
        let pp: JoltProverPreprocessing<Fr, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);

        // --- Proving ---
        let execution_trace = text_classification_model.trace();
        let snark: JoltSNARK<Fr, KeccakTranscript> = JoltSNARK::prove(pp.clone(), execution_trace);

        // --- Verification ---
        snark.verify((&pp).into()).unwrap();
    }

    // TODO(Forpee): refactor duplicate code in these tests
    #[test]
    fn test_simple_classification() {
        // --- Preprocessing ---
        init_logger();
        let text_classification_model = ONNXProgram::new(
            "../onnx-tracer/models/simple_text_classification/network.onnx".into(),
            Tensor::new(Some(&[1, 2, 3, 4, 5]), &[1, 5]).unwrap(), // Example input
        );
        let program_bytecode = text_classification_model.decode();
        let pp: JoltProverPreprocessing<Fr, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);

        // --- Proving ---
        let execution_trace = text_classification_model.trace();
        let snark: JoltSNARK<Fr, KeccakTranscript> = JoltSNARK::prove(pp.clone(), execution_trace);

        // --- Verification ---
        snark.verify((&pp).into()).unwrap();
    }

    // TODO(Forpee): There is a runtime bug here related to the pow operator in the ONNX model
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

    #[test]
    fn test_subgraph() {
        init_logger();
        let subgraph_program = ONNXProgram {
            model_path: "../onnx-tracer/models/subgraph/network.onnx".into(),
            inputs: Tensor::new(Some(&[1, 2, 3, 4]), &[1, 4]).unwrap(), // Example input
        };
        let program_bytecode = subgraph_program.decode();

        println!("Program decoded");
        println!("Program code: {program_bytecode:#?}",);

        // Test that the addresses of a subgraph are monotonically increasing
        let mut i = 0;
        for instr in program_bytecode {
            assert!(instr.address > i);
            i = instr.address;
        }

        subgraph_program.trace();
    }
}
