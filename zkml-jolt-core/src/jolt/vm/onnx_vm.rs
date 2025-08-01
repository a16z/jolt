#[cfg(test)]
mod e2e_tests {
    use crate::{
        jolt::vm::{JoltProverPreprocessing, JoltSNARK},
        program::ONNXProgram,
    };
    use ark_bn254::Fr;
    use jolt_core::utils::transcript::KeccakTranscript;
    use log::info;
    use serde_json::Value;
    use std::fs::File;
    use std::io::Read;

    use onnx_tracer::{custom_addsubmul_model, logger::init_logger, model, tensor::Tensor};

    fn load_input_vector(path: &str) -> Vec<i128> {
        let mut file = File::open(path).expect("Unable to open input.json");
        let mut data = String::new();
        file.read_to_string(&mut data).expect("Unable to read file");

        let v: Value = serde_json::from_str(&data).expect("Invalid JSON");
        let arr = v["input_data"][0]
            .as_array()
            .expect("Bad input_data format");

        let mut vec: Vec<i128> = arr.iter().map(|x| x.as_i64().unwrap() as i128).collect();
        vec.resize(1000, 0);
        vec
    }

    #[test]
    fn test_addsubmul0() {
        // --- Preprocessing ---
        init_logger();
        let text_classification_model = ONNXProgram::new(
            "../onnx-tracer/models/addsubmul0/network.onnx".into(),
            Tensor::new(Some(&[10]), &[1]).unwrap(),
        );
        let model = model(&text_classification_model.model_path);
        println!("Model: {model:#?}");
        // let program_bytecode = text_classification_model.decode();
        // println!("Program code: {program_bytecode:#?}",);
        // let pp: JoltProverPreprocessing<Fr, KeccakTranscript> =
        //     JoltSNARK::prover_preprocess(program_bytecode);

        // // --- Proving ---
        // let execution_trace = text_classification_model.trace();
        // // println!("{execution_trace:#?}");
        // let snark: JoltSNARK<Fr, KeccakTranscript> = JoltSNARK::prove(pp.clone(), execution_trace);

        // // --- Verification ---
        // snark.verify((&pp).into()).unwrap();
    }

    #[test]
    fn test_custom_addsubmul() {
        let custom_addsubmul_model = custom_addsubmul_model();
        let program_bytecode = onnx_tracer::decode_model(custom_addsubmul_model.clone());
        println!("Program code: {program_bytecode:#?}",);
        let execution_trace = onnx_tracer::execution_trace(
            custom_addsubmul_model,
            &Tensor::new(Some(&[10]), &[1]).unwrap(),
        );
        println!("Execution trace: {execution_trace:#?}",);
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
        println!("Program code: {program_bytecode:#?}",);
        // let pp: JoltProverPreprocessing<Fr, KeccakTranscript> =
        //     JoltSNARK::prover_preprocess(program_bytecode);

        // --- Proving ---
        let execution_trace = text_classification_model.trace();
        println!("Execution trace: {execution_trace:#?}",);
        // let snark: JoltSNARK<Fr, KeccakTranscript> = JoltSNARK::prove(pp.clone(), execution_trace);

        // --- Verification ---
        // snark.verify((&pp).into()).unwrap();
    }

    #[test]

    pub fn test_article_classification_output() {
        init_logger();
        let working_dir: &str = "../onnx-tracer/models/article_classification/";
        // Load input
        let input_vector = load_input_vector(&format!("{working_dir}input.json"));

        // Prepare ONNX program
        let text_classification = ONNXProgram {
            model_path: format!("{working_dir}network.onnx").into(),
            inputs: Tensor::new(Some(&input_vector), &[1, 1000]).unwrap(),
        };

        // Decode to program bytecode (for EZKL use)
        let program_bytecode = text_classification.decode();
        println!("Program code: {program_bytecode:#?}");

        // Load model
        let model = model(&text_classification.model_path);

        // Run inference
        let result = model
            .forward(&[text_classification.inputs.clone()])
            .unwrap();
        let output = result.outputs[0].clone();

        // Map index to label
        let classes = ["business", "entertainment", "politics", "sport", "tech"];
        let (pred_idx, _) = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        info!("Predicted class: {}", classes[pred_idx]);
    }

    #[test]
    fn test_medium_classification() {
        init_logger();
        let mut input_vector = vec![846, 3, 195, 4, 374, 14, 259];
        input_vector.resize(100, 0); // Resize to match the input shape

        let text_classification = ONNXProgram {
            model_path: "../onnx-tracer/models/medium_text_classification/network.onnx".into(),
            inputs: Tensor::new(Some(&input_vector), &[1, 100]).unwrap(), // Example input
        };
        let program_bytecode = text_classification.decode();
        println!("Program code: {program_bytecode:#?}",);
        text_classification.trace();
    }

    #[test]
    fn test_medium_classification_output() {
        init_logger();
        let mut input_vector = vec![197, 10, 862, 8, 23, 53, 2, 319, 34, 122, 100, 53, 33];
        input_vector.resize(100, 0); // Resize to match the input shape

        let text_classification = ONNXProgram {
            model_path: "../onnx-tracer/models/medium_text_classification/network.onnx".into(),
            inputs: Tensor::new(Some(&input_vector), &[1, 100]).unwrap(), // Example input
        };
        let program_bytecode = text_classification.decode();
        println!("Program code: {program_bytecode:#?}",);
        let model = model(&text_classification.model_path);

        let result = model
            .forward(&[text_classification.inputs.clone()])
            .unwrap();
        let output = result.outputs[0].clone();
        info!("Output: {output:#?}",);
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
