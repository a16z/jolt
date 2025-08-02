use jolt_core::jolt::{
    instruction::{InstructionLookup, LookupQuery},
    lookup_table::LookupTables,
};

use crate::jolt::instruction::{add::ADD, mul::MUL, sub::SUB};

pub const WORD_SIZE: usize = 32;

macro_rules! define_lookup_enum {
    (
        enum $enum_name:ident,
        const $word_size:ident,
        trait $trait_name:ident,
        $($variant:ident : $inner:ty),+ $(,)?
    ) => {
        #[derive(Debug)]
        pub enum $enum_name {
            $(
                $variant($inner),
            )+
        }

        impl $trait_name<$word_size> for $enum_name {
            fn to_instruction_inputs(&self) -> (u64, i64) {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_instruction_inputs(),
                    )+
                }
            }

            fn to_lookup_index(&self) -> u64 {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_index(),
                    )+
                }
            }

            fn to_lookup_operands(&self) -> (u64, u64) {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_operands(),
                    )+
                }
            }

            fn to_lookup_output(&self) -> u64 {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_output(),
                    )+
                }
            }
        }
    };
}

define_lookup_enum!(
    enum ONNXLookup,
    const WORD_SIZE,
    trait LookupQuery,
    Add: ADD<WORD_SIZE>,
    Sub: SUB<WORD_SIZE>,
    Mul: MUL<WORD_SIZE>,
);

impl InstructionLookup<WORD_SIZE> for ONNXLookup {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        match self {
            ONNXLookup::Add(add) => add.lookup_table(),
            ONNXLookup::Sub(sub) => sub.lookup_table(),
            ONNXLookup::Mul(mul) => mul.lookup_table(),
        }
    }
}

#[cfg(test)]
mod e2e_tests {

    use crate::{
        jolt::zkvm::{JoltProverPreprocessing, JoltSNARK},
        program::ONNXProgram,
    };
    use ark_bn254::Fr;
    use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
    use jolt_core::utils::transcript::KeccakTranscript;
    use log::info;
    use onnx_tracer::{custom_addsubmul_model, logger::init_logger, model, tensor::Tensor};
    use serde_json::Value;
    use std::{collections::HashMap, fs::File, io::Read};

    type PCS = DoryCommitmentScheme<KeccakTranscript>;

    #[test]
    fn test_custom_addsubmul() {
        // --- Preprocessing ---
        let custom_addsubmul_model = custom_addsubmul_model();
        let program_bytecode = onnx_tracer::decode_model(custom_addsubmul_model.clone());
        println!("Program code: {program_bytecode:#?}");
        let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);

        // --- Proving ---
        let input = Tensor::new(Some(&[60]), &[1]).unwrap();
        let execution_trace = onnx_tracer::execution_trace(custom_addsubmul_model, &input);
        println!("Execution trace: {execution_trace:#?}");
        let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prove(pp.clone(), execution_trace);

        // --- Verification ---
        snark.verify((&pp).into()).unwrap();
    }

    /// Load vocab.json into HashMap<String, (usize, i32)>
    pub fn load_vocab(
        path: &str,
    ) -> Result<HashMap<String, (usize, i32)>, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let json_value: Value = serde_json::from_str(&contents)?;
        let mut vocab = HashMap::new();

        if let Value::Object(map) = json_value {
            for (word, data) in map {
                if let (Some(index), Some(idf)) = (
                    data.get("index").and_then(|v| v.as_u64()),
                    data.get("idf").and_then(|v| v.as_i64()),
                ) {
                    vocab.insert(word, (index as usize, idf as i32));
                }
            }
        }

        Ok(vocab)
    }

    /// Tokenize and convert text to vector of length 1000
    pub fn build_input_vector(text: &str, vocab: &HashMap<String, (usize, i32)>) -> Vec<i128> {
        let mut vec = vec![0; 1000];

        // Split text into tokens (preserve punctuation as tokens)
        let re = regex::Regex::new(r"\w+|[^\w\s]").unwrap();
        for cap in re.captures_iter(text) {
            let token = cap.get(0).unwrap().as_str().to_lowercase();
            if let Some(&(index, idf)) = vocab.get(&token) {
                if index < 1000 {
                    vec[index] += idf as i128; // accumulate idf value
                }
            }
        }

        vec
    }

    #[test]

    pub fn test_article_classification_output() {
        init_logger();
        let working_dir: &str = "../onnx-tracer/models/article_classification/";

        // Load the vocab mapping from JSON
        let vocab_path = format!("{}/vocab.json", working_dir);
        let vocab = load_vocab(&vocab_path).expect("Failed to load vocab");

        // Input text string to classify
        let input_texts = [
            "The government plans new trade policies.",
            "The latest computer model has impressive features.",
            "The football match ended in a thrilling draw.",
            "The new movie has received rave reviews from critics.",
            "The stock market saw a significant drop today.",
        ];

        let expected_classes = ["politics", "tech", "sport", "entertainment", "business"];

        let mut predicted_classes = Vec::new();

        for input_text in &input_texts {
            // Build input vector from the input text
            let input_vector = build_input_vector(&input_text, &vocab);

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

            info!("Output: {}", output.show());
            info!("Predicted class: {}", classes[pred_idx]);

            predicted_classes.push(classes[pred_idx]);
        }
        // Check if predicted classes match expected classes
        for (predicted, expected) in predicted_classes.iter().zip(expected_classes.iter()) {
            assert_eq!(predicted, expected, "Mismatch in predicted class");
        }
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
