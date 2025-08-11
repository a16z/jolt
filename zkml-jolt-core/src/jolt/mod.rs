pub mod bytecode;
pub mod execution_trace;
pub mod instruction;
pub mod instruction_lookups;
pub mod r1cs;
pub mod tensor_heap;

use crate::jolt::{
    bytecode::{BytecodePreprocessing, BytecodeProof},
    execution_trace::JoltONNXCycle,
    instruction::{VirtualInstructionSequence, div::DIVInstruction},
    instruction_lookups::LookupsProof,
    r1cs::{
        constraints::{JoltONNXConstraints, R1CSConstraints},
        spartan::UniformSpartanProof,
    },
    tensor_heap::TensorHeapTwistProof,
};
use execution_trace::WORD_SIZE;
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, dory::DoryGlobals},
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    utils::{errors::ProofVerifyError, transcript::Transcript},
};
use onnx_tracer::{
    constants::MAX_TENSOR_SIZE,
    trace_types::{ONNXInstr, ONNXOpcode},
};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Clone, Serialize, Deserialize)]
pub struct JoltProverPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub shared: JoltSharedPreprocessing,
    field: F::SmallValueLookupTables,
    _p: PhantomData<(ProofTranscript, PCS)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoltSharedPreprocessing {
    pub bytecode: BytecodePreprocessing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoltVerifierPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    pub shared: JoltSharedPreprocessing,
    _p: PhantomData<(F, PCS, ProofTranscript)>,
}

impl<F, PCS, ProofTranscript> From<&JoltProverPreprocessing<F, PCS, ProofTranscript>>
    for JoltVerifierPreprocessing<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    fn from(preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>) -> Self {
        JoltVerifierPreprocessing {
            shared: preprocessing.shared.clone(),
            _p: PhantomData,
        }
    }
}

pub struct JoltSNARK<F, PCS, ProofTranscript>
where
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    F: JoltField,
{
    pub trace_length: usize,
    bytecode: BytecodeProof<F, ProofTranscript>,
    instruction_lookups: LookupsProof<WORD_SIZE, F, PCS, ProofTranscript>,
    tensor_heap: TensorHeapTwistProof<F, ProofTranscript>,
    r1cs: UniformSpartanProof<F, ProofTranscript>,
    _p: PhantomData<PCS>,
}

impl<F, PCS, ProofTranscript> JoltSNARK<F, PCS, ProofTranscript>
where
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    F: JoltField,
{
    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    pub fn shared_preprocess(bytecode: Vec<ONNXInstr>) -> JoltSharedPreprocessing {
        let bytecode = bytecode
            .into_iter()
            .flat_map(|instr| match instr.opcode {
                ONNXOpcode::Div => DIVInstruction::<32>::virtual_sequence(instr),
                _ => vec![instr],
            })
            .collect();
        let bytecode_preprocessing = BytecodePreprocessing::preprocess(bytecode);
        JoltSharedPreprocessing {
            bytecode: bytecode_preprocessing,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    pub fn prover_preprocess(
        bytecode: Vec<ONNXInstr>,
    ) -> JoltProverPreprocessing<F, PCS, ProofTranscript> {
        let small_value_lookup_tables = F::compute_lookup_tables();
        F::initialize_lookup_tables(small_value_lookup_tables.clone());
        let shared = Self::shared_preprocess(bytecode);
        JoltProverPreprocessing {
            shared,
            field: small_value_lookup_tables,
            _p: PhantomData,
        }
    }

    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    pub fn prove(
        mut preprocessing: JoltProverPreprocessing<F, PCS, ProofTranscript>,
        mut trace: Vec<JoltONNXCycle>,
    ) -> Self {
        let trace_length = trace.len();
        println!("Trace length: {trace_length}");
        F::initialize_lookup_tables(std::mem::take(&mut preprocessing.field));
        // pad trace to the next power of two
        let padded_trace_length = trace_length.next_power_of_two();
        let padding = padded_trace_length - trace_length;
        let last_address = trace.last().unwrap().instr().address;
        if padding != 0 {
            // Pad with NoOps (with sequential addresses)
            trace.extend((0..padding - 1).map(|i| {
                let mut no_op = JoltONNXCycle::no_op();
                no_op.instr.address = last_address + i + 1;
                no_op
            }));

            // HACK(Forpee): Not sure if this is correct. RV pushes a jump instr:
            // ```
            // // Final JALR sets NextUnexpandedPC = 0
            // trace.push(RV32IMCycle::last_jalr(last_address + 4 * (padding - 1)));
            // ```
            trace.push(JoltONNXCycle::no_op());
        };

        let tensor_heap_addresses: Vec<usize> = trace
            .iter()
            .map(|cycle| cycle.td_write().0.last().unwrap() + 1)
            .collect();
        let tensor_heap_K = tensor_heap_addresses
            .iter()
            .max()
            .unwrap()
            .next_power_of_two();
        let K = [
            preprocessing.shared.bytecode.code_size,
            tensor_heap_K,
            1 << 16, // K for instruction lookups Shout
        ]
        .into_iter()
        .max()
        .unwrap();
        println!("T = {padded_trace_length}, K = {K}");
        let _guard = DoryGlobals::initialize(K, padded_trace_length);
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: ProverOpeningAccumulator<F, PCS, ProofTranscript> =
            ProverOpeningAccumulator::new();
        let constraint_builder = JoltONNXConstraints::construct_constraints(padded_trace_length);
        let spartan_key = UniformSpartanProof::<F, ProofTranscript>::setup(
            &constraint_builder,
            padded_trace_length,
        );
        transcript.append_scalar(&spartan_key.vk_digest);
        let r1cs_snark = UniformSpartanProof::prove::<PCS>(
            &preprocessing,
            &constraint_builder,
            &spartan_key,
            &trace,
            &mut transcript,
        )
        .ok()
        .unwrap();
        let instruction_lookups_snark: LookupsProof<WORD_SIZE, F, PCS, ProofTranscript> =
            LookupsProof::prove(
                &preprocessing,
                &trace,
                &mut opening_accumulator,
                &mut transcript,
            );
        let bytecode_snark =
            BytecodeProof::prove(&preprocessing.shared.bytecode, &trace, &mut transcript);
        let tensor_heap_snark = TensorHeapTwistProof::prove(
            &preprocessing,
            &trace,
            tensor_heap_K,
            &mut opening_accumulator,
            &mut transcript,
        );
        JoltSNARK {
            trace_length,
            r1cs: r1cs_snark,
            tensor_heap: tensor_heap_snark,
            instruction_lookups: instruction_lookups_snark,
            bytecode: bytecode_snark,
            _p: PhantomData,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn verify(
        &self,
        preprocessing: JoltVerifierPreprocessing<F, PCS, ProofTranscript>,
    ) -> Result<(), ProofVerifyError> {
        let mut transcript = ProofTranscript::new(b"Jolt transcript");
        let mut opening_accumulator: VerifierOpeningAccumulator<F, PCS, ProofTranscript> =
            VerifierOpeningAccumulator::new();
        // Regenerate the uniform Spartan key
        let padded_trace_length = self.trace_length.next_power_of_two();
        let r1cs_builder = JoltONNXConstraints::construct_constraints(padded_trace_length);
        let spartan_key =
            UniformSpartanProof::<F, ProofTranscript>::setup(&r1cs_builder, padded_trace_length);
        transcript.append_scalar(&spartan_key.vk_digest);
        self.r1cs
            .verify::<PCS>(&spartan_key, &mut transcript)
            .map_err(|e| ProofVerifyError::SpartanError(e.to_string()))?;
        self.instruction_lookups
            .verify(&mut opening_accumulator, &mut transcript)?;
        self.bytecode.verify(
            &preprocessing.shared.bytecode,
            padded_trace_length,
            &mut transcript,
        )?;
        self.tensor_heap.verify(
            padded_trace_length * MAX_TENSOR_SIZE,
            &mut opening_accumulator,
            &mut transcript,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod e2e_tests {
    use crate::{
        jolt::{
            JoltProverPreprocessing, JoltSNARK,
            execution_trace::{check_mcc, jolt_execution_trace},
        },
        program::ONNXProgram,
    };
    use ark_bn254::Fr;
    use jolt_core::{
        poly::commitment::dory::DoryCommitmentScheme, utils::transcript::KeccakTranscript,
    };
    use log::{debug, info};
    use onnx_tracer::{builder, logger::init_logger, model, tensor::Tensor};
    use serde_json::Value;
    use serial_test::serial;
    use std::{collections::HashMap, fs::File, io::Read};

    type PCS = DoryCommitmentScheme<KeccakTranscript>;

    // TODO: Refactor duplicate code in tests

    /*
        vocab.json:
        {
            "i": 1,
            "love": 2,
            "this": 3,
            "is": 4,
            "great": 5,
            "happy": 6,
            "with": 7,
            "the": 8,
            "result": 9,
            "hate": 10,
            "bad": 11,
            "not": 12,
            "satisfied": 13
        }
    */

    /// const: [I, love, this, 0, 0]
    const I_LOVE_THIS: [i128; 5] = [1, 2, 3, 0, 0];

    /// const: [I, hate, this, 0, 0]
    const I_HATE_THIS: [i128; 5] = [1, 10, 3, 0, 0];

    /// const: [This, is, great, 0, 0]
    const THIS_IS_GREAT: [i128; 5] = [3, 4, 5, 0, 0];

    /// const: [This, is, bad, 0, 0]
    const THIS_IS_BAD: [i128; 5] = [3, 4, 11, 0, 0];

    const TEST_SENTIMENT_INPUTS: [[i128; 5]; 4] =
        [I_LOVE_THIS, I_HATE_THIS, THIS_IS_GREAT, THIS_IS_BAD];

    const TEST_SENTIMENT_OUTPUTS: [i128; 4] = [1, 0, 1, 0];

    #[test]
    #[serial]
    fn test_custom_sentiment_sum() {
        init_logger();

        // Load model
        let mut sentiment_model = builder::embedding_sentiment_model();
        let program_bytecode = onnx_tracer::decode_model(sentiment_model.clone());
        debug!("Program code: {program_bytecode:#?}");
        let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);

        // Run inference
        for (i, input) in TEST_SENTIMENT_INPUTS.iter().enumerate() {
            let result = sentiment_model
                .forward(&[Tensor::new(Some(input), &[1, 5]).unwrap()])
                .unwrap();
            let output = result.outputs[0].clone();
            assert_eq!(output.inner[0], TEST_SENTIMENT_OUTPUTS[i]);
        }
        sentiment_model.clear_execution_trace();

        let raw_trace = onnx_tracer::execution_trace(
            sentiment_model,
            &Tensor::new(Some(&THIS_IS_GREAT), &[1, 5]).unwrap(),
        );
        info!("Raw trace: {raw_trace:#?}");

        // Prove
        let execution_trace = jolt_execution_trace(raw_trace);
        info!("Execution trace: {execution_trace:#?}");
        info!("Execution trace length: {}", execution_trace.len());
        check_mcc(&execution_trace);

        // let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
        //     JoltSNARK::prove(pp.clone(), execution_trace);

        // // Verify
        // snark.verify((&pp).into()).unwrap();
    }

    #[test]
    fn test_sentiment_sum() {
        let input_vector = THIS_IS_BAD;

        let text_classification = ONNXProgram {
            model_path: "../onnx-tracer/models/sentiment_sum/network.onnx".into(),
            inputs: Tensor::new(Some(&input_vector), &[1, 5]).unwrap(), // Example input
        };
        let program_bytecode = text_classification.decode();
        info!("Program code: {program_bytecode:#?}",);

        // Load model
        let model = model(&text_classification.model_path);

        // Run inference
        for (i, input) in TEST_SENTIMENT_INPUTS.iter().enumerate() {
            let result = model
                .forward(&[Tensor::new(Some(input), &[1, 5]).unwrap()])
                .unwrap();
            let output = result.outputs[0].clone();
            assert_eq!(
                output.inner[0], TEST_SENTIMENT_OUTPUTS[i],
                "Input: {input:?}, Output: {output:?}, Expected: {}",
                TEST_SENTIMENT_OUTPUTS[i]
            );
        }

        // let result = model
        //     .forward(&[text_classification.inputs.clone()])
        //     .unwrap();
        // let output = result.outputs[0].clone();
        // info!("Output: {output:#?}",);

        // let raw_trace = text_classification.trace();
        // debug!("Raw trace: {raw_trace:#?}",);
    }

    #[test]
    fn test_sentiment_simple() {
        let mut input_vector = vec![3, 4, 5, 0, 0];
        input_vector.resize(5, 0); // Resize to match the input shape

        let text_classification = ONNXProgram {
            model_path: "../onnx-tracer/models/sentiment_simple/network.onnx".into(),
            inputs: Tensor::new(Some(&input_vector), &[1, 5]).unwrap(), // Example input
        };
        let program_bytecode = text_classification.decode();
        info!("Program code: {program_bytecode:#?}",);

        // Load model
        let model = model(&text_classification.model_path);

        // Run inference
        let result = model
            .forward(&[text_classification.inputs.clone()])
            .unwrap();
        let output = result.outputs[0].clone();
        info!("Output: {output:#?}",);

        let raw_trace = text_classification.trace();
        info!("Raw trace: {raw_trace:#?}",);
    }

    #[serial]
    #[test]
    fn test_addsubmuldivdiv() {
        // --- Preprocessing ---
        let custom_addsubmul_model = builder::custom_addsubmuldivdiv_model();
        let program_bytecode = onnx_tracer::decode_model(custom_addsubmul_model.clone());
        debug!("Program code: {program_bytecode:#?}");
        let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);

        // --- Proving ---
        // Get execution trace
        let input = Tensor::new(Some(&[10, 20, 30, 40]), &[1, 4]).unwrap();
        let raw_trace = onnx_tracer::execution_trace(custom_addsubmul_model, &input);
        let execution_trace = jolt_execution_trace(raw_trace);
        let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prove(pp.clone(), execution_trace);

        // --- Verification ---
        snark.verify((&pp).into()).unwrap();
    }

    #[serial]
    #[test]
    fn test_addsubmuldiv() {
        // --- Preprocessing ---
        let custom_addsubmul_model = builder::custom_addsubmuldiv_model();
        let program_bytecode = onnx_tracer::decode_model(custom_addsubmul_model.clone());
        // debug!("Program code: {program_bytecode:#?}");
        let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);

        // --- Proving ---
        // Get execution trace
        let input = Tensor::new(Some(&[10, 20, 30, 40]), &[1, 4]).unwrap();
        let raw_trace = onnx_tracer::execution_trace(custom_addsubmul_model, &input);
        // debug!("raw trace: {raw_trace:#?}");
        let execution_trace = jolt_execution_trace(raw_trace);
        let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prove(pp.clone(), execution_trace);

        // --- Verification ---
        snark.verify((&pp).into()).unwrap();
    }

    #[serial]
    #[test]
    fn test_custom_addsubmulconst() {
        // --- Preprocessing ---
        let custom_addsubmul_model = builder::custom_addsubmulconst_model();
        let program_bytecode = onnx_tracer::decode_model(custom_addsubmul_model.clone());
        // debug!("Program code: {program_bytecode:#?}");
        let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);

        // --- Proving ---
        // Get execution trace
        let input = Tensor::new(Some(&[10, 20, 30, 40]), &[1, 4]).unwrap();
        let raw_trace = onnx_tracer::execution_trace(custom_addsubmul_model, &input);
        // debug!("raw trace: {raw_trace:#?}");
        let execution_trace = jolt_execution_trace(raw_trace);
        debug!("Execution trace: {execution_trace:#?}");
        let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prove(pp.clone(), execution_trace);

        // --- Verification ---
        snark.verify((&pp).into()).unwrap();
    }

    #[serial]
    #[test]
    fn test_custom_addsubmul() {
        // --- Preprocessing ---
        let custom_addsubmul_model = builder::custom_addsubmul_model();
        let program_bytecode = onnx_tracer::decode_model(custom_addsubmul_model.clone());
        debug!("Program code: {program_bytecode:#?}");
        let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);

        // --- Proving ---
        // Get execution trace
        let input = Tensor::new(Some(&[10, 20, 30, 40]), &[1, 4]).unwrap();
        let raw_trace = onnx_tracer::execution_trace(custom_addsubmul_model, &input);
        debug!("raw trace: {raw_trace:#?}");
        let execution_trace = jolt_execution_trace(raw_trace);
        let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prove(pp.clone(), execution_trace);

        // --- Verification ---
        snark.verify((&pp).into()).unwrap();
    }

    #[serial]
    #[test]
    fn test_scalar_addsubmul() {
        // --- Preprocessing ---
        let scalar_addsubmul_model = builder::scalar_addsubmul_model();
        let program_bytecode = onnx_tracer::decode_model(scalar_addsubmul_model.clone());
        debug!("Program code: {program_bytecode:#?}");
        let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);

        // --- Proving ---
        let input = Tensor::new(Some(&[60]), &[1]).unwrap();
        let raw_trace = onnx_tracer::execution_trace(scalar_addsubmul_model, &input);
        debug!("Execution trace: {raw_trace:#?}");
        let execution_trace = jolt_execution_trace(raw_trace);

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
        let working_dir: &str = "../onnx-tracer/models/article_classification/";

        // Load the vocab mapping from JSON
        let vocab_path = format!("{working_dir}/vocab.json",);
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
            let input_vector = build_input_vector(input_text, &vocab);

            // Prepare ONNX program
            let text_classification = ONNXProgram {
                model_path: format!("{working_dir}network.onnx").into(),
                inputs: Tensor::new(Some(&input_vector), &[1, 1000]).unwrap(),
            };

            // Decode to program bytecode (for EZKL use)
            let program_bytecode = text_classification.decode();
            debug!("Program code: {program_bytecode:#?}");

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
        let mut input_vector = vec![846, 3, 195, 4, 374, 14, 259];
        input_vector.resize(100, 0); // Resize to match the input shape

        let text_classification = ONNXProgram {
            model_path: "../onnx-tracer/models/medium_text_classification/network.onnx".into(),
            inputs: Tensor::new(Some(&input_vector), &[1, 100]).unwrap(), // Example input
        };
        let program_bytecode = text_classification.decode();
        debug!("Program code: {program_bytecode:#?}",);
        text_classification.trace();
    }

    #[test]
    fn test_medium_classification_output() {
        let mut input_vector = vec![197, 10, 862, 8, 23, 53, 2, 319, 34, 122, 100, 53, 33];
        input_vector.resize(100, 0); // Resize to match the input shape

        let text_classification = ONNXProgram {
            model_path: "../onnx-tracer/models/medium_text_classification/network.onnx".into(),
            inputs: Tensor::new(Some(&input_vector), &[1, 100]).unwrap(), // Example input
        };
        let program_bytecode = text_classification.decode();
        debug!("Program code: {program_bytecode:#?}",);
        let model = model(&text_classification.model_path);

        let result = model
            .forward(&[text_classification.inputs.clone()])
            .unwrap();
        let output = result.outputs[0].clone();
        info!("Output: {output:#?}",);
    }

    #[should_panic(expected = "not yet implemented")]
    #[test]
    fn test_subgraph() {
        let subgraph_program = ONNXProgram {
            model_path: "../onnx-tracer/models/subgraph/network.onnx".into(),
            inputs: Tensor::new(Some(&[1, 2, 3, 4]), &[1, 4]).unwrap(), // Example input
        };
        let program_bytecode = subgraph_program.decode();

        debug!("Program decoded");
        debug!("Program code: {program_bytecode:#?}",);

        // Test that the addresses of a subgraph are monotonically increasing
        let mut i = 0;
        for instr in program_bytecode {
            assert!(instr.address > i);
            i = instr.address;
        }

        subgraph_program.trace();
    }
}
