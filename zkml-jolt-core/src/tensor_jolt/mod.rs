pub mod bytecode;
pub mod execution_trace;
pub mod instruction;
pub mod instruction_lookups;
pub mod r1cs;
pub mod sparse_dense_shout;
pub mod tensor_heap;

use crate::tensor_jolt::{
    bytecode::{BytecodePreprocessing, BytecodeProof},
    execution_trace::JoltONNXCycle,
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
use onnx_tracer::{constants::MAX_TENSOR_SIZE, trace_types::ONNXInstr};
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
        trace.resize(trace.len().next_power_of_two(), JoltONNXCycle::no_op());
        let padded_trace_length = trace.len();
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
mod tests {
    use ark_bn254::Fr;
    use jolt_core::{
        poly::commitment::dory::DoryCommitmentScheme,
        utils::{
            math::Math,
            transcript::{KeccakTranscript, Transcript},
        },
    };
    use onnx_tracer::{
        constants::MAX_TENSOR_SIZE, custom_addsubmul_model, scalar_addsubmul_model, tensor::Tensor,
    };

    use crate::tensor_jolt::{
        JoltProverPreprocessing, JoltSNARK,
        execution_trace::{JoltONNXCycle, jolt_execution_trace},
        r1cs::{
            constraints::{JoltONNXConstraints, R1CSConstraints},
            spartan::UniformSpartanProof,
        },
        sparse_dense_shout::{prove_sparse_dense_shout, verify_sparse_dense_shout},
    };

    type PCS = DoryCommitmentScheme<KeccakTranscript>;

    #[test]
    fn test_custom_addsubmul() {
        // --- Preprocessing ---
        let custom_addsubmul_model = custom_addsubmul_model();
        let program_bytecode = onnx_tracer::decode_model(custom_addsubmul_model.clone());
        let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);

        // --- Proving ---
        // Get execution trace
        let input = Tensor::new(Some(&[10, 20, 30, 40]), &[1, 4]).unwrap();
        let raw_trace = onnx_tracer::execution_trace(custom_addsubmul_model, &input);
        let execution_trace = jolt_execution_trace(raw_trace);
        println!("Execution trace: {execution_trace:#?}");
        let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prove(pp.clone(), execution_trace);

        // --- Verification ---
        snark.verify((&pp).into()).unwrap();
    }

    #[test]
    fn test_scalar_addsubmul() {
        // --- Preprocessing ---
        let scalar_addsubmul_model = scalar_addsubmul_model();
        let program_bytecode = onnx_tracer::decode_model(scalar_addsubmul_model.clone());
        println!("Program code: {program_bytecode:#?}");
        let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);

        // --- Proving ---
        let input = Tensor::new(Some(&[60]), &[1]).unwrap();
        let raw_trace = onnx_tracer::execution_trace(scalar_addsubmul_model, &input);
        println!("Execution trace: {raw_trace:#?}");
        let execution_trace = jolt_execution_trace(raw_trace);

        let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prove(pp.clone(), execution_trace);

        // --- Verification ---
        snark.verify((&pp).into()).unwrap();
    }
}
