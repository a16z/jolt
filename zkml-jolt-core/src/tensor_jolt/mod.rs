pub mod execution_trace;
// pub mod tensor_heap;
pub mod bytecode;
pub mod instruction;
pub mod instruction_lookups;
pub mod r1cs;
pub mod sparse_dense_shout;

use crate::tensor_jolt::{
    bytecode::{BytecodePreprocessing, BytecodeProof},
    execution_trace::JoltONNXCycle,
    instruction_lookups::LookupsProof,
    r1cs::{
        constraints::{JoltONNXConstraints, R1CSConstraints},
        spartan::UniformSpartanProof,
    },
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
use onnx_tracer::trace_types::{ONNXCycle, ONNXInstr};
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
    // registers: RegistersTwistProof<F, ProofTranscript>,
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
        let ram_addresses: Vec<usize> = trace.iter().map(|cycle| cycle.td_write().0 + 1).collect();
        let ram_K = ram_addresses.iter().max().unwrap().next_power_of_two();
        let K = [
            preprocessing.shared.bytecode.code_size,
            ram_K,
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
        // let registers_snark = RegistersTwistProof::prove(
        //     &preprocessing,
        //     &trace,
        //     ram_K,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // );
        let bytecode_snark =
            BytecodeProof::prove(&preprocessing.shared.bytecode, &trace, &mut transcript);
        JoltSNARK {
            trace_length,
            r1cs: r1cs_snark,
            // registers: registers_snark,
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
        // self.registers.verify(
        //     padded_trace_length,
        //     &mut opening_accumulator,
        //     &mut transcript,
        // )?;
        self.bytecode.verify(
            &preprocessing.shared.bytecode,
            self.trace_length,
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
    use onnx_tracer::{constants::MAX_TENSOR_SIZE, custom_addsubmul_model, tensor::Tensor};

    use crate::{
        tensor_jolt::{JoltProverPreprocessing, JoltSNARK},
        tensor_jolt::{
            execution_trace::{JoltONNXCycle, jolt_trace},
            r1cs::{
                constraints::{JoltONNXConstraints, R1CSConstraints},
                spartan::UniformSpartanProof,
            },
            sparse_dense_shout::{prove_sparse_dense_shout, verify_sparse_dense_shout},
        },
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
        let mut execution_trace = jolt_trace(raw_trace);
        println!("Execution trace: {execution_trace:#?}");
        let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prove(pp.clone(), execution_trace);

        // --- Verification ---
        snark.verify((&pp).into()).unwrap();
    }

    #[test]
    fn test_custom_addsubmul_expanded() {
        // --- Preprocessing ---
        let custom_addsubmul_model = custom_addsubmul_model();
        let program_bytecode = onnx_tracer::decode_model(custom_addsubmul_model.clone());
        let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
            JoltSNARK::prover_preprocess(program_bytecode);

        // --- Proving ---
        // Get execution trace
        let input = Tensor::new(Some(&[10, 20, 30, 40]), &[1, 4]).unwrap();
        let raw_trace = onnx_tracer::execution_trace(custom_addsubmul_model, &input);
        let mut execution_trace = jolt_trace(raw_trace);
        execution_trace.resize(
            execution_trace.len().next_power_of_two(),
            JoltONNXCycle::no_op(),
        );
        let padded_trace_length = execution_trace.len();
        println!("Execution trace: {execution_trace:#?}");
        // Setup transcript
        let mut prover_transcript = KeccakTranscript::new(b"Jolt transcript");
        // --- Execution proof ---
        // --- Spartan ---
        let constraint_builder = JoltONNXConstraints::construct_constraints(padded_trace_length);
        let spartan_key = UniformSpartanProof::<Fr, KeccakTranscript>::setup(
            &constraint_builder,
            padded_trace_length,
        );
        prover_transcript.append_scalar(&spartan_key.vk_digest);
        let r1cs_snark = UniformSpartanProof::prove::<PCS>(
            &pp,
            &constraint_builder,
            &spartan_key,
            &execution_trace,
            &mut prover_transcript,
        )
        .ok()
        .unwrap();
        // --- Instruction lookups snark ---
        let T = execution_trace.len() * MAX_TENSOR_SIZE;
        let log_T = T.log_2();
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(log_T);
        let (proof, rv_claim, ra_claims, add_mul_sub_claim, flag_claims, _) =
            prove_sparse_dense_shout::<_, _>(&execution_trace, &r_cycle, &mut prover_transcript);

        // --- Verification ---
        // Setup transcript
        let mut verifier_transcript = KeccakTranscript::new(b"Jolt transcript");
        verifier_transcript.compare_to(prover_transcript);
        verifier_transcript.append_scalar(&spartan_key.vk_digest);
        // --- Execution proof ---
        // --- Spartan ---
        r1cs_snark
            .verify::<PCS>(&spartan_key, &mut verifier_transcript)
            .unwrap();
        // --- Instruction lookups snark ---
        let r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(log_T);
        let verification_result = verify_sparse_dense_shout::<32, _, _>(
            &proof,
            log_T,
            r_cycle,
            rv_claim,
            ra_claims,
            add_mul_sub_claim,
            &flag_claims,
            &mut verifier_transcript,
        );
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }
}
