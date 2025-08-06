pub mod execution_trace;
// pub mod tensor_heap;
pub mod r1cs;
pub mod sparse_dense_shout;

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
        jolt::{JoltProverPreprocessing, JoltSNARK},
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
