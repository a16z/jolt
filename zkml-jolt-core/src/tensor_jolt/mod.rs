pub mod execution_trace;
// pub mod tensor_heap;
pub mod r1cs;
pub mod sparse_dense_shout;

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use jolt_core::utils::{
        math::Math,
        transcript::{KeccakTranscript, Transcript},
    };
    use onnx_tracer::{constants::MAX_TENSOR_SIZE, custom_addsubmul_model, tensor::Tensor};

    use crate::tensor_jolt::{
        execution_trace::{JoltONNXCycle, jolt_trace},
        sparse_dense_shout::{prove_sparse_dense_shout, verify_sparse_dense_shout},
    };

    #[test]
    fn test_custom_addsubmul() {
        // --- Preprocessing ---
        let custom_addsubmul_model = custom_addsubmul_model();
        let program_bytecode = onnx_tracer::decode_model(custom_addsubmul_model.clone());

        // --- Proving ---
        let input = Tensor::new(Some(&[10, 20, 30, 40]), &[1, 4]).unwrap();
        let raw_trace = onnx_tracer::execution_trace(custom_addsubmul_model, &input);
        let mut execution_trace = jolt_trace(raw_trace);
        execution_trace.resize(
            execution_trace.len().next_power_of_two(),
            JoltONNXCycle::no_op(),
        );
        println!("Execution trace: {execution_trace:#?}");
        let T = execution_trace.len() * MAX_TENSOR_SIZE;
        let log_T = T.log_2();
        let mut prover_transcript = KeccakTranscript::new(b"Jolt transcript");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(log_T);
        let (proof, rv_claim, ra_claims, add_mul_sub_claim, flag_claims, _) =
            prove_sparse_dense_shout::<_, _>(&execution_trace, &r_cycle, &mut prover_transcript);
        let mut verifier_transcript = KeccakTranscript::new(b"Jolt transcript");
        verifier_transcript.compare_to(prover_transcript);
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
