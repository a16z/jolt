#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
mod tests {
    use common::constants::MAX_BLINDFOLD_GENERATORS;
    use jolt_backends::cpu::{CpuBackend, CpuBackendConfig};
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_prover_harness::{trace_sdk_guest, SdkGuestTraceRequest};
    use jolt_transcript::Blake2bTranscript;
    use jolt_verifier::JoltVerifierPreprocessing;
    use jolt_witness::protocols::jolt_vm::{
        JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackedJoltVmWitness,
    };

    #[test]
    fn top_level_zk_prover_outputs_verify() -> Result<(), String> {
        let inputs = postcard::to_stdvec(&[9_u32, 5, 3])
            .map_err(|error| format!("serialize input: {error}"))?;
        let fixture = trace_sdk_guest(
            SdkGuestTraceRequest::new("muldiv-guest", inputs).with_max_padded_trace_length(1 << 9),
        )
        .map_err(|error| error.to_string())?;

        let one_hot = JoltOneHotConfig {
            log_k_chunk: 4,
            lookups_ra_virtual_log_k_chunk: 16,
        };
        let log_t = fixture.padded_trace_length.trailing_zeros() as usize;
        let committed_chunk_bits = one_hot.committed_chunk_bits();
        let bytecode_start = fixture
            .preprocessing
            .memory_layout
            .remapped_word_address(fixture.preprocessing.ram.min_bytecode_address)
            .map_err(|error| error.to_string())? as usize;
        let ram_k = (bytecode_start + fixture.preprocessing.ram.bytecode_words.len())
            .next_power_of_two()
            .max(1);
        let witness_config = JoltVmWitnessConfig::new(log_t, ram_k, one_hot);
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&fixture.program, &fixture.preprocessing, fixture.trace),
        );
        let max_num_vars =
            (log_t + committed_chunk_bits).max(MAX_BLINDFOLD_GENERATORS.ilog2() as usize);
        let pcs_setup = DoryScheme::setup_prover(max_num_vars);
        let verifier_preprocessing =
            JoltVerifierPreprocessing::<DoryScheme, Pedersen<Bn254G1>>::from_pcs_prover_setup(
                fixture.preprocessing.clone(),
                [7; 32],
                &pcs_setup,
                MAX_BLINDFOLD_GENERATORS,
            );
        let prover_preprocessing =
            jolt_prover::JoltProverPreprocessing::new(verifier_preprocessing.clone(), pcs_setup);
        let public_io = witness.trace.device.clone();
        let proof_shape = jolt_prover::ProverProofShape::new(
            fixture.padded_trace_length,
            ram_k,
            JoltReadWriteConfig {
                ram_rw_phase1_num_rounds: 1,
                ram_rw_phase2_num_rounds: 1,
                registers_rw_phase1_num_rounds: 1,
                registers_rw_phase2_num_rounds: 1,
            },
            one_hot,
            jolt_verifier::proof::TracePolynomialOrder::AddressMajor,
        );
        let config = jolt_prover::ProverConfig::default().with_proof_shape(proof_shape);
        let mut backend = CpuBackend::new(CpuBackendConfig {
            preserve_core_fast_path: true,
            commitment_chunk_size: 1 << 10,
        });

        let output = jolt_prover::prove_with_output::<DoryScheme, Pedersen<Bn254G1>, _, _>(
            &prover_preprocessing,
            &public_io,
            &witness,
            config,
            &mut backend,
        )
        .map_err(|error| error.to_string())?;
        assert!(output.trusted_advice_commitment.is_none());

        jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript<Fr>>(
            &verifier_preprocessing,
            &public_io,
            &output.proof,
            None,
            true,
        )
        .map_err(|error| error.to_string())?;

        Ok(())
    }
}
