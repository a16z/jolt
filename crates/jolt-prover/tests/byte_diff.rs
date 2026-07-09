//! The stage-granular byte-diff harness against `jolt-prover-legacy`.
//!
//! Both provers run from the same guest program and inputs. Legacy's
//! per-stage-boundary transcript states are recovered WITHOUT instrumenting
//! legacy: its accepted proof is replayed through the verifier
//! (`verify_until_stage1`, then stage verifies as stages land), whose
//! Fiat-Shamir transcript is byte-identical to the prover's by soundness of
//! the accepted proof. The harness ratchets one stage at a time; today it
//! pins stage 0 (config derivation, preamble, witness commitments).

#[cfg(feature = "prover-fixtures")]
#[expect(clippy::expect_used)]
mod stage0 {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_program::execution::{
        ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
    };
    use jolt_prover::stages::stage0::prove_stage0;
    use jolt_prover::{JoltProverPreprocessing, ProverConfig};
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
    use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_prover_legacy::zkvm::RV64IMACProver;
    use jolt_transcript::{LegacyBlake2bTranscript as Blake2bTranscript, Transcript};
    use jolt_verifier::verify_until_stage1;
    use jolt_witness::protocols::jolt_vm::{
        JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackedJoltVmWitness,
    };
    use tracer::execution_backend::TracerBackend;

    use common::jolt_device::MemoryConfig;

    const MAX_PADDED_TRACE_LENGTH: usize = 1 << 16;

    /// Prove muldiv with both provers from the same guest and inputs; assert
    /// the new prover's stage-0 transcript state and commitment bytes equal
    /// legacy's.
    #[test]
    fn stage0_matches_legacy_on_muldiv() {
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");

        // --- Legacy side: preprocess, prove, and recover the pre-stage-1
        // transcript state by replaying the proof through the verifier.
        let (bytecode, init_memory_state, _, entry_address) = program.decode();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);
        let elf_contents = program.get_elf_contents().expect("elf contents");

        let legacy_program =
            LegacyProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
                .expect("legacy preprocess");
        let shared = JoltSharedPreprocessing::new(
            legacy_program,
            io_device.memory_layout.clone(),
            MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing = LegacyProverPreprocessing::new(shared);
        let legacy_prover = RV64IMACProver::gen_from_elf(
            &legacy_preprocessing,
            &elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
            None,
        );
        let public_io = legacy_prover.program_io.clone();
        let (legacy_proof, _) = legacy_prover.prove().expect("legacy prove");
        let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);

        let legacy_pre_stage1 =
            verify_until_stage1::<DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                &verifier_preprocessing,
                &public_io,
                &legacy_proof,
                None,
                false,
            )
            .expect("legacy proof must verify through stage 0");

        // --- New-prover side: trace independently through the modular stack.
        let jolt_program = JoltProgram::from_elf_bytes(elf_contents);
        let memory_layout = &public_io.memory_layout;
        let memory_config = MemoryConfig {
            max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
            max_trusted_advice_size: memory_layout.max_trusted_advice_size,
            max_input_size: memory_layout.max_input_size,
            max_output_size: memory_layout.max_output_size,
            stack_size: memory_layout.stack_size,
            heap_size: memory_layout.heap_size,
            program_size: Some(memory_layout.program_size),
        };
        let trace_output = TracerBackend::new()
            .trace(
                &jolt_program,
                TraceInputs {
                    inputs: inputs.clone(),
                    untrusted_advice: Vec::new(),
                    trusted_advice: Vec::new(),
                    memory_config,
                },
            )
            .expect("modular trace");

        let program_preprocessing = verifier_preprocessing
            .program
            .as_full()
            .expect("full program preprocessing")
            .clone();

        let config = ProverConfig::derive::<Fr>(
            trace_output.trace.rows(),
            memory_layout,
            verifier_preprocessing.program.min_bytecode_address(),
            verifier_preprocessing.program.program_image_len_words(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive config");

        // The derived proof shape must equal what legacy wrote on the wire.
        assert_eq!(config.trace_length, legacy_proof.trace_length);
        assert_eq!(config.ram_K, legacy_proof.ram_K);
        assert_eq!(config.rw_config, legacy_proof.rw_config);
        assert_eq!(config.one_hot_config, legacy_proof.one_hot_config);
        assert_eq!(
            config.trace_polynomial_order,
            legacy_proof.trace_polynomial_order
        );

        // Pad to the padded trace length with no-op rows, as legacy does.
        let mut rows = trace_output.trace.rows().to_vec();
        rows.resize(config.trace_length, TraceRow::default());
        let padded_output = TraceOutput::new(
            OwnedTrace::new(rows),
            trace_output.device,
            trace_output.final_memory,
        );

        let witness_config = JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        );
        let witness = TraceBackedJoltVmWitness::new(
            witness_config,
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
        );

        let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
            verifier: verifier_preprocessing,
            pcs_setup: DoryScheme::setup_prover(setup_total_vars(memory_layout)),
        };

        let stage0 = prove_stage0::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
            &prover_preprocessing,
            &config,
            &witness,
            &public_io,
        )
        .expect("stage 0 proves");

        // The ratchet: commitment bytes, then the stage-boundary transcript state.
        assert_eq!(stage0.commitments, legacy_proof.commitments);
        assert_eq!(
            stage0.transcript.state(),
            legacy_pre_stage1.transcript.state(),
            "stage-0 transcript state diverged from legacy",
        );
        assert_eq!(
            stage0.hints.len(),
            2 + legacy_proof.commitments.instruction_ra.len()
                + legacy_proof.commitments.ram_ra.len()
                + legacy_proof.commitments.bytecode_ra.len(),
        );
    }

    /// The PCS setup sizing legacy uses: the maximum embedding over the
    /// largest supported trace and both advice candidates (always included in
    /// setup sizing, present or not).
    fn setup_total_vars(memory_layout: &common::jolt_device::MemoryLayout) -> usize {
        let max_log_t = MAX_PADDED_TRACE_LENGTH.ilog2() as usize;
        let max_log_k_chunk = 4usize; // max_log_t = 16 < the 25-bit threshold
        let advice =
            |bytes: u64| ((bytes / 8) as usize).next_power_of_two().max(1).ilog2() as usize;
        (max_log_k_chunk + max_log_t)
            .max(advice(memory_layout.max_trusted_advice_size))
            .max(advice(memory_layout.max_untrusted_advice_size))
    }
}

#[cfg(not(feature = "prover-fixtures"))]
#[test]
#[ignore = "enable --features prover-fixtures to run the legacy byte-diff harness"]
fn stage0_matches_legacy_on_muldiv() {}
