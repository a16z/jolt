//! The stage-granular byte-diff harness against `jolt-prover-legacy`.
//!
//! Both provers run from the same guest program and inputs. Legacy's
//! per-stage-boundary transcript states are recovered WITHOUT instrumenting
//! legacy: its accepted proof is replayed through the verifier
//! (`verify_until_stage1`, then stage verifies as stages land), whose
//! Fiat-Shamir transcript is byte-identical to the prover's by soundness of
//! the accepted proof. The harness ratchets one stage at a time; today it
//! pins stages 0 through 6b (config derivation, preamble, witness
//! commitments, both uni-skips, all seven sumcheck batches, all claims, and
//! every stage-boundary transcript state).

#[cfg(feature = "prover-fixtures")]
#[expect(clippy::expect_used, clippy::panic)]
mod muldiv {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_program::execution::{
        ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
    };
    use jolt_prover::stages::stage0::prove_stage0;
    use jolt_prover::stages::stage1::prove_stage1;
    use jolt_prover::stages::stage2::prove_stage2;
    use jolt_prover::stages::stage3::prove_stage3;
    use jolt_prover::stages::stage4::prove_stage4;
    use jolt_prover::stages::stage5::prove_stage5;
    use jolt_prover::stages::stage6a::prove_stage6a;
    use jolt_prover::stages::stage6b::prove_stage6b;
    use jolt_prover::{JoltBackend, JoltProverPreprocessing, ProverConfig};
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
    /// byte equality of every proof component and stage-boundary transcript
    /// state the ratchet covers.
    #[test]
    fn prover_matches_legacy_on_muldiv() {
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

        let jolt_verifier::proof::JoltProofClaims::Clear(legacy_claims) = &legacy_proof.claims
        else {
            panic!("legacy transparent proof must carry clear claims");
        };

        // The per-backend ratchet: proof bytes must be identical to legacy's
        // for ANY backend (spec invariant 8) — run once per backend under
        // test, against the legacy oracle computed above.
        let assert_backend_matches_legacy = |backend: &JoltBackend<Fr, DoryScheme>| {
            let mut session = backend.begin_proof();
            let stage0 = prove_stage0::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                backend,
                &mut session,
                &prover_preprocessing,
                &config,
                &witness,
                &public_io,
            )
            .expect("stage 0 proves");

            // The stage-0 ratchet: commitment bytes, then the stage-boundary
            // transcript state.
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

            // The stage-1 ratchet: prove, then replay legacy's proof through
            // the verifier's stage 1 for the boundary state.
            let mut new_transcript = stage0.transcript;
            let stage1 = prove_stage1::<Fr, DoryScheme, Bn254G1, Blake2bTranscript>(
                backend,
                &mut session,
                config.trace_length.ilog2() as usize,
                &witness,
                &mut new_transcript,
            )
            .expect("stage 1 proves");

            assert_eq!(
                stage1.uniskip_proof, legacy_proof.stages.stage1_uni_skip_first_round_proof,
                "stage-1 uni-skip proof bytes diverged from legacy",
            );
            assert_eq!(
                stage1.sumcheck_proof, legacy_proof.stages.stage1_sumcheck_proof,
                "stage-1 sumcheck proof bytes diverged from legacy",
            );
            assert_eq!(stage1.claims, legacy_claims.stage1);

            let mut legacy_transcript = legacy_pre_stage1.transcript.clone();
            let legacy_stage1 = jolt_verifier::stages::stage1::verify(
                &legacy_pre_stage1.checked,
                &legacy_proof,
                &mut legacy_transcript,
            )
            .expect("legacy proof must verify through stage 1");
            assert_eq!(
                new_transcript.state(),
                legacy_transcript.state(),
                "stage-1 transcript state diverged from legacy",
            );

            // The stage-2 ratchet: prove, then replay legacy's proof through
            // the verifier's stage 2 for the boundary state.
            let stage2 = prove_stage2::<Fr, DoryScheme, Bn254G1, Blake2bTranscript>(
                backend,
                &mut session,
                &config,
                &public_io,
                &stage1.clear_output,
                &witness,
                &mut new_transcript,
            )
            .expect("stage 2 proves");

            assert_eq!(
                stage2.uniskip_proof, legacy_proof.stages.stage2_uni_skip_first_round_proof,
                "stage-2 uni-skip proof bytes diverged from legacy",
            );
            assert_eq!(
                stage2.sumcheck_proof, legacy_proof.stages.stage2_sumcheck_proof,
                "stage-2 sumcheck proof bytes diverged from legacy",
            );
            assert_eq!(stage2.claims, legacy_claims.stage2);

            let legacy_stage2 = jolt_verifier::stages::stage2::verify(
                &legacy_pre_stage1.checked,
                &legacy_proof,
                &mut legacy_transcript,
                &legacy_stage1,
            )
            .expect("legacy proof must verify through stage 2");
            assert_eq!(
                new_transcript.state(),
                legacy_transcript.state(),
                "stage-2 transcript state diverged from legacy",
            );

            // The stage-3 ratchet: prove, then replay legacy's proof through
            // the verifier's stage 3 for the boundary state.
            let stage3 = prove_stage3::<Fr, DoryScheme, Bn254G1, Blake2bTranscript>(
                backend,
                &mut session,
                &config,
                &stage1.clear_output,
                &stage2.clear_output,
                &witness,
                &mut new_transcript,
            )
            .expect("stage 3 proves");

            assert_eq!(
                stage3.sumcheck_proof, legacy_proof.stages.stage3_sumcheck_proof,
                "stage-3 sumcheck proof bytes diverged from legacy",
            );
            assert_eq!(stage3.claims, legacy_claims.stage3);

            let legacy_stage3 = jolt_verifier::stages::stage3::verify(
                &legacy_pre_stage1.checked,
                &legacy_proof,
                &mut legacy_transcript,
                &legacy_stage1,
                &legacy_stage2,
            )
            .expect("legacy proof must verify through stage 3");
            assert_eq!(
                new_transcript.state(),
                legacy_transcript.state(),
                "stage-3 transcript state diverged from legacy",
            );

            // The stage-4 ratchet: prove, then replay legacy's proof through
            // the verifier's stage 4 for the boundary state.
            let stage4 =
                prove_stage4::<Fr, DoryScheme, Pedersen<Bn254G1>, Bn254G1, Blake2bTranscript>(
                    backend,
                    &mut session,
                    &legacy_pre_stage1.checked,
                    &config,
                    &prover_preprocessing,
                    &stage2.clear_output,
                    &stage3.clear_output,
                    &witness,
                    &mut new_transcript,
                )
                .expect("stage 4 proves");

            assert_eq!(
                stage4.sumcheck_proof, legacy_proof.stages.stage4_sumcheck_proof,
                "stage-4 sumcheck proof bytes diverged from legacy",
            );
            assert_eq!(stage4.claims, legacy_claims.stage4);

            let legacy_stage4 = jolt_verifier::stages::stage4::verify(
                &legacy_pre_stage1.checked,
                &prover_preprocessing.verifier,
                &legacy_proof,
                &mut legacy_transcript,
                &legacy_stage2,
                &legacy_stage3,
            )
            .expect("legacy proof must verify through stage 4");
            assert_eq!(
                new_transcript.state(),
                legacy_transcript.state(),
                "stage-4 transcript state diverged from legacy",
            );

            // The stage-5 ratchet: prove, then replay legacy's proof through
            // the verifier's stage 5 for the boundary state.
            let stage5 =
                prove_stage5::<Fr, DoryScheme, Pedersen<Bn254G1>, Bn254G1, Blake2bTranscript, _>(
                    backend,
                    &mut session,
                    &legacy_pre_stage1.checked,
                    &config,
                    &prover_preprocessing,
                    &stage2.clear_output,
                    &stage4.clear_output,
                    &witness,
                    &mut new_transcript,
                )
                .expect("stage 5 proves");

            assert_eq!(
                stage5.sumcheck_proof, legacy_proof.stages.stage5_sumcheck_proof,
                "stage-5 sumcheck proof bytes diverged from legacy",
            );
            assert_eq!(stage5.claims, legacy_claims.stage5);

            let formula_dimensions = jolt_verifier::stages::build_formula_dimensions(
                &legacy_proof,
                &prover_preprocessing.verifier,
                &legacy_pre_stage1.checked,
                config.trace_length.ilog2() as usize,
                jolt_claims::protocols::jolt::JoltRelationId::InstructionReadRaf,
            )
            .expect("legacy formula dimensions");
            let legacy_stage5 = jolt_verifier::stages::stage5::verify(
                &legacy_pre_stage1.checked,
                &legacy_proof,
                &formula_dimensions,
                &mut legacy_transcript,
                &legacy_stage2,
                &legacy_stage4,
            )
            .expect("legacy proof must verify through stage 5");
            assert_eq!(
                new_transcript.state(),
                legacy_transcript.state(),
                "stage-5 transcript state diverged from legacy",
            );

            // The stage-6a ratchet: prove, then replay legacy's proof through
            // the verifier's stage 6a for the boundary state.
            let stage6a =
                prove_stage6a::<Fr, DoryScheme, Pedersen<Bn254G1>, Bn254G1, Blake2bTranscript, _>(
                    backend,
                    &mut session,
                    &legacy_pre_stage1.checked,
                    &config,
                    &prover_preprocessing,
                    &stage1.clear_output,
                    &stage2.clear_output,
                    &stage3.clear_output,
                    &stage4.clear_output,
                    &stage5.clear_output,
                    &witness,
                    &mut new_transcript,
                )
                .expect("stage 6a proves");

            assert_eq!(
                stage6a.sumcheck_proof, legacy_proof.stages.stage6a_sumcheck_proof,
                "stage-6a sumcheck proof bytes diverged from legacy",
            );
            assert_eq!(stage6a.claims, legacy_claims.stage6a);

            let legacy_stage6a = jolt_verifier::stages::stage6a::verify(
                &legacy_pre_stage1.checked,
                &legacy_proof,
                &formula_dimensions,
                &mut legacy_transcript,
                &legacy_stage1,
                &legacy_stage2,
                &legacy_stage3,
                &legacy_stage4,
                &legacy_stage5,
            )
            .expect("legacy proof must verify through stage 6a");
            assert_eq!(
                new_transcript.state(),
                legacy_transcript.state(),
                "stage-6a transcript state diverged from legacy",
            );

            // The stage-6b ratchet: prove, then replay legacy's proof through
            // the verifier's stage 6b for the boundary state.
            let stage6b =
                prove_stage6b::<Fr, DoryScheme, Pedersen<Bn254G1>, Bn254G1, Blake2bTranscript>(
                    backend,
                    &mut session,
                    &legacy_pre_stage1.checked,
                    &config,
                    &prover_preprocessing,
                    &stage1.clear_output,
                    &stage2.clear_output,
                    &stage3.clear_output,
                    &stage4.clear_output,
                    &stage5.clear_output,
                    &stage6a.claims,
                    &stage6a.clear_output,
                    &witness,
                    &mut new_transcript,
                )
                .expect("stage 6b proves");

            assert_eq!(
                stage6b.sumcheck_proof, legacy_proof.stages.stage6b_sumcheck_proof,
                "stage-6b sumcheck proof bytes diverged from legacy",
            );
            assert_eq!(stage6b.claims, legacy_claims.stage6b);

            let _legacy_stage6b = jolt_verifier::stages::stage6b::verify(
                &legacy_pre_stage1.checked,
                &prover_preprocessing.verifier,
                &legacy_proof,
                &formula_dimensions,
                &mut legacy_transcript,
                &legacy_stage1,
                &legacy_stage2,
                &legacy_stage3,
                &legacy_stage4,
                &legacy_stage5,
                &legacy_stage6a,
            )
            .expect("legacy proof must verify through stage 6b");
            assert_eq!(
                new_transcript.state(),
                legacy_transcript.state(),
                "stage-6b transcript state diverged from legacy",
            );
        };

        assert_backend_matches_legacy(&JoltBackend::reference());
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
fn prover_matches_legacy_on_muldiv() {}
