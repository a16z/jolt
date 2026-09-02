//! The packed (Akita) byte-diff harness against `jolt-prover-legacy`.
//!
//! Both provers run from the same guest program, inputs, and packed
//! preprocessing artifacts; the modular proof must equal legacy's
//! wire-for-wire (structural equality on the shared `AkitaJoltProof` wire
//! types) and verify end-to-end. Every module is a whole-proof ratchet with
//! component-wise asserts; the packed proof's per-stage sumcheck fields and
//! native grouped opening already give stage-level granularity when bytes
//! diverge. The stage-granular
//! verifier-replay ratchet (`byte_diff::muldiv` style, per-stage prove +
//! legacy verifier replay of stage-boundary transcript states) lands once
//! the port exposes per-stage packed drivers.
//!
//! The Dory harness lives in `dory_byte_diff.rs` — one compiled prover
//! proves exactly one protocol, so the two harnesses are mutually exclusive
//! by feature.

/// Shared scaffolding: every test runs the same legacy-side packed pipeline
/// (decode + trace + preprocess + prove) and the same modular-side pipeline
/// (trace + config + witness + prove + verify); the per-mode differences —
/// advice, committed program — stay in the test bodies.
#[cfg(all(feature = "prover-fixtures", feature = "akita"))]
#[expect(clippy::expect_used)]
mod support {
    use std::sync::Arc;

    use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
    use jolt_program::execution::{JoltProgram, TraceInputs, TraceOutput};
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_prover::ProverConfig;
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::packed::{
        AkitaField, AkitaJoltProof, AkitaPackedScheme, AkitaScheme, AkitaTranscript, AkitaVc,
    };
    use jolt_prover_legacy::zkvm::program::{
        CommittedProgramProverData as LegacyCommittedProgramProverData,
        ProgramPreprocessing as LegacyProgramPreprocessing,
    };
    use jolt_riscv::JoltTraceRow;
    use jolt_verifier::JoltVerifierPreprocessing;
    use jolt_witness::JoltVmWitnessConfig;
    use tracer::execution_backend::TracerBackend;

    pub const MAX_PADDED_TRACE_LENGTH: usize = 1 << 16;

    pub type AkitaCommitmentOutput = <AkitaScheme as jolt_crypto::Commitment>::Output;

    /// The legacy-side guest artifacts every test starts from: the program
    /// preprocessing, the traced I/O device (for the memory layout), and the
    /// raw ELF the modular side re-traces from.
    pub struct PackedGuest {
        pub program_data: LegacyProgramPreprocessing<AkitaPackedScheme>,
        pub io_device: JoltDevice,
        pub elf_contents: Vec<u8>,
    }

    pub fn packed_guest(
        program: &mut host::Program,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> PackedGuest {
        let (bytecode, init_memory_state, _, entry_address) = program.decode();
        let (_, _, _, io_device) = program.trace(inputs, untrusted_advice, trusted_advice);
        let elf_contents = program.get_elf_contents().expect("elf contents");
        let program_data =
            LegacyProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
                .expect("legacy preprocess");
        PackedGuest {
            program_data,
            io_device,
            elf_contents,
        }
    }

    /// Trace the guest through the modular stack (`TracerBackend`), with the
    /// memory config mirrored off the legacy run's layout.
    pub fn trace_modular(
        program: &JoltProgram,
        preprocessing: &JoltProgramPreprocessing,
        memory_layout: &MemoryLayout,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> TraceOutput<Arc<Vec<JoltTraceRow>>> {
        let memory_config = MemoryConfig {
            max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
            max_trusted_advice_size: memory_layout.max_trusted_advice_size,
            max_input_size: memory_layout.max_input_size,
            max_output_size: memory_layout.max_output_size,
            stack_size: memory_layout.stack_size,
            heap_size: memory_layout.heap_size,
            program_size: Some(memory_layout.program_size),
        };
        TracerBackend::new()
            .trace_compact(
                program,
                TraceInputs {
                    inputs: inputs.to_vec(),
                    untrusted_advice: untrusted_advice.to_vec(),
                    trusted_advice: trusted_advice.to_vec(),
                    memory_config,
                    advice_tape: None,
                },
                &preprocessing.bytecode,
            )
            .expect("modular trace")
    }

    /// Derive the modular config over the packed field and pin every wire
    /// config field against what legacy wrote on the proof. The packed
    /// pipeline is cycle-major only — derivation's default.
    pub fn derive_config_pinned(
        trace_output: &TraceOutput<Arc<Vec<JoltTraceRow>>>,
        memory_layout: &MemoryLayout,
        verifier_preprocessing: &JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
        legacy_proof: &AkitaJoltProof,
    ) -> ProverConfig {
        let config = ProverConfig::derive_compact::<AkitaField>(
            trace_output.trace.as_slice(),
            memory_layout,
            verifier_preprocessing.program.min_bytecode_address(),
            verifier_preprocessing.program.program_image_len_words(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive config");
        assert_eq!(config.trace_length, legacy_proof.trace_length);
        assert_eq!(config.ram_K, legacy_proof.ram_K);
        assert_eq!(config.rw_config, legacy_proof.rw_config);
        assert_eq!(config.one_hot_config, legacy_proof.one_hot_config);
        assert_eq!(
            config.trace_polynomial_order,
            legacy_proof.trace_polynomial_order
        );
        config
    }

    pub fn witness_config(config: &ProverConfig) -> JoltVmWitnessConfig {
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        )
    }

    /// Rebuild the full program preprocessing from the legacy prover data's
    /// retained copy (the verifier preprocessing carries only the
    /// direct-program commitments in committed mode).
    pub fn rebuild_full_program(
        prover_data: &LegacyCommittedProgramProverData<AkitaPackedScheme>,
        memory_layout: &MemoryLayout,
    ) -> Arc<JoltProgramPreprocessing> {
        Arc::new(JoltProgramPreprocessing {
            bytecode: prover_data.full.bytecode.as_ref().clone(),
            ram: prover_data.full.ram.clone(),
            memory_layout: memory_layout.clone(),
            max_padded_trace_length: MAX_PADDED_TRACE_LENGTH,
        })
    }

    /// Component-wise asserts give per-stage granularity when bytes diverge;
    /// the final whole-struct assert is the ratchet.
    pub fn assert_proof_matches_legacy(proof: &AkitaJoltProof, legacy_proof: &AkitaJoltProof) {
        assert_eq!(
            proof.commitments, legacy_proof.commitments,
            "the packed OneHotTrace commitment diverged from legacy",
        );
        assert_eq!(
            proof.untrusted_advice_commitment,
            legacy_proof.untrusted_advice_commitment
        );
        assert_eq!(
            proof.stages.stage1_uni_skip_first_round_proof,
            legacy_proof.stages.stage1_uni_skip_first_round_proof
        );
        assert_eq!(
            proof.stages.stage1_sumcheck_proof, legacy_proof.stages.stage1_sumcheck_proof,
            "stage-1 bytes diverged (the preamble seeds every challenge)",
        );
        assert_eq!(
            proof.stages.stage2_uni_skip_first_round_proof,
            legacy_proof.stages.stage2_uni_skip_first_round_proof
        );
        assert_eq!(
            proof.stages.stage2_sumcheck_proof,
            legacy_proof.stages.stage2_sumcheck_proof
        );
        assert_eq!(
            proof.stages.stage3_sumcheck_proof,
            legacy_proof.stages.stage3_sumcheck_proof
        );
        assert_eq!(
            proof.stages.stage4_sumcheck_proof,
            legacy_proof.stages.stage4_sumcheck_proof
        );
        assert_eq!(
            proof.stages.stage5_sumcheck_proof,
            legacy_proof.stages.stage5_sumcheck_proof
        );
        assert_eq!(
            proof.stages.stage6a_sumcheck_proof, legacy_proof.stages.stage6a_sumcheck_proof,
            "stage-6a bytes diverged (the nine-stage bytecode read-raf address phase runs here)",
        );
        assert_eq!(
            proof.stages.stage6b_sumcheck_proof, legacy_proof.stages.stage6b_sumcheck_proof,
            "stage-6b bytes diverged (fused-inc read-raf + lattice booleanity cycle phases run here)",
        );
        assert_eq!(
            proof.stages.stage7_sumcheck_proof, legacy_proof.stages.stage7_sumcheck_proof,
            "stage-7 bytes diverged (the hamming-weight inc fold runs here)",
        );
        assert_eq!(
            proof.joint_opening_proof, legacy_proof.joint_opening_proof,
            "the native grouped opening diverged from legacy",
        );
        assert_eq!(proof.claims, legacy_proof.claims);
        assert_eq!(
            proof, legacy_proof,
            "assembled packed proof diverged from legacy"
        );
    }

    pub fn verify_modular(
        preprocessing: &JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
        public_io: &JoltDevice,
        proof: &AkitaJoltProof,
        trusted_advice_commitment: Option<&AkitaCommitmentOutput>,
    ) {
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            preprocessing,
            public_io,
            proof,
            trusted_advice_commitment,
        )
        .expect("modular packed proof must verify end-to-end");
    }
}

#[cfg(all(feature = "prover-fixtures", feature = "akita"))]
#[expect(clippy::expect_used)]
mod muldiv {
    use std::sync::Arc;

    use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
    use jolt_program::execution::{JoltProgram, OwnedTrace};
    use jolt_prover::akita;
    use jolt_prover::JoltProverPreprocessing;
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::packed::{
        akita_verifier_preprocessing, AkitaField, AkitaPackedProver, AkitaPackedScheme,
        AkitaScheme, AkitaTranscript, AkitaVc,
    };
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::prover::{
        JoltCpuProver, JoltProverPreprocessing as LegacyProverPreprocessing,
    };
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    /// Prove muldiv over the packed stack with both provers from the same
    /// guest, inputs, and OneHotTrace setup; assert wire-for-wire equality
    /// of the assembled packed proofs and verify the modular proof
    /// end-to-end.
    #[test]
    fn prover_matches_legacy_on_muldiv_akita() {
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
        let guest = support::packed_guest(&mut program, &inputs, &[], &[]);

        // --- Legacy side: packed preprocessing, the transparent OneHotTrace
        // setup, and the oracle proof.
        let shared: JoltSharedPreprocessing<AkitaPackedScheme> = JoltSharedPreprocessing::new(
            guest.program_data,
            guest.io_device.memory_layout.clone(),
            support::MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing = LegacyProverPreprocessing::new(shared);
        let legacy_prover: AkitaPackedProver<'_> = JoltCpuProver::gen_from_elf(
            &legacy_preprocessing,
            &guest.elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
            None,
        )
        .expect("legacy prover construction");
        let public_io = legacy_prover.program_io.clone();
        let (object_setup, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(
            legacy_prover.one_hot_trace_setup_params(),
        )
        .expect("the transparent packed setup must derive");
        let legacy_proof = legacy_prover
            .prove_packed(&object_setup, None, None)
            .expect("legacy packed prove");
        let verifier_preprocessing =
            akita_verifier_preprocessing(&legacy_preprocessing, verifier_setup, None);

        // --- Modular side: trace independently through the modular stack.
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(guest.elf_contents));
        let memory_layout = &public_io.memory_layout;
        let program_preprocessing = verifier_preprocessing
            .program
            .as_full_arc()
            .expect("full program preprocessing");
        let trace_output = support::trace_modular(
            &jolt_program,
            &program_preprocessing,
            memory_layout,
            &inputs,
            &[],
            &[],
        );
        // The derived proof shape must equal what legacy wrote on the wire
        // (asserted inside).
        let config = support::derive_config_pinned(
            &trace_output,
            memory_layout,
            &verifier_preprocessing,
            &legacy_proof,
        );
        let witness = TraceBackend::<OwnedTrace>::from_compact(
            support::witness_config(&config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, trace_output),
        );
        let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier: verifier_preprocessing,
            pcs_setup: object_setup,
            committed_program: None,
        };

        for backend in [
            akita::JoltAkitaBackend::reference(),
            akita::JoltAkitaBackend::optimized(),
        ] {
            let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
                &backend,
                &prover_preprocessing,
                &config,
                None,
                &witness,
                &public_io,
            )
            .expect("modular packed prove");

            support::assert_proof_matches_legacy(&proof, &legacy_proof);
            support::verify_modular(&prover_preprocessing.verifier, &public_io, &proof, None);
        }
    }
}

#[cfg(all(feature = "prover-fixtures", feature = "akita"))]
#[expect(clippy::expect_used)]
mod advice_consumer {
    use std::sync::Arc;

    use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
    use jolt_program::execution::{JoltProgram, OwnedTrace};
    use jolt_prover::akita;
    use jolt_prover::JoltProverPreprocessing;
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::packed::{
        akita_verifier_preprocessing, commit_trusted_advice, AkitaField, AkitaPackedProver,
        AkitaPackedScheme, AkitaScheme, AkitaTranscript, AkitaVc,
    };
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::prover::{
        JoltCpuProver, JoltProverPreprocessing as LegacyProverPreprocessing,
    };
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    /// Prove the advice-consumer guest (trusted AND untrusted advice) over
    /// the packed stack with both provers — three commitment objects
    /// (`OneHotTrace`, `UntrustedAdvice`, `TrustedAdvice`), the
    /// trusted object committed once at preprocessing time and shared by
    /// both sides; assert wire-for-wire equality and verify the modular
    /// proof against the trusted commitment.
    #[test]
    fn prover_matches_legacy_on_advice_consumer_akita() {
        let mut program = host::Program::new("advice-consumer-guest");
        let inputs = postcard::to_stdvec(&12u64).expect("serialize inputs");
        let untrusted_advice = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
        let trusted_advice = postcard::to_stdvec(&7u64).expect("serialize trusted advice");
        let guest =
            support::packed_guest(&mut program, &inputs, &untrusted_advice, &trusted_advice);

        // --- Legacy side: the trusted-advice object is produced at
        // preprocessing time (before any proving) and handed to the prover.
        let shared: JoltSharedPreprocessing<AkitaPackedScheme> = JoltSharedPreprocessing::new(
            guest.program_data,
            guest.io_device.memory_layout.clone(),
            support::MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing = LegacyProverPreprocessing::new(shared);
        let trusted_object = commit_trusted_advice(
            &trusted_advice,
            guest.io_device.memory_layout.max_trusted_advice_size as usize,
        )
        .expect("trusted advice object must commit");
        let trusted_commitment = trusted_object.commitment.clone();

        let legacy_prover: AkitaPackedProver<'_> = JoltCpuProver::gen_from_elf(
            &legacy_preprocessing,
            &guest.elf_contents,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
            None,
            None,
            None,
        )
        .expect("legacy prover construction");
        let public_io = legacy_prover.program_io.clone();
        let (object_setup, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(
            legacy_prover.one_hot_trace_setup_params(),
        )
        .expect("the transparent packed setup must derive");
        let legacy_proof = legacy_prover
            .prove_packed(&object_setup, Some(&trusted_object), None)
            .expect("legacy packed prove");
        let verifier_preprocessing =
            akita_verifier_preprocessing(&legacy_preprocessing, verifier_setup, None);

        // --- Modular side: trace independently with the advice inputs,
        // prove with an independently precommitted trusted object (its
        // commitment must land byte-identical to legacy's).
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(guest.elf_contents));
        let memory_layout = &public_io.memory_layout;
        let program_preprocessing = verifier_preprocessing
            .program
            .as_full_arc()
            .expect("full program preprocessing");
        let trace_output = support::trace_modular(
            &jolt_program,
            &program_preprocessing,
            memory_layout,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
        );
        let config = support::derive_config_pinned(
            &trace_output,
            memory_layout,
            &verifier_preprocessing,
            &legacy_proof,
        );
        let witness = TraceBackend::<OwnedTrace>::from_compact(
            support::witness_config(&config)
                .include_trusted_advice(true)
                .include_untrusted_advice(true),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, trace_output),
        );
        let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier: verifier_preprocessing,
            pcs_setup: object_setup,
            committed_program: None,
        };
        let modular_trusted_object = akita::witness::AdviceObject {
            plan: trusted_object.plan.clone(),
            commitment: trusted_object.commitment.clone(),
            hint: trusted_object.hint.clone(),
        };

        for backend in [
            akita::JoltAkitaBackend::reference(),
            akita::JoltAkitaBackend::optimized(),
        ] {
            let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
                &backend,
                &prover_preprocessing,
                &config,
                Some(&modular_trusted_object),
                &witness,
                &public_io,
            )
            .expect("modular packed prove");

            support::assert_proof_matches_legacy(&proof, &legacy_proof);
            support::verify_modular(
                &prover_preprocessing.verifier,
                &public_io,
                &proof,
                Some(&trusted_commitment),
            );
        }
    }
}

#[cfg(all(feature = "prover-fixtures", feature = "akita"))]
#[expect(clippy::expect_used)]
mod committed_muldiv {
    use std::sync::Arc;

    use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
    use jolt_program::execution::{JoltProgram, OwnedTrace};
    use jolt_prover::akita;
    use jolt_prover::JoltProverPreprocessing;
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::packed::{
        akita_verifier_preprocessing, shared_preprocessing_with_direct_program, AkitaField,
        AkitaPackedProver, AkitaPackedScheme, AkitaScheme, AkitaTranscript, AkitaVc,
    };
    use jolt_prover_legacy::zkvm::prover::{
        JoltCpuProver, JoltProverPreprocessing as LegacyProverPreprocessing,
    };
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    /// Prove muldiv under packed committed-program preprocessing with both
    /// provers: direct-program objects join the native commitment group,
    /// assembled and committed once at preprocessing time and shared by
    /// both sides.
    #[test]
    fn prover_matches_legacy_on_committed_muldiv_akita() {
        committed_muldiv_matches_legacy(1);
    }

    /// The multi-chunk arm: the bytecode splits across two objects in the
    /// precommitted packing.
    #[test]
    fn prover_matches_legacy_on_committed_muldiv_akita_two_chunks() {
        committed_muldiv_matches_legacy(2);
    }

    fn committed_muldiv_matches_legacy(bytecode_chunk_count: usize) {
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
        let guest = support::packed_guest(&mut program, &inputs, &[], &[]);

        // --- Legacy side: packed committed preprocessing (direct program is
        // assembled and committed here, before any proving), then prove.
        let (shared, prover_data, direct_program) = shared_preprocessing_with_direct_program(
            guest.program_data,
            guest.io_device.memory_layout.clone(),
            support::MAX_PADDED_TRACE_LENGTH,
            bytecode_chunk_count,
        )
        .expect("packed committed preprocessing");
        let legacy_preprocessing =
            LegacyProverPreprocessing::new_committed(shared, prover_data, AkitaPackedScheme);
        let legacy_prover: AkitaPackedProver<'_> = JoltCpuProver::gen_from_elf(
            &legacy_preprocessing,
            &guest.elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
            None,
        )
        .expect("legacy prover construction");
        let public_io = legacy_prover.program_io.clone();
        let (object_setup, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(
            legacy_prover.one_hot_trace_setup_params(),
        )
        .expect("the transparent packed setup must derive");
        let legacy_proof = legacy_prover
            .prove_packed(&object_setup, None, Some(&direct_program))
            .expect("legacy packed prove");
        let verifier_preprocessing = akita_verifier_preprocessing(
            &legacy_preprocessing,
            verifier_setup,
            Some(&direct_program),
        );

        // --- Modular side: the full program is rebuilt from the legacy
        // prover data's retained copy, and the precommitted direct-program
        // objects are independently re-committed at preprocessing time and
        // retained in the packed prover data.
        let memory_layout = &public_io.memory_layout;
        let full_program = support::rebuild_full_program(
            legacy_preprocessing
                .committed_program_prover_data
                .as_ref()
                .expect("legacy committed prover data"),
            memory_layout,
        );
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(guest.elf_contents));
        let trace_output = support::trace_modular(
            &jolt_program,
            &full_program,
            memory_layout,
            &inputs,
            &[],
            &[],
        );
        let config = support::derive_config_pinned(
            &trace_output,
            memory_layout,
            &verifier_preprocessing,
            &legacy_proof,
        );
        let witness = TraceBackend::<OwnedTrace>::from_compact(
            support::witness_config(&config),
            JoltVmWitnessInputs::new(&jolt_program, &full_program, trace_output),
        );
        let modular_direct_program =
            jolt_prover::akita::witness::commit_direct_program::<AkitaScheme>(
                &full_program,
                bytecode_chunk_count,
                config.trace_polynomial_order,
            )
            .expect("modular direct program objects must commit");
        let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier: verifier_preprocessing,
            pcs_setup: object_setup,
            committed_program: Some(jolt_prover::CommittedProgramProverData {
                full: (*full_program).clone(),
                direct_program: modular_direct_program,
                trace_order: config.trace_polynomial_order,
            }),
        };

        for backend in [
            akita::JoltAkitaBackend::reference(),
            akita::JoltAkitaBackend::optimized(),
        ] {
            let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
                &backend,
                &prover_preprocessing,
                &config,
                None,
                &witness,
                &public_io,
            )
            .expect("modular packed prove");

            support::assert_proof_matches_legacy(&proof, &legacy_proof);
            support::verify_modular(&prover_preprocessing.verifier, &public_io, &proof, None);
        }
    }
}

#[cfg(not(all(feature = "prover-fixtures", feature = "akita")))]
#[test]
#[ignore = "enable --features akita,prover-fixtures to build the packed (Akita) byte-diff harness"]
fn prover_matches_legacy_on_muldiv_akita() {}
