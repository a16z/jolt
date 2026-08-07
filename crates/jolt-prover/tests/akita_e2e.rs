//! End-to-end packed (Akita) tests for the modular prover: the analogs of
//! `jolt-prover-legacy`'s `zkvm::packed` e2e suite (`muldiv_e2e_akita`,
//! `muldiv_e2e_akita_forced_k256`, `advice_e2e_akita`,
//! `advice_e2e_akita_full_advice`, `muldiv_e2e_akita_committed_program`).
//!
//! Legacy hosts the guest compilation and the packed preprocessing
//! artifacts (shared preprocessing, transparent OneHotTrace setup, verifier
//! preprocessing); the modular stack traces, derives the config, generates
//! the witness, and proves.
//!
//! Deviations from the legacy suite: the advice tests run the purpose-built
//! `advice-consumer-guest` (asserts `trusted + untrusted == public`) instead
//! of the merkle guest — the established external-crate choice (`byte_diff`,
//! the jolt-verifier akita fixtures), avoiding the force-linked inline
//! crates the merkle guest needs. The release-only `sha2_chain_akita_perf`
//! harness has no analog here.

/// Shared scaffolding: the legacy-side guest artifacts every packed test
/// starts from, and the modular-side trace/config/witness pipeline pieces.
#[cfg(all(feature = "prover-fixtures", feature = "akita"))]
#[expect(clippy::expect_used)]
mod support {
    use std::sync::Arc;

    use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
    use jolt_program::execution::{
        ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
    };
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_prover::ProverConfig;
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::packed::{AkitaField, AkitaPackedScheme, AkitaScheme, AkitaVc};
    use jolt_prover_legacy::zkvm::program::{
        CommittedProgramProverData as LegacyCommittedProgramProverData,
        ProgramPreprocessing as LegacyProgramPreprocessing,
    };
    use jolt_verifier::JoltVerifierPreprocessing;
    use jolt_witness::JoltVmWitnessConfig;
    use tracer::execution_backend::TracerBackend;

    pub const MAX_PADDED_TRACE_LENGTH: usize = 1 << 16;

    /// The legacy-side guest artifacts every packed test starts from: the
    /// program preprocessing, the traced I/O device (for the memory layout),
    /// and the raw ELF the modular side re-traces from.
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
        memory_layout: &MemoryLayout,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> TraceOutput<OwnedTrace> {
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
            .trace(
                program,
                TraceInputs {
                    inputs: inputs.to_vec(),
                    untrusted_advice: untrusted_advice.to_vec(),
                    trusted_advice: trusted_advice.to_vec(),
                    memory_config,
                    advice_tape: None,
                },
            )
            .expect("modular trace")
    }

    /// Derive the modular config over the packed field. The packed pipeline
    /// is cycle-major only — derivation's default.
    pub fn derive_config(
        trace_output: &TraceOutput<OwnedTrace>,
        memory_layout: &MemoryLayout,
        verifier_preprocessing: &JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
    ) -> ProverConfig {
        ProverConfig::derive::<AkitaField>(
            trace_output.trace.rows(),
            memory_layout,
            verifier_preprocessing.program.min_bytecode_address(),
            verifier_preprocessing.program.program_image_len_words(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive config")
    }

    /// Pad to the padded trace length with no-op rows, as legacy does.
    pub fn pad_trace(
        trace_output: TraceOutput<OwnedTrace>,
        trace_length: usize,
    ) -> TraceOutput<OwnedTrace> {
        let mut rows = trace_output.trace.rows().to_vec();
        rows.resize(trace_length, TraceRow::default());
        TraceOutput::new(
            OwnedTrace::new(rows),
            trace_output.device,
            trace_output.final_memory,
            trace_output.advice_tape,
        )
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
    /// `ProgramOneHot` commitment in committed mode).
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
}

#[cfg(all(feature = "prover-fixtures", feature = "akita"))]
#[expect(clippy::expect_used, clippy::panic)]
mod muldiv {
    use std::sync::Arc;

    use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
    use jolt_program::execution::JoltProgram;
    use jolt_prover::akita;
    use jolt_prover::JoltProverPreprocessing;
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::config::{
        OneHotConfig as LegacyOneHotConfig, OneHotParams as LegacyOneHotParams,
    };
    use jolt_prover_legacy::zkvm::packed::{
        akita_verifier_preprocessing, AkitaField, AkitaJoltProof, AkitaPackedProver,
        AkitaPackedScheme, AkitaScheme, AkitaTranscript, AkitaVc,
    };
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::prover::{
        JoltCpuProver, JoltProverPreprocessing as LegacyProverPreprocessing,
    };
    use jolt_verifier::proof::{ClearProofClaims, JoltProofClaims};
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    /// Proves and verifies muldiv end to end over the packed (Akita) stack
    /// with the MODULAR prover: the full-program packed pipeline, one
    /// `OneHotTrace` commitment object, and the joint packed opening — the
    /// analog of legacy's `muldiv_e2e_akita`, including its live tampers on
    /// the fused-inc pipeline's claim wires.
    #[test]
    fn muldiv_e2e_akita() {
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
        let guest = support::packed_guest(&mut program, &inputs, &[], &[]);

        // Legacy hosts the packed preprocessing artifacts; the proof comes
        // from the modular prover below. TODO(port): derive the OneHotTrace
        // setup params from `ProverConfig` + the program shape instead of a
        // legacy prover instance.
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
        );
        let public_io = legacy_prover.program_io.clone();
        let setup_params = legacy_prover.one_hot_trace_setup_params();
        assert_eq!(setup_params.one_hot_k(), 16);
        let (object_setup, verifier_setup) =
            <AkitaScheme as VerifierCommitmentScheme>::setup(setup_params)
                .expect("the transparent packed setup must derive");
        let verifier_preprocessing =
            akita_verifier_preprocessing(&legacy_preprocessing, verifier_setup, None);

        // --- Modular side: trace independently, derive the config, prove.
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(guest.elf_contents));
        let memory_layout = &public_io.memory_layout;
        let trace_output = support::trace_modular(&jolt_program, memory_layout, &inputs, &[], &[]);
        let program_preprocessing = verifier_preprocessing
            .program
            .as_full_arc()
            .expect("full program preprocessing");
        let config = support::derive_config(&trace_output, memory_layout, &verifier_preprocessing);
        let padded_output = support::pad_trace(trace_output, config.trace_length);
        let witness = TraceBackend::new(
            support::witness_config(&config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
        );
        let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier: verifier_preprocessing,
            pcs_setup: object_setup,
            committed_program: None,
        };

        let backend = akita::JoltAkitaBackend::reference();
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
            None,
            None,
            &witness,
            &public_io,
        )
        .expect("packed prover should produce a verifier-native proof");

        let verify = |proof: &AkitaJoltProof| {
            jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
                &prover_preprocessing.verifier,
                &public_io,
                proof,
                None,
            )
        };
        verify(&proof).expect("packed verifier should accept the packed proof");

        // Live tampers on the fused-inc pipeline's claim wires: the fused
        // increment's reduced claim and the hamming-reduction chunk/msb
        // finals each participate in a batched output fold — an offset on
        // any of them must be rejected.
        let tamper = |mutate: &dyn Fn(&mut ClearProofClaims<AkitaField>)| {
            let mut tampered = proof.clone();
            let JoltProofClaims::Clear(claims) = &mut tampered.claims else {
                panic!("packed proofs carry clear claims");
            };
            mutate(claims);
            tampered
        };
        let one = AkitaField::from_u64(1);
        assert!(
            verify(&tamper(&|claims| claims
                .stage6b
                .bytecode_read_raf
                .fused_inc += one))
            .is_err(),
            "tampered read-raf fused-inc opening must be rejected"
        );
        assert!(
            verify(&tamper(&|claims| claims
                .stage7
                .hamming_weight_claim_reduction
                .unsigned_inc_chunks[0] += one))
            .is_err(),
            "tampered unsigned-inc chunk final must be rejected"
        );
        assert!(
            verify(&tamper(&|claims| claims
                .stage7
                .hamming_weight_claim_reduction
                .unsigned_inc_msb += one))
            .is_err(),
            "tampered unsigned-inc msb final must be rejected"
        );
    }

    /// The large-trace regime at e2e scale: small traces select K = 16 by
    /// the shared toggle, so this pins the K = 256 arm by forcing the
    /// one-hot params on both sides before proving (the verifier accepts
    /// either regime at any trace length; the choice is carried by the
    /// proof's one-hot config and bound by the digest) — the analog of
    /// legacy's `muldiv_e2e_akita_forced_k256`.
    #[test]
    fn muldiv_e2e_akita_forced_k256() {
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
        let guest = support::packed_guest(&mut program, &inputs, &[], &[]);

        let shared: JoltSharedPreprocessing<AkitaPackedScheme> = JoltSharedPreprocessing::new(
            guest.program_data,
            guest.io_device.memory_layout.clone(),
            support::MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing = LegacyProverPreprocessing::new(shared);
        let mut legacy_prover: AkitaPackedProver<'_> = JoltCpuProver::gen_from_elf(
            &legacy_preprocessing,
            &guest.elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
            None,
        );
        // The forced K = 256 regime; the setup params must be derived AFTER
        // the override (they carry K and the layout digest).
        let forced = LegacyOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        };
        legacy_prover.one_hot_params = LegacyOneHotParams::from_config(
            &forced,
            legacy_preprocessing.shared.bytecode_size(),
            legacy_prover.one_hot_params.ram_k,
        );
        let public_io = legacy_prover.program_io.clone();
        let (object_setup, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(
            legacy_prover.one_hot_trace_setup_params(),
        )
        .expect("the transparent packed setup must derive");
        let verifier_preprocessing =
            akita_verifier_preprocessing(&legacy_preprocessing, verifier_setup, None);

        // --- Modular side, with the same forced regime on the wire config.
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(guest.elf_contents));
        let memory_layout = &public_io.memory_layout;
        let trace_output = support::trace_modular(&jolt_program, memory_layout, &inputs, &[], &[]);
        let program_preprocessing = verifier_preprocessing
            .program
            .as_full_arc()
            .expect("full program preprocessing");
        let mut config =
            support::derive_config(&trace_output, memory_layout, &verifier_preprocessing);
        config.one_hot_config = jolt_claims::protocols::jolt::JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        };
        let padded_output = support::pad_trace(trace_output, config.trace_length);
        let witness = TraceBackend::new(
            support::witness_config(&config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
        );
        let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier: verifier_preprocessing,
            pcs_setup: object_setup,
            committed_program: None,
        };

        let backend = akita::JoltAkitaBackend::reference();
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
            None,
            None,
            &witness,
            &public_io,
        )
        .expect("packed prover should produce a verifier-native proof");

        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &prover_preprocessing.verifier,
            &public_io,
            &proof,
            None,
        )
        .expect("packed verifier should accept the forced-K256 proof");
    }
}

#[cfg(all(feature = "prover-fixtures", feature = "akita"))]
#[expect(clippy::expect_used)]
mod advice {
    use std::sync::Arc;

    use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
    use jolt_program::execution::JoltProgram;
    use jolt_prover::akita;
    use jolt_prover::JoltProverPreprocessing;
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::packed::{
        akita_verifier_preprocessing, commit_trusted_advice_one_hot, AkitaField, AkitaJoltProof,
        AkitaPackedProver, AkitaPackedScheme, AkitaScheme, AkitaTranscript, AkitaVc,
    };
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::prover::{
        JoltCpuProver, JoltProverPreprocessing as LegacyProverPreprocessing,
    };
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    /// The packed advice e2e: a guest consuming both advice kinds, proved
    /// over three commitment objects (`OneHotTrace`, `UntrustedAdviceOneHot`,
    /// `TrustedAdviceOneHot`), with per-object tamper rejection — the analog
    /// of legacy's `advice_e2e_akita` (7 + 5 == 12 on the advice-consumer
    /// guest instead of the merkle leaves).
    #[test]
    fn advice_e2e_akita() {
        let mut program = host::Program::new("advice-consumer-guest");
        let inputs = postcard::to_stdvec(&12u64).expect("serialize inputs");
        let untrusted_advice = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
        let trusted_advice = postcard::to_stdvec(&7u64).expect("serialize trusted advice");
        let guest =
            support::packed_guest(&mut program, &inputs, &untrusted_advice, &trusted_advice);

        let shared: JoltSharedPreprocessing<AkitaPackedScheme> = JoltSharedPreprocessing::new(
            guest.program_data,
            guest.io_device.memory_layout.clone(),
            support::MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing = LegacyProverPreprocessing::new(shared);

        // The trusted-advice object commits at preprocessing time, out of
        // band; its commitment goes to both the prover and the verifier.
        let trusted_object = commit_trusted_advice_one_hot(
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
        );
        let public_io = legacy_prover.program_io.clone();
        let (object_setup, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(
            legacy_prover.one_hot_trace_setup_params(),
        )
        .expect("the transparent packed setup must derive");
        let verifier_preprocessing =
            akita_verifier_preprocessing(&legacy_preprocessing, verifier_setup, None);

        // --- Modular side: trace with the advice inputs, prove with the
        // precommitted trusted object's commitment (the port will carry the
        // full object's opening material through a packed prover-data shape).
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(guest.elf_contents));
        let memory_layout = &public_io.memory_layout;
        let trace_output = support::trace_modular(
            &jolt_program,
            memory_layout,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
        );
        let program_preprocessing = verifier_preprocessing
            .program
            .as_full_arc()
            .expect("full program preprocessing");
        let config = support::derive_config(&trace_output, memory_layout, &verifier_preprocessing);
        let padded_output = support::pad_trace(trace_output, config.trace_length);
        let witness = TraceBackend::new(
            support::witness_config(&config)
                .include_trusted_advice(true)
                .include_untrusted_advice(true),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
        );
        let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier: verifier_preprocessing,
            pcs_setup: object_setup,
            committed_program: None,
        };

        let backend = akita::JoltAkitaBackend::reference();
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
            Some(&trusted_commitment),
            None,
            &witness,
            &public_io,
        )
        .expect("packed prover should produce a verifier-native proof");
        assert!(proof.untrusted_advice_commitment.is_some());
        assert!(proof.stages.reconstruction_sumcheck_proof.is_some());
        // OneHotTrace is discharged by its native same-point batch. The two
        // advice commitment objects remain in the auxiliary packed opening.
        let auxiliary = proof
            .joint_opening_proof
            .auxiliary
            .as_ref()
            .expect("advice requires an auxiliary opening");
        assert_eq!(auxiliary.openings.len(), 2);
        assert_eq!(auxiliary.evaluations.len(), 2);

        let verify = |proof: &AkitaJoltProof| {
            jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
                &prover_preprocessing.verifier,
                &public_io,
                proof,
                Some(&trusted_commitment),
            )
        };
        verify(&proof).expect("packed verifier should accept the packed proof");

        // Per-object tampers: a mutated claimed evaluation breaks that
        // object's native opening; a dropped reconstruction proof breaks the
        // fail-closed presence rule. The two advice objects hold the last two
        // per-object evaluations.
        for object in 0..2 {
            let mut tampered = proof.clone();
            tampered
                .joint_opening_proof
                .auxiliary
                .as_mut()
                .expect("advice requires an auxiliary opening")
                .evaluations[object] += AkitaField::from_u64(1);
            assert!(
                verify(&tampered).is_err(),
                "tampered object-{object} evaluation must be rejected"
            );
        }
        let mut tampered = proof.clone();
        tampered.stages.reconstruction_sumcheck_proof = None;
        assert!(
            verify(&tampered).is_err(),
            "a dropped reconstruction proof must be rejected"
        );
        let mut tampered = proof.clone();
        tampered.joint_opening_proof.auxiliary = None;
        assert!(
            verify(&tampered).is_err(),
            "a dropped auxiliary opening proof must be rejected"
        );
    }

    /// The advice-size boundary e2e: the untrusted advice buffer fills
    /// `max_untrusted_advice_size` exactly, so the byte column carries
    /// non-degenerate lane content on every row and the exact-capacity edge
    /// is exercised end to end. The guest reads only its postcard-encoded
    /// prefix; the remaining filler bytes still enter the committed column —
    /// the analog of legacy's `advice_e2e_akita_full_advice`.
    #[test]
    fn advice_e2e_akita_full_advice() {
        let mut program = host::Program::new("advice-consumer-guest");
        let inputs = postcard::to_stdvec(&12u64).expect("serialize inputs");
        let trusted_advice = postcard::to_stdvec(&7u64).expect("serialize trusted advice");

        // Fill the advice capacity exactly (the test never overrides the
        // default, and the traced layout below re-confirms the size).
        let max_untrusted = common::constants::DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE as usize;
        let mut untrusted_advice = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
        untrusted_advice
            .extend((untrusted_advice.len()..max_untrusted).map(|index| (index * 31 + 7) as u8));
        assert_eq!(untrusted_advice.len(), max_untrusted);

        let guest =
            support::packed_guest(&mut program, &inputs, &untrusted_advice, &trusted_advice);
        assert_eq!(
            guest.io_device.memory_layout.max_untrusted_advice_size as usize,
            max_untrusted
        );

        let shared: JoltSharedPreprocessing<AkitaPackedScheme> = JoltSharedPreprocessing::new(
            guest.program_data,
            guest.io_device.memory_layout.clone(),
            support::MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing = LegacyProverPreprocessing::new(shared);
        let trusted_object = commit_trusted_advice_one_hot(
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
        );
        let public_io = legacy_prover.program_io.clone();
        let (object_setup, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(
            legacy_prover.one_hot_trace_setup_params(),
        )
        .expect("the transparent packed setup must derive");
        let verifier_preprocessing =
            akita_verifier_preprocessing(&legacy_preprocessing, verifier_setup, None);

        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(guest.elf_contents));
        let memory_layout = &public_io.memory_layout;
        let trace_output = support::trace_modular(
            &jolt_program,
            memory_layout,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
        );
        let program_preprocessing = verifier_preprocessing
            .program
            .as_full_arc()
            .expect("full program preprocessing");
        let config = support::derive_config(&trace_output, memory_layout, &verifier_preprocessing);
        let padded_output = support::pad_trace(trace_output, config.trace_length);
        let witness = TraceBackend::new(
            support::witness_config(&config)
                .include_trusted_advice(true)
                .include_untrusted_advice(true),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
        );
        let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier: verifier_preprocessing,
            pcs_setup: object_setup,
            committed_program: None,
        };

        let backend = akita::JoltAkitaBackend::reference();
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
            Some(&trusted_commitment),
            None,
            &witness,
            &public_io,
        )
        .expect("packed prover should produce a verifier-native proof");

        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &prover_preprocessing.verifier,
            &public_io,
            &proof,
            Some(&trusted_commitment),
        )
        .expect("packed verifier should accept the full-advice proof");
    }
}

#[cfg(all(feature = "prover-fixtures", feature = "akita"))]
#[expect(clippy::expect_used, clippy::panic)]
mod committed {
    use std::sync::Arc;

    use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
    use jolt_program::execution::JoltProgram;
    use jolt_prover::akita;
    use jolt_prover::JoltProverPreprocessing;
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::packed::{
        akita_verifier_preprocessing, shared_preprocessing_with_program_one_hot, AkitaField,
        AkitaJoltProof, AkitaPackedProver, AkitaPackedScheme, AkitaScheme, AkitaTranscript,
        AkitaVc,
    };
    use jolt_prover_legacy::zkvm::prover::{
        JoltCpuProver, JoltProverPreprocessing as LegacyProverPreprocessing,
    };
    use jolt_verifier::proof::JoltProofClaims;
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    /// The committed-program packed e2e: `ProgramOneHot` joins as the second
    /// commitment object (muldiv carries no advice), with tamper rejection
    /// on its claimed evaluation and a reconstruction wire — the analog of
    /// legacy's `muldiv_e2e_akita_committed_program`.
    fn committed_e2e(bytecode_chunk_count: usize) {
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
        let guest = support::packed_guest(&mut program, &inputs, &[], &[]);

        let (shared, prover_data, program_one_hot) = shared_preprocessing_with_program_one_hot(
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
        );
        let public_io = legacy_prover.program_io.clone();
        let (object_setup, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(
            legacy_prover.one_hot_trace_setup_params(),
        )
        .expect("the transparent packed setup must derive");
        let program_one_hot_commitment = program_one_hot.commitment.clone();
        let verifier_preprocessing = akita_verifier_preprocessing(
            &legacy_preprocessing,
            verifier_setup,
            Some(program_one_hot_commitment.clone()),
        );

        // --- Modular side. The full program is rebuilt from the legacy
        // prover data's retained copy for witness generation. NOTE(port): the
        // `ProgramOneHot` opening material does not fit the modular
        // `CommittedProgramProverData` chunk/image shape, so the commitment
        // rides the `prove_packed` argument until the port defines the packed
        // committed-program prover-data shape (`committed_program` stays
        // `None` here).
        let memory_layout = &public_io.memory_layout;
        let full_program = support::rebuild_full_program(
            legacy_preprocessing
                .committed_program_prover_data
                .as_ref()
                .expect("legacy committed prover data"),
            memory_layout,
        );
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(guest.elf_contents));
        let trace_output = support::trace_modular(&jolt_program, memory_layout, &inputs, &[], &[]);
        let config = support::derive_config(&trace_output, memory_layout, &verifier_preprocessing);
        let padded_output = support::pad_trace(trace_output, config.trace_length);
        let witness = TraceBackend::new(
            support::witness_config(&config),
            JoltVmWitnessInputs::new(&jolt_program, &full_program, padded_output),
        );
        let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier: verifier_preprocessing,
            pcs_setup: object_setup,
            committed_program: None,
        };

        let backend = akita::JoltAkitaBackend::reference();
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
            None,
            Some(&program_one_hot_commitment),
            &witness,
            &public_io,
        )
        .expect("packed prover should produce a verifier-native proof");
        assert!(proof.stages.reconstruction_sumcheck_proof.is_some());
        // OneHotTrace is discharged by its native same-point batch;
        // ProgramOneHot is the only auxiliary packed object.
        let auxiliary = proof
            .joint_opening_proof
            .auxiliary
            .as_ref()
            .expect("committed-program mode requires an auxiliary opening");
        assert_eq!(auxiliary.openings.len(), 1);
        assert_eq!(auxiliary.evaluations.len(), 1);

        let verify = |proof: &AkitaJoltProof| {
            jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
                &prover_preprocessing.verifier,
                &public_io,
                proof,
                None,
            )
        };
        verify(&proof).expect("packed verifier should accept the committed packed proof");

        // Tampers: the ProgramOneHot claimed evaluation (last object) breaks
        // its native opening; a mutated reconstruction wire breaks the
        // batched output check.
        let mut tampered = proof.clone();
        tampered
            .joint_opening_proof
            .auxiliary
            .as_mut()
            .expect("committed-program mode requires an auxiliary opening")
            .evaluations[0] += AkitaField::from_u64(1);
        assert!(
            verify(&tampered).is_err(),
            "tampered ProgramOneHot evaluation must be rejected"
        );
        let mut tampered = proof.clone();
        let JoltProofClaims::Clear(claims) = &mut tampered.claims else {
            panic!("packed proofs carry clear claims");
        };
        let bytecode_cell = claims
            .reconstruction
            .bytecode
            .as_mut()
            .expect("committed proofs carry the bytecode reconstruction cell");
        bytecode_cell.pc_bytes[0] += AkitaField::from_u64(1);
        assert!(
            verify(&tampered).is_err(),
            "tampered bytecode reconstruction wire must be rejected"
        );
    }

    #[test]
    fn muldiv_e2e_akita_committed_program() {
        committed_e2e(1);
        committed_e2e(2);
    }
}

#[cfg(not(all(feature = "prover-fixtures", feature = "akita")))]
#[test]
#[ignore = "enable --features akita,prover-fixtures to build the packed (Akita) e2e suite"]
fn muldiv_e2e_akita() {}
