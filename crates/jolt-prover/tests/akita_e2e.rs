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

#![cfg_attr(
    all(feature = "prover-fixtures", feature = "akita"),
    expect(
        clippy::unwrap_used,
        reason = "end-to-end fixtures fail loudly when guest construction is invalid"
    )
)]

/// Shared scaffolding: the legacy-side guest artifacts every packed test
/// starts from, and the modular-side trace/config/witness pipeline pieces.
#[cfg(all(feature = "prover-fixtures", feature = "akita"))]
#[expect(clippy::expect_used)]
mod support {
    use std::sync::Arc;

    use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
    use jolt_program::execution::{JoltProgram, TraceInputs, TraceOutput};
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_prover::ProverConfig;
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::packed::{AkitaField, AkitaPackedScheme, AkitaScheme, AkitaVc};
    use jolt_prover_legacy::zkvm::program::{
        CommittedProgramProverData as LegacyCommittedProgramProverData,
        ProgramPreprocessing as LegacyProgramPreprocessing,
    };
    use jolt_riscv::JoltTraceRow;
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

    /// Derive the modular config over the packed field. The packed pipeline
    /// is cycle-major only — derivation's default.
    pub fn derive_config(
        trace_output: &TraceOutput<Arc<Vec<JoltTraceRow>>>,
        memory_layout: &MemoryLayout,
        verifier_preprocessing: &JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
    ) -> ProverConfig {
        ProverConfig::derive_compact::<AkitaField>(
            trace_output.trace.as_slice(),
            memory_layout,
            verifier_preprocessing.program.min_bytecode_address(),
            verifier_preprocessing.program.program_image_len_words(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive config")
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
}

#[cfg(all(feature = "prover-fixtures", feature = "akita"))]
#[expect(clippy::expect_used, clippy::panic)]
mod muldiv {
    use std::sync::Arc;

    use jolt_field::Ring;
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
        )
        .unwrap();
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
        let config = support::derive_config(&trace_output, memory_layout, &verifier_preprocessing);
        let witness = TraceBackend::<jolt_program::execution::OwnedTrace>::from_compact(
            support::witness_config(&config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, trace_output),
        );
        let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier: verifier_preprocessing,
            pcs_setup: object_setup,
            committed_program: None,
        };
        let backend = akita::JoltAkitaBackend::optimized();
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
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

        let encoded = bincode::serde::encode_to_vec(&proof, bincode::config::standard())
            .expect("serialize packed proof");
        let (decoded, consumed): (AkitaJoltProof, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard())
                .expect("deserialize packed proof");
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, proof);
        let backend_proof_body_size = proof.joint_opening_proof.backend_proof_body_size();
        let unframed_payload_size = proof
            .joint_opening_proof
            .unframed_payload_size()
            .expect("packed proof component sizes should fit in usize");
        assert!(
            unframed_payload_size >= backend_proof_body_size,
            "the unframed Akita opening contains its headerless proof body"
        );
        assert!(
            encoded.len() > unframed_payload_size,
            "the full Jolt wire proof contains framing beyond the Akita opening"
        );
        verify(&decoded).expect("deserialized packed proof should verify");

        // Live tampers on the fused-inc pipeline's claim wires: the fused
        // increment's reduced claim and the hamming-reduction digit/carry
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
                .balanced_inc_digits[0] += one))
            .is_err(),
            "tampered balanced-inc digit final must be rejected"
        );
        assert!(
            verify(&tamper(&|claims| claims
                .stage7
                .hamming_weight_claim_reduction
                .balanced_inc_carry += one))
            .is_err(),
            "tampered balanced-inc carry final must be rejected"
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
        )
        .unwrap();
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
        let mut config =
            support::derive_config(&trace_output, memory_layout, &verifier_preprocessing);
        config.one_hot_config = jolt_claims::protocols::jolt::JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        };
        let witness = TraceBackend::<jolt_program::execution::OwnedTrace>::from_compact(
            support::witness_config(&config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, trace_output),
        );
        let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier: verifier_preprocessing,
            pcs_setup: object_setup,
            committed_program: None,
        };
        let backend = akita::JoltAkitaBackend::optimized();
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
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
        akita_verifier_preprocessing, commit_trusted_advice, AkitaField, AkitaJoltProof,
        AkitaPackedProver, AkitaPackedScheme, AkitaScheme, AkitaTranscript, AkitaVc,
    };
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::prover::{
        JoltCpuProver, JoltProverPreprocessing as LegacyProverPreprocessing,
    };
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    /// The packed advice e2e: a guest consuming both advice kinds, proved
    /// over three commitment objects (`OneHotTrace`, `UntrustedAdvice`,
    /// `TrustedAdvice`), with per-object tamper rejection — the analog
    /// of legacy's `advice_e2e_akita` (7 + 5 == 12 on the advice-consumer
    /// guest instead of the merkle leaves).
    #[test]
    fn advice_e2e_akita() {
        run_advice_e2e_akita(true);
    }

    /// Untrusted advice is a precommitted batch group in its own right, so it
    /// joins the joint opening even with no trusted precommit present.
    #[test]
    fn untrusted_only_advice_e2e_akita() {
        run_advice_e2e_akita(false);
    }

    fn run_advice_e2e_akita(with_trusted: bool) {
        let mut program = host::Program::new("advice-consumer-guest");
        let inputs = postcard::to_stdvec(&(if with_trusted { 12u64 } else { 5u64 }))
            .expect("serialize inputs");
        let untrusted_advice = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
        let trusted_advice = if with_trusted {
            postcard::to_stdvec(&7u64).expect("serialize trusted advice")
        } else {
            Vec::new()
        };
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
        let trusted_object = with_trusted.then(|| {
            commit_trusted_advice(
                &trusted_advice,
                guest.io_device.memory_layout.max_trusted_advice_size as usize,
            )
            .expect("trusted advice object must commit")
        });
        let trusted_commitment = trusted_object
            .as_ref()
            .map(|object| object.commitment.clone());

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
        .unwrap();
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
        let config = support::derive_config(&trace_output, memory_layout, &verifier_preprocessing);
        let witness = TraceBackend::<jolt_program::execution::OwnedTrace>::from_compact(
            support::witness_config(&config)
                .include_trusted_advice(with_trusted)
                .include_untrusted_advice(true),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, trace_output),
        );
        let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier: verifier_preprocessing,
            pcs_setup: object_setup,
            committed_program: None,
        };

        let modular_trusted_object =
            trusted_object
                .as_ref()
                .map(|object| akita::witness::AdviceObject {
                    plan: object.plan.clone(),
                    commitment: object.commitment.clone(),
                    hint: object.hint.clone(),
                });

        let backend = akita::JoltAkitaBackend::optimized();
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
            modular_trusted_object.as_ref(),
            &witness,
            &public_io,
        )
        .expect("packed prover should produce a verifier-native proof");
        assert!(proof.untrusted_advice_commitment.is_some());

        let verify = |proof: &AkitaJoltProof| {
            jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
                &prover_preprocessing.verifier,
                &public_io,
                proof,
                trusted_commitment.as_ref(),
            )
        };
        verify(&proof).expect("packed verifier should accept the packed proof");

        assert!(
            jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
                &prover_preprocessing.verifier,
                &public_io,
                &proof,
                proof.untrusted_advice_commitment.as_ref(),
            )
            .is_err(),
            "substituting the untrusted commitment for the externally supplied trusted commitment must be rejected"
        );

        let mut tampered = proof.clone();
        let mut encoded_main_batch = serde_json::to_value(&tampered.joint_opening_proof)
            .expect("serialize the main batch opening");
        let schedule_selection = encoded_main_batch
            .get_mut("serialized_schedule_selection")
            .and_then(serde_json::Value::as_array_mut)
            .expect("main batch carries a byte-encoded schedule selection");
        let first_byte = schedule_selection[0]
            .as_u64()
            .expect("schedule selection bytes serialize as integers");
        schedule_selection[0] = serde_json::Value::from(first_byte ^ 1);
        tampered.joint_opening_proof = serde_json::from_value(encoded_main_batch)
            .expect("deserialize the tampered main batch opening");
        assert!(
            verify(&tampered).is_err(),
            "a tampered main-batch schedule selection must be rejected"
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
        .unwrap();
        let public_io = legacy_prover.program_io.clone();
        let (object_setup, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(
            legacy_prover.one_hot_trace_setup_params(),
        )
        .expect("the transparent packed setup must derive");
        let verifier_preprocessing =
            akita_verifier_preprocessing(&legacy_preprocessing, verifier_setup, None);

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
        let config = support::derive_config(&trace_output, memory_layout, &verifier_preprocessing);
        let witness = TraceBackend::<jolt_program::execution::OwnedTrace>::from_compact(
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

        let backend = akita::JoltAkitaBackend::optimized();
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
            Some(&modular_trusted_object),
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

    use jolt_field::Ring;
    use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
    use jolt_program::execution::JoltProgram;
    use jolt_prover::akita;
    use jolt_prover::JoltProverPreprocessing;
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::packed::{
        akita_verifier_preprocessing, shared_preprocessing_with_direct_program, AkitaField,
        AkitaJoltProof, AkitaPackedProver, AkitaPackedScheme, AkitaScheme, AkitaTranscript,
        AkitaVc,
    };
    use jolt_prover_legacy::zkvm::prover::{
        JoltCpuProver, JoltProverPreprocessing as LegacyProverPreprocessing,
    };
    use jolt_verifier::proof::JoltProofClaims;
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    /// The committed-program packed e2e: direct bytecode and program-image
    /// objects join the main trace in one grouped opening — the analog of
    /// legacy's `muldiv_e2e_akita_committed_program`.
    fn committed_e2e(bytecode_chunk_count: usize) {
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
        let guest = support::packed_guest(&mut program, &inputs, &[], &[]);

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
        .unwrap();
        let public_io = legacy_prover.program_io.clone();
        let (object_setup, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(
            legacy_prover.one_hot_trace_setup_params(),
        )
        .expect("the transparent packed setup must derive");
        let verifier_preprocessing = akita_verifier_preprocessing(
            &legacy_preprocessing,
            verifier_setup,
            Some(&direct_program),
        );

        // --- Modular side. The full program is rebuilt from the legacy
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
        let config = support::derive_config(&trace_output, memory_layout, &verifier_preprocessing);
        let witness = TraceBackend::<jolt_program::execution::OwnedTrace>::from_compact(
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

        let backend = akita::JoltAkitaBackend::optimized();
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
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
        verify(&proof).expect("packed verifier should accept the committed packed proof");

        // A mutated direct bytecode claim breaks the grouped opening.
        let mut tampered = proof.clone();
        let JoltProofClaims::Clear(claims) = &mut tampered.claims else {
            panic!("packed proofs carry clear claims");
        };
        claims
            .stage7
            .bytecode_address_phase
            .as_mut()
            .expect("committed fixture carries the bytecode address phase")
            .chunks[0] += AkitaField::from_u64(1);
        assert!(
            verify(&tampered).is_err(),
            "tampered direct bytecode claim must be rejected"
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
