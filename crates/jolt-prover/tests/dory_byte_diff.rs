//! The Dory (elliptic-curve) stage-granular byte-diff harness against
//! `jolt-prover-legacy`.
//!
//! Both provers run from the same guest program and inputs. Legacy's
//! per-stage-boundary transcript states are recovered WITHOUT instrumenting
//! legacy: its accepted proof is replayed through the verifier
//! (`verify_until_stage1`, then stage verifies as stages land), whose
//! Fiat-Shamir transcript is byte-identical to the prover's by soundness of
//! the accepted proof. The harness ratchets one stage at a time; today it
//! pins the ENTIRE clear proof: stages 0 through 8 (config derivation,
//! preamble, witness commitments, both uni-skips, all eight sumcheck
//! batches, all claims, the joint batched opening, and every stage-boundary
//! transcript state), plus the assembled `JoltProof` from the top-level
//! `prove()` — asserted equal to legacy's wire-for-wire and verified
//! end-to-end.
//!
//! `muldiv` is the stage-granular ratchet; the other modules
//! (`advice_consumer`, `committed_muldiv`, `address_major`,
//! `advice_committed`) are whole-proof ratchets over the mode ×
//! trace-order matrix, sharing the `support` scaffolding.
//! `chunk_boundary` is the scale arm: a real 2^17-cycle trace, one power
//! past the optimized backend's 2^16-row streaming chunk, proved both
//! slice-backed and behind a re-emulating source (the forced chunk walk).
//! `wide_one_hot` is the geometry arm: the `{8, 32}` one-hot config the
//! provers derive at log_T ≥ 25, injected at small T under a 2^25-cap SRS.
//!
//! This harness covers the homomorphic (Dory) protocol in clear mode only:
//! under the `akita` feature the wire types swap to the packed envelope and
//! legacy's Dory prove path is compiled out (the packed harness lives in
//! `akita_byte_diff.rs`), and under the `zk` feature both provers emit
//! randomized committed proofs (fresh Pedersen blinds), so byte equality is
//! undefined — the ZK correctness gate is `zk_e2e.rs` instead.

/// Shared scaffolding for the byte-diff modules: every test runs the same
/// legacy-side guest pipeline (decode + trace + preprocess + prove + replay)
/// and the same modular-side pipeline (trace + config + witness + prove +
/// verify); the per-mode differences — advice, committed program, trace
/// order — stay in the test bodies.
#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used)]
mod support {
    use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
    use jolt_claims::protocols::jolt::geometry::claim_reductions::{bytecode, program_image};
    use jolt_claims::protocols::jolt::geometry::dimensions::CommitmentMatrixShape;
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltOneHotConfig, TracePolynomialOrder,
    };
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::{DoryCommitment, DoryScheme};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_kernels::committed_program::{
        build_committed_bytecode_chunk_coeffs, program_image_words_padded,
    };
    use jolt_kernels::CommitmentGrid;
    use jolt_openings::{CommitmentScheme, StreamingCommitment};
    use jolt_program::execution::{
        ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
    };
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_prover::dory::stages::stage0::TrustedAdviceCommitment;
    use jolt_prover::{JoltBackend, ProverConfig};
    use jolt_prover_legacy::curve::Bn254Curve;
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::poly::commitment::commitment_scheme::CommitmentScheme as LegacyCommitmentScheme;
    use jolt_prover_legacy::poly::commitment::dory::{
        DoryCommitmentScheme, DoryContext, DoryGlobals, DoryLayout,
    };
    use jolt_prover_legacy::poly::multilinear_polynomial::MultilinearPolynomial;
    use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
    use jolt_prover_legacy::zkvm::proof::ProofCommitmentScheme;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_prover_legacy::zkvm::ram::populate_memory_states;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_verifier::proof::JoltProof;
    use jolt_verifier::JoltVerifierPreprocessing;
    use jolt_witness::{JoltVmWitnessConfig, TraceBackend};
    use tracer::execution_backend::TracerBackend;

    pub const MAX_PADDED_TRACE_LENGTH: usize = 1 << 16;

    pub type Proof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;
    pub type VerifierPreprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;
    pub type LegacyPreprocessing =
        LegacyProverPreprocessing<LegacyField, Bn254Curve, DoryCommitmentScheme>;
    type LegacyField = jolt_prover_legacy::ark_bn254::Fr;

    /// Force the legacy process-global Dory layout BEFORE any legacy
    /// preprocessing or proving (committed preprocessing bakes it into the
    /// chunk commitments). `DoryGlobals::set_layout` is `cfg(test)`
    /// (unreachable from an external integration test); the pub
    /// initializer's layout parameter stores the same process-global, and
    /// the placeholder dims are overwritten when the legacy prover
    /// re-initializes the main context (preserving the current layout). The
    /// flipped layout is never restored — nextest's process-per-test model
    /// (this workspace's mandated runner) isolates sibling tests.
    pub fn force_legacy_layout(order: TracePolynomialOrder) {
        if order == TracePolynomialOrder::AddressMajor {
            DoryGlobals::initialize_context(
                1,
                2,
                DoryContext::Main,
                Some(DoryLayout::AddressMajor),
            )
            .expect("initialize the main Dory context");
        }
    }

    /// The legacy-side guest artifacts every test starts from: the program
    /// preprocessing, the traced I/O device (for the memory layout), and the
    /// raw ELF the modular side re-traces from.
    pub struct LegacyGuest {
        pub program: LegacyProgramPreprocessing,
        pub io_device: JoltDevice,
        pub elf_contents: Vec<u8>,
    }

    pub fn legacy_guest(
        program: &mut host::Program,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> LegacyGuest {
        let (bytecode, init_memory_state, _, entry_address) = program.decode();
        let (_, _, _, io_device) = program.trace(inputs, untrusted_advice, trusted_advice);
        let elf_contents = program.get_elf_contents().expect("elf contents");
        let preprocessed =
            LegacyProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
                .expect("legacy preprocess");
        LegacyGuest {
            program: preprocessed,
            io_device,
            elf_contents,
        }
    }

    /// Legacy's preprocessing-only trusted-advice commitment, with its
    /// opening hint and its conversion into the new verifier's wire type.
    pub struct LegacyTrustedAdvice {
        pub commitment: <DoryCommitmentScheme as LegacyCommitmentScheme>::Commitment,
        pub hint: <DoryCommitmentScheme as LegacyCommitmentScheme>::OpeningProofHint,
        pub converted: DoryCommitment,
    }

    /// The `commit_trusted_advice_preprocessing_only` replica: pad the bytes
    /// to the layout's maximum advice words, commit in a dedicated balanced
    /// Dory context.
    pub fn legacy_trusted_advice_commit(
        preprocessing: &LegacyPreprocessing,
        trusted_advice: &[u8],
    ) -> LegacyTrustedAdvice {
        let max_trusted_advice_size = preprocessing.shared.memory_layout.max_trusted_advice_size;
        let mut trusted_advice_words = vec![0u64; (max_trusted_advice_size as usize) / 8];
        populate_memory_states(0, trusted_advice, Some(&mut trusted_advice_words), None);
        let poly = MultilinearPolynomial::<LegacyField>::from(trusted_advice_words);
        let advice_len = poly.len().next_power_of_two().max(1);
        let _guard =
            DoryGlobals::initialize_context(1, advice_len, DoryContext::TrustedAdvice, None);
        let _ctx = DoryGlobals::with_context(DoryContext::TrustedAdvice);
        let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &preprocessing.generators);
        let converted =
            <DoryCommitmentScheme as ProofCommitmentScheme<LegacyField>>::commitment_into_verifier(
                commitment,
            );
        LegacyTrustedAdvice {
            commitment,
            hint,
            converted,
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

    /// Derive the modular config, apply the trace order (always a caller
    /// override — derivation picks cycle-major) and any one-hot config
    /// override (the wide-geometry arm injects `{8, 32}` below the 2^25
    /// derivation threshold), and pin every wire config field against what
    /// legacy wrote on the proof.
    pub fn derive_config_pinned(
        trace_output: &TraceOutput<OwnedTrace>,
        memory_layout: &MemoryLayout,
        verifier_preprocessing: &VerifierPreprocessing,
        order: TracePolynomialOrder,
        one_hot_override: Option<JoltOneHotConfig>,
        legacy_proof: &Proof,
        max_padded_trace_length: usize,
    ) -> ProverConfig {
        let mut config = ProverConfig::derive::<Fr>(
            trace_output.trace.rows(),
            memory_layout,
            verifier_preprocessing.program.min_bytecode_address(),
            verifier_preprocessing.program.program_image_len_words(),
            max_padded_trace_length,
        )
        .expect("derive config");
        config.trace_polynomial_order = order;
        if let Some(one_hot_config) = one_hot_override {
            config.one_hot_config = one_hot_config;
        }
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

    /// A word-aligned advice buffer's balanced Dory matrix variable count.
    pub fn advice_vars(max_advice_size_bytes: u64) -> usize {
        ((max_advice_size_bytes / 8) as usize)
            .next_power_of_two()
            .max(1)
            .ilog2() as usize
    }

    /// The PCS setup sizing legacy uses: the main matrix at the largest
    /// supported trace, both advice candidates (always included in setup
    /// sizing, present or not), plus any committed-program candidates. The
    /// SRS is prefix-stable, so an over-sized setup commits identical bytes.
    pub fn setup_total_vars(
        memory_layout: &MemoryLayout,
        extra_candidates: &[usize],
        max_padded_trace_length: usize,
    ) -> usize {
        let max_log_t = max_padded_trace_length.ilog2() as usize;
        // Mirrors legacy's sizing (`compute_max_total_vars`): the chunk
        // width derives from the TRACE CAP — 8-bit chunks at and beyond the
        // 2^25 threshold, 4-bit below.
        let max_log_k_chunk = if max_log_t >= common::constants::ONEHOT_CHUNK_THRESHOLD_LOG_T {
            8usize
        } else {
            4usize
        };
        extra_candidates.iter().copied().fold(
            (max_log_k_chunk + max_log_t)
                .max(advice_vars(memory_layout.max_trusted_advice_size))
                .max(advice_vars(memory_layout.max_untrusted_advice_size)),
            usize::max,
        )
    }

    /// The new-side trusted-advice commit (preprocessing-time in a real
    /// deployment): the commit slot over the advice grid must reproduce
    /// legacy's dedicated-context commitment bytes exactly.
    pub fn modular_trusted_advice_commitment(
        backend: &JoltBackend<Fr, DoryScheme>,
        witness: &TraceBackend<OwnedTrace>,
        memory_layout: &MemoryLayout,
        setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
        expected: &DoryCommitment,
    ) -> TrustedAdviceCommitment<DoryScheme> {
        let mut session = backend.begin_proof();
        let advice_grid = CommitmentGrid {
            total_vars: advice_vars(memory_layout.max_trusted_advice_size),
            log_t: 0,
            log_k_chunk: 0,
            // Advice grids always place cycle-major — see `CommitmentGrid`.
            order: TracePolynomialOrder::CycleMajor,
        };
        let entry = backend
            .commit
            .commit_advice(
                &mut session,
                witness,
                JoltCommittedPolynomial::TrustedAdvice,
                advice_grid,
                setup,
            )
            .expect("trusted advice commit");
        assert_eq!(
            &entry.commitment, expected,
            "new-side trusted-advice commitment diverged from legacy's",
        );
        TrustedAdviceCommitment::<DoryScheme> {
            commitment: entry.commitment,
            hint: entry.hint,
        }
    }

    /// Rebuild the full program preprocessing from the legacy prover data's
    /// retained copy (the verifier preprocessing carries only commitments in
    /// committed mode).
    pub fn rebuild_full_program(
        legacy_preprocessing: &LegacyPreprocessing,
        memory_layout: &MemoryLayout,
    ) -> JoltProgramPreprocessing {
        let legacy_full = &legacy_preprocessing
            .committed_program_prover_data
            .as_ref()
            .expect("legacy committed prover data")
            .full;
        JoltProgramPreprocessing {
            bytecode: legacy_full.bytecode.as_ref().clone(),
            ram: legacy_full.ram.clone(),
            memory_layout: memory_layout.clone(),
            max_padded_trace_length: MAX_PADDED_TRACE_LENGTH,
        }
    }

    /// The committed-program candidate widths (bytecode chunk, program
    /// image) that size the shared grid and the PCS setup.
    pub fn precommitted_candidates(
        verifier_preprocessing: &VerifierPreprocessing,
        bytecode_chunk_count: usize,
    ) -> (usize, usize) {
        let bytecode_candidate = bytecode::precommitted_candidate(
            verifier_preprocessing.program.bytecode_len(),
            bytecode_chunk_count,
        )
        .expect("valid chunking");
        let image_candidate = program_image::precommitted_candidate(
            verifier_preprocessing.program.program_image_len_words(),
        );
        (bytecode_candidate, image_candidate)
    }

    /// Commit one dense table over its balanced matrix through the streaming
    /// path (one row per `row_width` coefficients) — the preprocessing-time
    /// counterpart of the stage-0 advice commit.
    pub fn commit_table(
        table: &[Fr],
        row_width: usize,
        setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
    ) -> (
        DoryCommitment,
        <DoryScheme as CommitmentScheme>::OpeningHint,
    ) {
        let mut partial = DoryScheme::begin(setup);
        for row in table.chunks(row_width) {
            DoryScheme::feed(&mut partial, row, setup);
        }
        DoryScheme::finish_with_hint(partial, setup)
    }

    /// The new-side chunk/image commits (preprocessing-time in a real
    /// deployment): must reproduce legacy's commitment bytes exactly; the
    /// returned hints feed the stage-8 joint opening.
    pub fn commit_committed_program(
        verifier_preprocessing: &VerifierPreprocessing,
        full_program: &JoltProgramPreprocessing,
        bytecode_chunk_count: usize,
        order: TracePolynomialOrder,
        setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
    ) -> (
        Vec<<DoryScheme as CommitmentScheme>::OpeningHint>,
        <DoryScheme as CommitmentScheme>::OpeningHint,
    ) {
        let committed_view = verifier_preprocessing
            .program
            .committed()
            .expect("committed verifier preprocessing");
        let (bytecode_candidate, image_candidate) =
            precommitted_candidates(verifier_preprocessing, bytecode_chunk_count);
        let chunk_tables = build_committed_bytecode_chunk_coeffs::<Fr>(
            &full_program.bytecode.bytecode,
            bytecode_chunk_count,
            order,
        )
        .expect("chunk grids");
        let chunk_shape = CommitmentMatrixShape::balanced(bytecode_candidate);
        let mut bytecode_chunk_hints = Vec::new();
        for (index, table) in chunk_tables.iter().enumerate() {
            let (commitment, hint) =
                commit_table(table, 1usize << chunk_shape.column_vars(), setup);
            assert_eq!(
                commitment, committed_view.bytecode_chunk_commitments[index],
                "bytecode chunk {index} commitment diverged from legacy's",
            );
            bytecode_chunk_hints.push(hint);
        }
        let image_words = program_image_words_padded(&full_program.ram.bytecode_words);
        let image_table: Vec<Fr> = image_words.into_iter().map(Fr::from_u64).collect();
        let image_shape = CommitmentMatrixShape::balanced(image_candidate);
        let (image_commitment, program_image_hint) =
            commit_table(&image_table, 1usize << image_shape.column_vars(), setup);
        assert_eq!(
            image_commitment, committed_view.program_image_commitment,
            "program image commitment diverged from legacy's",
        );
        (bytecode_chunk_hints, program_image_hint)
    }

    /// A re-emulating view of an owned trace: `TraceSource::rows()` (and
    /// `shared_rows`) stay at their `None` defaults, so every witness
    /// collect and the commit stream take the sequential chunked walk —
    /// `COLLECT_ROWS_CHUNK` boundaries, carried lookahead rows, and the
    /// streamed (non-pipelined) consumers — instead of short-circuiting to
    /// the index-parallel slice paths. The walk shape the chunk-boundary
    /// gate exists to pin.
    #[derive(Clone)]
    pub struct HiddenRows(pub OwnedTrace);

    impl jolt_program::execution::TraceSource for HiddenRows {
        fn next_row(&mut self) -> Option<TraceRow> {
            jolt_program::execution::TraceSource::next_row(&mut self.0)
        }
    }

    pub fn verify_modular(
        preprocessing: &VerifierPreprocessing,
        public_io: &JoltDevice,
        proof: &Proof,
        trusted_advice_commitment: Option<&DoryCommitment>,
    ) {
        jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            preprocessing,
            public_io,
            proof,
            trusted_advice_commitment,
        )
        .expect("modular proof must verify end-to-end");
    }
}

#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used, clippy::panic)]
mod muldiv {
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::TracePolynomialOrder;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_program::execution::JoltProgram;
    use jolt_prover::dory::stages::stage0::prove_stage0;
    use jolt_prover::dory::stages::stage8::prove_stage8;
    use jolt_prover::stages::stage1::prove_stage1;
    use jolt_prover::stages::stage2::prove_stage2;
    use jolt_prover::stages::stage3::prove_stage3;
    use jolt_prover::stages::stage4::prove_stage4;
    use jolt_prover::stages::stage5::prove_stage5;
    use jolt_prover::stages::stage6a::prove_stage6a;
    use jolt_prover::stages::stage6b::prove_stage6b;
    use jolt_prover::stages::stage7::prove_stage7;
    use jolt_prover::{JoltBackend, JoltProverPreprocessing};
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_prover_legacy::zkvm::RV64IMACProver;
    use jolt_transcript::{LegacyBlake2bTranscript as Blake2bTranscript, Transcript};
    use jolt_verifier::verify_until_stage1;
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    /// Prove muldiv with both provers from the same guest and inputs; assert
    /// byte equality of every proof component and stage-boundary transcript
    /// state the ratchet covers.
    #[test]
    fn prover_matches_legacy_on_muldiv() {
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");

        // --- Legacy side: preprocess, prove, and recover the pre-stage-1
        // transcript state by replaying the proof through the verifier.
        let guest = support::legacy_guest(&mut program, &inputs, &[], &[]);
        let shared = JoltSharedPreprocessing::new(
            guest.program,
            guest.io_device.memory_layout.clone(),
            support::MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing = LegacyProverPreprocessing::new(shared);
        let legacy_prover = RV64IMACProver::gen_from_elf(
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
        let (legacy_proof, _) = legacy_prover.prove().expect("legacy prove");
        let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);

        let legacy_pre_stage1 = verify_until_stage1::<
            DoryScheme,
            Pedersen<Bn254G1>,
            Blake2bTranscript,
            _,
        >(&verifier_preprocessing, &public_io, &legacy_proof, None)
        .expect("legacy proof must verify through stage 0");

        // --- New-prover side: trace independently through the modular stack.
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(guest.elf_contents));
        let memory_layout = &public_io.memory_layout;
        let trace_output = support::trace_modular(&jolt_program, memory_layout, &inputs, &[], &[]);

        let program_preprocessing = verifier_preprocessing
            .program
            .as_full_arc()
            .expect("full program preprocessing");

        // The derived proof shape must equal what legacy wrote on the wire
        // (asserted inside).
        let config = support::derive_config_pinned(
            &trace_output,
            memory_layout,
            &verifier_preprocessing,
            TracePolynomialOrder::CycleMajor,
            None,
            &legacy_proof,
            support::MAX_PADDED_TRACE_LENGTH,
        );
        let padded_output = support::pad_trace(trace_output, config.trace_length);
        let witness = Arc::new(TraceBackend::new(
            support::witness_config(&config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
        ));

        let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
            verifier: verifier_preprocessing,
            pcs_setup: DoryScheme::setup_prover(support::setup_total_vars(
                memory_layout,
                &[],
                support::MAX_PADDED_TRACE_LENGTH,
            )),
            committed_program: None,
        };

        let jolt_verifier::proof::JoltProofClaims::Clear(legacy_claims) = &legacy_proof.claims
        else {
            panic!("legacy transparent proof must carry clear claims");
        };

        // The per-backend ratchet: proof bytes must be identical to legacy's
        // for ANY backend (spec invariant 8) — run once per backend under
        // test, against the legacy oracle computed above.
        let assert_backend_matches_legacy = |backend: &JoltBackend<Fr, DoryScheme>| {
            let mode = jolt_prover::ProofMode::<Pedersen<Bn254G1>>::new(
                prover_preprocessing.verifier.vc_setup.as_ref(),
            )
            .expect("clear mode needs no vector-commitment setup");
            let mut session = backend.begin_proof();
            let stage0 = prove_stage0::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                backend,
                &mut session,
                &prover_preprocessing,
                &config,
                None,
                witness.as_ref(),
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
            let stage1 = prove_stage1::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                backend,
                &mut session,
                &mode,
                config.trace_length.ilog2() as usize,
                witness.as_ref(),
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
            let stage2 = prove_stage2::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                backend,
                &mut session,
                &mode,
                &config,
                &public_io,
                &stage1.clear_output,
                witness.as_ref(),
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
            let stage3 = prove_stage3::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                backend,
                &mut session,
                &mode,
                &config,
                &stage1.clear_output,
                &stage2.clear_output,
                witness.as_ref(),
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
            let stage4 = prove_stage4::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                backend,
                &mut session,
                &mode,
                &legacy_pre_stage1.checked,
                &config,
                &prover_preprocessing,
                &stage2.clear_output,
                &stage3.clear_output,
                witness.as_ref(),
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
            let stage5 = prove_stage5::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                backend,
                &mut session,
                &mode,
                &legacy_pre_stage1.checked,
                &config,
                &prover_preprocessing,
                &stage2.clear_output,
                &stage4.clear_output,
                witness.as_ref(),
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
            let stage6a = prove_stage6a::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                backend,
                &mut session,
                &mode,
                &legacy_pre_stage1.checked,
                &config,
                &prover_preprocessing,
                &stage1.clear_output,
                &stage2.clear_output,
                &stage3.clear_output,
                &stage4.clear_output,
                &stage5.clear_output,
                witness.as_ref(),
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
                &prover_preprocessing.verifier,
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
            let stage6b = prove_stage6b::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                backend,
                &mut session,
                &mode,
                &legacy_pre_stage1.checked,
                &config,
                &prover_preprocessing,
                &stage1.clear_output,
                &stage2.clear_output,
                &stage3.clear_output,
                &stage4.clear_output,
                &stage5.clear_output,
                &stage6a.clear_output,
                witness.as_ref(),
                &mut new_transcript,
            )
            .expect("stage 6b proves");

            assert_eq!(
                stage6b.sumcheck_proof, legacy_proof.stages.stage6b_sumcheck_proof,
                "stage-6b sumcheck proof bytes diverged from legacy",
            );
            assert_eq!(stage6b.claims, legacy_claims.stage6b);

            let legacy_stage6b = jolt_verifier::stages::stage6b::verify(
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

            // The stage-7 ratchet: prove, then replay legacy's proof through
            // the verifier's stage 7 for the boundary state.
            let stage7 = prove_stage7::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                backend,
                &mut session,
                &mode,
                &legacy_pre_stage1.checked,
                &config,
                &prover_preprocessing,
                &stage4.clear_output,
                &stage6b.clear_output,
                witness.as_ref(),
                &mut new_transcript,
            )
            .expect("stage 7 proves");

            assert_eq!(
                stage7.sumcheck_proof, legacy_proof.stages.stage7_sumcheck_proof,
                "stage-7 sumcheck proof bytes diverged from legacy",
            );
            assert_eq!(stage7.claims, legacy_claims.stage7);

            let legacy_stage7 = jolt_verifier::stages::stage7::verify(
                &legacy_pre_stage1.checked,
                &legacy_proof,
                &formula_dimensions,
                &mut legacy_transcript,
                &legacy_stage4,
                &legacy_stage6b,
            )
            .expect("legacy proof must verify through stage 7");
            assert_eq!(
                new_transcript.state(),
                legacy_transcript.state(),
                "stage-7 transcript state diverged from legacy",
            );

            // The stage-8 ratchet: the joint batched opening, then the final
            // end-of-proof transcript state — the whole clear proof.
            let stage8 = prove_stage8::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                backend,
                &mut session,
                &legacy_pre_stage1.checked,
                &config,
                &prover_preprocessing,
                &stage0.commitments,
                None,
                None,
                &stage0.hints,
                &stage6b.clear_output,
                &stage7.clear_output,
                witness.as_ref(),
                &mut new_transcript,
            )
            .expect("stage 8 proves");

            assert_eq!(
                stage8.joint_opening_proof, legacy_proof.joint_opening_proof,
                "joint opening proof bytes diverged from legacy",
            );

            let _legacy_stage8 = jolt_verifier::stages::stage8::verify(
                &legacy_pre_stage1.checked,
                &prover_preprocessing.verifier,
                &legacy_proof,
                &formula_dimensions,
                None,
                &mut legacy_transcript,
                &legacy_stage6b,
                &legacy_stage7,
            )
            .expect("legacy proof must verify through stage 8");
            assert_eq!(
                new_transcript.state(),
                legacy_transcript.state(),
                "end-of-proof transcript state diverged from legacy",
            );
        };

        assert_backend_matches_legacy(&JoltBackend::reference());
        // The optimized backend must reproduce the same bytes stage by
        // stage (its RAM kernels replace the naive grid materialization).
        assert_backend_matches_legacy(&JoltBackend::optimized());

        // The full-proof ratchet: the top-level prove() runs the same stage
        // sequence on a fresh session and assembles the complete JoltProof —
        // it must equal legacy's wire-for-wire and verify end-to-end.
        for backend in [JoltBackend::reference(), JoltBackend::optimized()] {
            let proof = jolt_prover::dory::prove::<
                Fr,
                DoryScheme,
                Pedersen<Bn254G1>,
                Blake2bTranscript,
                _,
            >(
                &backend,
                &prover_preprocessing,
                &config,
                None,
                Arc::clone(&witness),
                &public_io,
            )
            .expect("top-level prove");
            assert_eq!(proof, legacy_proof, "assembled proof diverged from legacy");
            support::verify_modular(&prover_preprocessing.verifier, &public_io, &proof, None);
        }
    }
}

#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used)]
mod advice_consumer {
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::TracePolynomialOrder;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_program::execution::JoltProgram;
    use jolt_prover::{JoltBackend, JoltProverPreprocessing};
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_prover_legacy::zkvm::RV64IMACProver;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    /// Prove the advice-consumer guest (trusted AND untrusted advice) with
    /// both provers; assert component-wise byte equality of the assembled
    /// proofs (per-stage granularity comes from the component asserts) and
    /// verify the modular proof end-to-end against the trusted commitment.
    #[test]
    fn prover_matches_legacy_on_advice_consumer() {
        advice_consumer_matches_legacy(TracePolynomialOrder::CycleMajor);
    }

    /// The address-major arm (legacy oracle: `advice_e2e_dory_address_major`):
    /// the advice commits and block embeddings are layout-invariant, but the
    /// claim-reduction round schedules and the main-grid commit placement are
    /// not.
    #[test]
    fn prover_matches_legacy_on_advice_consumer_address_major() {
        advice_consumer_matches_legacy(TracePolynomialOrder::AddressMajor);
    }

    fn advice_consumer_matches_legacy(order: TracePolynomialOrder) {
        support::force_legacy_layout(order);
        let mut program = host::Program::new("advice-consumer-guest");
        let inputs = postcard::to_stdvec(&12u64).expect("serialize inputs");
        let untrusted_advice = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
        let trusted_advice = postcard::to_stdvec(&7u64).expect("serialize trusted advice");

        // --- Legacy side, mirroring the jolt-verifier advice fixture: the
        // trusted commitment is produced at preprocessing time (before any
        // proving) and handed to the prover.
        let guest =
            support::legacy_guest(&mut program, &inputs, &untrusted_advice, &trusted_advice);
        let shared = JoltSharedPreprocessing::new(
            guest.program,
            guest.io_device.memory_layout.clone(),
            support::MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing = LegacyProverPreprocessing::new(shared);
        let trusted = support::legacy_trusted_advice_commit(&legacy_preprocessing, &trusted_advice);

        let legacy_prover = RV64IMACProver::gen_from_elf(
            &legacy_preprocessing,
            &guest.elf_contents,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
            Some(trusted.commitment),
            Some(trusted.hint.clone()),
            None,
        );
        let public_io = legacy_prover.program_io.clone();
        let (legacy_proof, _) = legacy_prover.prove().expect("legacy prove");
        let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);

        // --- New-prover side: trace independently with the advice inputs.
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
        let config = support::derive_config_pinned(
            &trace_output,
            memory_layout,
            &verifier_preprocessing,
            order,
            None,
            &legacy_proof,
            support::MAX_PADDED_TRACE_LENGTH,
        );
        let padded_output = support::pad_trace(trace_output, config.trace_length);
        let witness = Arc::new(TraceBackend::new(
            support::witness_config(&config)
                .include_trusted_advice(true)
                .include_untrusted_advice(true),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
        ));

        let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
            verifier: verifier_preprocessing,
            pcs_setup: DoryScheme::setup_prover(support::setup_total_vars(
                memory_layout,
                &[],
                support::MAX_PADDED_TRACE_LENGTH,
            )),
            committed_program: None,
        };

        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let trusted_advice_commitment = support::modular_trusted_advice_commitment(
            &backend,
            &witness,
            memory_layout,
            &prover_preprocessing.pcs_setup,
            &trusted.converted,
        );

        let proof =
            jolt_prover::dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                &backend,
                &prover_preprocessing,
                &config,
                Some(&trusted_advice_commitment),
                Arc::clone(&witness),
                &public_io,
            )
            .expect("top-level prove");

        // Component-wise asserts give per-stage granularity when bytes
        // diverge; the final whole-struct assert is the ratchet.
        assert_eq!(proof.commitments, legacy_proof.commitments);
        assert_eq!(
            proof.untrusted_advice_commitment,
            legacy_proof.untrusted_advice_commitment
        );
        assert_eq!(
            proof.stages.stage1_sumcheck_proof,
            legacy_proof.stages.stage1_sumcheck_proof
        );
        assert_eq!(
            proof.stages.stage4_sumcheck_proof, legacy_proof.stages.stage4_sumcheck_proof,
            "stage-4 bytes diverged (advice openings stage here)",
        );
        assert_eq!(
            proof.stages.stage5_sumcheck_proof,
            legacy_proof.stages.stage5_sumcheck_proof
        );
        assert_eq!(
            proof.stages.stage6a_sumcheck_proof,
            legacy_proof.stages.stage6a_sumcheck_proof
        );
        assert_eq!(
            proof.stages.stage6b_sumcheck_proof, legacy_proof.stages.stage6b_sumcheck_proof,
            "stage-6b bytes diverged (advice cycle phase runs here)",
        );
        assert_eq!(
            proof.stages.stage7_sumcheck_proof, legacy_proof.stages.stage7_sumcheck_proof,
            "stage-7 bytes diverged (advice address phase runs here)",
        );
        assert_eq!(proof.claims, legacy_proof.claims);
        assert_eq!(proof, legacy_proof, "assembled proof diverged from legacy");

        support::verify_modular(
            &prover_preprocessing.verifier,
            &public_io,
            &proof,
            Some(&trusted.converted),
        );

        // The optimized backend must assemble the identical proof — advice
        // exercises the RAM val-check advice cells and the optimized
        // val_init reconstruction against advice-populated initial memory.
        let backend = JoltBackend::<Fr, DoryScheme>::optimized();
        let proof =
            jolt_prover::dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                &backend,
                &prover_preprocessing,
                &config,
                Some(&trusted_advice_commitment),
                Arc::clone(&witness),
                &public_io,
            )
            .expect("optimized-backend prove");
        assert_eq!(
            proof, legacy_proof,
            "optimized-backend proof diverged from legacy"
        );
    }
}

#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used)]
mod committed_muldiv {
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::TracePolynomialOrder;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_program::execution::JoltProgram;
    use jolt_prover::{CommittedProgramProverData, JoltBackend, JoltProverPreprocessing};
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_prover_legacy::zkvm::RV64IMACProver;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    /// Prove muldiv under committed-program preprocessing (chunk count 2,
    /// matching the jolt-verifier committed fixture): the bytecode-chunk
    /// candidate widens the commitment grid columns past the trace length,
    /// exercising the materialized wide-row one-hot commit fallback.
    #[test]
    fn prover_matches_legacy_on_committed_muldiv() {
        committed_muldiv_matches_legacy(2, TracePolynomialOrder::CycleMajor);
    }

    /// The mild-widening arm: at chunk count 64 the bytecode candidate still
    /// exceeds the main matrix (widening the grid) but the columns fit the
    /// trace, so the streaming one-hot commit path runs over the widened
    /// grid.
    #[test]
    fn prover_matches_legacy_on_committed_muldiv_many_chunks() {
        committed_muldiv_matches_legacy(64, TracePolynomialOrder::CycleMajor);
    }

    /// The address-major arms (no legacy test exists, but legacy supports
    /// the combination — the harness's live run IS the oracle): the widened
    /// grid exercises the embedding-extra strides (`one_hot_stride = 2^e`),
    /// and the chunk grids interleave lane/cycle transposed.
    #[test]
    fn prover_matches_legacy_on_committed_muldiv_address_major() {
        committed_muldiv_matches_legacy(2, TracePolynomialOrder::AddressMajor);
    }

    #[test]
    fn prover_matches_legacy_on_committed_muldiv_many_chunks_address_major() {
        committed_muldiv_matches_legacy(64, TracePolynomialOrder::AddressMajor);
    }

    fn committed_muldiv_matches_legacy(bytecode_chunk_count: usize, order: TracePolynomialOrder) {
        support::force_legacy_layout(order);
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");

        // --- Legacy side: committed preprocessing (the chunk/image commits
        // happen here, before any proving), then prove.
        let guest = support::legacy_guest(&mut program, &inputs, &[], &[]);
        let (shared, committed_program_prover_data, generators) =
            JoltSharedPreprocessing::new_committed(
                guest.program,
                guest.io_device.memory_layout.clone(),
                support::MAX_PADDED_TRACE_LENGTH,
                bytecode_chunk_count,
            );
        let legacy_preprocessing = LegacyProverPreprocessing::new_committed(
            shared,
            committed_program_prover_data,
            generators,
        );
        let legacy_prover = RV64IMACProver::gen_from_elf(
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
        let (legacy_proof, _) = legacy_prover.prove().expect("legacy prove");
        let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);

        // --- New-prover side: the full program is rebuilt from the legacy
        // prover data's retained copy (the verifier preprocessing carries only
        // commitments).
        let memory_layout = &public_io.memory_layout;
        let full_program = support::rebuild_full_program(&legacy_preprocessing, memory_layout);
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(guest.elf_contents));
        let trace_output = support::trace_modular(&jolt_program, memory_layout, &inputs, &[], &[]);
        let config = support::derive_config_pinned(
            &trace_output,
            memory_layout,
            &verifier_preprocessing,
            order,
            None,
            &legacy_proof,
            support::MAX_PADDED_TRACE_LENGTH,
        );
        let padded_output = support::pad_trace(trace_output, config.trace_length);
        // The witness owns its own copy: `full_program` itself moves into
        // the prover preprocessing below.
        let witness_program = Arc::new(full_program.clone());
        let witness = Arc::new(TraceBackend::new(
            support::witness_config(&config),
            JoltVmWitnessInputs::new(&jolt_program, &witness_program, padded_output),
        ));

        // Setup sizing: the committed candidates can exceed the main grid.
        let (bytecode_candidate, image_candidate) =
            support::precommitted_candidates(&verifier_preprocessing, bytecode_chunk_count);
        let pcs_setup = DoryScheme::setup_prover(support::setup_total_vars(
            memory_layout,
            &[bytecode_candidate, image_candidate],
            support::MAX_PADDED_TRACE_LENGTH,
        ));
        let (bytecode_chunk_hints, program_image_hint) = support::commit_committed_program(
            &verifier_preprocessing,
            &full_program,
            bytecode_chunk_count,
            order,
            &pcs_setup,
        );

        let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
            verifier: verifier_preprocessing,
            pcs_setup,
            committed_program: Some(CommittedProgramProverData {
                full: full_program,
                bytecode_chunk_hints,
                program_image_hint,
                trace_order: order,
            }),
        };

        // Both backends: the optimized tier ships committed-only slots
        // (bytecode/program-image reductions, the joint opening's
        // block-embedded committed views) that only these arms anchor to
        // legacy bytes.
        for backend in [
            JoltBackend::<Fr, DoryScheme>::reference(),
            JoltBackend::<Fr, DoryScheme>::optimized(),
        ] {
            let proof = jolt_prover::dory::prove::<
                Fr,
                DoryScheme,
                Pedersen<Bn254G1>,
                Blake2bTranscript,
                _,
            >(
                &backend,
                &prover_preprocessing,
                &config,
                None,
                Arc::clone(&witness),
                &public_io,
            )
            .expect("top-level prove");

            // Component-wise asserts give per-stage granularity when bytes
            // diverge; the final whole-struct assert is the ratchet.
            assert_eq!(proof.commitments, legacy_proof.commitments);
            assert_eq!(
                proof.stages.stage4_sumcheck_proof, legacy_proof.stages.stage4_sumcheck_proof,
                "stage-4 bytes diverged (program-image contribution stages here)",
            );
            assert_eq!(
                proof.stages.stage6a_sumcheck_proof, legacy_proof.stages.stage6a_sumcheck_proof,
                "stage-6a bytes diverged (raw val stages staged here)",
            );
            assert_eq!(
                proof.stages.stage6b_sumcheck_proof, legacy_proof.stages.stage6b_sumcheck_proof,
                "stage-6b bytes diverged (committed read-RAF + reduction cycle phases run here)",
            );
            assert_eq!(
                proof.stages.stage7_sumcheck_proof, legacy_proof.stages.stage7_sumcheck_proof,
                "stage-7 bytes diverged (reduction address phases run here)",
            );
            assert_eq!(proof.claims, legacy_proof.claims);
            assert_eq!(proof, legacy_proof, "assembled proof diverged from legacy");

            support::verify_modular(&prover_preprocessing.verifier, &public_io, &proof, None);
        }
    }
}

#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used)]
mod address_major {
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::TracePolynomialOrder;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_program::execution::JoltProgram;
    use jolt_prover::{JoltBackend, JoltProverPreprocessing};
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_prover_legacy::zkvm::RV64IMACProver;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    /// Prove fibonacci under the address-major trace layout with both provers
    /// (legacy oracle: `fib_e2e_dory_address_major`); assert component-wise
    /// byte equality of the assembled proofs and verify the modular proof
    /// end-to-end. Address-major changes the Fiat-Shamir preamble scalar, the
    /// witness commitment placement (cycle-block-strided), and the stage-8
    /// unified point/embeddings; stages 1-7 differ only through the
    /// challenges.
    #[test]
    fn prover_matches_legacy_on_address_major_fibonacci() {
        support::force_legacy_layout(TracePolynomialOrder::AddressMajor);

        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&50u32).expect("serialize inputs");

        // --- Legacy side: standard full preprocessing; the layout rides the
        // process-global set above.
        let guest = support::legacy_guest(&mut program, &inputs, &[], &[]);
        let shared = JoltSharedPreprocessing::new(
            guest.program,
            guest.io_device.memory_layout.clone(),
            support::MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing = LegacyProverPreprocessing::new(shared);
        let legacy_prover = RV64IMACProver::gen_from_elf(
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
        let (legacy_proof, _) = legacy_prover.prove().expect("legacy prove");
        let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);

        // --- New-prover side: trace independently through the modular stack.
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(guest.elf_contents));
        let memory_layout = &public_io.memory_layout;
        let trace_output = support::trace_modular(&jolt_program, memory_layout, &inputs, &[], &[]);
        let program_preprocessing = verifier_preprocessing
            .program
            .as_full_arc()
            .expect("full program preprocessing");
        let config = support::derive_config_pinned(
            &trace_output,
            memory_layout,
            &verifier_preprocessing,
            TracePolynomialOrder::AddressMajor,
            None,
            &legacy_proof,
            support::MAX_PADDED_TRACE_LENGTH,
        );
        let padded_output = support::pad_trace(trace_output, config.trace_length);
        let witness = Arc::new(TraceBackend::new(
            support::witness_config(&config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
        ));

        let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
            verifier: verifier_preprocessing,
            pcs_setup: DoryScheme::setup_prover(support::setup_total_vars(
                memory_layout,
                &[],
                support::MAX_PADDED_TRACE_LENGTH,
            )),
            committed_program: None,
        };

        // Both backends: the optimized commit/opening slots handle the
        // address-major strided placement themselves, so this arm anchors
        // them to legacy bytes too.
        for backend in [
            JoltBackend::<Fr, DoryScheme>::reference(),
            JoltBackend::<Fr, DoryScheme>::optimized(),
        ] {
            let proof = jolt_prover::dory::prove::<
                Fr,
                DoryScheme,
                Pedersen<Bn254G1>,
                Blake2bTranscript,
                _,
            >(
                &backend,
                &prover_preprocessing,
                &config,
                None,
                Arc::clone(&witness),
                &public_io,
            )
            .expect("top-level prove");

            // Component-wise asserts give per-stage granularity when bytes
            // diverge; the final whole-struct assert is the ratchet.
            assert_eq!(
                proof.commitments, legacy_proof.commitments,
                "witness commitments diverged (address-major strided placement commits here)",
            );
            assert_eq!(
                proof.stages.stage1_sumcheck_proof, legacy_proof.stages.stage1_sumcheck_proof,
                "stage-1 bytes diverged (the preamble layout scalar seeds every challenge)",
            );
            assert_eq!(
                proof.stages.stage6b_sumcheck_proof,
                legacy_proof.stages.stage6b_sumcheck_proof
            );
            assert_eq!(
                proof.stages.stage7_sumcheck_proof,
                legacy_proof.stages.stage7_sumcheck_proof
            );
            assert_eq!(
                proof.joint_opening_proof, legacy_proof.joint_opening_proof,
                "joint opening diverged (the [cycle ‖ address] unified point and strided embeddings)",
            );
            assert_eq!(proof.claims, legacy_proof.claims);
            assert_eq!(proof, legacy_proof, "assembled proof diverged from legacy");

            support::verify_modular(&prover_preprocessing.verifier, &public_io, &proof, None);
        }
    }
}

#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used)]
mod advice_committed {
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::TracePolynomialOrder;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_program::execution::JoltProgram;
    use jolt_prover::{CommittedProgramProverData, JoltBackend, JoltProverPreprocessing};
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_prover_legacy::zkvm::RV64IMACProver;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    const BYTECODE_CHUNK_COUNT: usize = 2;

    /// Advice and a committed program TOGETHER — the mixed precommitted
    /// geometry no legacy test exercises (the harness's live legacy run is
    /// the oracle): the advice anchors coexist with the bytecode-chunk and
    /// program-image anchors in the stage-8 dominant-anchor selection, the
    /// advice and committed claim-reduction members share the stage-6b/7
    /// batches, and the advice block embeds ride a committed-widened grid.
    #[test]
    fn prover_matches_legacy_on_committed_advice_consumer() {
        committed_advice_consumer_matches_legacy(TracePolynomialOrder::CycleMajor);
    }

    #[test]
    fn prover_matches_legacy_on_committed_advice_consumer_address_major() {
        committed_advice_consumer_matches_legacy(TracePolynomialOrder::AddressMajor);
    }

    fn committed_advice_consumer_matches_legacy(order: TracePolynomialOrder) {
        support::force_legacy_layout(order);
        let mut program = host::Program::new("advice-consumer-guest");
        let inputs = postcard::to_stdvec(&12u64).expect("serialize inputs");
        let untrusted_advice = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
        let trusted_advice = postcard::to_stdvec(&7u64).expect("serialize trusted advice");

        // --- Legacy side: committed preprocessing AND the preprocessing-time
        // trusted-advice commitment.
        let guest =
            support::legacy_guest(&mut program, &inputs, &untrusted_advice, &trusted_advice);
        let (shared, committed_program_prover_data, generators) =
            JoltSharedPreprocessing::new_committed(
                guest.program,
                guest.io_device.memory_layout.clone(),
                support::MAX_PADDED_TRACE_LENGTH,
                BYTECODE_CHUNK_COUNT,
            );
        let legacy_preprocessing = LegacyProverPreprocessing::new_committed(
            shared,
            committed_program_prover_data,
            generators,
        );
        let trusted = support::legacy_trusted_advice_commit(&legacy_preprocessing, &trusted_advice);
        let legacy_prover = RV64IMACProver::gen_from_elf(
            &legacy_preprocessing,
            &guest.elf_contents,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
            Some(trusted.commitment),
            Some(trusted.hint.clone()),
            None,
        );
        let public_io = legacy_prover.program_io.clone();
        let (legacy_proof, _) = legacy_prover.prove().expect("legacy prove");
        let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);

        // --- New-prover side.
        let memory_layout = &public_io.memory_layout;
        let full_program = support::rebuild_full_program(&legacy_preprocessing, memory_layout);
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(guest.elf_contents));
        let trace_output = support::trace_modular(
            &jolt_program,
            memory_layout,
            &inputs,
            &untrusted_advice,
            &trusted_advice,
        );
        let config = support::derive_config_pinned(
            &trace_output,
            memory_layout,
            &verifier_preprocessing,
            order,
            None,
            &legacy_proof,
            support::MAX_PADDED_TRACE_LENGTH,
        );
        let padded_output = support::pad_trace(trace_output, config.trace_length);
        // The witness owns its own copy: `full_program` itself moves into
        // the prover preprocessing below.
        let witness_program = Arc::new(full_program.clone());
        let witness = Arc::new(TraceBackend::new(
            support::witness_config(&config)
                .include_trusted_advice(true)
                .include_untrusted_advice(true),
            JoltVmWitnessInputs::new(&jolt_program, &witness_program, padded_output),
        ));

        let (bytecode_candidate, image_candidate) =
            support::precommitted_candidates(&verifier_preprocessing, BYTECODE_CHUNK_COUNT);
        let pcs_setup = DoryScheme::setup_prover(support::setup_total_vars(
            memory_layout,
            &[bytecode_candidate, image_candidate],
            support::MAX_PADDED_TRACE_LENGTH,
        ));
        let (bytecode_chunk_hints, program_image_hint) = support::commit_committed_program(
            &verifier_preprocessing,
            &full_program,
            BYTECODE_CHUNK_COUNT,
            order,
            &pcs_setup,
        );
        let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
            verifier: verifier_preprocessing,
            pcs_setup,
            committed_program: Some(CommittedProgramProverData {
                full: full_program,
                bytecode_chunk_hints,
                program_image_hint,
                trace_order: order,
            }),
        };

        // The trusted-advice commit is backend-independent (the optimized
        // backend delegates advice grids to the reference pass) and pinned
        // against legacy's bytes inside; produce it once.
        let trusted_advice_commitment = support::modular_trusted_advice_commitment(
            &JoltBackend::<Fr, DoryScheme>::reference(),
            &witness,
            memory_layout,
            &prover_preprocessing.pcs_setup,
            &trusted.converted,
        );

        // Both backends: the mixed advice + committed-program geometry is
        // the only anchor for the optimized committed slots sharing batches
        // with advice members.
        for backend in [
            JoltBackend::<Fr, DoryScheme>::reference(),
            JoltBackend::<Fr, DoryScheme>::optimized(),
        ] {
            let proof = jolt_prover::dory::prove::<
                Fr,
                DoryScheme,
                Pedersen<Bn254G1>,
                Blake2bTranscript,
                _,
            >(
                &backend,
                &prover_preprocessing,
                &config,
                Some(&trusted_advice_commitment),
                Arc::clone(&witness),
                &public_io,
            )
            .expect("top-level prove");

            // Component-wise asserts give per-stage granularity when bytes
            // diverge; the final whole-struct assert is the ratchet.
            assert_eq!(proof.commitments, legacy_proof.commitments);
            assert_eq!(
                proof.untrusted_advice_commitment,
                legacy_proof.untrusted_advice_commitment
            );
            assert_eq!(
                proof.stages.stage4_sumcheck_proof, legacy_proof.stages.stage4_sumcheck_proof,
                "stage-4 bytes diverged (advice openings and the program-image contribution stage here)",
            );
            assert_eq!(
                proof.stages.stage6a_sumcheck_proof, legacy_proof.stages.stage6a_sumcheck_proof,
                "stage-6a bytes diverged (raw val stages staged here)",
            );
            assert_eq!(
                proof.stages.stage6b_sumcheck_proof, legacy_proof.stages.stage6b_sumcheck_proof,
                "stage-6b bytes diverged (advice AND committed reduction cycle phases share this batch)",
            );
            assert_eq!(
                proof.stages.stage7_sumcheck_proof, legacy_proof.stages.stage7_sumcheck_proof,
                "stage-7 bytes diverged (advice AND committed reduction address phases share this batch)",
            );
            assert_eq!(proof.claims, legacy_proof.claims);
            assert_eq!(proof, legacy_proof, "assembled proof diverged from legacy");

            support::verify_modular(
                &prover_preprocessing.verifier,
                &public_io,
                &proof,
                Some(&trusted.converted),
            );
        }
    }
}

/// The chunk-boundary gate: the optimized backend's streamed trace walks
/// buffer 2^16-row chunks (`jolt-kernels` `COLLECT_ROWS_CHUNK`), and every
/// other byte-parity arm caps at a 2^16 padded trace — no arm crosses a
/// chunk boundary. This one pads to 2^17 (asserted) and proves twice: once
/// slice-backed (the production shape — index-parallel collects and the
/// pipelined commit), and once behind [`support::HiddenRows`], which forces
/// the sequential chunked walk so the boundary carry and the streamed
/// fallback consumers actually produce the pinned bytes.
#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used)]
mod chunk_boundary {
    // Anchor the sha2 inline's inventory registration into this test binary;
    // without it the guest's inline instructions fail bytecode expansion.
    extern crate jolt_inlines_sha2;

    use std::sync::Arc;

    use jolt_claims::protocols::jolt::TracePolynomialOrder;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_program::execution::JoltProgram;
    use jolt_prover::{JoltBackend, JoltProverPreprocessing};
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_prover_legacy::zkvm::RV64IMACProver;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    /// One power past the streaming chunk (and past the other arms' cap).
    const MAX_PADDED_TRACE_LENGTH: usize = 1 << 17;

    /// sha2 iterations landing the raw trace in `(2^16, 2^17]` at ~3396
    /// cycles per inlined hash (the legacy perf harness's calibration), so
    /// the padded trace is exactly 2^17 — asserted below.
    const SHA2_ITERATIONS: u32 = 30;

    #[test]
    fn prover_matches_legacy_on_sha2_chain_across_collect_rows_chunks() {
        let mut program = host::Program::new("sha2-chain-guest");
        let inputs = [
            postcard::to_stdvec(&[5u8; 32]).expect("serialize seed"),
            postcard::to_stdvec(&SHA2_ITERATIONS).expect("serialize iterations"),
        ]
        .concat();

        // --- Legacy side: preprocess and prove at the widened trace cap.
        let guest = support::legacy_guest(&mut program, &inputs, &[], &[]);
        let shared = JoltSharedPreprocessing::new(
            guest.program,
            guest.io_device.memory_layout.clone(),
            MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing = LegacyProverPreprocessing::new(shared);
        let legacy_prover = RV64IMACProver::gen_from_elf(
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
        let (legacy_proof, _) = legacy_prover.prove().expect("legacy prove");
        let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);

        // --- New-prover side: trace independently through the modular stack.
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(guest.elf_contents));
        let memory_layout = &public_io.memory_layout;
        let trace_output = support::trace_modular(&jolt_program, memory_layout, &inputs, &[], &[]);
        let program_preprocessing = verifier_preprocessing
            .program
            .as_full_arc()
            .expect("full program preprocessing");
        let config = support::derive_config_pinned(
            &trace_output,
            memory_layout,
            &verifier_preprocessing,
            TracePolynomialOrder::CycleMajor,
            None,
            &legacy_proof,
            MAX_PADDED_TRACE_LENGTH,
        );
        // The gate's precondition — a shorter trace never crosses the
        // 2^16-row streaming chunk this module exists to exercise.
        assert_eq!(
            config.trace_length,
            1usize << 17,
            "the chunk-boundary gate needs a 2^17 padded trace; retune SHA2_ITERATIONS",
        );
        let padded_output = support::pad_trace(trace_output, config.trace_length);
        // The forced-walk twin of the same padded trace: hiding
        // `TraceSource::rows()` makes every collect and the commit stream
        // take the sequential chunked walk (`COLLECT_ROWS_CHUNK` boundary
        // carries, streamed consumers) — slice-backed sources short-circuit
        // all of it to index-parallel whole-range collection, which the
        // first prove below covers.
        let hidden_output = jolt_program::execution::TraceOutput::new(
            support::HiddenRows(jolt_program::execution::OwnedTrace::new(
                padded_output.trace.rows().to_vec(),
            )),
            padded_output.device.clone(),
            padded_output.final_memory.clone(),
            padded_output.advice_tape.clone(),
        );
        let witness = Arc::new(TraceBackend::new(
            support::witness_config(&config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
        ));

        let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
            verifier: verifier_preprocessing,
            pcs_setup: DoryScheme::setup_prover(support::setup_total_vars(
                memory_layout,
                &[],
                MAX_PADDED_TRACE_LENGTH,
            )),
            committed_program: None,
        };

        // Optimized backend only: the reference tier materializes dense
        // `(K × T)` grids per RAM/instruction polynomial — at a real 2^17
        // trace that is tens of gigabytes and hours of naive-prover walking,
        // and the reference tier has no streaming chunk to gate. The chunked
        // walks under test are the optimized tier's.
        let backend = JoltBackend::<Fr, DoryScheme>::optimized();
        let proof =
            jolt_prover::dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                &backend,
                &prover_preprocessing,
                &config,
                None,
                Arc::clone(&witness),
                &public_io,
            )
            .expect("top-level prove");

        // Component-wise asserts give per-stage granularity when bytes
        // diverge; the final whole-struct assert is the ratchet.
        assert_eq!(proof.commitments, legacy_proof.commitments);
        assert_eq!(
            proof.stages.stage1_sumcheck_proof,
            legacy_proof.stages.stage1_sumcheck_proof
        );
        assert_eq!(
            proof.stages.stage2_sumcheck_proof, legacy_proof.stages.stage2_sumcheck_proof,
            "stage-2 bytes diverged (RAM access columns collect here)",
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
            proof.stages.stage6a_sumcheck_proof,
            legacy_proof.stages.stage6a_sumcheck_proof
        );
        assert_eq!(
            proof.stages.stage6b_sumcheck_proof,
            legacy_proof.stages.stage6b_sumcheck_proof
        );
        assert_eq!(
            proof.stages.stage7_sumcheck_proof,
            legacy_proof.stages.stage7_sumcheck_proof
        );
        assert_eq!(proof.claims, legacy_proof.claims);
        assert_eq!(proof, legacy_proof, "assembled proof diverged from legacy");

        support::verify_modular(&prover_preprocessing.verifier, &public_io, &proof, None);

        // The forced-walk arm: the same trace behind a re-emulating source,
        // so the sequential chunked walk (and its 2^16-row boundary carry)
        // is what actually produces these bytes.
        let hidden_witness = Arc::new(TraceBackend::new(
            support::witness_config(&config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, hidden_output),
        ));
        let hidden_proof =
            jolt_prover::dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                &backend,
                &prover_preprocessing,
                &config,
                None,
                Arc::clone(&hidden_witness),
                &public_io,
            )
            .expect("top-level prove over the re-emulating source");
        assert_eq!(
            hidden_proof, legacy_proof,
            "chunk-walk proof diverged from legacy",
        );
    }
}

/// The wide one-hot geometry gate: at log_T ≥ 25 both provers derive
/// `{log_k_chunk: 8, lookups_ra_virtual_log_k_chunk: 32}`, so the
/// 2^25–2^27 half of the PR's parity claim runs under a geometry no
/// small-trace arm witnesses. Both provers accept an injected config
/// (legacy validates `log_k_chunk ∈ {4, 8}` plus divisibility, the modular
/// wire config carries it), so this arm forces `{8, 32}` on a muldiv trace
/// under a 2^25 trace cap — the cap makes both SRS sizings derive the
/// 8-bit chunk width exactly as a real 2^25 deployment would, while the
/// actual trace stays small.
///
/// NOT covered here: legacy's instruction read-raf phase split (8 × 16-bit
/// at log_T ≥ 24 vs the modular tiers' fixed 16 × 8-bit) derives from the
/// ACTUAL trace length inside the sumcheck, not from any injectable config
/// — that argument rests on the reference kernel's exactness note alone.
#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "zk")
))]
#[expect(clippy::expect_used)]
mod wide_one_hot {
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::{JoltOneHotConfig, TracePolynomialOrder};
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_program::execution::JoltProgram;
    use jolt_prover::{JoltBackend, JoltProverPreprocessing};
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::config::{
        OneHotConfig as LegacyOneHotConfig, OneHotParams as LegacyOneHotParams,
    };
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_prover_legacy::zkvm::RV64IMACProver;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_witness::{JoltVmWitnessInputs, TraceBackend};

    use super::support;

    /// The trace cap of a real wide-geometry deployment: at this cap both
    /// setup sizings pick the 8-bit chunk width.
    const MAX_PADDED_TRACE_LENGTH: usize = 1 << 25;

    const WIDE_LOG_K_CHUNK: u8 = 8;
    const WIDE_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK: u8 = 32;

    #[test]
    fn prover_matches_legacy_on_muldiv_wide_one_hot() {
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");

        // --- Legacy side: preprocess at the 2^25 cap, then inject the wide
        // one-hot params before proving (the prover derives {4, 16} from the
        // small ACTUAL trace; the injected config is what a log_T ≥ 25 run
        // uses, and the verifier accepts it from the proof wire).
        let guest = support::legacy_guest(&mut program, &inputs, &[], &[]);
        let shared = JoltSharedPreprocessing::new(
            guest.program,
            guest.io_device.memory_layout.clone(),
            MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing = LegacyProverPreprocessing::new(shared);
        let mut legacy_prover = RV64IMACProver::gen_from_elf(
            &legacy_preprocessing,
            &guest.elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
            None,
        );
        legacy_prover.one_hot_params = LegacyOneHotParams::from_config(
            &LegacyOneHotConfig {
                log_k_chunk: WIDE_LOG_K_CHUNK,
                lookups_ra_virtual_log_k_chunk: WIDE_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK,
            },
            legacy_prover.one_hot_params.bytecode_len,
            legacy_prover.one_hot_params.ram_k,
        );
        let public_io = legacy_prover.program_io.clone();
        let (legacy_proof, _) = legacy_prover.prove().expect("legacy prove");
        assert_eq!(
            legacy_proof.one_hot_config.log_k_chunk, WIDE_LOG_K_CHUNK,
            "injected wide config must reach the proof wire",
        );
        let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);

        // --- New-prover side: trace independently, inject the same config.
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(guest.elf_contents));
        let memory_layout = &public_io.memory_layout;
        let trace_output = support::trace_modular(&jolt_program, memory_layout, &inputs, &[], &[]);
        let program_preprocessing = verifier_preprocessing
            .program
            .as_full_arc()
            .expect("full program preprocessing");
        let config = support::derive_config_pinned(
            &trace_output,
            memory_layout,
            &verifier_preprocessing,
            TracePolynomialOrder::CycleMajor,
            Some(JoltOneHotConfig {
                log_k_chunk: WIDE_LOG_K_CHUNK,
                lookups_ra_virtual_log_k_chunk: WIDE_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK,
            }),
            &legacy_proof,
            MAX_PADDED_TRACE_LENGTH,
        );
        let padded_output = support::pad_trace(trace_output, config.trace_length);
        let witness = Arc::new(TraceBackend::new(
            support::witness_config(&config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
        ));

        let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
            verifier: verifier_preprocessing,
            pcs_setup: DoryScheme::setup_prover(support::setup_total_vars(
                memory_layout,
                &[],
                MAX_PADDED_TRACE_LENGTH,
            )),
            committed_program: None,
        };

        // Both backends: the optimized kernels' chunk arithmetic (8-bit RA
        // chunks, 32-bit virtual batches, the halved d counts) gets its
        // legacy byte anchor here.
        for backend in [
            JoltBackend::<Fr, DoryScheme>::reference(),
            JoltBackend::<Fr, DoryScheme>::optimized(),
        ] {
            let proof = jolt_prover::dory::prove::<
                Fr,
                DoryScheme,
                Pedersen<Bn254G1>,
                Blake2bTranscript,
                _,
            >(
                &backend,
                &prover_preprocessing,
                &config,
                None,
                Arc::clone(&witness),
                &public_io,
            )
            .expect("top-level prove");

            // Component-wise asserts give per-stage granularity when bytes
            // diverge; the final whole-struct assert is the ratchet.
            assert_eq!(
                proof.commitments, legacy_proof.commitments,
                "witness commitments diverged (16 instruction RA chunk grids commit here)",
            );
            assert_eq!(
                proof.stages.stage5_sumcheck_proof, legacy_proof.stages.stage5_sumcheck_proof,
                "stage-5 bytes diverged (the 32-bit virtual RA batches run here)",
            );
            assert_eq!(
                proof.stages.stage6a_sumcheck_proof,
                legacy_proof.stages.stage6a_sumcheck_proof
            );
            assert_eq!(
                proof.stages.stage6b_sumcheck_proof, legacy_proof.stages.stage6b_sumcheck_proof,
                "stage-6b bytes diverged (8-bit committed RA virtualization runs here)",
            );
            assert_eq!(
                proof.stages.stage7_sumcheck_proof,
                legacy_proof.stages.stage7_sumcheck_proof
            );
            assert_eq!(proof.claims, legacy_proof.claims);
            assert_eq!(proof, legacy_proof, "assembled proof diverged from legacy");

            support::verify_modular(&prover_preprocessing.verifier, &public_io, &proof, None);
        }
    }
}

#[cfg(not(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "zk")
)))]
#[test]
#[ignore = "enable --features prover-fixtures (without akita or zk — the harness byte-compares \
            clear Dory proofs; one compiled prover proves one protocol) to run the legacy \
            byte-diff harness"]
fn prover_matches_legacy_on_muldiv() {}
