//! ZK end-to-end: the modular prover's BlindFold proofs against the
//! unmodified `jolt-verifier` ZK path.
//!
//! ZK proofs are randomized (fresh Pedersen blinds, the Nova random
//! instance, hiding commitments), so there is no byte oracle — the
//! correctness bar is acceptance by the verifier the legacy ZK prover
//! already targets, plus a tamper rejection. The committed-program variant
//! commits the chunk/image tables through the modular hiding streaming path
//! and carries those commitments in the verifier preprocessing, as a real
//! ZK deployment's preprocessing would.

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[expect(clippy::expect_used, reason = "integration tests should fail loudly")]
mod support {
    use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
    use jolt_claims::protocols::jolt::geometry::claim_reductions::{bytecode, program_image};
    use jolt_claims::protocols::jolt::geometry::dimensions::CommitmentMatrixShape;
    use jolt_claims::protocols::jolt::TracePolynomialOrder;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::{DoryCommitment, DoryScheme};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_kernels::committed_program::{
        build_committed_bytecode_chunk_coeffs, program_image_words_padded,
    };
    use jolt_openings::{CommitmentScheme, StreamingCommitment, ZkStreamingCommitment};
    use jolt_program::execution::{
        ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
    };
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_prover::ProverConfig;
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_verifier::proof::JoltProof;
    use jolt_verifier::JoltVerifierPreprocessing;
    use jolt_witness::protocols::jolt_vm::JoltVmWitnessConfig;
    use tracer::execution_backend::TracerBackend;

    pub const MAX_PADDED_TRACE_LENGTH: usize = 1 << 16;

    pub type Proof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;
    pub type VerifierPreprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;
    pub type LegacyPreprocessing = jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing<
        jolt_prover_legacy::ark_bn254::Fr,
        jolt_prover_legacy::curve::Bn254Curve,
        jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme,
    >;

    /// The legacy-side guest artifacts: program preprocessing, the traced
    /// I/O device (for the memory layout), and the raw ELF the modular side
    /// re-traces from.
    pub struct LegacyGuest {
        pub program: LegacyProgramPreprocessing,
        pub io_device: JoltDevice,
        pub elf_contents: Vec<u8>,
    }

    pub fn legacy_guest(program: &mut host::Program, inputs: &[u8]) -> LegacyGuest {
        let (bytecode, init_memory_state, _, entry_address) = program.decode();
        let (_, _, _, io_device) = program.trace(inputs, &[], &[]);
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

    /// Trace the guest through the modular stack, memory config mirrored off
    /// the legacy run's layout.
    pub fn trace_modular(
        program: &JoltProgram,
        memory_layout: &MemoryLayout,
        inputs: &[u8],
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
                    untrusted_advice: Vec::new(),
                    trusted_advice: Vec::new(),
                    memory_config,
                },
            )
            .expect("modular trace")
    }

    pub fn derive_config(
        trace_output: &TraceOutput<OwnedTrace>,
        memory_layout: &MemoryLayout,
        verifier_preprocessing: &VerifierPreprocessing,
    ) -> ProverConfig {
        ProverConfig::derive::<Fr>(
            trace_output.trace.rows(),
            memory_layout,
            verifier_preprocessing.program.min_bytecode_address(),
            verifier_preprocessing.program.program_image_len_words(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive config")
    }

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
        )
    }

    pub fn witness_config(config: &ProverConfig) -> JoltVmWitnessConfig {
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        )
    }

    fn advice_vars(max_advice_size_bytes: u64) -> usize {
        ((max_advice_size_bytes / 8) as usize)
            .next_power_of_two()
            .max(1)
            .ilog2() as usize
    }

    pub fn setup_total_vars(memory_layout: &MemoryLayout, extra_candidates: &[usize]) -> usize {
        let max_log_t = MAX_PADDED_TRACE_LENGTH.ilog2() as usize;
        let max_log_k_chunk = 4usize; // max_log_t = 16 < the 25-bit threshold
        extra_candidates.iter().copied().fold(
            (max_log_k_chunk + max_log_t)
                .max(advice_vars(memory_layout.max_trusted_advice_size))
                .max(advice_vars(memory_layout.max_untrusted_advice_size)),
            usize::max,
        )
    }

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

    /// Commit one dense table through the hiding streaming path — the
    /// ZK-preprocessing-time counterpart of the byte-diff harness's
    /// transparent `commit_table`.
    fn commit_table_zk(
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
        DoryScheme::finish_zk_with_hint(partial, setup)
    }

    /// The ZK committed-program commits: chunk/image tables through the
    /// hiding path. Unlike the transparent byte-diff harness (which asserts
    /// byte equality against legacy's preprocessing commitments), hiding
    /// commits are randomized — the returned commitments REPLACE the
    /// preprocessing's, and the hints (which carry the blinds) feed the
    /// prover.
    #[expect(clippy::type_complexity, reason = "the two hint families")]
    pub fn commit_committed_program_zk(
        verifier_preprocessing: &VerifierPreprocessing,
        full_program: &JoltProgramPreprocessing,
        bytecode_chunk_count: usize,
        order: TracePolynomialOrder,
        setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
    ) -> (
        Vec<(
            DoryCommitment,
            <DoryScheme as CommitmentScheme>::OpeningHint,
        )>,
        (
            DoryCommitment,
            <DoryScheme as CommitmentScheme>::OpeningHint,
        ),
    ) {
        let (bytecode_candidate, image_candidate) =
            precommitted_candidates(verifier_preprocessing, bytecode_chunk_count);
        let chunk_tables = build_committed_bytecode_chunk_coeffs::<Fr>(
            &full_program.bytecode.bytecode,
            bytecode_chunk_count,
            order,
        )
        .expect("chunk grids");
        let chunk_shape = CommitmentMatrixShape::balanced(bytecode_candidate);
        let chunks = chunk_tables
            .iter()
            .map(|table| commit_table_zk(table, 1usize << chunk_shape.column_vars(), setup))
            .collect();
        let image_words = program_image_words_padded(&full_program.ram.bytecode_words);
        let image_table: Vec<Fr> = image_words.into_iter().map(Fr::from_u64).collect();
        let image_shape = CommitmentMatrixShape::balanced(image_candidate);
        let image = commit_table_zk(&image_table, 1usize << image_shape.column_vars(), setup);
        (chunks, image)
    }

    pub fn verify_modular(
        preprocessing: &VerifierPreprocessing,
        public_io: &JoltDevice,
        proof: &Proof,
    ) -> Result<(), jolt_verifier::VerifierError> {
        jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            preprocessing,
            public_io,
            proof,
            None,
        )
    }

    /// BlindFold verification (and the prover's replay of it) recurses over
    /// a large folded R1CS — run on a dedicated wide stack like the
    /// jolt-verifier ZK suites.
    pub fn with_zk_stack<R: Send + 'static>(body: impl FnOnce() -> R + Send + 'static) -> R {
        std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(body)
            .expect("spawn ZK test thread")
            .join()
            .expect("ZK test thread panicked")
    }
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests should fail loudly"
)]
mod zk {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_program::execution::JoltProgram;
    use jolt_prover::{CommittedProgramProverData, JoltBackend, JoltProverPreprocessing};
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_verifier::preprocessing::ProgramPreprocessing;
    use jolt_verifier::proof::JoltProofClaims;
    use jolt_witness::protocols::jolt_vm::{JoltVmWitnessInputs, TraceBackedJoltVmWitness};

    use super::support;

    fn prove_muldiv_zk() -> (
        support::VerifierPreprocessing,
        common::jolt_device::JoltDevice,
        support::Proof,
    ) {
        let mut program = host::Program::new("muldiv-guest");
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");

        // Legacy host preprocessing carries the program metadata and — under
        // the zk feature — the BlindFold vector-commitment setup.
        let guest = support::legacy_guest(&mut program, &inputs);
        let io_device = guest.io_device.clone();
        let shared = JoltSharedPreprocessing::new(
            guest.program,
            guest.io_device.memory_layout.clone(),
            support::MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing: support::LegacyPreprocessing =
            LegacyProverPreprocessing::new(shared);
        let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);
        assert!(
            verifier_preprocessing.vc_setup.is_some(),
            "zk-compiled legacy preprocessing must carry the BlindFold setup",
        );

        let jolt_program = JoltProgram::from_elf_bytes(guest.elf_contents);
        let memory_layout = io_device.memory_layout.clone();
        let trace_output = support::trace_modular(&jolt_program, &memory_layout, &inputs);
        let public_io = trace_output.device.clone();
        let program_preprocessing = verifier_preprocessing
            .program
            .as_full()
            .expect("full program preprocessing")
            .clone();
        let config = support::derive_config(&trace_output, &memory_layout, &verifier_preprocessing);
        let padded_output = support::pad_trace(trace_output, config.trace_length);
        let witness = TraceBackedJoltVmWitness::new(
            support::witness_config(&config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
        );

        let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
            verifier: verifier_preprocessing,
            pcs_setup: DoryScheme::setup_prover(support::setup_total_vars(&memory_layout, &[])),
            committed_program: None,
        };
        let backend = JoltBackend::<Fr, DoryScheme>::reference();
        let proof = jolt_prover::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
            None,
            &witness,
            &public_io,
        )
        .expect("modular ZK prove");
        (prover_preprocessing.verifier, public_io, proof)
    }

    #[test]
    fn zk_muldiv_modular_proof_is_accepted() {
        support::with_zk_stack(|| {
            let (preprocessing, public_io, proof) = prove_muldiv_zk();
            assert!(matches!(proof.claims, JoltProofClaims::Zk { .. }));
            support::verify_modular(&preprocessing, &public_io, &proof)
                .expect("modular ZK proof must verify");
        });
    }

    #[test]
    fn zk_muldiv_tampered_blindfold_is_rejected() {
        support::with_zk_stack(|| {
            let (preprocessing, public_io, mut proof) = prove_muldiv_zk();
            let JoltProofClaims::Zk { blindfold_proof } = &mut proof.claims else {
                panic!("ZK proof must carry the BlindFold claims variant");
            };
            blindfold_proof.random_u += Fr::from(1u64);
            assert!(
                support::verify_modular(&preprocessing, &public_io, &proof).is_err(),
                "a tampered BlindFold proof must be rejected",
            );
        });
    }

    #[test]
    fn zk_committed_muldiv_modular_proof_is_accepted() {
        support::with_zk_stack(|| {
            let bytecode_chunk_count = 2usize;
            let order = jolt_claims::protocols::jolt::TracePolynomialOrder::CycleMajor;
            let mut program = host::Program::new("muldiv-guest");
            let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");

            let guest = support::legacy_guest(&mut program, &inputs);
            let io_device = guest.io_device.clone();
            let (shared, committed_program_prover_data, generators) =
                JoltSharedPreprocessing::new_committed(
                    guest.program,
                    guest.io_device.memory_layout.clone(),
                    support::MAX_PADDED_TRACE_LENGTH,
                    bytecode_chunk_count,
                );
            let legacy_preprocessing: support::LegacyPreprocessing =
                LegacyProverPreprocessing::new_committed(
                    shared,
                    committed_program_prover_data,
                    generators,
                );
            let mut verifier_preprocessing =
                verifier_preprocessing_from_prover(&legacy_preprocessing);

            // Rebuild the full program from legacy's retained copy (the
            // committed verifier preprocessing carries only commitments).
            let memory_layout = io_device.memory_layout.clone();
            let legacy_full = &legacy_preprocessing
                .committed_program_prover_data
                .as_ref()
                .expect("legacy committed prover data")
                .full;
            let full_program = jolt_program::preprocess::JoltProgramPreprocessing {
                bytecode: legacy_full.bytecode.as_ref().clone(),
                ram: legacy_full.ram.clone(),
                memory_layout: memory_layout.clone(),
                max_padded_trace_length: support::MAX_PADDED_TRACE_LENGTH,
            };
            let jolt_program = JoltProgram::from_elf_bytes(guest.elf_contents);
            let trace_output = support::trace_modular(&jolt_program, &memory_layout, &inputs);
            let public_io = trace_output.device.clone();
            let config =
                support::derive_config(&trace_output, &memory_layout, &verifier_preprocessing);
            let padded_output = support::pad_trace(trace_output, config.trace_length);
            let witness_program = full_program.clone();
            let witness = TraceBackedJoltVmWitness::new(
                support::witness_config(&config),
                JoltVmWitnessInputs::new(&jolt_program, &witness_program, padded_output),
            );

            // ZK committed-program preprocessing: hiding chunk/image commits
            // replace legacy's (randomized blinds make byte reuse impossible);
            // the hints carry the matching blinds into the stage-8 opening.
            let (bytecode_candidate, image_candidate) =
                support::precommitted_candidates(&verifier_preprocessing, bytecode_chunk_count);
            let pcs_setup = DoryScheme::setup_prover(support::setup_total_vars(
                &memory_layout,
                &[bytecode_candidate, image_candidate],
            ));
            let (chunks, image) = support::commit_committed_program_zk(
                &verifier_preprocessing,
                &full_program,
                bytecode_chunk_count,
                order,
                &pcs_setup,
            );
            {
                let ProgramPreprocessing::Committed(committed) =
                    &mut verifier_preprocessing.program
                else {
                    panic!("committed preprocessing expected");
                };
                committed.bytecode_chunk_commitments = chunks
                    .iter()
                    .map(|(commitment, _)| commitment.clone())
                    .collect();
                committed.program_image_commitment = image.0;
            }

            let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
                verifier: verifier_preprocessing,
                pcs_setup,
                committed_program: Some(CommittedProgramProverData {
                    full: full_program,
                    bytecode_chunk_hints: chunks.into_iter().map(|(_, hint)| hint).collect(),
                    program_image_hint: image.1,
                    trace_order: order,
                }),
            };
            let backend = JoltBackend::<Fr, DoryScheme>::reference();
            let proof =
                jolt_prover::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                    &backend,
                    &prover_preprocessing,
                    &config,
                    None,
                    &witness,
                    &public_io,
                )
                .expect("modular committed ZK prove");
            support::verify_modular(&prover_preprocessing.verifier, &public_io, &proof)
                .expect("modular committed ZK proof must verify");
        });
    }
}

#[cfg(not(all(feature = "prover-fixtures", feature = "zk")))]
#[test]
#[ignore = "enable --features prover-fixtures,zk to run the modular ZK e2e"]
fn zk_muldiv_modular_proof_is_accepted() {}
