//! ZK end-to-end coverage for the modular prover and verifier.

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests should fail loudly"
)]
mod zk {
    extern crate jolt_inlines_keccak256;

    use std::sync::Arc;

    use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::{DoryCommitment, DoryScheme};
    use jolt_field::{Fr, Ring};
    use jolt_host::{JoltProgramSource, Program};
    use jolt_program::execution::{
        ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
    };
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_prover::dory::DoryProverPreprocessing;
    use jolt_prover::{JoltBackend, JoltSharedPreprocessing, ProverConfig};
    use jolt_riscv::JoltInstructionKind;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_verifier::proof::{JoltProof, JoltProofClaims};
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use tracer::execution_backend::TracerBackend;

    const MAX_PADDED_TRACE_LENGTH: usize = 1 << 16;
    const KECCAK_ROTRI_ROWS: usize = 696;

    type Proof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;

    struct GuestRun {
        program: Arc<JoltProgram>,
        preprocessing: JoltProgramPreprocessing,
        trace: TraceOutput<OwnedTrace>,
    }

    struct ProvedGuest {
        preprocessing: DoryProverPreprocessing,
        public_io: JoltDevice,
        proof: Proof,
        trusted_advice_commitment: Option<DoryCommitment>,
    }

    fn memory_config(layout: &MemoryLayout) -> MemoryConfig {
        MemoryConfig {
            max_untrusted_advice_size: layout.max_untrusted_advice_size,
            max_trusted_advice_size: layout.max_trusted_advice_size,
            max_input_size: layout.max_input_size,
            max_output_size: layout.max_output_size,
            stack_size: layout.stack_size,
            heap_size: layout.heap_size,
            program_size: Some(layout.program_size),
        }
    }

    fn guest_run(
        guest_name: &str,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> GuestRun {
        let mut source = Program::new(guest_name);
        let (_, sizing_trace, _, device) = source.trace(inputs, untrusted_advice, trusted_advice);
        assert!(
            sizing_trace.len().next_power_of_two() <= MAX_PADDED_TRACE_LENGTH,
            "trace exceeds the fixture limit",
        );
        let layout = device.memory_layout;
        let program = Arc::new(source.build_jolt_program().expect("build Jolt program"));
        let preprocessing = JoltProgramPreprocessing::new(
            program.expanded_bytecode.clone(),
            program.memory_init.clone(),
            layout.clone(),
            program.entry_address,
            MAX_PADDED_TRACE_LENGTH,
            source.instruction_profile(),
        )
        .expect("program preprocessing");
        let trace = TracerBackend::new()
            .trace(
                &program,
                TraceInputs::new(
                    inputs.to_vec(),
                    untrusted_advice.to_vec(),
                    trusted_advice.to_vec(),
                    memory_config(&layout),
                ),
            )
            .expect("modular trace");
        GuestRun {
            program,
            preprocessing,
            trace,
        }
    }

    fn derive_config(
        trace: &TraceOutput<OwnedTrace>,
        program: &JoltProgramPreprocessing,
    ) -> ProverConfig {
        ProverConfig::derive::<Fr>(
            trace.trace.rows(),
            &program.memory_layout,
            program.ram.min_bytecode_address,
            program.ram.bytecode_words.len(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive config")
    }

    fn pad_trace(trace: TraceOutput<OwnedTrace>, trace_length: usize) -> TraceOutput<OwnedTrace> {
        let mut rows = trace.trace.rows().to_vec();
        rows.resize(trace_length, TraceRow::default());
        TraceOutput::new(
            OwnedTrace::new(rows),
            trace.device,
            trace.final_memory,
            trace.advice_tape,
        )
    }

    fn prove_guest(
        guest_name: &str,
        inputs: Vec<u8>,
        untrusted_advice: Vec<u8>,
        trusted_advice: Vec<u8>,
        backend: JoltBackend<Fr, DoryScheme>,
        inspect_trace: impl FnOnce(&[TraceRow]),
    ) -> ProvedGuest {
        let run = guest_run(guest_name, &inputs, &untrusted_advice, &trusted_advice);
        inspect_trace(run.trace.trace.rows());
        let config = derive_config(&run.trace, &run.preprocessing);
        let shared = JoltSharedPreprocessing::new(run.preprocessing).expect("shared preprocessing");
        let preprocessing = jolt_prover::dory::from_shared(shared);
        assert!(preprocessing.verifier.vc_setup.is_some());
        let program_preprocessing = preprocessing
            .program_arc()
            .expect("full program preprocessing");
        let public_io = run.trace.device.clone();
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            )
            .include_trusted_advice(!trusted_advice.is_empty())
            .include_untrusted_advice(!untrusted_advice.is_empty()),
            JoltVmWitnessInputs::new(
                &run.program,
                &program_preprocessing,
                pad_trace(run.trace, config.trace_length),
            ),
        );
        let trusted = (!trusted_advice.is_empty()).then(|| {
            jolt_prover::dory::commit_trusted_advice(&preprocessing, &trusted_advice)
                .expect("trusted advice commitment")
        });
        let proof =
            jolt_prover::dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                &backend,
                &preprocessing,
                &config,
                trusted.as_ref(),
                &witness,
                &public_io,
            )
            .expect("modular ZK prove");
        ProvedGuest {
            trusted_advice_commitment: trusted.map(|entry| entry.commitment),
            preprocessing,
            public_io,
            proof,
        }
    }

    fn prove_muldiv(backend: JoltBackend<Fr, DoryScheme>) -> ProvedGuest {
        prove_guest(
            "muldiv-guest",
            postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs"),
            Vec::new(),
            Vec::new(),
            backend,
            |_| {},
        )
    }

    fn verify(proved: &ProvedGuest) -> Result<(), jolt_verifier::VerifierError> {
        jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &proved.preprocessing.verifier,
            &proved.public_io,
            &proved.proof,
            proved.trusted_advice_commitment.as_ref(),
        )
    }

    fn with_zk_stack(body: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(body)
            .expect("spawn ZK test thread")
            .join()
            .expect("ZK test thread panicked");
    }

    #[test]
    fn zk_muldiv_modular_proof_is_accepted() {
        with_zk_stack(|| {
            let proved = prove_muldiv(JoltBackend::reference());
            assert!(matches!(proved.proof.claims, JoltProofClaims::Zk { .. }));
            verify(&proved).expect("modular ZK proof must verify");
        });
    }

    #[test]
    fn zk_muldiv_optimized_backend_proof_is_accepted() {
        with_zk_stack(|| {
            let proved = prove_muldiv(JoltBackend::optimized());
            verify(&proved).expect("optimized ZK proof must verify");
        });
    }

    #[test]
    fn zk_sha3_inline_modular_proof_is_accepted() {
        with_zk_stack(|| {
            let proved = prove_guest(
                "sha3-guest",
                postcard::to_stdvec(&[5u8; 32]).expect("serialize input"),
                Vec::new(),
                Vec::new(),
                JoltBackend::optimized(),
                |rows| {
                    assert_eq!(
                        rows.iter()
                            .filter(|row| {
                                row.instruction.instruction_kind
                                    == JoltInstructionKind::VirtualROTRI
                            })
                            .count(),
                        KECCAK_ROTRI_ROWS,
                    );
                },
            );
            verify(&proved).expect("modular SHA3 ZK proof must verify");
        });
    }

    #[test]
    fn zk_muldiv_tampered_blindfold_is_rejected() {
        with_zk_stack(|| {
            let mut proved = prove_muldiv(JoltBackend::reference());
            let JoltProofClaims::Zk { blindfold_proof } = &mut proved.proof.claims else {
                panic!("ZK proof must carry BlindFold claims");
            };
            blindfold_proof.random_u += Fr::from_u64(1);
            assert!(verify(&proved).is_err());
        });
    }

    #[test]
    fn zk_advice_consumer_modular_proof_is_accepted() {
        with_zk_stack(|| {
            let proved = prove_guest(
                "advice-consumer-guest",
                postcard::to_stdvec(&12u64).expect("serialize input"),
                postcard::to_stdvec(&5u64).expect("serialize untrusted advice"),
                postcard::to_stdvec(&7u64).expect("serialize trusted advice"),
                JoltBackend::reference(),
                |_| {},
            );
            assert!(proved.proof.untrusted_advice_commitment.is_some());
            verify(&proved).expect("modular ZK advice proof must verify");
        });
    }

    #[test]
    fn zk_committed_muldiv_modular_proof_is_accepted() {
        with_zk_stack(|| {
            let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
            let run = guest_run("muldiv-guest", &inputs, &[], &[]);
            let config = derive_config(&run.trace, &run.preprocessing);
            let preprocessing = jolt_prover::dory::preprocess_committed(run.preprocessing, 2)
                .expect("committed preprocessing");
            let program_preprocessing = preprocessing.program_arc().expect("retained full program");
            let public_io = run.trace.device.clone();
            let witness = TraceBackend::new(
                JoltVmWitnessConfig::new(
                    config.trace_length.ilog2() as usize,
                    config.ram_K,
                    config.one_hot_config,
                ),
                JoltVmWitnessInputs::new(
                    &run.program,
                    &program_preprocessing,
                    pad_trace(run.trace, config.trace_length),
                ),
            );
            let proof = jolt_prover::dory::prove::<
                Fr,
                DoryScheme,
                Pedersen<Bn254G1>,
                Blake2bTranscript,
                _,
            >(
                &JoltBackend::reference(),
                &preprocessing,
                &config,
                None,
                &witness,
                &public_io,
            )
            .expect("committed ZK prove");
            jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                &preprocessing.verifier,
                &public_io,
                &proof,
                None,
            )
            .expect("committed ZK proof must verify");
        });
    }
}

#[cfg(not(all(feature = "prover-fixtures", feature = "zk")))]
#[test]
#[ignore = "enable --features prover-fixtures,zk to run the modular ZK e2e"]
fn zk_muldiv_modular_proof_is_accepted() {}
