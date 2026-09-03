//! End-to-end coverage for the modular Akita prover and verifier.

#[cfg(all(feature = "prover-fixtures", feature = "akita"))]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests should fail loudly"
)]
mod akita_tests {
    use std::sync::Arc;

    use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
    use jolt_akita::{AkitaCommitment, AkitaField, AkitaScheme};
    use jolt_field::Ring;
    use jolt_claims::protocols::jolt::{JoltOneHotConfig, TracePolynomialOrder};
    use jolt_host::{JoltProgramSource, Program};
    use jolt_program::execution::{JoltProgram, OwnedTrace, TraceInputs, TraceOutput};
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_prover::akita::preprocessing::{
        self, AkitaProverPreprocessing, AkitaTranscript, AkitaVc,
    };
    use jolt_prover::akita::{self, JoltAkitaBackend};
    use jolt_prover::ProverConfig;
    use jolt_riscv::JoltTraceRow;
    use jolt_verifier::proof::{ClearProofClaims, JoltProof, JoltProofClaims};
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use tracer::execution_backend::TracerBackend;

    const MAX_PADDED_TRACE_LENGTH: usize = 1 << 16;

    type Proof = JoltProof<AkitaScheme, AkitaVc>;

    struct GuestRun {
        program: Arc<JoltProgram>,
        preprocessing: JoltProgramPreprocessing,
        trace: TraceOutput<Arc<Vec<JoltTraceRow>>>,
    }

    struct ProvedGuest {
        preprocessing: AkitaProverPreprocessing,
        public_io: JoltDevice,
        proof: Proof,
        trusted_advice_commitment: Option<AkitaCommitment>,
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
            .trace_compact(
                &program,
                TraceInputs::new(
                    inputs.to_vec(),
                    untrusted_advice.to_vec(),
                    trusted_advice.to_vec(),
                    memory_config(&layout),
                ),
                &preprocessing.bytecode,
            )
            .expect("modular trace");
        GuestRun {
            program,
            preprocessing,
            trace,
        }
    }

    fn derive_config(run: &GuestRun) -> ProverConfig {
        ProverConfig::derive_compact::<AkitaField>(
            run.trace.trace.as_slice(),
            &run.preprocessing.memory_layout,
            run.preprocessing.ram.min_bytecode_address,
            run.preprocessing.ram.bytecode_words.len(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive config")
    }

    fn witness_config(
        config: &ProverConfig,
        untrusted_advice: bool,
        trusted_advice: bool,
    ) -> JoltVmWitnessConfig {
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        )
        .include_untrusted_advice(untrusted_advice)
        .include_trusted_advice(trusted_advice)
    }

    fn prove_guest(
        run: GuestRun,
        config: ProverConfig,
        untrusted_advice: bool,
        trusted_advice: &[u8],
    ) -> ProvedGuest {
        let has_trusted_advice = !trusted_advice.is_empty();
        let preprocessing = preprocessing::preprocess_full_with_advice(
            run.preprocessing,
            &config,
            untrusted_advice,
            has_trusted_advice,
        )
        .expect("Akita preprocessing");
        let trusted = has_trusted_advice.then(|| {
            preprocessing::commit_trusted_advice(&preprocessing, trusted_advice)
                .expect("trusted advice commitment")
        });
        let trusted_advice_commitment = trusted.as_ref().map(|object| object.commitment.clone());
        let program_preprocessing = preprocessing
            .program_arc()
            .expect("full program preprocessing");
        let public_io = run.trace.device.clone();
        let witness = TraceBackend::<OwnedTrace>::from_compact(
            witness_config(&config, untrusted_advice, has_trusted_advice),
            JoltVmWitnessInputs::new(&run.program, &program_preprocessing, run.trace),
        );
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &JoltAkitaBackend::optimized(),
            &preprocessing,
            &config,
            trusted.as_ref(),
            &witness,
            &public_io,
        )
        .expect("Akita proof");
        ProvedGuest {
            preprocessing,
            public_io,
            proof,
            trusted_advice_commitment,
        }
    }

    fn verify(proved: &ProvedGuest) -> Result<(), jolt_verifier::VerifierError> {
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &proved.preprocessing.verifier,
            &proved.public_io,
            &proved.proof,
            proved.trusted_advice_commitment.as_ref(),
        )
    }

    fn muldiv_run() -> (GuestRun, ProverConfig) {
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
        let run = guest_run("muldiv-guest", &inputs, &[], &[]);
        let config = derive_config(&run);
        (run, config)
    }

    #[test]
    fn muldiv_e2e_akita() {
        let (run, config) = muldiv_run();
        assert_eq!(config.one_hot_config.committed_chunk_bits(), 4);
        let proved = prove_guest(run, config, false, &[]);
        verify(&proved).expect("Akita proof must verify");

        let tamper = |mutate: &dyn Fn(&mut ClearProofClaims<AkitaField>)| {
            let mut proof = proved.proof.clone();
            let JoltProofClaims::Clear(claims) = &mut proof.claims else {
                panic!("Akita proofs carry clear claims");
            };
            mutate(claims);
            proof
        };
        let one = AkitaField::from_u64(1);
        for proof in [
            tamper(&|claims| claims.stage6b.bytecode_read_raf.fused_inc += one),
            tamper(&|claims| {
                claims
                    .stage7
                    .hamming_weight_claim_reduction
                    .balanced_inc_digits[0] += one;
            }),
            tamper(&|claims| {
                claims
                    .stage7
                    .hamming_weight_claim_reduction
                    .balanced_inc_carry += one;
            }),
        ] {
            let tampered = ProvedGuest {
                preprocessing: proved.preprocessing.clone(),
                public_io: proved.public_io.clone(),
                proof,
                trusted_advice_commitment: None,
            };
            assert!(verify(&tampered).is_err());
        }
    }

    #[test]
    fn muldiv_e2e_akita_forced_k256() {
        let (run, mut config) = muldiv_run();
        config.one_hot_config = JoltOneHotConfig {
            log_k_chunk: 8,
            lookups_ra_virtual_log_k_chunk: 32,
        };
        let proved = prove_guest(run, config, false, &[]);
        verify(&proved).expect("forced-K256 proof must verify");
    }

    #[test]
    fn akita_rejects_address_major_preprocessing_and_proving() {
        let (run, mut config) = muldiv_run();
        config.trace_polynomial_order = TracePolynomialOrder::AddressMajor;

        let result = preprocessing::preprocess_full(run.preprocessing, &config);
        assert!(matches!(
            result,
            Err(jolt_prover::PreprocessingError::InvalidConfiguration { .. })
        ));

        let (run, mut config) = muldiv_run();
        let preprocessing = preprocessing::preprocess_full(run.preprocessing, &config)
            .expect("cycle-major preprocessing");
        config.trace_polynomial_order = TracePolynomialOrder::AddressMajor;
        let program_preprocessing = preprocessing.program_arc().expect("full program");
        let public_io = run.trace.device.clone();
        let witness = TraceBackend::<OwnedTrace>::from_compact(
            witness_config(&config, false, false),
            JoltVmWitnessInputs::new(&run.program, &program_preprocessing, run.trace),
        );
        let result = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &JoltAkitaBackend::optimized(),
            &preprocessing,
            &config,
            None,
            &witness,
            &public_io,
        );
        assert!(matches!(
            result,
            Err(jolt_prover::ProverError::Unsupported {
                reason: "Akita supports only cycle-major trace polynomials"
            })
        ));
    }

    #[test]
    fn advice_e2e_akita() {
        for with_trusted in [false, true] {
            let inputs = postcard::to_stdvec(&(if with_trusted { 12u64 } else { 5u64 }))
                .expect("serialize inputs");
            let untrusted = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
            let trusted = if with_trusted {
                postcard::to_stdvec(&7u64).expect("serialize trusted advice")
            } else {
                Vec::new()
            };
            let run = guest_run("advice-consumer-guest", &inputs, &untrusted, &trusted);
            let config = derive_config(&run);
            let proved = prove_guest(run, config, true, &trusted);
            assert!(proved.proof.untrusted_advice_commitment.is_some());
            verify(&proved).expect("advice proof must verify");
        }
    }

    #[test]
    fn advice_e2e_akita_full_advice() {
        let inputs = postcard::to_stdvec(&12u64).expect("serialize inputs");
        let trusted = postcard::to_stdvec(&7u64).expect("serialize trusted advice");
        let capacity = common::constants::DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE as usize;
        let mut untrusted = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
        untrusted.extend((untrusted.len()..capacity).map(|index| (index * 31 + 7) as u8));
        let run = guest_run("advice-consumer-guest", &inputs, &untrusted, &trusted);
        let config = derive_config(&run);
        let proved = prove_guest(run, config, true, &trusted);
        verify(&proved).expect("full-advice proof must verify");
    }

    fn committed_e2e(bytecode_chunk_count: usize) {
        let (run, config) = muldiv_run();
        let preprocessing =
            preprocessing::preprocess_committed(run.preprocessing, &config, bytecode_chunk_count)
                .expect("committed Akita preprocessing");
        let program_preprocessing = preprocessing.program_arc().expect("retained full program");
        let public_io = run.trace.device.clone();
        let witness = TraceBackend::<OwnedTrace>::from_compact(
            witness_config(&config, false, false),
            JoltVmWitnessInputs::new(&run.program, &program_preprocessing, run.trace),
        );
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &JoltAkitaBackend::optimized(),
            &preprocessing,
            &config,
            None,
            &witness,
            &public_io,
        )
        .expect("committed Akita proof");
        let verify = |proof: &Proof| {
            jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
                &preprocessing.verifier,
                &public_io,
                proof,
                None,
            )
        };
        verify(&proof).expect("committed Akita proof must verify");

        // A mutated direct bytecode claim breaks the grouped opening.
        let mut tampered = proof;
        let JoltProofClaims::Clear(claims) = &mut tampered.claims else {
            panic!("Akita proofs carry clear claims");
        };
        claims
            .stage7
            .bytecode_address_phase
            .as_mut()
            .expect("committed proofs carry the bytecode address phase")
            .chunks[0] += AkitaField::from_u64(1);
        assert!(verify(&tampered).is_err());
    }

    #[test]
    fn muldiv_e2e_akita_committed_program() {
        committed_e2e(1);
        committed_e2e(2);
    }

    #[test]
    fn advice_e2e_akita_committed_program() {
        let inputs = postcard::to_stdvec(&12u64).expect("serialize inputs");
        let untrusted = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
        let trusted = postcard::to_stdvec(&7u64).expect("serialize trusted advice");
        let run = guest_run("advice-consumer-guest", &inputs, &untrusted, &trusted);
        let config = derive_config(&run);
        let preprocessing = preprocessing::preprocess_committed_with_advice(
            run.preprocessing,
            &config,
            1,
            true,
            true,
        )
        .expect("committed Akita preprocessing");
        let trusted_object = preprocessing::commit_trusted_advice(&preprocessing, &trusted)
            .expect("trusted advice commitment");
        let program_preprocessing = preprocessing.program_arc().expect("retained full program");
        let public_io = run.trace.device.clone();
        let witness = TraceBackend::<OwnedTrace>::from_compact(
            witness_config(&config, true, true),
            JoltVmWitnessInputs::new(&run.program, &program_preprocessing, run.trace),
        );
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &JoltAkitaBackend::optimized(),
            &preprocessing,
            &config,
            Some(&trusted_object),
            &witness,
            &public_io,
        )
        .expect("committed advice Akita proof");

        assert!(proof.untrusted_advice_commitment.is_some());
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &preprocessing.verifier,
            &public_io,
            &proof,
            Some(&trusted_object.commitment),
        )
        .expect("committed advice Akita proof must verify");
    }
}

#[cfg(not(all(feature = "prover-fixtures", feature = "akita")))]
#[test]
#[ignore = "enable --features akita,prover-fixtures to run the Akita e2e"]
fn muldiv_e2e_akita() {}
