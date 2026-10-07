//! End-to-end coverage for the modular Akita prover and verifier: the
//! mode-specific checks (tampering, forced one-hot sizes, committed programs,
//! trace-order rejection). Plain acceptance across guests is `e2e_matrix.rs`.

#[cfg(all(
    feature = "prover-fixtures",
    feature = "akita",
    not(feature = "field-inline")
))]
#[path = "support/akita_backend.rs"]
mod backend_contract;

#[cfg(all(
    feature = "prover-fixtures",
    feature = "akita",
    not(feature = "field-inline")
))]
mod support;

#[cfg(all(
    feature = "prover-fixtures",
    feature = "akita",
    not(feature = "field-inline")
))]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests should fail loudly"
)]
mod akita_tests {
    use common::constants::{DEFAULT_MAX_TRUSTED_ADVICE_SIZE, DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE};
    use common::jolt_device::JoltDevice;
    use jolt_akita::{
        AkitaChunkProfile, AkitaCommitment, AkitaField, AkitaScheduleArtifacts, AkitaScheme,
    };
    use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltOneHotConfig, TracePolynomialOrder};
    use jolt_field::Ring;
    use jolt_program::execution::OwnedTrace;
    use jolt_prover::akita;
    use jolt_prover::akita::preprocessing::{
        self, AkitaProverPreprocessing, AkitaTranscript, AkitaVc,
    };
    use jolt_prover::akita::witness::commit_advice;
    use jolt_prover::JoltBackend;
    use jolt_prover::{PreprocessingError, ProverConfig, ProverError};
    use jolt_verifier::proof::{ClearProofClaims, JoltProof, JoltProofClaims};
    use jolt_verifier::VerifierError;
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

    use crate::support::{self, GuestCase, PreparedGuest};

    type Proof = JoltProof<AkitaScheme, AkitaVc>;

    struct ProvedGuest {
        preprocessing: AkitaProverPreprocessing,
        public_io: JoltDevice,
        proof: Proof,
        trusted_advice_commitment: Option<AkitaCommitment>,
    }

    fn guest_run(
        guest_name: &'static str,
        inputs: &[u8],
        untrusted_advice: &[u8],
        trusted_advice: &[u8],
    ) -> PreparedGuest {
        support::prepare(&GuestCase {
            inputs: inputs.to_vec(),
            untrusted_advice: untrusted_advice.to_vec(),
            trusted_advice: trusted_advice.to_vec(),
            ..GuestCase::new(guest_name)
        })
    }

    fn derive_config(run: &PreparedGuest) -> ProverConfig {
        ProverConfig::derive_from_dimensions::<AkitaField>(
            run.trace.dimensions,
            &run.preprocessing.memory_layout,
            run.preprocessing.ram.min_bytecode_address,
            run.preprocessing.ram.bytecode_words.len(),
            run.preprocessing.max_padded_trace_length,
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
        run: PreparedGuest,
        config: ProverConfig,
        untrusted_advice: bool,
        trusted_advice: &[u8],
    ) -> ProvedGuest {
        let has_trusted_advice = !trusted_advice.is_empty();
        let preprocessing = preprocessing::preprocess_full_with_advice(
            &AkitaScheduleArtifacts::shared_from_default_directory(),
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
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            JoltBackend::optimized().with_witness(&witness),
            &preprocessing,
            &config,
            trusted.as_ref(),
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

    fn verify(proved: &ProvedGuest) -> Result<(), VerifierError> {
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &proved.preprocessing.verifier,
            &proved.public_io,
            &proved.proof,
            proved.trusted_advice_commitment.as_ref(),
        )
    }

    fn muldiv_run() -> (PreparedGuest, ProverConfig) {
        let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
        let run = guest_run("muldiv-guest", &inputs, &[], &[]);
        let config = derive_config(&run);
        (run, config)
    }

    #[test]
    fn commitment_request_rejects_a_different_layout_and_excess_groups() {
        use jolt_claims::protocols::jolt::lattice::{OneHotTraceShape, ONE_HOT_TRACE_LAYOUT};
        use jolt_claims::protocols::jolt::JoltFormulaDimensions;
        use jolt_kernels::akita::commitment::WitnessCommitRequest;
        use jolt_openings::{CommitmentScheme, OpeningsError};

        let (run, config) = muldiv_run();
        let log_t = config.trace_length.ilog2() as usize;
        let dimensions = JoltFormulaDimensions::try_from(config.one_hot_config.dimensions(
            log_t,
            128,
            run.preprocessing.bytecode.code_size,
            config.ram_K,
        ))
        .expect("formula dimensions");
        let shape = OneHotTraceShape {
            ra_layout: dimensions.ra_layout,
            log_t,
            log_k_chunk: config.one_hot_config.committed_chunk_bits(),
        };
        let preprocessing = preprocessing::preprocess_full_with_advice(
            &AkitaScheduleArtifacts::shared_from_default_directory(),
            run.preprocessing,
            &config,
            false,
            false,
        )
        .expect("Akita preprocessing");
        let setup = &preprocessing.pcs_setup;
        let request = WitnessCommitRequest::<AkitaScheme>::new(setup, shape, &[])
            .expect("matching canonical geometry");
        assert_eq!(
            request.plan(),
            &ONE_HOT_TRACE_LAYOUT.plan(&shape).expect("canonical layout")
        );

        let different_shape = OneHotTraceShape {
            log_t: log_t + 1,
            ..shape
        };
        assert!(matches!(
            WitnessCommitRequest::<AkitaScheme>::new(setup, different_shape, &[]),
            Err(OpeningsError::InvalidSetup(reason))
                if reason.contains("layout digest")
        ));

        // The trace itself consumes one group, so filling the entire setup
        // capacity with auxiliary groups must be rejected before dispatch.
        let hint = <AkitaScheme as CommitmentScheme>::OpeningHint::default();
        let hints = vec![&hint; setup.max_total_batch_polys()];
        assert!(matches!(
            WitnessCommitRequest::<AkitaScheme>::new(setup, shape, &hints),
            Err(OpeningsError::InvalidSetup(reason))
                if reason.contains("dimensions")
        ));
    }

    #[test]
    fn muldiv_e2e_akita() {
        check_muldiv_e2e(false);
    }

    #[test]
    fn muldiv_address_first_e2e_akita() {
        check_muldiv_e2e(true);
    }

    fn check_muldiv_e2e(address_first: bool) {
        let (run, mut config) = muldiv_run();
        if address_first {
            config.rw_config.ram_rw_phase1_num_rounds = 0;
            config.rw_config.registers_rw_phase1_num_rounds = 0;
        }
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

        let result = preprocessing::preprocess_full(
            &AkitaScheduleArtifacts::shared_from_default_directory(),
            run.preprocessing,
            &config,
        );
        assert!(matches!(
            result,
            Err(PreprocessingError::InvalidConfiguration { .. })
        ));

        let (run, mut config) = muldiv_run();
        let preprocessing = preprocessing::preprocess_full(
            &AkitaScheduleArtifacts::shared_from_default_directory(),
            run.preprocessing,
            &config,
        )
        .expect("cycle-major preprocessing");
        config.trace_polynomial_order = TracePolynomialOrder::AddressMajor;
        let program_preprocessing = preprocessing.program_arc().expect("full program");
        let public_io = run.trace.device.clone();
        let witness = TraceBackend::<OwnedTrace>::from_compact(
            witness_config(&config, false, false),
            JoltVmWitnessInputs::new(&run.program, &program_preprocessing, run.trace),
        );
        let result = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            JoltBackend::optimized().with_witness(&witness),
            &preprocessing,
            &config,
            None,
            &public_io,
        );
        assert!(matches!(
            result,
            Err(ProverError::Unsupported {
                reason: "Akita supports only cycle-major trace polynomials"
            })
        ));
    }

    #[test]
    fn akita_rejects_chunk_profile_mismatch() {
        for setup_profile in [AkitaChunkProfile::Single, AkitaChunkProfile::Four] {
            let (run, mut config) = muldiv_run();
            config.akita_chunk_profile = setup_profile;
            let preprocessing = preprocessing::preprocess_full(
                &AkitaScheduleArtifacts::shared_from_default_directory(),
                run.preprocessing,
                &config,
            )
            .expect("preprocess selected chunk profile");
            let program = preprocessing.program_arc().expect("full program");
            let public_io = run.trace.device.clone();
            let witness = TraceBackend::<OwnedTrace>::from_compact(
                witness_config(&config, false, false),
                JoltVmWitnessInputs::new(&run.program, &program, run.trace),
            );
            for requested in [
                AkitaChunkProfile::Single,
                AkitaChunkProfile::Two,
                AkitaChunkProfile::Four,
                AkitaChunkProfile::Eight,
            ] {
                if requested == setup_profile {
                    continue;
                }
                config.akita_chunk_profile = requested;
                let result = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
                    JoltBackend::optimized().with_witness(&witness),
                    &preprocessing,
                    &config,
                    None,
                    &public_io,
                );
                assert!(matches!(
                    result,
                    Err(ProverError::Unsupported {
                        reason: "Akita chunk profile differs from preprocessing; reuse its configuration or regenerate preprocessing"
                    })
                ));
            }
        }
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
    fn advice_e2e_akita_two_chunks() {
        advice_chunk_roundtrip(AkitaChunkProfile::Two);
    }

    #[test]
    fn advice_e2e_akita_four_chunks() {
        advice_chunk_roundtrip(AkitaChunkProfile::Four);
    }

    #[test]
    fn advice_e2e_akita_eight_chunks() {
        advice_chunk_roundtrip(AkitaChunkProfile::Eight);
    }

    fn advice_chunk_roundtrip(profile: AkitaChunkProfile) {
        let inputs = postcard::to_stdvec(&12u64).expect("serialize inputs");
        let mut untrusted = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
        untrusted.resize(DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE as usize, u8::MAX);
        let trusted = postcard::to_stdvec(&7u64).expect("serialize trusted advice");
        let run = guest_run("advice-consumer-guest", &inputs, &untrusted, &trusted);
        let mut config = derive_config(&run);
        config.akita_chunk_profile = profile;
        let proved = prove_guest(run, config, true, &trusted);
        assert_eq!(
            proved
                .preprocessing
                .verifier
                .pcs_setup
                .akita_chunk_profile(),
            profile
        );
        verify(&proved).expect("chunked grouped advice proof must verify");
    }

    #[test]
    fn trusted_advice_commitment_reused_across_chunk_profiles() {
        let artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
        let inputs = postcard::to_stdvec(&12u64).expect("serialize inputs");
        let untrusted = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
        let trusted = postcard::to_stdvec(&7u64).expect("serialize trusted advice");
        let object = commit_advice::<AkitaScheme>(
            &artifacts,
            JoltAdviceKind::Trusted,
            &trusted,
            DEFAULT_MAX_TRUSTED_ADVICE_SIZE as usize,
        )
        .expect("commit trusted advice before selecting a trace chunk profile");
        for profile in [
            AkitaChunkProfile::Single,
            AkitaChunkProfile::Two,
            AkitaChunkProfile::Four,
            AkitaChunkProfile::Eight,
        ] {
            let run = guest_run("advice-consumer-guest", &inputs, &untrusted, &trusted);
            let mut config = derive_config(&run);
            config.akita_chunk_profile = profile;
            let preprocessing = preprocessing::preprocess_full_with_advice(
                &artifacts,
                run.preprocessing,
                &config,
                true,
                true,
            )
            .expect("preprocess with the fixed advice producer");
            let program = preprocessing
                .program_arc()
                .expect("full program preprocessing");
            let public_io = run.trace.device.clone();
            let witness = TraceBackend::<OwnedTrace>::from_compact(
                witness_config(&config, true, true),
                JoltVmWitnessInputs::new(&run.program, &program, run.trace),
            );
            let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
                JoltBackend::optimized().with_witness(&witness),
                &preprocessing,
                &config,
                Some(&object),
                &public_io,
            )
            .expect("reuse the original advice commitment and opening hint");
            jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
                &preprocessing.verifier,
                &public_io,
                &proof,
                Some(&object.commitment),
            )
            .expect("verify with the original advice commitment");
        }
    }

    #[test]
    fn advice_e2e_akita_full_advice() {
        let inputs = postcard::to_stdvec(&12u64).expect("serialize inputs");
        let trusted = postcard::to_stdvec(&7u64).expect("serialize trusted advice");
        let capacity = DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE as usize;
        let mut untrusted = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
        untrusted.extend((untrusted.len()..capacity).map(|index| (index * 31 + 7) as u8));
        let run = guest_run("advice-consumer-guest", &inputs, &untrusted, &trusted);
        let config = derive_config(&run);
        let proved = prove_guest(run, config, true, &trusted);
        verify(&proved).expect("full-advice proof must verify");
    }

    fn committed_e2e(bytecode_chunk_count: usize, profile: AkitaChunkProfile) {
        let (run, mut config) = muldiv_run();
        config.akita_chunk_profile = profile;
        let preprocessing = preprocessing::preprocess_committed(
            &AkitaScheduleArtifacts::shared_from_default_directory(),
            run.preprocessing,
            &config,
            bytecode_chunk_count,
        )
        .expect("committed Akita preprocessing");
        let program_preprocessing = preprocessing.program_arc().expect("retained full program");
        let public_io = run.trace.device.clone();
        let witness = TraceBackend::<OwnedTrace>::from_compact(
            witness_config(&config, false, false),
            JoltVmWitnessInputs::new(&run.program, &program_preprocessing, run.trace),
        );
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            JoltBackend::optimized().with_witness(&witness),
            &preprocessing,
            &config,
            None,
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
        for profile in [
            AkitaChunkProfile::Single,
            AkitaChunkProfile::Two,
            AkitaChunkProfile::Four,
            AkitaChunkProfile::Eight,
        ] {
            committed_e2e(1, profile);
            committed_e2e(2, profile);
        }
    }

    #[test]
    fn advice_e2e_akita_committed_program() {
        let inputs = postcard::to_stdvec(&12u64).expect("serialize inputs");
        let untrusted = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
        let trusted = postcard::to_stdvec(&7u64).expect("serialize trusted advice");
        let run = guest_run("advice-consumer-guest", &inputs, &untrusted, &trusted);
        let config = derive_config(&run);
        let preprocessing = preprocessing::preprocess_committed_with_advice(
            &AkitaScheduleArtifacts::shared_from_default_directory(),
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
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            JoltBackend::optimized().with_witness(&witness),
            &preprocessing,
            &config,
            Some(&trusted_object),
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
