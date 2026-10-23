//! Whiteout end to end on the Akita build (`specs/whiteout-jolt-integration.md`):
//! an honest muldiv proof verifies and tampering rejects, and the shapes
//! this prototype does not open reject with typed errors. The verifier-side
//! suites (census, sweeps, compensating shifts) are jolt-verifier's
//! `soundness::whiteout`. The Σ ranks are Akita A40's test-only ones; the production setup
//! rejects until W5e prices them.

#[cfg(all(
    feature = "prover-fixtures",
    feature = "akita",
    feature = "zk",
    not(feature = "field-inline")
))]
mod support;

#[cfg(all(
    feature = "prover-fixtures",
    feature = "akita",
    feature = "zk",
    not(feature = "field-inline")
))]
#[expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "integration tests should fail loudly"
)]
mod whiteout {
    use common::jolt_device::JoltDevice;
    use jolt_akita::{AkitaField, AkitaScheduleArtifacts, AkitaScheme};
    use jolt_prover::akita;
    use jolt_prover::akita::preprocessing::{self, AkitaProverPreprocessing, AkitaVc};
    use jolt_prover::{JoltBackend, ProverConfig, ProverError};
    use jolt_verifier::proof::JoltProof;
    use jolt_verifier::{JoltSponge, VerifierError};
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

    use crate::support::{self, GuestCase, PreparedGuest};

    type F = AkitaField;

    /// How a case is preprocessed.
    #[derive(Clone, Copy)]
    enum Setup {
        /// A40's test-only Σ ranks.
        A40TestOnly,
        /// The production setup, which prices the Σ ranks.
        Priced,
        /// Production, with a committed program.
        Committed,
        /// Production, with untrusted and trusted advice.
        Advice,
    }

    struct Proved {
        preprocessing: AkitaProverPreprocessing,
        public_io: JoltDevice,
        proof: Result<JoltProof, ProverError<F>>,
    }

    fn prepare(setup: Setup) -> PreparedGuest {
        match setup {
            Setup::A40TestOnly | Setup::Priced | Setup::Committed => support::prepare(&GuestCase {
                inputs: postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs"),
                ..GuestCase::new("muldiv-guest")
            }),
            Setup::Advice => support::prepare(&GuestCase {
                inputs: postcard::to_stdvec(&12u64).expect("serialize inputs"),
                untrusted_advice: postcard::to_stdvec(&5u64).expect("serialize advice"),
                trusted_advice: postcard::to_stdvec(&7u64).expect("serialize advice"),
                ..GuestCase::new("advice-consumer-guest")
            }),
        }
    }

    fn prove(setup: Setup) -> Proved {
        let run = prepare(setup);
        let config = ProverConfig::derive_from_dimensions::<F>(
            run.trace.dimensions,
            &run.preprocessing.memory_layout,
            run.preprocessing.ram.min_bytecode_address,
            run.preprocessing.ram.bytecode_words.len(),
            run.preprocessing.max_padded_trace_length,
        )
        .expect("derive config");
        let artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
        let advice = matches!(setup, Setup::Advice);
        let preprocessing = match setup {
            Setup::A40TestOnly => preprocessing::preprocess_full_whiteout_a40_test_only(
                &artifacts,
                run.preprocessing,
                &config,
            ),
            Setup::Priced | Setup::Advice => preprocessing::preprocess_full_with_advice(
                &artifacts,
                run.preprocessing,
                &config,
                advice,
                advice,
            ),
            Setup::Committed => {
                preprocessing::preprocess_committed(&artifacts, run.preprocessing, &config)
            }
        }
        .expect("Akita preprocessing");
        let trusted = advice.then(|| {
            preprocessing::commit_trusted_advice(
                &preprocessing,
                &postcard::to_stdvec(&7u64).expect("serialize advice"),
            )
            .expect("trusted advice commitment")
        });
        let program = preprocessing.program_arc().expect("full program");
        let public_io = run.trace.device.clone();
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            )
            .include_untrusted_advice(advice)
            .include_trusted_advice(advice),
            JoltVmWitnessInputs::new(&run.program, &program, run.trace),
        );
        let proof = akita::prove::<F, AkitaScheme, AkitaVc, JoltSponge>(
            JoltBackend::optimized().with_witness(&witness),
            &preprocessing,
            &config,
            trusted.as_ref(),
            &public_io,
        );
        Proved {
            preprocessing,
            public_io,
            proof,
        }
    }

    fn verify(proved: &Proved, proof: &JoltProof) -> Result<(), VerifierError> {
        jolt_verifier::verify::<F, AkitaScheme, AkitaVc, JoltSponge>(
            &proved.preprocessing.verifier,
            &proved.public_io,
            proof,
            None,
        )
    }

    fn honest() -> (Proved, JoltProof) {
        let mut proved = prove(Setup::A40TestOnly);
        let proof = std::mem::replace(
            &mut proved.proof,
            Err(ProverError::InvariantViolation { reason: "taken" }),
        )
        .expect("Whiteout proof");
        (proved, proof)
    }

    #[test]
    fn whiteout_muldiv_verifies_and_rejects_tampering() {
        let (proved, proof) = honest();
        verify(&proved, &proof).expect("an honest Whiteout proof verifies");

        let len = proof.narg.len();
        for position in [len / 4, len / 2, 3 * len / 4, len - 1] {
            let mut tampered = proof.clone();
            tampered.narg[position] ^= 1;
            assert!(
                verify(&proved, &tampered).is_err(),
                "a flipped bit at {position} of {len} must reject"
            );
        }
        let mut truncated = proof.clone();
        let _ = truncated.narg.pop();
        assert!(verify(&proved, &truncated).is_err());
        let mut extended = proof.clone();
        extended.narg.push(0);
        assert!(verify(&proved, &extended).is_err());

        // HT3: a second proof of the same statement draws fresh hiding
        // randomness, pads and Σ coins.
        let (again, second) = honest();
        verify(&again, &second).expect("a second Whiteout proof verifies");
        assert_ne!(proof.narg, second.narg);
    }

    /// The production setup is gated on A40 and A41 by name, so it reads as
    /// pending pricing rather than a capability gap, and pricing the Σ ranks
    /// alone cannot open it.
    #[test]
    fn whiteout_production_ranks_reject_until_w5e() {
        let error = prove(Setup::Priced)
            .proof
            .expect_err("unpriced Σ ranks reject");
        let message = error.to_string();
        assert!(
            message.contains("A40") && message.contains("A41"),
            "{message}"
        );
    }

    /// Precommitted groups (advice, committed programs) are not opened yet.
    #[test]
    fn whiteout_rejects_precommitted_groups() {
        for setup in [Setup::Committed, Setup::Advice] {
            let error = prove(setup).proof.expect_err("precommitted groups reject");
            assert!(error.to_string().contains("precommitted"), "{error}");
        }
    }
}
