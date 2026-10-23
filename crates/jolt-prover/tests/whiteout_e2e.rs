//! Whiteout end to end on the Akita build (`specs/whiteout-jolt-integration.md`):
//! honest proofs verify and tampering rejects, for muldiv alone, beside
//! untrusted and trusted advice, and beside a committed program. The
//! verifier-side suites (census, sweeps, compensating shifts) are
//! jolt-verifier's `soundness::whiteout`.

#[cfg(all(feature = "prover-fixtures", feature = "akita", feature = "zk"))]
#[cfg_attr(
    feature = "field-inline",
    expect(
        dead_code,
        reason = "the field-inline Whiteout test proves only field-ops"
    )
)]
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
    use jolt_akita::{AkitaCommitment, AkitaField, AkitaScheduleArtifacts, AkitaScheme};
    use jolt_prover::akita;
    use jolt_prover::akita::preprocessing::{
        self, AkitaProverPreprocessing, AkitaVc, AkitaVerifierPreprocessing,
    };
    use jolt_prover::{JoltBackend, ProverConfig, ProverError};
    use jolt_verifier::proof::JoltProof;
    use jolt_verifier::{JoltSponge, VerifierError};
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

    use crate::support::{self, GuestCase, PreparedGuest};

    type F = AkitaField;

    /// How a case is preprocessed.
    #[derive(Clone, Copy)]
    enum Setup {
        /// The production setup.
        Muldiv,
        /// Production, with a committed program.
        Committed,
        /// Production, with untrusted and trusted advice.
        Advice,
    }

    struct Proved {
        preprocessing: AkitaProverPreprocessing,
        public_io: JoltDevice,
        trusted: Option<AkitaCommitment>,
        proof: Result<JoltProof, ProverError<F>>,
    }

    fn prepare(setup: Setup) -> PreparedGuest {
        match setup {
            Setup::Muldiv | Setup::Committed => support::prepare(&GuestCase {
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

    fn config(run: &PreparedGuest) -> ProverConfig {
        ProverConfig::derive_from_dimensions::<F>(
            run.trace.dimensions,
            &run.preprocessing.memory_layout,
            run.preprocessing.ram.min_bytecode_address,
            run.preprocessing.ram.bytecode_words.len(),
            run.preprocessing.max_padded_trace_length,
        )
        .expect("derive config")
    }

    fn preprocess(
        setup: Setup,
        run: &PreparedGuest,
        config: &ProverConfig,
    ) -> AkitaProverPreprocessing {
        let artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
        let advice = matches!(setup, Setup::Advice);
        match setup {
            Setup::Muldiv | Setup::Advice => preprocessing::preprocess_full_with_advice(
                &artifacts,
                run.preprocessing.clone(),
                config,
                advice,
                advice,
            ),
            Setup::Committed => {
                preprocessing::preprocess_committed(&artifacts, run.preprocessing.clone(), config)
            }
        }
        .expect("Akita preprocessing")
    }

    fn prove(setup: Setup) -> Proved {
        let run = prepare(setup);
        let config = config(&run);
        let advice = matches!(setup, Setup::Advice);
        let preprocessing = preprocess(setup, &run, &config);
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
            trusted: trusted.map(|object| object.commitment),
            proof,
        }
    }

    fn verify(proved: &Proved, proof: &JoltProof) -> Result<(), VerifierError> {
        verify_with(proved, proof, proved.trusted.as_ref())
    }

    fn verify_with(
        proved: &Proved,
        proof: &JoltProof,
        trusted: Option<&AkitaCommitment>,
    ) -> Result<(), VerifierError> {
        jolt_verifier::verify::<F, AkitaScheme, AkitaVc, JoltSponge>(
            &proved.preprocessing.verifier,
            &proved.public_io,
            proof,
            trusted,
        )
    }

    /// The verifier preprocessing after a serde round trip: its Whiteout
    /// catalog is derived state (`serde(skip)`), so the copy plans its own
    /// instead of sharing the prover's.
    fn transported(preprocessing: &AkitaVerifierPreprocessing) -> AkitaVerifierPreprocessing {
        let encoded = bincode::serde::encode_to_vec(preprocessing, bincode::config::standard())
            .expect("encode verifier preprocessing");
        bincode::serde::decode_from_slice(&encoded, bincode::config::standard())
            .expect("decode verifier preprocessing")
            .0
    }

    /// An honest proof verifies against transported preprocessing.
    fn assert_verifies_transported(proved: &Proved, proof: &JoltProof) {
        jolt_verifier::verify::<F, AkitaScheme, AkitaVc, JoltSponge>(
            &transported(&proved.preprocessing.verifier),
            &proved.public_io,
            proof,
            proved.trusted.as_ref(),
        )
        .expect("an honest proof verifies against transported preprocessing");
    }

    fn honest(setup: Setup) -> (Proved, JoltProof) {
        let mut proved = prove(setup);
        let proof = std::mem::replace(
            &mut proved.proof,
            Err(ProverError::InvariantViolation { reason: "taken" }),
        )
        .expect("Whiteout proof");
        (proved, proof)
    }

    #[test]
    fn whiteout_muldiv_verifies_and_rejects_tampering() {
        let (proved, proof) = honest(Setup::Muldiv);
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
        let (again, second) = honest(Setup::Muldiv);
        verify(&again, &second).expect("a second Whiteout proof verifies");
        assert_ne!(proof.narg, second.narg);
    }

    /// Flipped bits across the argument string, truncation and an appended
    /// byte reject.
    fn assert_byte_tampers_reject(proved: &Proved, proof: &JoltProof) {
        let len = proof.narg.len();
        for position in [len / 8, len / 4, len / 2, 3 * len / 4, len - 1] {
            let mut tampered = proof.clone();
            tampered.narg[position] ^= 1;
            assert!(
                verify(proved, &tampered).is_err(),
                "a flipped bit at {position} of {len} must reject"
            );
        }
        let mut truncated = proof.clone();
        let _ = truncated.narg.pop();
        assert!(verify(proved, &truncated).is_err());
    }

    /// Untrusted and trusted advice open as hiding precommitted groups: an
    /// honest proof verifies; byte tampers reject, as do a trusted-advice
    /// commitment of other advice, the same advice recommitted (fresh
    /// hiding randomness, so another commitment), and a missing one.
    #[test]
    fn whiteout_advice_verifies_and_rejects_tampering() {
        let (proved, proof) = honest(Setup::Advice);
        verify(&proved, &proof).expect("an honest Whiteout advice proof verifies");
        assert_verifies_transported(&proved, &proof);
        assert_byte_tampers_reject(&proved, &proof);
        let trusted = proved.trusted.as_ref().expect("trusted advice");
        let other = preprocessing::commit_trusted_advice(
            &proved.preprocessing,
            &postcard::to_stdvec(&8u64).expect("serialize advice"),
        )
        .expect("other trusted advice")
        .commitment;
        let again = preprocessing::commit_trusted_advice(
            &proved.preprocessing,
            &postcard::to_stdvec(&7u64).expect("serialize advice"),
        )
        .expect("the same trusted advice")
        .commitment;
        assert_ne!(&again, trusted, "a hiding recommit draws fresh randomness");
        for (label, commitment) in [("other advice", Some(&other)), ("a recommit", Some(&again))] {
            assert!(
                verify_with(&proved, &proof, commitment).is_err(),
                "a trusted-advice commitment of {label} is accepted"
            );
        }
        assert!(verify_with(&proved, &proof, None).is_err());
    }

    /// A committed program opens as two hiding precommitted groups: an
    /// honest proof verifies; byte tampers reject, as does verifier
    /// preprocessing that holds a recommitted program (fresh randomness).
    #[test]
    fn whiteout_committed_program_verifies_and_rejects_tampering() {
        let (proved, proof) = honest(Setup::Committed);
        verify(&proved, &proof).expect("an honest Whiteout committed-program proof verifies");
        assert_verifies_transported(&proved, &proof);
        assert_byte_tampers_reject(&proved, &proof);
        let run = prepare(Setup::Committed);
        let recommitted = preprocess(Setup::Committed, &run, &config(&run));
        assert_ne!(
            recommitted.verifier.program.committed(),
            proved.preprocessing.verifier.program.committed(),
            "a hiding recommit draws fresh randomness"
        );
        assert!(
            jolt_verifier::verify::<F, AkitaScheme, AkitaVc, JoltSponge>(
                &recommitted.verifier,
                &proved.public_io,
                &proof,
                None,
            )
            .is_err(),
            "a proof verifies against another commitment to its program"
        );
    }
}

#[cfg(all(
    feature = "prover-fixtures",
    feature = "akita",
    feature = "zk",
    feature = "field-inline"
))]
#[expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "integration tests should fail loudly"
)]
mod whiteout_field_inline {
    use jolt_prover::JoltBackend;

    use crate::support::field_inline::{akita, field_ops};

    /// The field increments open as a full-field hiding precommitted group
    /// beside untrusted advice: an honest field-ops proof, whose increments
    /// leave the `u64` envelope, verifies, and flipped bits across the
    /// argument string and truncation reject.
    #[test]
    fn whiteout_field_inline_verifies_and_rejects_tampering() {
        let mut case = field_ops();
        case.untrusted_advice = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (output, ()) = akita::prove(&case, JoltBackend::optimized(), |_, _, _| ());
        let verify = |proof: &akita::Proof| {
            akita::verify_full(&output.verifier_preprocessing, &output.public_io, proof)
        };
        verify(&output.proof).expect("an honest Whiteout field-inline proof verifies");
        let len = output.proof.narg.len();
        for position in [len / 8, len / 4, len / 2, 3 * len / 4, len - 1] {
            let mut tampered = output.proof.clone();
            tampered.narg[position] ^= 1;
            assert!(
                verify(&tampered).is_err(),
                "a flipped bit at {position} of {len} must reject"
            );
        }
        let mut truncated = output.proof.clone();
        let _ = truncated.narg.pop();
        assert!(verify(&truncated).is_err());
    }
}
