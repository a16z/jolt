//! Live Fiat-Shamir attacks: each test shows one transcript binding is
//! load-bearing.
//!
//! An attack is valid only when all four steps hold:
//! 1. the honest fixture verifies and its squeeze stream is recorded;
//! 2. a coordinated mutation of the argument string (or statement) makes an
//!    individual protocol claim false;
//! 3. verification with the recorded squeezes replayed accepts, so the
//!    mutation survives every algebraic check once the binding is removed;
//! 4. live verification changes the squeeze stream and rejects.

#![cfg(all(feature = "logging", feature = "prover-fixtures"))]
#![expect(
    dead_code,
    reason = "the shared support module is compiled into every integration-test target but only partially used per feature configuration"
)]

mod support;

#[path = "support/fs_mutations.rs"]
mod fs_mutations;
#[path = "support/fs_transcript.rs"]
mod fs_transcript;

use fs_transcript::{record, replay, AuditSponge, ChallengeTape};
use jolt_verifier::VerifierError;

/// Step 1: the honest proof verifies on the audit sponge.
fn record_honest(verify: impl FnOnce() -> Result<(), VerifierError>) -> ChallengeTape {
    let (honest, tape) = record(verify);
    assert!(honest.is_ok(), "honest fixture rejected: {honest:?}");
    tape
}

/// Steps 3 and 4 for an attacked verification.
fn assert_binding_load_bearing(
    tape: &ChallengeTape,
    attacked: impl Fn() -> Result<(), VerifierError>,
    attack: &str,
) {
    let frozen = replay(tape, &attacked);
    assert!(
        frozen.output.is_ok(),
        "frozen-challenge verifier rejected {attack}: {:?}",
        frozen.output
    );
    assert_eq!(
        frozen.consumed, frozen.recorded,
        "{attack} changed the squeeze schedule under frozen challenges"
    );

    let (live, live_tape) = record(&attacked);
    assert!(live.is_err(), "production verifier accepted {attack}");
    assert!(
        tape.first_divergence(&live_tape).is_some(),
        "{attack} did not alter a production challenge"
    );
}

mod audit_sponge {
    use jolt_transcript::DuplexSpongeInterface;
    use jolt_transcript::{Blake2b512, Fork, FORK_SEED_LEN};

    use super::fs_transcript::{record, replay, AuditSponge};

    fn squeeze<const N: usize>(sponge: &mut impl DuplexSpongeInterface<U = u8>) -> [u8; N] {
        let mut out = [0u8; N];
        let _ = sponge.squeeze(&mut out);
        out
    }

    #[test]
    fn records_the_blake2b_stream() {
        let mut reference = Blake2b512::default();
        let _ = reference.absorb(b"statement");
        let expected: [u8; 48] = squeeze(&mut reference);

        let (squeezed, tape) = record(|| {
            let mut sponge = AuditSponge::default();
            let _ = sponge.absorb(b"statement");
            let head: [u8; 16] = squeeze(&mut sponge);
            let tail: [u8; 32] = squeeze(&mut sponge);
            [head.as_slice(), tail.as_slice()].concat()
        });
        assert_eq!(squeezed, expected);
        assert_eq!(tape.bytes, expected);
    }

    /// Forks are untaped: their squeezes match plain Blake2b forks in and out
    /// of a session, on any thread, and leave the session's tape untouched.
    #[test]
    #[expect(clippy::expect_used, reason = "a panicking fork thread fails the test")]
    fn forks_pass_through_untaped() {
        let seed = [7u8; FORK_SEED_LEN];
        let expected: [u8; 32] = Fork::<Blake2b512>::new(&seed, 3).squeeze();
        let (squeezed, tape) = record(|| {
            let on_session_thread: [u8; 32] = Fork::<AuditSponge>::new(&seed, 3).squeeze();
            let on_worker_thread: [u8; 32] =
                std::thread::spawn(move || Fork::<AuditSponge>::new(&seed, 3).squeeze())
                    .join()
                    .expect("fork thread");
            assert_eq!(on_worker_thread, on_session_thread);
            on_session_thread
        });
        assert_eq!(squeezed, expected);
        assert!(tape.bytes.is_empty());
    }

    #[test]
    #[should_panic(expected = "outside a record or replay session")]
    fn transcript_sponge_needs_a_session() {
        let _ = AuditSponge::default().absorb(b"statement");
    }

    #[test]
    #[should_panic(expected = "challenge replay exhausted")]
    fn replay_past_the_tape_panics() {
        let (_, tape) = record(|| squeeze::<16>(&mut AuditSponge::default()));
        let _ = replay(&tape, || squeeze::<17>(&mut AuditSponge::default()));
    }
}

#[cfg(not(feature = "akita"))]
mod dory {
    use common::jolt_device::JoltDevice;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::{DoryCommitment, DoryScheme};
    use jolt_field::Fr;
    use jolt_verifier::{
        verify, verify_stages, JoltProof, JoltSponge, JoltVerifierPreprocessing, VerifierError,
    };

    use super::fs_mutations::{cancel_dory_final_opening_commitments, locate_events, LocatedEvent};
    use super::{assert_binding_load_bearing, record_honest, AuditSponge};

    type Preprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;

    pub(super) struct Statement<'a> {
        pub preprocessing: &'a Preprocessing,
        pub public_io: &'a JoltDevice,
        pub trusted_advice_commitment: Option<&'a DoryCommitment>,
    }

    impl Statement<'_> {
        pub(super) fn verify(&self, proof: &JoltProof) -> Result<(), VerifierError> {
            verify::<Fr, DoryScheme, Pedersen<Bn254G1>, AuditSponge>(
                self.preprocessing,
                self.public_io,
                proof,
                self.trusted_advice_commitment,
            )
        }

        /// The honest proof's events through stage 8 (BlindFold excluded).
        pub(super) fn events(&self, proof: &JoltProof) -> Vec<LocatedEvent> {
            locate_events(&proof.narg, |transcript| {
                verify_stages::<Fr, DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
                    self.preprocessing,
                    self.public_io,
                    self.trusted_advice_commitment,
                    transcript,
                )
                .map(drop)
            })
        }
    }

    pub(super) fn preprocessing_digest_attack(statement: &Statement<'_>, proof: &JoltProof) {
        let tape = record_honest(|| statement.verify(proof));
        let mut preprocessing = statement.preprocessing.clone();
        preprocessing.preprocessing_digest[0] ^= 1;
        let attacked = Statement {
            preprocessing: &preprocessing,
            ..*statement
        };
        assert_binding_load_bearing(
            &tape,
            || attacked.verify(proof),
            "a proof under different preprocessing",
        );
    }

    pub(super) fn final_opening_commitment_attack(statement: &Statement<'_>, proof: &JoltProof) {
        let tape = record_honest(|| statement.verify(proof));
        let events = statement.events(proof);
        let mut attacked = proof.clone();
        cancel_dory_final_opening_commitments::<DoryCommitment>(&mut attacked.narg, &events, &tape);
        assert_binding_load_bearing(
            &tape,
            || statement.verify(&attacked),
            "commitments cancelled in the final-opening batch",
        );
    }

    #[cfg(not(feature = "zk"))]
    mod clear {
        use jolt_field::{Fr, Ring};

        use super::super::fs_mutations::equivocate_stage1_clear;
        use super::super::support::verifier_fixtures::{standard_muldiv_case, VerifierFixtureCase};
        use super::super::{assert_binding_load_bearing, record_honest};
        use super::{final_opening_commitment_attack, preprocessing_digest_attack, Statement};

        fn statement(case: &VerifierFixtureCase) -> Statement<'_> {
            Statement {
                preprocessing: &case.preprocessing,
                public_io: &case.public_io,
                trusted_advice_commitment: case.trusted_advice_commitment.as_ref(),
            }
        }

        #[test]
        fn dory_clear_preprocessing_digest_requires_fiat_shamir_binding() {
            let case = standard_muldiv_case();
            preprocessing_digest_attack(&statement(&case), &case.proof);
        }

        #[test]
        fn dory_clear_stage1_sumcheck_requires_fiat_shamir_challenges() {
            let case = standard_muldiv_case();
            let statement = statement(&case);
            let tape = record_honest(|| statement.verify(&case.proof));
            let events = statement.events(&case.proof);
            let mut attacked = case.proof.clone();
            equivocate_stage1_clear(&mut attacked.narg, &events, &tape, Fr::from_u64(1));
            assert_binding_load_bearing(
                &tape,
                || statement.verify(&attacked),
                "a stage-1 sumcheck equivocation",
            );
        }

        #[test]
        fn dory_clear_final_opening_batch_requires_commitment_binding() {
            let case = standard_muldiv_case();
            final_opening_commitment_attack(&statement(&case), &case.proof);
        }
    }

    #[cfg(feature = "zk")]
    mod zk {
        use super::super::support::verifier_fixtures::{zk_muldiv_case, ZkVerifierFixtureCase};
        use super::{final_opening_commitment_attack, preprocessing_digest_attack, Statement};

        fn statement(case: &ZkVerifierFixtureCase) -> Statement<'_> {
            Statement {
                preprocessing: &case.preprocessing,
                public_io: &case.public_io,
                trusted_advice_commitment: case.trusted_advice_commitment.as_ref(),
            }
        }

        #[test]
        fn dory_zk_preprocessing_digest_requires_fiat_shamir_binding() {
            let case = zk_muldiv_case();
            preprocessing_digest_attack(&statement(&case), &case.proof);
        }

        #[test]
        fn dory_zk_final_opening_batch_requires_commitment_binding() {
            let case = zk_muldiv_case();
            final_opening_commitment_attack(&statement(&case), &case.proof);
        }
    }
}

#[cfg(feature = "akita")]
mod akita {
    use jolt_akita::{AkitaField, AkitaScheme};
    use jolt_crypto::Commitment;
    use jolt_field::Ring;
    use jolt_prover::akita::preprocessing::AkitaVc;
    use jolt_verifier::{
        seed_transcript, stages::stage1, verify, JoltProof, JoltSponge, JoltVerifierPreprocessing,
        VerifierError,
    };

    use super::fs_mutations::{equivocate_stage1_clear, locate_events};
    use super::support::akita_fixtures::{akita_muldiv_case, AkitaFixtureCase};
    use super::{assert_binding_load_bearing, record_honest, AuditSponge};

    fn verify_with(
        case: &AkitaFixtureCase,
        preprocessing: &JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
        proof: &JoltProof,
    ) -> Result<(), VerifierError> {
        verify::<AkitaField, AkitaScheme, AkitaVc, AuditSponge>(
            preprocessing,
            &case.public_io,
            proof,
            case.trusted_advice_commitment.as_ref(),
        )
    }

    #[test]
    fn akita_clear_preprocessing_digest_requires_fiat_shamir_binding() {
        let case = akita_muldiv_case();
        let tape = record_honest(|| verify_with(case, &case.preprocessing, &case.proof));
        let mut preprocessing = case.preprocessing.clone();
        preprocessing.preprocessing_digest[0] ^= 1;
        assert_binding_load_bearing(
            &tape,
            || verify_with(case, &preprocessing, &case.proof),
            "a proof under different preprocessing",
        );
    }

    #[test]
    fn akita_clear_stage1_sumcheck_requires_fiat_shamir_challenges() {
        let case = akita_muldiv_case();
        let tape = record_honest(|| verify_with(case, &case.preprocessing, &case.proof));
        // Stage 1 is the attack's last transcript dependency, so the event log
        // stops there.
        let events = locate_events(&case.proof.narg, |transcript| {
            let seeded = seed_transcript::<AkitaScheme, AkitaVc, JoltSponge>(
                &case.preprocessing,
                &case.public_io,
                case.trusted_advice_commitment.as_ref(),
                transcript,
            )?;
            stage1::verify::<AkitaField, <AkitaVc as Commitment>::Output, JoltSponge>(
                &seeded.checked,
                transcript,
            )
            .map(drop)
        });
        let mut attacked = case.proof.clone();
        equivocate_stage1_clear(&mut attacked.narg, &events, &tape, AkitaField::from_u64(1));
        assert_binding_load_bearing(
            &tape,
            || verify_with(case, &case.preprocessing, &attacked),
            "a stage-1 sumcheck equivocation",
        );
    }
}
