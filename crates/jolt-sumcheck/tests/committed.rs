#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]

use jolt_crypto::{Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_sumcheck::round_proof::RoundMessage;
use jolt_sumcheck::{
    CommittedOutputClaims, CommittedRound, CommittedRoundWitness, SumcheckError, SumcheckStatement,
    SumcheckVerifier,
};
use jolt_transcript::{prover_transcript, verifier_transcript, Blake2b512, FsAbsorb, FsChallenge};

type F = Fr;
type VC = Pedersen<Bn254G1>;

const INSTANCE: [u8; 32] = [0u8; 32];

fn pedersen_setup(capacity: usize) -> PedersenSetup<Bn254G1> {
    let generator = Bn254::g1_generator();
    let message_generators = (1..=capacity)
        .map(|i| generator.scalar_mul(&F::from_u64(i as u64)))
        .collect();
    PedersenSetup::new(message_generators, generator.scalar_mul(&F::from_u64(99)))
}

fn committed_rounds(
    setup: &PedersenSetup<Bn254G1>,
    coefficients: &[Vec<F>],
) -> Vec<CommittedRound<Bn254G1>> {
    coefficients
        .iter()
        .enumerate()
        .map(|(round, coefficients)| {
            CommittedRoundWitness {
                coefficients: coefficients.clone(),
                blinding: F::from_u64(round as u64 + 17),
            }
            .commit::<VC>(setup)
            .unwrap()
        })
        .collect()
}

#[test]
fn committed_rounds_complete_with_pedersen_commitments() {
    let setup = pedersen_setup(3);
    let rounds = committed_rounds(
        &setup,
        &[
            vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)],
            vec![F::from_u64(7), F::from_u64(11)],
            vec![F::from_u64(13), F::from_u64(17), F::from_u64(19)],
        ],
    );

    let mut prover_transcript =
        prover_transcript(b"committed-roundtrip", INSTANCE, Blake2b512::default());
    let mut expected_challenges = Vec::new();
    for round in &rounds {
        RoundMessage::<F>::append_to_transcript(round, &mut prover_transcript);
        expected_challenges.push(FsChallenge::<F>::challenge(&mut prover_transcript));
    }

    let mut verifier_transcript =
        verifier_transcript(b"committed-roundtrip", INSTANCE, Blake2b512::default(), &[]);
    let consistency = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(rounds.len(), 2),
        &rounds,
        &mut verifier_transcript,
    )
    .unwrap();

    assert_eq!(consistency.challenges(), expected_challenges);
    assert_eq!(
        consistency
            .rounds
            .iter()
            .map(|round| round.degree)
            .collect::<Vec<_>>(),
        vec![2, 1, 2]
    );

    let verifier_next: F = FsChallenge::<F>::challenge(&mut verifier_transcript);
    let prover_next: F = FsChallenge::<F>::challenge(&mut prover_transcript);
    assert_eq!(verifier_next, prover_next);
}

#[test]
fn committed_rounds_reject_wrong_count_and_degree() {
    let setup = pedersen_setup(3);
    let rounds = committed_rounds(
        &setup,
        &[
            vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)],
            vec![F::from_u64(7), F::from_u64(11)],
        ],
    );

    let mut wrong_count_transcript =
        prover_transcript(b"committed-roundtrip", INSTANCE, Blake2b512::default());
    let wrong_count = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(3, 2),
        &rounds,
        &mut wrong_count_transcript,
    );
    assert!(matches!(
        wrong_count,
        Err(SumcheckError::WrongNumberOfRounds {
            expected: 3,
            got: 2
        })
    ));

    let mut degree_transcript =
        prover_transcript(b"committed-roundtrip", INSTANCE, Blake2b512::default());
    let degree = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(2, 1),
        &rounds,
        &mut degree_transcript,
    );
    assert!(matches!(
        degree,
        Err(SumcheckError::DegreeBoundExceeded { got: 2, max: 1 })
    ));
}

#[test]
fn tampered_committed_round_changes_challenges() {
    let setup = pedersen_setup(3);
    let rounds = committed_rounds(
        &setup,
        &[
            vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)],
            vec![F::from_u64(7), F::from_u64(11), F::from_u64(13)],
        ],
    );
    let mut tampered = rounds.clone();
    tampered[1] = committed_rounds(
        &setup,
        &[vec![F::from_u64(101), F::from_u64(103), F::from_u64(107)]],
    )
    .remove(0);

    let mut original_transcript =
        prover_transcript(b"committed-roundtrip", INSTANCE, Blake2b512::default());
    let original = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(2, 2),
        &rounds,
        &mut original_transcript,
    )
    .unwrap();

    let mut tampered_transcript =
        prover_transcript(b"committed-roundtrip", INSTANCE, Blake2b512::default());
    let tampered = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(2, 2),
        &tampered,
        &mut tampered_transcript,
    )
    .unwrap();

    assert_ne!(tampered.challenges(), original.challenges());
    // A tampered commitment desyncs the sponge; the next squeezed challenge
    // from each transcript must differ.
    let tampered_next: F = FsChallenge::<F>::challenge(&mut tampered_transcript);
    let original_next: F = FsChallenge::<F>::challenge(&mut original_transcript);
    assert_ne!(tampered_next, original_next);
}

#[test]
fn committed_output_claims_keep_length_and_order() {
    let setup = pedersen_setup(2);
    let rounds = committed_rounds(
        &setup,
        &[
            vec![F::from_u64(2), F::from_u64(3)],
            vec![F::from_u64(5), F::from_u64(7)],
        ],
    );
    let output_claims = CommittedOutputClaims {
        commitments: rounds
            .iter()
            .map(|round| round.commitment)
            .collect::<Vec<_>>(),
    };

    let mut actual = prover_transcript(b"committed-output", INSTANCE, Blake2b512::default());
    output_claims.append_to_transcript(&mut actual);

    let mut expected = prover_transcript(b"committed-output", INSTANCE, Blake2b512::default());
    expected.absorb(&output_claims.commitments);

    let actual_next: F = FsChallenge::<F>::challenge(&mut actual);
    let expected_next: F = FsChallenge::<F>::challenge(&mut expected);
    assert_eq!(actual_next, expected_next);
}
