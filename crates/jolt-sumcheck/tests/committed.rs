#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]

use jolt_crypto::{Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup};
use jolt_field::{Fr, Ring};
use jolt_sumcheck::round_proof::RoundMessage;
use jolt_sumcheck::{CommittedRound, CommittedRoundWitness, SumcheckStatement, SumcheckVerifier};
use jolt_transcript::{Blake2bTranscript, Transcript};

type F = Fr;
type VC = Pedersen<Bn254G1>;

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

    let mut prover_transcript = Blake2bTranscript::<F>::new(b"committed-roundtrip");
    let mut expected_challenges = Vec::new();
    for round in &rounds {
        round.append_to_transcript(&mut prover_transcript);
        expected_challenges.push(prover_transcript.challenge());
    }

    let mut verifier_transcript = Blake2bTranscript::<F>::new(b"committed-roundtrip");
    let consistency = SumcheckVerifier::verify_committed_round_consistency(
        SumcheckStatement::new(rounds.len(), 2),
        &rounds,
        &mut verifier_transcript,
    )
    .unwrap();

    assert_eq!(consistency.challenges(), expected_challenges);
    assert_eq!(consistency.round_degrees(), vec![2, 1, 2]);
    assert_eq!(verifier_transcript.state(), prover_transcript.state());
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

    let mut original_transcript = Blake2bTranscript::<F>::new(b"committed-roundtrip");
    let original = SumcheckVerifier::verify_committed_round_consistency(
        SumcheckStatement::new(2, 2),
        &rounds,
        &mut original_transcript,
    )
    .unwrap();

    let mut tampered_transcript = Blake2bTranscript::<F>::new(b"committed-roundtrip");
    let tampered = SumcheckVerifier::verify_committed_round_consistency(
        SumcheckStatement::new(2, 2),
        &tampered,
        &mut tampered_transcript,
    )
    .unwrap();

    assert_ne!(tampered.challenges(), original.challenges());
    assert_ne!(tampered_transcript.state(), original_transcript.state());
}
