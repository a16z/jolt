#![expect(clippy::unwrap_used, reason = "tests may panic on assertion failures")]

use jolt_crypto::{Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup};
use jolt_field::{Fr, Ring};
use jolt_sumcheck::round_proof::RoundMessage;
use jolt_sumcheck::{
    round_pow_bits, CommittedRound, CommittedRoundWitness, SumcheckStatement, SumcheckVerifier,
};
use jolt_transcript::{
    prover_transcript, Blake2b512, FieldTranscript, ProverState, ProverTranscript,
};

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

/// Replays the verifier's round loop and grinds each round's nonce at the
/// difficulty for the statement's `degree`.
fn grind_nonces(
    rounds: &[CommittedRound<Bn254G1>],
    degree: usize,
    session: &'static [u8],
) -> Vec<u64> {
    let bits = round_pow_bits::<F, ProverState<Blake2b512>>(degree).unwrap();
    let mut transcript =
        prover_transcript::<Blake2b512>(&jolt_transcript::protocol_tag(session), b"");
    rounds
        .iter()
        .filter_map(|round| {
            round.append_to_transcript(&mut transcript);
            let nonce = transcript.public_grind(bits).unwrap();
            let _challenge: F = transcript.challenge().unwrap();
            nonce
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

    let bits = round_pow_bits::<F, ProverState<Blake2b512>>(2).unwrap();
    let mut prover_state = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"committed-roundtrip"),
        b"",
    );
    let mut expected_challenges: Vec<F> = Vec::new();
    let mut pow_nonces = Vec::new();
    for round in &rounds {
        round.append_to_transcript(&mut prover_state);
        pow_nonces.extend(prover_state.public_grind(bits).unwrap());
        expected_challenges.push(prover_state.challenge().unwrap());
    }

    let mut verifier_transcript = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"committed-roundtrip"),
        b"",
    );
    let consistency = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(rounds.len(), 2),
        &rounds,
        &pow_nonces,
        &mut verifier_transcript,
    )
    .unwrap();

    assert_eq!(consistency.challenges(), expected_challenges);
    assert_eq!(consistency.round_degrees(), vec![2, 1, 2]);
    assert_eq!(
        verifier_transcript.verifier_message::<[u8; 32]>(),
        prover_state.verifier_message::<[u8; 32]>()
    );
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

    let mut original_transcript = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"committed-roundtrip"),
        b"",
    );
    let original = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(2, 2),
        &rounds,
        &grind_nonces(&rounds, 2, b"committed-roundtrip"),
        &mut original_transcript,
    )
    .unwrap();

    let mut tampered_transcript = prover_transcript::<Blake2b512>(
        &jolt_transcript::protocol_tag(b"committed-roundtrip"),
        b"",
    );
    let tampered = SumcheckVerifier::verify_committed_round_consistency::<F, _, _>(
        SumcheckStatement::new(2, 2),
        &tampered,
        &grind_nonces(&tampered, 2, b"committed-roundtrip"),
        &mut tampered_transcript,
    )
    .unwrap();

    assert_ne!(tampered.challenges(), original.challenges());
    assert_ne!(
        tampered_transcript.verifier_message::<[u8; 32]>(),
        original_transcript.verifier_message::<[u8; 32]>()
    );
}
