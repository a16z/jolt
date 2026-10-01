#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests may panic on assertion failures and index fixture data"
)]

use jolt_crypto::{
    Bn254, Bn254G1, JoltGroup, NoVectorCommitment, Pedersen, PedersenSetup, VectorCommitment,
};
use jolt_field::{CanonicalBytes, Fr, Ring};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    CommittedOutputClaims, CommittedSumcheckBuilder, CommittedSumcheckConsistency,
    CommittedSumcheckWitness, SumcheckError, SumcheckStatement, SumcheckVerifier,
};
use jolt_transcript::{
    Blake2b512, Channel, ProtocolId, ProverTranscript, TranscriptError, VerifierTranscript,
};
use rand_core::OsRng;

type F = Fr;
type VC = Pedersen<Bn254G1>;

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-sumcheck/tests/committed");
const SESSION: &[u8] = b"committed-roundtrip";
const COMMITMENT_BYTES: usize = <Bn254G1 as CanonicalBytes>::NUM_BYTES;

fn pedersen_setup(capacity: usize) -> PedersenSetup<Bn254G1> {
    let generator = Bn254::g1_generator();
    let message_generators = (1..=capacity)
        .map(|i| generator.scalar_mul(&F::from_u64(i as u64)))
        .collect();
    PedersenSetup::new(message_generators, generator.scalar_mul(&F::from_u64(99)))
}

fn poly(coefficients: &[u64]) -> UnivariatePoly<F> {
    UnivariatePoly::new(coefficients.iter().copied().map(F::from_u64).collect())
}

struct Proved {
    narg: Vec<u8>,
    challenges: Vec<F>,
    witness: CommittedSumcheckWitness<F>,
    fingerprint: [u8; 32],
}

/// Commits three rounds at degree bound 2 (the middle one of degree 1) and
/// four output-claim values in rows of the setup's capacity 3.
fn prove(setup: &PedersenSetup<Bn254G1>) -> Proved {
    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    let mut builder = CommittedSumcheckBuilder::<F, VC, _>::new(setup, OsRng).unwrap();
    let challenges = [poly(&[2, 3, 5]), poly(&[7, 11]), poly(&[13, 17, 19])]
        .iter()
        .map(|round| builder.commit_round(round, 2, &mut transcript).unwrap())
        .collect();
    let outputs: Vec<F> = [23, 29, 31, 37].map(F::from_u64).to_vec();
    let witness = builder.finish(&outputs, &mut transcript).unwrap();
    let fingerprint = transcript.preview().squeeze();
    Proved {
        narg: transcript.finish(),
        challenges,
        witness,
        fingerprint,
    }
}

type Verified = (
    CommittedSumcheckConsistency<F, Bn254G1>,
    CommittedOutputClaims<Bn254G1>,
);

fn verify(
    narg: &[u8],
    statement: SumcheckStatement,
    num_output_commitments: usize,
) -> Result<(Verified, [u8; 32]), SumcheckError<F>> {
    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, narg);
    let verified =
        SumcheckVerifier::verify_committed(statement, num_output_commitments, &mut transcript)?;
    let fingerprint = transcript.preview().squeeze();
    transcript.finish()?;
    Ok((verified, fingerprint))
}

#[test]
fn committed_rounds_pad_to_the_statement_degree_and_open() {
    let setup = pedersen_setup(3);
    let proved = prove(&setup);
    assert_eq!(proved.narg.len(), 5 * COMMITMENT_BYTES);

    let ((consistency, output_claims), fingerprint) =
        verify(&proved.narg, SumcheckStatement::new(3, 2), 2).unwrap();
    assert_eq!(fingerprint, proved.fingerprint);
    assert_eq!(consistency.challenges(), proved.challenges);
    assert_eq!(consistency.round_degrees(), vec![2, 2, 2]);

    let witness = &proved.witness;
    assert_eq!(
        witness.round_coefficients[1],
        poly(&[7, 11, 0]).coefficients()
    );
    for ((commitment, coefficients), blinding) in consistency
        .round_commitments()
        .iter()
        .zip(&witness.round_coefficients)
        .zip(&witness.round_blindings)
    {
        assert!(VC::verify(&setup, commitment, coefficients, blinding));
    }

    assert_eq!(
        witness.output_claim_rows,
        vec![
            [23, 29, 31].map(F::from_u64).to_vec(),
            vec![F::from_u64(37)]
        ]
    );
    assert_eq!(output_claims.commitments.len(), 2);
    for ((commitment, row), blinding) in output_claims
        .commitments
        .iter()
        .zip(&witness.output_claim_rows)
        .zip(&witness.output_claim_blindings)
    {
        assert!(VC::verify(&setup, commitment, row, blinding));
    }
}

#[test]
fn committed_proof_rejects_mismatched_public_dimensions() {
    let proved = prove(&pedersen_setup(3));
    for (statement, outputs, expected) in [
        (SumcheckStatement::new(4, 2), 2, TranscriptError::Truncated),
        (SumcheckStatement::new(3, 2), 3, TranscriptError::Truncated),
        (
            SumcheckStatement::new(2, 2),
            2,
            TranscriptError::TrailingBytes,
        ),
        (
            SumcheckStatement::new(3, 2),
            1,
            TranscriptError::TrailingBytes,
        ),
    ] {
        assert!(
            matches!(
                verify(&proved.narg, statement, outputs),
                Err(SumcheckError::Transcript(error)) if error == expected
            ),
            "{statement:?} with {outputs} output commitments"
        );
    }
}

#[test]
fn replaced_round_commitment_changes_later_challenges() {
    let setup = pedersen_setup(3);
    let proved = prove(&setup);
    let replacement = VC::commit(
        &setup,
        poly(&[101, 103, 107]).coefficients(),
        &F::from_u64(1),
    );
    let mut tampered = proved.narg.clone();
    replacement.to_bytes_le(&mut tampered[COMMITMENT_BYTES..2 * COMMITMENT_BYTES]);

    let ((consistency, _), fingerprint) =
        verify(&tampered, SumcheckStatement::new(3, 2), 2).unwrap();
    let challenges = consistency.challenges();
    assert_eq!(challenges[0], proved.challenges[0]);
    assert_ne!(challenges[1], proved.challenges[1]);
    assert_ne!(challenges[2], proved.challenges[2]);
    assert_ne!(fingerprint, proved.fingerprint);
}

#[test]
fn builder_rejects_rounds_it_cannot_commit_before_writing() {
    let setup = pedersen_setup(3);
    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    let mut builder = CommittedSumcheckBuilder::<F, VC, _>::new(&setup, OsRng).unwrap();
    assert!(matches!(
        builder.commit_round(&poly(&[1, 2, 3, 4]), 2, &mut transcript),
        Err(SumcheckError::DegreeBoundExceeded { got: 3, max: 2 })
    ));
    assert!(matches!(
        builder.commit_round(&poly(&[1, 2]), 3, &mut transcript),
        Err(SumcheckError::RoundExceedsCommitmentCapacity {
            coefficients: 4,
            capacity: 3
        })
    ));
    let witness = builder.finish(&[], &mut transcript).unwrap();
    assert!(witness.round_coefficients.is_empty());
    assert!(transcript.finish().is_empty());

    assert!(matches!(
        CommittedSumcheckBuilder::<F, NoVectorCommitment<F>, _>::new(&(), OsRng),
        Err(SumcheckError::ZeroCommitmentCapacity)
    ));
}
