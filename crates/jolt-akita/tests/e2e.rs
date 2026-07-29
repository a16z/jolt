#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]
#![expect(
    clippy::unwrap_used,
    reason = "benchmarks and tests unwrap successful PCS operations"
)]

mod support;

use jolt_akita::{AkitaScheme, AkitaSetupParams};
use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use support::{f, layout, polynomial, setup_for};

#[test]
fn akita_single_opening_roundtrips_across_dimensions() {
    single_opening_roundtrip(13, 10, (2..15).map(f).collect(), b"akita-e2e-arity-13");
    single_opening_roundtrip(14, 20, (3..17).map(f).collect(), b"akita-e2e-arity-14");
    single_opening_roundtrip(15, 30, (5..20).map(f).collect(), b"akita-e2e-arity-15");
}

#[test]
fn akita_single_opening_rejects_wrong_eval_point_setup_and_transcript() {
    let (prover_setup, verifier_setup) = setup_for(13, 1, layout(7));
    let poly = polynomial(13, 100);
    let point: Vec<_> = (0..13).map(|i| f(2 + 3 * i)).collect();
    let eval = poly.evaluate(&point);
    let (commitment, hint) = AkitaScheme::commit(&poly, &prover_setup).unwrap();

    let mut prover_transcript = Blake2bTranscript::new(b"akita-e2e-rejects");
    let proof = AkitaScheme::open(
        &poly,
        &point,
        eval,
        &prover_setup,
        Some(hint),
        &mut prover_transcript,
    )
    .unwrap();

    let mut transcript = Blake2bTranscript::new(b"akita-e2e-rejects");
    assert!(
        AkitaScheme::verify(
            &commitment,
            &point,
            eval + f(1),
            &proof,
            &verifier_setup,
            &mut transcript,
        )
        .is_err(),
        "wrong evaluation should reject"
    );

    let mut wrong_point = point.clone();
    wrong_point[2] += f(1);
    let mut transcript = Blake2bTranscript::new(b"akita-e2e-rejects");
    assert!(
        AkitaScheme::verify(
            &commitment,
            &wrong_point,
            eval,
            &proof,
            &verifier_setup,
            &mut transcript,
        )
        .is_err(),
        "wrong opening point should reject"
    );

    let mut wrong_transcript = Blake2bTranscript::new(b"akita-e2e-wrong-domain");
    assert!(
        AkitaScheme::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &verifier_setup,
            &mut wrong_transcript,
        )
        .is_err(),
        "different transcript domain should reject"
    );

    let (_, wrong_layout_setup) = setup_for(13, 1, layout(8));
    let mut transcript = Blake2bTranscript::new(b"akita-e2e-rejects");
    assert!(
        AkitaScheme::verify(
            &commitment,
            &point,
            eval,
            &proof,
            &wrong_layout_setup,
            &mut transcript,
        )
        .is_err(),
        "wrong verifier setup key should reject"
    );
}

#[test]
fn akita_commit_group_rejects_shape_pathologies() {
    let (prover_setup, _) = setup_for(14, 2, layout(7));
    let poly_a = polynomial(14, 1);
    let mixed_vars = polynomial(13, 40);

    let empty: Vec<Polynomial<_>> = Vec::new();
    assert!(matches!(
        AkitaScheme::commit_group(&prover_setup, layout(7), &empty),
        Err(OpeningsError::InvalidBatch(_))
    ));

    assert!(matches!(
        AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), mixed_vars]),
        Err(OpeningsError::InvalidBatch(_))
    ));

    assert!(matches!(
        AkitaScheme::commit_group(
            &prover_setup,
            layout(7),
            &[poly_a.clone(), polynomial(14, 20), polynomial(14, 40)],
        ),
        Err(OpeningsError::InvalidBatch(_))
    ));

    let (wrong_dimension_setup, _) =
        AkitaScheme::setup(AkitaSetupParams::new(13, 2, layout(7))).unwrap();
    assert!(matches!(
        AkitaScheme::commit_group(&wrong_dimension_setup, layout(7), &[poly_a]),
        Err(OpeningsError::InvalidBatch(_))
    ));
}

#[test]
fn akita_commit_group_preserves_statement_layout_digest() {
    let (prover_setup, _) = setup_for(14, 2, layout(7));
    let (commitment, _) =
        AkitaScheme::commit_group(&prover_setup, layout(11), &[polynomial(14, 1)])
            .expect("direct commitments carry their statement layout digest");

    assert_eq!(commitment.layout_digest(), layout(11));
    assert_eq!(commitment.num_vars(), 14);
    assert_eq!(commitment.poly_count(), 1);
}

fn single_opening_roundtrip(
    num_vars: usize,
    offset: u64,
    point: Vec<jolt_akita::AkitaField>,
    label: &'static [u8],
) {
    let (prover_setup, verifier_setup) = setup_for(num_vars, 1, layout(7));
    let poly = polynomial(num_vars, offset);
    let eval = poly.evaluate(&point);
    let (commitment, hint) = AkitaScheme::commit(&poly, &prover_setup).unwrap();

    let mut prover_transcript = Blake2bTranscript::new(label);
    let proof = AkitaScheme::open(
        &poly,
        &point,
        eval,
        &prover_setup,
        Some(hint),
        &mut prover_transcript,
    )
    .unwrap();

    let mut verifier_transcript = Blake2bTranscript::new(label);
    AkitaScheme::verify(
        &commitment,
        &point,
        eval,
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
    )
    .expect("single proof should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}
