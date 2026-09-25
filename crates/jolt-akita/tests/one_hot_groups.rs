//! The owned one-hot commitment surface: `commit_one_hot_group_owned` must
//! agree with the borrowed path, and its hint must drive native batching to
//! proofs the batch verifier accepts.

#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

#[expect(
    dead_code,
    reason = "shared integration-test support is compiled independently per test file"
)]
mod support;

use jolt_akita::{
    AkitaField, AkitaNativeBatchPolynomials, AkitaNativeBatching, AkitaScheduleArtifacts,
    AkitaScheme, AkitaSetupParams, AKITA_ONE_HOT_K16,
};
use jolt_openings::{BatchOpeningScheme, CommitmentScheme, OpeningsError};
use jolt_poly::{MultilinearPoly, OneHotIndexOrder, OneHotPolynomial};
use jolt_transcript::{Blake2bTranscript, Transcript};
use support::{f, layout, native_statement};

/// `log2(K) + 8`: the smallest K=16 one-hot dimension the folded-only
/// planner schedules (mirrors the scheme unit tests' roundtrip size).
const NUM_VARS: usize = 12;
const ROWS: usize = 1 << (NUM_VARS - 4);

fn one_hot_setup(
    max_polys: usize,
) -> (
    <AkitaScheme as CommitmentScheme>::ProverSetup,
    <AkitaScheme as CommitmentScheme>::VerifierSetup,
) {
    AkitaScheme::setup(AkitaSetupParams::one_hot_only(
        NUM_VARS,
        max_polys,
        layout(3),
        AKITA_ONE_HOT_K16,
        AkitaScheduleArtifacts::shared_from_default_directory(),
    ))
    .expect("one-hot-only setup should build")
}

fn hot_indices(rows: usize, stride: usize, gap: usize) -> Vec<Option<u8>> {
    (0..rows)
        .map(|row| {
            if row % gap == gap - 1 {
                None
            } else {
                Some(((row * stride) % 16) as u8)
            }
        })
        .collect()
}

fn group_polynomials() -> Vec<OneHotPolynomial> {
    vec![
        OneHotPolynomial::new(AKITA_ONE_HOT_K16, hot_indices(ROWS, 3, 5)),
        OneHotPolynomial::new(AKITA_ONE_HOT_K16, hot_indices(ROWS, 7, 9)),
    ]
}

fn opening_point() -> Vec<AkitaField> {
    (0..NUM_VARS).map(|index| f(index as u64 + 3)).collect()
}

#[test]
fn owned_group_commitment_matches_borrowed_group_commitment() {
    let (prover_setup, _) = one_hot_setup(2);
    let polynomials = group_polynomials();
    let (borrowed, _) = AkitaScheme::commit_one_hot_group(&prover_setup, layout(3), &polynomials)
        .expect("borrowed commit should succeed");
    let (owned, _) = AkitaScheme::commit_one_hot_group_owned(&prover_setup, layout(3), polynomials)
        .expect("owned commit should succeed");
    assert_eq!(
        owned, borrowed,
        "owned and borrowed one-hot commits must produce identical commitments"
    );
}

#[test]
fn owned_group_opens_through_native_batching_and_verifies() {
    let (prover_setup, verifier_setup) = one_hot_setup(2);
    let polynomials = group_polynomials();
    let point = opening_point();
    let evaluations: Vec<_> = polynomials
        .iter()
        .map(|polynomial| MultilinearPoly::evaluate(polynomial, &point))
        .collect();
    let (commitment, hint) =
        AkitaScheme::commit_one_hot_group_owned(&prover_setup, layout(3), polynomials.clone())
            .expect("owned commit should succeed");
    let witnesses: AkitaNativeBatchPolynomials<'_> = polynomials
        .iter()
        .map(|polynomial| polynomial as &dyn MultilinearPoly<AkitaField>)
        .collect();
    let statement = native_statement(commitment, &point, evaluations.iter().copied());

    let mut prover_transcript = Blake2bTranscript::new(b"akita-owned-one-hot");
    let proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        witnesses.clone(),
        hint.clone(),
        &mut prover_transcript,
    )
    .expect("owned one-hot group should prove");

    let mut verifier_transcript = Blake2bTranscript::new(b"akita-owned-one-hot");
    <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        &statement,
        &proof,
        &mut verifier_transcript,
    )
    .expect("owned one-hot group proof should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());

    let mut tampered = statement.clone();
    tampered[0].evaluation.value += f(1);
    let mut verifier_transcript = Blake2bTranscript::new(b"akita-owned-one-hot");
    assert!(
        <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            &tampered,
            &proof,
            &mut verifier_transcript,
        )
        .is_err(),
        "tampered evaluation must reject"
    );

    let mut transcript = Blake2bTranscript::new(b"akita-owned-one-hot");
    let err = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement[..1].to_vec(),
        witnesses,
        hint,
        &mut transcript,
    )
    .expect_err("one claim for a two-slot commitment must reject");
    assert!(
        matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("claims")),
        "unexpected error: {err}"
    );
}

#[test]
fn owned_group_rejects_polynomials_the_setup_cannot_open() {
    let (prover_setup, _) = one_hot_setup(1);

    let err = AkitaScheme::commit_one_hot_group_owned(&prover_setup, layout(3), Vec::new())
        .expect_err("empty group must reject");
    assert!(
        matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("must contain a polynomial")),
        "unexpected error: {err}"
    );

    // Same twelve-variable dimension as the setup, but chunk size 4 instead
    // of the setup's 16 — the shape check passes and the K check must fire.
    let wrong_k = vec![OneHotPolynomial::new(4, vec![Some(1); 1 << (NUM_VARS - 2)])];
    let err = AkitaScheme::commit_one_hot_group_owned(&prover_setup, layout(3), wrong_k)
        .expect_err("K=4 polynomial must reject against a K=16 setup");
    assert!(
        matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("row-major K=16")),
        "unexpected error: {err}"
    );

    let column_major = vec![OneHotPolynomial::new_with_index_order(
        AKITA_ONE_HOT_K16,
        hot_indices(ROWS, 3, 5),
        OneHotIndexOrder::ColumnMajor,
    )];
    let err = AkitaScheme::commit_one_hot_group_owned(&prover_setup, layout(3), column_major)
        .expect_err("column-major polynomial must reject");
    assert!(
        matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("row-major K=16")),
        "unexpected error: {err}"
    );

    let mixed_dimensions = vec![
        OneHotPolynomial::new(AKITA_ONE_HOT_K16, hot_indices(ROWS, 3, 5)),
        OneHotPolynomial::new(AKITA_ONE_HOT_K16, hot_indices(ROWS / 2, 3, 5)),
    ];
    let (two_slot_setup, _) = one_hot_setup(2);
    let err = AkitaScheme::commit_one_hot_group_owned(&two_slot_setup, layout(3), mixed_dimensions)
        .expect_err("mixed dimensions must reject");
    assert!(
        matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("mixes")),
        "unexpected error: {err}"
    );
}

#[test]
fn borrowed_group_rejects_column_major_polynomials() {
    let (prover_setup, _) = one_hot_setup(1);
    let column_major = OneHotPolynomial::new_with_index_order(
        AKITA_ONE_HOT_K16,
        hot_indices(ROWS, 3, 5),
        OneHotIndexOrder::ColumnMajor,
    );
    let err = AkitaScheme::commit_one_hot_group(
        &prover_setup,
        layout(3),
        std::slice::from_ref(&column_major),
    )
    .expect_err("column-major polynomial must reject on the borrowed path");
    assert!(
        matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("row-major K=16")),
        "unexpected error: {err}"
    );
}
