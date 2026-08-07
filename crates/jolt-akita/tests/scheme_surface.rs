//! Trait-surface coverage for `AkitaScheme`: setup parameter validation,
//! metadata trait consistency, hint-free openings, and the flavor-restricted
//! `one_hot_only` setup.

#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

mod support;

use jolt_akita::{
    AkitaBackendFlavor, AkitaScheme, AkitaSetupParams, AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256,
};
use jolt_openings::{
    CommitmentScheme, GroupCommitmentMetadata, GroupSetupMetadata, OpeningsError, ZkOpeningScheme,
};
use jolt_poly::{MultilinearPoly, OneHotPolynomial};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};
use support::{f, layout, polynomial, setup_for};

/// The smallest dense dimension the folded-only planner schedules.
const DENSE_VARS: usize = 13;
/// The smallest K=16 one-hot dimension (`log2(K) + 8`).
const ONE_HOT_VARS: usize = 12;

fn one_hot_indices() -> Vec<Option<u8>> {
    (0..1usize << (ONE_HOT_VARS - 4))
        .map(|row| {
            if row % 6 == 5 {
                None
            } else {
                Some(((row * 5) % 16) as u8)
            }
        })
        .collect()
}

fn k16_setup() -> (
    <AkitaScheme as CommitmentScheme>::ProverSetup,
    <AkitaScheme as CommitmentScheme>::VerifierSetup,
) {
    AkitaScheme::setup(AkitaSetupParams::one_hot_only(
        ONE_HOT_VARS,
        1,
        layout(2),
        AKITA_ONE_HOT_K16,
    ))
    .expect("one-hot setup should build")
}

#[test]
fn setup_rejects_unsupported_one_hot_chunk_sizes() {
    for bad_k in [0, 4, 32, 512] {
        let err = AkitaScheme::setup(AkitaSetupParams::one_hot_only(6, 1, layout(1), bad_k))
            .expect_err("only K=16 and K=256 are supported");
        assert!(
            matches!(&err, OpeningsError::InvalidSetup(message) if message.contains("must be 16 or 256")),
            "unexpected error for K={bad_k}: {err}"
        );
    }
}

#[test]
fn verifier_setup_accessor_matches_setup_output() {
    let params = AkitaSetupParams::new(DENSE_VARS, 2, layout(9));
    assert_eq!(params.one_hot_k(), AKITA_ONE_HOT_K256, "default chunk size");
    let (prover_setup, verifier_setup) = AkitaScheme::setup(params).expect("setup should succeed");
    assert_eq!(AkitaScheme::verifier_setup(&prover_setup), verifier_setup);
    assert_eq!(prover_setup.max_num_vars(), DENSE_VARS);
    assert_eq!(prover_setup.max_num_polys_per_commitment_group(), 2);
    assert_eq!(prover_setup.default_layout_digest(), layout(9));
    assert_eq!(prover_setup.one_hot_k(), AKITA_ONE_HOT_K256);
}

/// The `GroupSetupMetadata` / `GroupCommitmentMetadata` impls are the shape
/// vocabulary `jolt-openings` enforces before a native batch opening; they
/// must agree with the concrete accessors and the committed polynomials.
#[test]
fn metadata_traits_report_committed_shapes() {
    let (prover_setup, verifier_setup) = setup_for(DENSE_VARS, 2, layout(7));
    assert_eq!(
        GroupSetupMetadata::max_num_vars(&verifier_setup),
        DENSE_VARS
    );
    assert_eq!(
        GroupSetupMetadata::max_num_polys_per_commitment_group(&verifier_setup),
        2
    );
    assert_eq!(
        GroupSetupMetadata::default_layout_digest(&verifier_setup),
        layout(7)
    );
    assert_eq!(
        GroupSetupMetadata::one_hot_k(&verifier_setup),
        AKITA_ONE_HOT_K256
    );
    assert_eq!(verifier_setup.max_num_vars(), DENSE_VARS);
    assert_eq!(verifier_setup.max_num_polys_per_commitment_group(), 2);
    assert_eq!(verifier_setup.default_layout_digest(), layout(7));
    assert_eq!(verifier_setup.one_hot_k(), AKITA_ONE_HOT_K256);
    assert!(
        format!("{verifier_setup:?}").contains("BackendVerifierCache"),
        "derived state must appear as an opaque cache in Debug output"
    );

    let (dense_commitment, _) = AkitaScheme::commit_group(
        &prover_setup,
        layout(11),
        &[polynomial(DENSE_VARS, 1), polynomial(DENSE_VARS, 9)],
    )
    .expect("dense group commit should succeed");
    assert!(!GroupCommitmentMetadata::is_one_hot_backend(
        &dense_commitment
    ));
    assert_eq!(
        GroupCommitmentMetadata::layout_digest(&dense_commitment),
        layout(11)
    );
    assert_eq!(
        GroupCommitmentMetadata::num_vars(&dense_commitment),
        DENSE_VARS
    );
    assert_eq!(GroupCommitmentMetadata::poly_count(&dense_commitment), 2);
    assert_eq!(
        GroupCommitmentMetadata::one_hot_k(&dense_commitment),
        0,
        "dense-flavor commitments must not declare a chunk size"
    );

    let (one_hot_setup, _) = k16_setup();
    let one_hot = OneHotPolynomial::new(AKITA_ONE_HOT_K16, one_hot_indices());
    let (one_hot_commitment, _) = AkitaScheme::commit_one_hot_group(
        &one_hot_setup,
        layout(2),
        std::slice::from_ref(&one_hot),
    )
    .expect("one-hot commit should succeed");
    assert!(GroupCommitmentMetadata::is_one_hot_backend(
        &one_hot_commitment
    ));
    assert_eq!(
        GroupCommitmentMetadata::one_hot_k(&one_hot_commitment),
        AKITA_ONE_HOT_K16
    );
}

/// A `one_hot_only` setup skips the dense-flavor backend entirely, so a
/// dense polynomial cannot be committed through it.
#[test]
fn one_hot_only_setup_rejects_dense_commits() {
    let (prover_setup, _) = k16_setup();
    let dense = polynomial(ONE_HOT_VARS, 1);
    let err = AkitaScheme::commit(&dense, &prover_setup)
        .expect_err("dense commit must reject without a dense backend");
    assert!(
        matches!(&err, OpeningsError::InvalidSetup(message) if message.contains("without the dense-flavor backend")),
        "unexpected error: {err}"
    );
}

/// `commit` routes a row-major K=16 one-hot polynomial through the K=16
/// backend; the proof must verify against a serde-transported verifier setup,
/// which re-derives its one-hot backend key from shape alone.
#[test]
fn single_k16_one_hot_commit_roundtrips_with_transported_verifier_setup() {
    let (prover_setup, verifier_setup) = k16_setup();
    let one_hot = OneHotPolynomial::new(AKITA_ONE_HOT_K16, one_hot_indices());
    let (commitment, hint) =
        AkitaScheme::commit(&one_hot, &prover_setup).expect("one-hot commit should succeed");
    assert_eq!(commitment.backend_flavor(), AkitaBackendFlavor::OneHot);
    assert_eq!(commitment.one_hot_k(), AKITA_ONE_HOT_K16);

    let point: Vec<_> = (0..ONE_HOT_VARS).map(|index| f(index as u64 + 5)).collect();
    let eval = MultilinearPoly::evaluate(&one_hot, &point);
    let mut prover_transcript = Blake2bTranscript::new(b"akita-k16-transported");
    let proof = AkitaScheme::open(
        &one_hot,
        &point,
        eval,
        &prover_setup,
        Some(hint),
        &mut prover_transcript,
    )
    .expect("one-hot opening should prove");

    let json = serde_json::to_string(&verifier_setup).expect("verifier setup serializes");
    let transported: <AkitaScheme as CommitmentScheme>::VerifierSetup =
        serde_json::from_str(&json).expect("verifier setup deserializes");
    assert_eq!(transported, verifier_setup);

    let mut verifier_transcript = Blake2bTranscript::new(b"akita-k16-transported");
    AkitaScheme::verify(
        &commitment,
        &point,
        eval,
        &proof,
        &transported,
        &mut verifier_transcript,
    )
    .expect("transported setup must re-derive the one-hot backend key");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

/// Without a commit-time hint, `open` re-commits internally; the resulting
/// proof must still verify against the original commitment.
#[test]
fn open_without_hint_recommits_deterministically() {
    let (prover_setup, verifier_setup) = setup_for(DENSE_VARS, 1, layout(7));
    let poly = polynomial(DENSE_VARS, 33);
    let point: Vec<_> = (0..DENSE_VARS).map(|index| f(index as u64 + 2)).collect();
    let eval = poly.evaluate(&point);
    let (commitment, _) = AkitaScheme::commit(&poly, &prover_setup).expect("commit succeeds");

    let mut prover_transcript = Blake2bTranscript::new(b"akita-no-hint");
    let proof = AkitaScheme::open(
        &poly,
        &point,
        eval,
        &prover_setup,
        None,
        &mut prover_transcript,
    )
    .expect("hint-free opening should prove");

    let mut verifier_transcript = Blake2bTranscript::new(b"akita-no-hint");
    AkitaScheme::verify(
        &commitment,
        &point,
        eval,
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
    )
    .expect("hint-free proof should verify against the original commitment");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn open_batch_rejects_mismatched_claim_counts() {
    let (prover_setup, _) = setup_for(DENSE_VARS, 2, layout(7));
    let poly_a = polynomial(DENSE_VARS, 1);
    let poly_b = polynomial(DENSE_VARS, 20);
    let point: Vec<_> = (0..DENSE_VARS).map(|index| f(index as u64 + 2)).collect();
    let (_, hint) =
        AkitaScheme::commit_group(&prover_setup, layout(7), &[poly_a.clone(), poly_b.clone()])
            .expect("group commit should succeed");

    let mut transcript = Blake2bTranscript::new(b"akita-batch-mismatch");
    let err = AkitaScheme::open_batch(
        &[&poly_a, &poly_b],
        &point,
        &[poly_a.evaluate(&point)],
        &prover_setup,
        hint,
        &mut transcript,
    )
    .expect_err("two polynomials with one evaluation must reject");
    assert!(
        matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("2 polynomials but 1 evaluations")),
        "unexpected error: {err}"
    );
}

/// The hiding commitment binds the committed evaluation bytes into the
/// transcript: equal evaluations bind identically, distinct ones diverge.
#[test]
fn hiding_commitment_transcript_binding_tracks_the_evaluation() {
    let (prover_setup, _) = setup_for(DENSE_VARS, 1, layout(7));
    let poly = polynomial(DENSE_VARS, 50);
    let point: Vec<_> = (0..DENSE_VARS).map(|index| f(index as u64 + 2)).collect();
    let eval = poly.evaluate(&point);
    let (_, hint) = AkitaScheme::commit_zk(&poly, &prover_setup).expect("commit_zk succeeds");

    let mut transcript = Blake2bTranscript::new(b"akita-hiding");
    let (_, hiding, ()) = AkitaScheme::open_zk(
        &poly,
        &point,
        eval,
        &prover_setup,
        hint.clone(),
        &mut transcript,
    )
    .expect("open_zk should produce a hiding commitment");

    let mut first = Blake2bTranscript::<jolt_akita::AkitaField>::new(b"akita-hiding-bind");
    hiding.append_to_transcript(&mut first);
    let mut second = Blake2bTranscript::<jolt_akita::AkitaField>::new(b"akita-hiding-bind");
    hiding.append_to_transcript(&mut second);
    assert_eq!(first.state(), second.state(), "binding is deterministic");

    let mut other_point = point.clone();
    other_point[DENSE_VARS - 1] += f(1);
    let other_eval = poly.evaluate(&other_point);
    assert_ne!(eval, other_eval, "fixture needs distinct evaluations");
    let mut transcript = Blake2bTranscript::new(b"akita-hiding");
    let (_, other_hiding, ()) = AkitaScheme::open_zk(
        &poly,
        &other_point,
        other_eval,
        &prover_setup,
        hint,
        &mut transcript,
    )
    .expect("open_zk should produce a hiding commitment");
    let mut third = Blake2bTranscript::<jolt_akita::AkitaField>::new(b"akita-hiding-bind");
    other_hiding.append_to_transcript(&mut third);
    assert_ne!(
        first.state(),
        third.state(),
        "distinct evaluations must bind distinct transcript states"
    );
}
