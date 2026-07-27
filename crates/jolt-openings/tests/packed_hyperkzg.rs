#![expect(clippy::expect_used, reason = "tests assert successful proof paths")]
#![expect(
    clippy::unwrap_used,
    reason = "benchmarks and tests unwrap successful PCS operations"
)]

use jolt_crypto::{Bn254, Commitment};
use jolt_field::Fr;
use jolt_hyperkzg::HyperKZGScheme;
use jolt_openings::{
    prove_packed_openings, verify_packed_openings, CommitmentScheme, OpeningsError,
    PackedObjectGroup, PackedOpeningProof, PackedProverGroup, PackedProverObject,
    PackedVerifierObject, PrefixPackedStatement,
};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};

#[path = "support/common.rs"]
pub mod common;
#[path = "support/packed.rs"]
pub mod packed_support;

use common::{fr, kzg_setup};
use packed_support::{
    build_packed, packed_claims, packed_polynomials, MaterializedPackedWitness, PackedId,
};

type KzgPCS = HyperKZGScheme<Bn254>;
type KzgOutput = <KzgPCS as Commitment>::Output;
type KzgProof = <KzgPCS as CommitmentScheme>::Proof;
type KzgProverSetup = <KzgPCS as CommitmentScheme>::ProverSetup;
type KzgVerifierSetup = <KzgPCS as CommitmentScheme>::VerifierSetup;
type KzgOpeningHint = <KzgPCS as CommitmentScheme>::OpeningHint;
type PackedStatement = PrefixPackedStatement<Fr, PackedId, KzgOutput>;

fn prove_single(
    packed: &MaterializedPackedWitness<PackedId, Fr>,
    setup: &KzgProverSetup,
    statement: &PackedStatement,
    hint: KzgOpeningHint,
    label: &'static [u8],
) -> Result<PackedOpeningProof<Fr, KzgProof>, OpeningsError> {
    let mut transcript = Blake2bTranscript::new(label);
    prove_packed_openings::<KzgPCS, PackedId, _>(
        vec![PackedProverObject {
            packing: &packed.packing,
            statement,
            polynomial: &packed.polynomial,
            setup,
        }],
        vec![PackedProverGroup::singleton(0, Some(hint))],
        &mut transcript,
    )
}

fn verify_single(
    packed: &MaterializedPackedWitness<PackedId, Fr>,
    setup: &KzgVerifierSetup,
    statement: &PackedStatement,
    proof: &PackedOpeningProof<Fr, KzgProof>,
    transcript: &mut Blake2bTranscript,
) -> Result<(), OpeningsError> {
    verify_packed_openings::<KzgPCS, PackedId, _>(
        &[PackedVerifierObject {
            packing: &packed.packing,
            statement,
            setup,
        }],
        &[PackedObjectGroup::singleton(0)],
        proof,
        transcript,
    )
}

#[test]
fn hyperkzg_prefix_packed_batch_roundtrip_complex_mixed_arities() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    assert_eq!(packed.packing.packed_num_vars, 5);
    assert_eq!((&packed.packing).into_iter().count(), 5);

    let (prover_setup, verifier_setup) = kzg_setup(packed.packing.packed_num_vars);
    let (commitment, hint) =
        <KzgPCS as CommitmentScheme>::commit(&packed.polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(11), fr(13), fr(17), fr(19), fr(23)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let statement = PrefixPackedStatement::new(commitment, claims);

    let mut prover_transcript = Blake2bTranscript::new(b"hyperkzg-packed-complex");
    let proof = prove_packed_openings::<KzgPCS, PackedId, _>(
        vec![PackedProverObject {
            packing: &packed.packing,
            statement: &statement,
            polynomial: &packed.polynomial,
            setup: &prover_setup,
        }],
        vec![PackedProverGroup::singleton(0, Some(hint))],
        &mut prover_transcript,
    )
    .expect("HyperKZG prefix-packed opening proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"hyperkzg-packed-complex");
    verify_single(
        &packed,
        &verifier_setup,
        &statement,
        &proof,
        &mut verifier_transcript,
    )
    .expect("HyperKZG prefix-packed opening proof should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn hyperkzg_prefix_packed_batch_rejects_missing_logical_slot() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = kzg_setup(packed.packing.packed_num_vars);
    let (commitment, hint) =
        <KzgPCS as CommitmentScheme>::commit(&packed.polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(11), fr(13), fr(17), fr(19), fr(23)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let _dropped = claims.pop();

    let result = prove_single(
        &packed,
        &prover_setup,
        &PrefixPackedStatement::new(commitment, claims),
        hint,
        b"hyperkzg-packed-missing-slot",
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn hyperkzg_prefix_packed_batch_rejects_wrong_logical_arity() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = kzg_setup(packed.packing.packed_num_vars);
    let (commitment, hint) =
        <KzgPCS as CommitmentScheme>::commit(&packed.polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(11), fr(13), fr(17), fr(19), fr(23)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let wide = claims
        .iter_mut()
        .find(|claim| claim.0 == PackedId::Wide)
        .expect("wide claim should exist");
    let mut point = wide.1.point.clone().into_vec();
    let _removed = point.pop();
    wide.1.point = point.into();

    let result = prove_single(
        &packed,
        &prover_setup,
        &PrefixPackedStatement::new(commitment, claims),
        hint,
        b"hyperkzg-packed-wrong-arity",
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn hyperkzg_prefix_packed_batch_rejects_unknown_id() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = kzg_setup(packed.packing.packed_num_vars);
    let (commitment, hint) =
        <KzgPCS as CommitmentScheme>::commit(&packed.polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(11), fr(13), fr(17), fr(19), fr(23)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    claims[0].0 = PackedId::Unused;

    let result = prove_single(
        &packed,
        &prover_setup,
        &PrefixPackedStatement::new(commitment, claims),
        hint,
        b"hyperkzg-packed-unknown-id",
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn hyperkzg_prefix_packed_batch_rejects_tampered_value() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = kzg_setup(packed.packing.packed_num_vars);
    let (commitment, hint) =
        <KzgPCS as CommitmentScheme>::commit(&packed.polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(11), fr(13), fr(17), fr(19), fr(23)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let proof = prove_single(
        &packed,
        &prover_setup,
        &PrefixPackedStatement::new(commitment, claims.clone()),
        hint,
        b"hyperkzg-packed-tamper",
    )
    .expect("HyperKZG prefix-packed opening proof should be produced");

    let mut tampered = claims;
    tampered[0].1.value += fr(1);
    let mut verifier_transcript = Blake2bTranscript::new(b"hyperkzg-packed-tamper");
    let result = verify_single(
        &packed,
        &verifier_setup,
        &PrefixPackedStatement::new(commitment, tampered),
        &proof,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "tampered packed value should fail");
}

#[test]
fn hyperkzg_prefix_packed_batch_rejects_wrong_packed_commitment() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = kzg_setup(packed.packing.packed_num_vars);
    let (commitment, hint) =
        <KzgPCS as CommitmentScheme>::commit(&packed.polynomial, &prover_setup).unwrap();
    let mut other_evals = packed.polynomial.evaluations().to_vec();
    other_evals[0] += fr(1);
    let other_polynomial = Polynomial::new(other_evals);
    let (other_commitment, ()) =
        <KzgPCS as CommitmentScheme>::commit(&other_polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(11), fr(13), fr(17), fr(19), fr(23)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let proof = prove_single(
        &packed,
        &prover_setup,
        &PrefixPackedStatement::new(commitment, claims.clone()),
        hint,
        b"hyperkzg-packed-wrong-commitment",
    )
    .expect("HyperKZG prefix-packed opening proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"hyperkzg-packed-wrong-commitment");
    let result = verify_single(
        &packed,
        &verifier_setup,
        &PrefixPackedStatement::new(other_commitment, claims),
        &proof,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "wrong packed commitment should fail");
}
