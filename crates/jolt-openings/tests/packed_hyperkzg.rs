#![expect(clippy::expect_used, reason = "tests assert successful proof paths")]
#![expect(
    clippy::unwrap_used,
    reason = "benchmarks and tests unwrap successful PCS operations"
)]

use jolt_crypto::{Bn254, Commitment};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, OpeningsError, PackedBatch,
    PackedWitness, PrefixPackedClaim, PrefixPackedProverSetup, PrefixPackedStatement,
    PrefixPackedVerifierSetup, PrefixPacking,
};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

#[path = "support/packed.rs"]
mod packed_support;

use packed_support::{materialize_packed, MaterializedPackedWitness};

type KzgPCS = HyperKZGScheme<Bn254>;
type PackedKzgBatch = PackedBatch<KzgPCS, PackedId>;
type KzgOutput = <KzgPCS as Commitment>::Output;
type PackedStatement = PrefixPackedStatement<Fr, PackedId, KzgOutput>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum PackedId {
    Constant,
    NarrowA,
    NarrowB,
    Medium,
    Wide,
    Unused,
}

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn kzg_setup(max_num_vars: usize) -> (HyperKZGProverSetup<Bn254>, HyperKZGVerifierSetup<Bn254>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0xdead_beef);
    let prover = KzgPCS::setup(
        &mut rng,
        1usize << max_num_vars,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let verifier = KzgPCS::verifier_setup(&prover);
    (prover, verifier)
}

fn packed_polynomials() -> Vec<(PackedId, Polynomial<Fr>)> {
    let mut rng = ChaCha20Rng::seed_from_u64(0xca_fe_ba_be);
    vec![
        (PackedId::Wide, Polynomial::<Fr>::random(3, &mut rng)),
        (PackedId::Medium, Polynomial::<Fr>::random(2, &mut rng)),
        (PackedId::NarrowB, Polynomial::<Fr>::random(1, &mut rng)),
        (PackedId::NarrowA, Polynomial::<Fr>::random(1, &mut rng)),
        (PackedId::Constant, Polynomial::new(vec![fr(73)])),
    ]
}

fn build_packed(
    polynomials: &[(PackedId, Polynomial<Fr>)],
) -> MaterializedPackedWitness<PackedId, Fr> {
    materialize_packed(polynomials).expect("packed witness should build")
}

fn packed_claims(
    polynomials: &[(PackedId, Polynomial<Fr>)],
    packing: &PrefixPacking<PackedId>,
    packed_point: &[Fr],
) -> Vec<PrefixPackedClaim<Fr, PackedId>> {
    polynomials
        .iter()
        .map(|(id, polynomial)| {
            let logical_point = packing
                .logical_point(id, packed_point)
                .expect("packed point should produce logical suffix");
            PrefixPackedClaim::new(
                *id,
                EvaluationClaim::new(logical_point.clone(), polynomial.evaluate(&logical_point)),
            )
        })
        .collect()
}

fn packed_setup(
    packing: PrefixPacking<PackedId>,
) -> (
    PrefixPackedProverSetup<KzgPCS, PackedId>,
    PrefixPackedVerifierSetup<KzgPCS, PackedId>,
) {
    let (prover_pcs, verifier_pcs) = kzg_setup(packing.packed_num_vars);
    (
        PrefixPackedProverSetup {
            pcs: prover_pcs,
            packing: packing.clone(),
        },
        PrefixPackedVerifierSetup {
            pcs: verifier_pcs,
            packing,
        },
    )
}

fn prove_packed(
    packed: &MaterializedPackedWitness<PackedId, Fr>,
    setup: &PrefixPackedProverSetup<KzgPCS, PackedId>,
    statement: PackedStatement,
    hint: <KzgPCS as CommitmentScheme>::OpeningHint,
    label: &'static [u8],
) -> <PackedKzgBatch as BatchOpeningScheme>::Proof {
    let mut transcript = Blake2bTranscript::new(label);
    <PackedKzgBatch as BatchOpeningScheme>::prove_batch(
        setup,
        statement,
        PackedWitness::new(&packed.polynomial, hint),
        &mut transcript,
    )
    .expect("HyperKZG prefix-packed batch proof should be produced")
}

#[test]
fn hyperkzg_prefix_packed_batch_roundtrip_complex_mixed_arities() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    assert_eq!(packed.packing.packed_num_vars, 5);
    assert_eq!((&packed.packing).into_iter().count(), 5);

    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let (commitment, hint) =
        <KzgPCS as CommitmentScheme>::commit(&packed.polynomial, &prover_setup.pcs).unwrap();
    let packed_point = vec![fr(11), fr(13), fr(17), fr(19), fr(23)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let statement = PrefixPackedStatement::new(commitment, claims);

    let mut prover_transcript = Blake2bTranscript::new(b"hyperkzg-packed-complex");
    let proof = <PackedKzgBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        PackedWitness::new(&packed.polynomial, hint),
        &mut prover_transcript,
    )
    .expect("HyperKZG prefix-packed batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"hyperkzg-packed-complex");
    <PackedKzgBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        statement,
        &proof,
        &mut verifier_transcript,
    )
    .expect("HyperKZG prefix-packed batch proof should verify");

    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn hyperkzg_prefix_packed_batch_rejects_missing_logical_slot() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.clone());
    let (commitment, hint) =
        <KzgPCS as CommitmentScheme>::commit(&packed.polynomial, &prover_setup.pcs).unwrap();
    let packed_point = vec![fr(11), fr(13), fr(17), fr(19), fr(23)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let _dropped = claims.pop();

    let mut transcript = Blake2bTranscript::new(b"hyperkzg-packed-missing-slot");
    let result = <PackedKzgBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        PrefixPackedStatement::new(commitment, claims),
        PackedWitness::new(&packed.polynomial, hint),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn hyperkzg_prefix_packed_batch_rejects_wrong_logical_arity() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.clone());
    let (commitment, hint) =
        <KzgPCS as CommitmentScheme>::commit(&packed.polynomial, &prover_setup.pcs).unwrap();
    let packed_point = vec![fr(11), fr(13), fr(17), fr(19), fr(23)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let wide = claims
        .iter_mut()
        .find(|claim| claim.id == PackedId::Wide)
        .expect("wide claim should exist");
    let mut point = wide.evaluation.point.clone().into_vec();
    let _removed = point.pop();
    wide.evaluation.point = point.into();

    let mut transcript = Blake2bTranscript::new(b"hyperkzg-packed-wrong-arity");
    let result = <PackedKzgBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        PrefixPackedStatement::new(commitment, claims),
        PackedWitness::new(&packed.polynomial, hint),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn hyperkzg_prefix_packed_batch_rejects_unknown_id() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.clone());
    let (commitment, hint) =
        <KzgPCS as CommitmentScheme>::commit(&packed.polynomial, &prover_setup.pcs).unwrap();
    let packed_point = vec![fr(11), fr(13), fr(17), fr(19), fr(23)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    claims[0].id = PackedId::Unused;

    let mut transcript = Blake2bTranscript::new(b"hyperkzg-packed-unknown-id");
    let result = <PackedKzgBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        PrefixPackedStatement::new(commitment, claims),
        PackedWitness::new(&packed.polynomial, hint),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn hyperkzg_prefix_packed_batch_rejects_tampered_value() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let (commitment, hint) =
        <KzgPCS as CommitmentScheme>::commit(&packed.polynomial, &prover_setup.pcs).unwrap();
    let packed_point = vec![fr(11), fr(13), fr(17), fr(19), fr(23)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let proof = prove_packed(
        &packed,
        &prover_setup,
        PrefixPackedStatement::new(commitment, claims.clone()),
        hint,
        b"hyperkzg-packed-tamper",
    );

    let mut tampered = claims;
    tampered[0].evaluation.value += fr(1);
    let mut verifier_transcript = Blake2bTranscript::new(b"hyperkzg-packed-tamper");
    let result = <PackedKzgBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        PrefixPackedStatement::new(commitment, tampered),
        &proof,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "tampered packed value should fail");
}

#[test]
fn hyperkzg_prefix_packed_batch_rejects_wrong_packed_commitment() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let (commitment, hint) =
        <KzgPCS as CommitmentScheme>::commit(&packed.polynomial, &prover_setup.pcs).unwrap();
    let mut other_evals = packed.polynomial.evaluations().to_vec();
    other_evals[0] += fr(1);
    let other_polynomial = Polynomial::new(other_evals);
    let (other_commitment, ()) =
        <KzgPCS as CommitmentScheme>::commit(&other_polynomial, &prover_setup.pcs).unwrap();
    let packed_point = vec![fr(11), fr(13), fr(17), fr(19), fr(23)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let proof = prove_packed(
        &packed,
        &prover_setup,
        PrefixPackedStatement::new(commitment, claims.clone()),
        hint,
        b"hyperkzg-packed-wrong-commitment",
    );

    let mut verifier_transcript = Blake2bTranscript::new(b"hyperkzg-packed-wrong-commitment");
    let result = <PackedKzgBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        PrefixPackedStatement::new(other_commitment, claims),
        &proof,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "wrong packed commitment should fail");
}
