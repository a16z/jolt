#![expect(clippy::expect_used, reason = "tests assert successful proof paths")]

use jolt_crypto::Commitment;
use jolt_dory::DoryScheme;
use jolt_field::{Fr, FromPrimitiveInt};
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

type PackedDoryBatch = PackedBatch<DoryScheme, PackedId>;
type DoryOutput = <DoryScheme as Commitment>::Output;
type DoryOpeningHint = <DoryScheme as CommitmentScheme>::OpeningHint;
type PackedStatement = PrefixPackedStatement<Fr, PackedId, DoryOutput>;

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

fn packed_polynomials() -> Vec<(PackedId, Polynomial<Fr>)> {
    let mut rng = ChaCha20Rng::seed_from_u64(0x0a_11_ce);
    vec![
        (PackedId::Wide, Polynomial::<Fr>::random(3, &mut rng)),
        (PackedId::Medium, Polynomial::<Fr>::random(2, &mut rng)),
        (PackedId::NarrowB, Polynomial::<Fr>::random(1, &mut rng)),
        (PackedId::NarrowA, Polynomial::<Fr>::random(1, &mut rng)),
        (PackedId::Constant, Polynomial::new(vec![fr(41)])),
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
    PrefixPackedProverSetup<DoryScheme, PackedId>,
    PrefixPackedVerifierSetup<DoryScheme, PackedId>,
) {
    let prover_pcs = DoryScheme::setup_prover(packing.packed_num_vars);
    let verifier_pcs = DoryScheme::setup_verifier(packing.packed_num_vars);
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
    setup: &PrefixPackedProverSetup<DoryScheme, PackedId>,
    statement: PackedStatement,
    hint: DoryOpeningHint,
    label: &'static [u8],
) -> <PackedDoryBatch as BatchOpeningScheme>::Proof {
    let mut transcript = Blake2bTranscript::new(label);
    <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        setup,
        statement,
        PackedWitness::new(&packed.polynomial, hint),
        &mut transcript,
    )
    .expect("Dory prefix-packed batch proof should be produced")
}

#[test]
fn dory_prefix_packed_batch_roundtrip_complex_mixed_arities() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    assert_eq!(packed.packing.packed_num_vars, 5);
    assert_eq!((&packed.packing).into_iter().count(), 5);
    assert_eq!(packed.packing[&PackedId::Wide].num_vars, 3);
    assert_eq!(packed.packing[&PackedId::Medium].num_vars, 2);

    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let statement = PrefixPackedStatement::new(commitment, claims);

    let mut prover_transcript = Blake2bTranscript::new(b"dory-packed-complex");
    let proof = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        PackedWitness::new(&packed.polynomial, hint),
        &mut prover_transcript,
    )
    .expect("Dory prefix-packed batch proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-complex");
    <PackedDoryBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        statement,
        &proof,
        &mut verifier_transcript,
    )
    .expect("Dory prefix-packed batch proof should verify");

    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn dory_prefix_packed_batch_rejects_right_values_at_wrong_suffix_point() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);
    let original_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let original_claims = packed_claims(&polynomials, &packed.packing, &original_point);
    let proof = prove_packed(
        &packed,
        &prover_setup,
        PrefixPackedStatement::new(commitment.clone(), original_claims),
        hint,
        b"dory-packed-wrong-suffix-point",
    );

    let wrong_point = vec![fr(3), fr(5), fr(17), fr(11), fr(13)];
    let wrong_claims = packed_claims(&polynomials, &packed.packing, &wrong_point);
    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-wrong-suffix-point");
    let result = <PackedDoryBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        PrefixPackedStatement::new(commitment, wrong_claims),
        &proof,
        &mut verifier_transcript,
    );
    assert!(
        result.is_err(),
        "right values at a different suffix-compatible point should fail"
    );
}

#[test]
fn dory_prefix_packed_batch_rejects_known_id_prefix_tamper() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let proof = prove_packed(
        &packed,
        &prover_setup,
        PrefixPackedStatement::new(commitment.clone(), claims.clone()),
        hint,
        b"dory-packed-prefix-tamper",
    );

    let mut tampered = claims;
    for claim in &mut tampered {
        claim.id = match claim.id {
            PackedId::NarrowA => PackedId::NarrowB,
            PackedId::NarrowB => PackedId::NarrowA,
            id => id,
        };
    }
    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-prefix-tamper");
    let result = <PackedDoryBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        PrefixPackedStatement::new(commitment, tampered),
        &proof,
        &mut verifier_transcript,
    );
    assert!(
        result.is_err(),
        "swapping same-arity ids changes the prefix selectors and should fail"
    );
}

#[test]
fn dory_prefix_packed_batch_rejects_duplicate_known_id() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    claims[0].id = claims[1].id;

    let mut transcript = Blake2bTranscript::new(b"dory-packed-duplicate-id");
    let result = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        PrefixPackedStatement::new(commitment, claims),
        PackedWitness::new(&packed.polynomial, hint),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_prefix_packed_batch_rejects_unknown_id() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    claims[0].id = PackedId::Unused;

    let mut transcript = Blake2bTranscript::new(b"dory-packed-unknown-id");
    let result = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        PrefixPackedStatement::new(commitment, claims),
        PackedWitness::new(&packed.polynomial, hint),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_prefix_packed_batch_rejects_suffix_incompatible_claims() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let medium = claims
        .iter_mut()
        .find(|claim| claim.id == PackedId::Medium)
        .expect("medium claim should exist");
    let mut point = medium.evaluation.point.clone().into_vec();
    point[0] += fr(1);
    medium.evaluation.point = point.into();

    let mut transcript = Blake2bTranscript::new(b"dory-packed-suffix-incompat");
    let result = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        PrefixPackedStatement::new(commitment, claims),
        PackedWitness::new(&packed.polynomial, hint),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_prefix_packed_batch_rejects_tampered_value() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let proof = prove_packed(
        &packed,
        &prover_setup,
        PrefixPackedStatement::new(commitment.clone(), claims.clone()),
        hint,
        b"dory-packed-tampered-value",
    );

    let mut tampered = claims;
    tampered[0].evaluation.value += fr(1);
    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-tampered-value");
    let result = <PackedDoryBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        PrefixPackedStatement::new(commitment, tampered),
        &proof,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "tampered packed value should fail");
}

#[test]
fn dory_prefix_packed_batch_rejects_wrong_witness_dimension() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let wrong_witness = Polynomial::new(vec![fr(1), fr(2), fr(3), fr(4)]);

    let mut transcript = Blake2bTranscript::new(b"dory-packed-wrong-witness");
    let result = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        PrefixPackedStatement::new(commitment, claims),
        PackedWitness::new(&wrong_witness, hint),
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}
