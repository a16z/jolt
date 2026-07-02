#![expect(clippy::expect_used, reason = "tests assert successful proof paths")]

use jolt_crypto::Commitment;
use jolt_dory::DoryScheme;
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, OpeningsError, PackedBatch,
    PrefixPackedProverSetup, PrefixPackedStatement, PrefixPackedVerifierSetup, PrefixPacking,
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
) -> Vec<(PackedId, EvaluationClaim<Fr>)> {
    polynomials
        .iter()
        .map(|(id, polynomial)| {
            let logical_point = packing
                .logical_point(id, packed_point)
                .expect("packed point should produce logical suffix");
            (
                *id,
                EvaluationClaim::new(logical_point.clone(), polynomial.evaluate(&logical_point)),
            )
        })
        .collect()
}

fn independent_claims(
    polynomials: &[(PackedId, Polynomial<Fr>)],
    rng: &mut ChaCha20Rng,
) -> Vec<(PackedId, EvaluationClaim<Fr>)> {
    polynomials
        .iter()
        .map(|(id, polynomial)| {
            let logical_point = (0..polynomial.num_vars())
                .map(|_| Fr::random(rng))
                .collect::<Vec<_>>();
            (
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
        &packed.polynomial,
        hint,
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
        &packed.polynomial,
        hint,
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
fn dory_prefix_packed_batch_rejects_proof_for_different_claim_points() {
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
        "a proof for one claim set should not verify a statement with different points"
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
        claim.0 = match claim.0 {
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
    claims[0].0 = claims[1].0;

    let mut transcript = Blake2bTranscript::new(b"dory-packed-duplicate-id");
    let result = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        PrefixPackedStatement::new(commitment, claims),
        &packed.polynomial,
        hint,
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
    claims[0].0 = PackedId::Unused;

    let mut transcript = Blake2bTranscript::new(b"dory-packed-unknown-id");
    let result = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        PrefixPackedStatement::new(commitment, claims),
        &packed.polynomial,
        hint,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_prefix_packed_batch_roundtrip_independent_points_per_slot() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xdecaf);
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);
    let claims = independent_claims(&polynomials, &mut rng);
    let statement = PrefixPackedStatement::new(commitment, claims);

    let mut prover_transcript = Blake2bTranscript::new(b"dory-packed-independent-points");
    let proof = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        statement.clone(),
        &packed.polynomial,
        hint,
        &mut prover_transcript,
    )
    .expect("claims at independent per-slot points should be provable");

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-independent-points");
    <PackedDoryBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        statement,
        &proof,
        &mut verifier_transcript,
    )
    .expect("claims at independent per-slot points should verify");

    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

/// A malicious prover proving a stale value: the claim point moves but the
/// value is left at the old point's evaluation.
#[test]
fn dory_prefix_packed_batch_rejects_stale_value_at_shifted_point() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let medium = claims
        .iter_mut()
        .find(|claim| claim.0 == PackedId::Medium)
        .expect("medium claim should exist");
    let mut point = medium.1.point.clone().into_vec();
    point[0] += fr(1);
    medium.1.point = point.into();
    let statement = PrefixPackedStatement::new(commitment, claims);

    let proof = prove_packed(
        &packed,
        &prover_setup,
        statement.clone(),
        hint,
        b"dory-packed-stale-value",
    );
    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-stale-value");
    let result = <PackedDoryBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        statement,
        &proof,
        &mut verifier_transcript,
    );
    assert!(
        result.is_err(),
        "stale value at a shifted point should fail"
    );
}

/// Regression test for the pinned-point reduction soundness bug: two slots
/// sharing the shortest packing prefix (`Medium` = `010`, `NarrowA` = `0110`)
/// had challenge-independent relative eq-weights, so lies of the form
/// `Δ_medium·(1 - a_1) + Δ_narrow_a·a_1(1 - a_2) = 0` (with `a` the pinned
/// suffix coordinates) cancelled identically and verified. The sumcheck
/// reduction must reject them.
#[test]
fn dory_prefix_packed_batch_rejects_seesaw_value_cancellation() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    assert_eq!(
        packed.packing[&PackedId::Medium].prefix,
        vec![false, true, false]
    );
    assert_eq!(
        packed.packing[&PackedId::NarrowA].prefix,
        vec![false, true, true, false]
    );

    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);

    let a_1 = packed_point[2];
    let a_2 = packed_point[3];
    let weight_medium = fr(1) - a_1;
    let weight_narrow_a = a_1 * (fr(1) - a_2);
    let delta_medium = weight_narrow_a;
    let delta_narrow_a = -weight_medium;
    assert_eq!(
        delta_medium * weight_medium + delta_narrow_a * weight_narrow_a,
        fr(0),
        "the two lies must cancel under the old pinned-point reduction"
    );
    for claim in &mut claims {
        match claim.0 {
            PackedId::Medium => claim.1.value += delta_medium,
            PackedId::NarrowA => claim.1.value += delta_narrow_a,
            _ => {}
        }
    }
    let statement = PrefixPackedStatement::new(commitment, claims);

    let proof = prove_packed(
        &packed,
        &prover_setup,
        statement.clone(),
        hint,
        b"dory-packed-seesaw",
    );
    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-seesaw");
    let result = <PackedDoryBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        statement,
        &proof,
        &mut verifier_transcript,
    );
    assert!(
        result.is_err(),
        "cancelling value lies across same-bucket slots should fail"
    );
}

#[test]
fn dory_prefix_packed_batch_rejects_missing_slot_claim() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    claims.retain(|claim| claim.0 != PackedId::Constant);

    let mut transcript = Blake2bTranscript::new(b"dory-packed-missing-slot");
    let result = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        PrefixPackedStatement::new(commitment, claims),
        &packed.polynomial,
        hint,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_prefix_packed_batch_rejects_empty_claims() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);

    let mut transcript = Blake2bTranscript::new(b"dory-packed-empty-claims");
    let result = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        PrefixPackedStatement::new(commitment, Vec::new()),
        &packed.polynomial,
        hint,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_prefix_packed_batch_rejects_wrong_point_arity() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let medium = claims
        .iter_mut()
        .find(|claim| claim.0 == PackedId::Medium)
        .expect("medium claim should exist");
    let mut point = medium.1.point.clone().into_vec();
    point.push(fr(17));
    medium.1.point = point.into();

    let mut transcript = Blake2bTranscript::new(b"dory-packed-wrong-arity");
    let result = <PackedDoryBatch as BatchOpeningScheme>::prove_batch(
        &prover_setup,
        PrefixPackedStatement::new(commitment, claims),
        &packed.polynomial,
        hint,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_prefix_packed_batch_rejects_wrong_packed_commitment() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone());
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup.pcs);
    let mut rng = ChaCha20Rng::seed_from_u64(0x0bad_c0de);
    let other_polynomial = Polynomial::<Fr>::random(packed.packing.packed_num_vars, &mut rng);
    let (other_commitment, _) = DoryScheme::commit(&other_polynomial, &prover_setup.pcs);
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let proof = prove_packed(
        &packed,
        &prover_setup,
        PrefixPackedStatement::new(commitment, claims.clone()),
        hint,
        b"dory-packed-wrong-commitment",
    );

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-wrong-commitment");
    let result = <PackedDoryBatch as BatchOpeningScheme>::verify_batch(
        &verifier_setup,
        PrefixPackedStatement::new(other_commitment, claims),
        &proof,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "wrong packed commitment should reject");
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
    tampered[0].1.value += fr(1);
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
        &wrong_witness,
        hint,
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}
