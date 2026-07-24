#![expect(clippy::expect_used, reason = "tests assert successful proof paths")]
#![expect(
    clippy::unwrap_used,
    reason = "benchmarks and tests unwrap successful PCS operations"
)]

use jolt_crypto::Commitment;
use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_openings::{
    prove_packed_openings, verify_packed_openings, CommitmentScheme, OpeningsError,
    PackedObjectGroup, PackedOpeningProof, PackedProverGroup, PackedProverObject,
    PackedVerifierObject, PrefixPackedStatement,
};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

#[path = "support/common.rs"]
pub mod common;
#[path = "support/packed.rs"]
pub mod packed_support;

use common::fr;
use packed_support::{
    build_packed, independent_claims, materialize_packed, packed_claims, packed_polynomials,
    MaterializedPackedWitness, PackedId,
};

type DoryOutput = <DoryScheme as Commitment>::Output;
type DoryProof = <DoryScheme as CommitmentScheme>::Proof;
type DoryProverSetup = <DoryScheme as CommitmentScheme>::ProverSetup;
type DoryVerifierSetup = <DoryScheme as CommitmentScheme>::VerifierSetup;
type DoryOpeningHint = <DoryScheme as CommitmentScheme>::OpeningHint;
type PackedStatement = PrefixPackedStatement<Fr, PackedId, DoryOutput>;

fn packed_setup(num_vars: usize) -> (DoryProverSetup, DoryVerifierSetup) {
    (
        DoryScheme::setup_prover(num_vars),
        DoryScheme::setup_verifier(num_vars),
    )
}

fn prove_single(
    packed: &MaterializedPackedWitness<PackedId, Fr>,
    setup: &DoryProverSetup,
    statement: &PackedStatement,
    hint: DoryOpeningHint,
    label: &'static [u8],
) -> Result<PackedOpeningProof<Fr, DoryProof>, OpeningsError> {
    let mut transcript = Blake2bTranscript::new(label);
    prove_packed_openings::<DoryScheme, PackedId, _>(
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

fn prove_packed(
    packed: &MaterializedPackedWitness<PackedId, Fr>,
    setup: &DoryProverSetup,
    statement: &PackedStatement,
    hint: DoryOpeningHint,
    label: &'static [u8],
) -> PackedOpeningProof<Fr, DoryProof> {
    prove_single(packed, setup, statement, hint, label)
        .expect("Dory prefix-packed opening proof should be produced")
}

fn verify_single(
    packed: &MaterializedPackedWitness<PackedId, Fr>,
    setup: &DoryVerifierSetup,
    statement: &PackedStatement,
    proof: &PackedOpeningProof<Fr, DoryProof>,
    transcript: &mut Blake2bTranscript,
) -> Result<(), OpeningsError> {
    verify_packed_openings::<DoryScheme, PackedId, _>(
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
fn dory_prefix_packed_batch_roundtrip_complex_mixed_arities() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    assert_eq!(packed.packing.packed_num_vars, 5);
    assert_eq!((&packed.packing).into_iter().count(), 5);
    assert_eq!(packed.packing[&PackedId::Wide].num_vars, 3);
    assert_eq!(packed.packing[&PackedId::Medium].num_vars, 2);

    let (prover_setup, verifier_setup) = packed_setup(packed.packing.packed_num_vars);
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let statement = PrefixPackedStatement::new(commitment, claims);

    let mut prover_transcript = Blake2bTranscript::new(b"dory-packed-complex");
    let proof = prove_packed_openings::<DoryScheme, PackedId, _>(
        vec![PackedProverObject {
            packing: &packed.packing,
            statement: &statement,
            polynomial: &packed.polynomial,
            setup: &prover_setup,
        }],
        vec![PackedProverGroup::singleton(0, Some(hint))],
        &mut prover_transcript,
    )
    .expect("Dory prefix-packed opening proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-complex");
    verify_single(
        &packed,
        &verifier_setup,
        &statement,
        &proof,
        &mut verifier_transcript,
    )
    .expect("Dory prefix-packed opening proof should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn dory_prefix_packed_batch_rejects_proof_for_different_claim_points() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.packed_num_vars);
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let original_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let original_claims = packed_claims(&polynomials, &packed.packing, &original_point);
    let proof = prove_packed(
        &packed,
        &prover_setup,
        &PrefixPackedStatement::new(commitment.clone(), original_claims),
        hint,
        b"dory-packed-wrong-suffix-point",
    );

    let wrong_point = vec![fr(3), fr(5), fr(17), fr(11), fr(13)];
    let wrong_claims = packed_claims(&polynomials, &packed.packing, &wrong_point);
    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-wrong-suffix-point");
    let result = verify_single(
        &packed,
        &verifier_setup,
        &PrefixPackedStatement::new(commitment, wrong_claims),
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
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.packed_num_vars);
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let proof = prove_packed(
        &packed,
        &prover_setup,
        &PrefixPackedStatement::new(commitment.clone(), claims.clone()),
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
    let result = verify_single(
        &packed,
        &verifier_setup,
        &PrefixPackedStatement::new(commitment, tampered),
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
    let (prover_setup, _) = packed_setup(packed.packing.packed_num_vars);
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    claims[0].0 = claims[1].0;

    let result = prove_single(
        &packed,
        &prover_setup,
        &PrefixPackedStatement::new(commitment, claims),
        hint,
        b"dory-packed-duplicate-id",
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_prefix_packed_batch_rejects_unknown_id() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.packed_num_vars);
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    claims[0].0 = PackedId::Unused;

    let result = prove_single(
        &packed,
        &prover_setup,
        &PrefixPackedStatement::new(commitment, claims),
        hint,
        b"dory-packed-unknown-id",
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_prefix_packed_batch_roundtrip_independent_points_per_slot() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xdecaf);
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.packed_num_vars);
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let claims = independent_claims(&polynomials, &mut rng);
    let statement = PrefixPackedStatement::new(commitment, claims);

    let proof = prove_packed(
        &packed,
        &prover_setup,
        &statement,
        hint,
        b"dory-packed-independent-points",
    );
    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-independent-points");
    verify_single(
        &packed,
        &verifier_setup,
        &statement,
        &proof,
        &mut verifier_transcript,
    )
    .expect("claims at independent per-slot points should verify");
}

/// A malicious prover proving a stale value: the claim point moves but the
/// value is left at the old point's evaluation.
#[test]
fn dory_prefix_packed_batch_rejects_stale_value_at_shifted_point() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.packed_num_vars);
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup).unwrap();
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
        &statement,
        hint,
        b"dory-packed-stale-value",
    );
    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-stale-value");
    let result = verify_single(
        &packed,
        &verifier_setup,
        &statement,
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

    let (prover_setup, verifier_setup) = packed_setup(packed.packing.packed_num_vars);
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup).unwrap();
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
        &statement,
        hint,
        b"dory-packed-seesaw",
    );
    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-seesaw");
    let result = verify_single(
        &packed,
        &verifier_setup,
        &statement,
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
    let (prover_setup, _) = packed_setup(packed.packing.packed_num_vars);
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    claims.retain(|claim| claim.0 != PackedId::Constant);

    let result = prove_single(
        &packed,
        &prover_setup,
        &PrefixPackedStatement::new(commitment, claims),
        hint,
        b"dory-packed-missing-slot",
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_prefix_packed_batch_rejects_empty_claims() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.packed_num_vars);
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup).unwrap();

    let result = prove_single(
        &packed,
        &prover_setup,
        &PrefixPackedStatement::new(commitment, Vec::new()),
        hint,
        b"dory-packed-empty-claims",
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_prefix_packed_batch_rejects_wrong_point_arity() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.packed_num_vars);
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let mut claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let medium = claims
        .iter_mut()
        .find(|claim| claim.0 == PackedId::Medium)
        .expect("medium claim should exist");
    let mut point = medium.1.point.clone().into_vec();
    point.push(fr(17));
    medium.1.point = point.into();

    let result = prove_single(
        &packed,
        &prover_setup,
        &PrefixPackedStatement::new(commitment, claims),
        hint,
        b"dory-packed-wrong-arity",
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

#[test]
fn dory_prefix_packed_batch_rejects_wrong_packed_commitment() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.packed_num_vars);
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(0x0bad_c0de);
    let other_polynomial = Polynomial::<Fr>::random(packed.packing.packed_num_vars, &mut rng);
    let (other_commitment, _) = DoryScheme::commit(&other_polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let proof = prove_packed(
        &packed,
        &prover_setup,
        &PrefixPackedStatement::new(commitment, claims.clone()),
        hint,
        b"dory-packed-wrong-commitment",
    );

    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-wrong-commitment");
    let result = verify_single(
        &packed,
        &verifier_setup,
        &PrefixPackedStatement::new(other_commitment, claims),
        &proof,
        &mut verifier_transcript,
    );
    assert!(result.is_err(), "wrong packed commitment should reject");
}

#[test]
fn dory_prefix_packed_batch_rejects_tampered_value() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.packed_num_vars);
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let proof = prove_packed(
        &packed,
        &prover_setup,
        &PrefixPackedStatement::new(commitment.clone(), claims.clone()),
        hint,
        b"dory-packed-tampered-value",
    );

    let mut tampered = claims;
    tampered[0].1.value += fr(1);
    let mut verifier_transcript = Blake2bTranscript::new(b"dory-packed-tampered-value");
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
fn dory_prefix_packed_batch_rejects_wrong_witness_dimension() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.packed_num_vars);
    let (commitment, hint) = DoryScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let packed_point = vec![fr(3), fr(5), fr(7), fr(11), fr(13)];
    let claims = packed_claims(&polynomials, &packed.packing, &packed_point);
    let wrong_witness = Polynomial::new(vec![fr(1), fr(2), fr(3), fr(4)]);

    let mut transcript = Blake2bTranscript::new(b"dory-packed-wrong-witness");
    let result = prove_packed_openings::<DoryScheme, PackedId, _>(
        vec![PackedProverObject {
            packing: &packed.packing,
            statement: &PrefixPackedStatement::new(commitment, claims),
            polynomial: &wrong_witness,
            setup: &prover_setup,
        }],
        vec![PackedProverGroup::singleton(0, Some(hint))],
        &mut transcript,
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

/// The second fixture object for the joint tests: two polynomials of arities
/// 2 and 1 packing into 3 variables, so the joint reduction pads it by two
/// rounds against the 5-variable object.
fn narrow_object_polynomials() -> Vec<(PackedId, Polynomial<Fr>)> {
    let mut rng = ChaCha20Rng::seed_from_u64(0x0b_0b_0b);
    vec![
        (PackedId::Medium, Polynomial::<Fr>::random(2, &mut rng)),
        (PackedId::NarrowA, Polynomial::<Fr>::random(1, &mut rng)),
    ]
}

/// Joint opening across two commitment objects of different widths: one
/// reduction sumcheck, per-object PCS openings, and the padded (narrower)
/// object bound at the suffix of the shared point.
#[test]
fn dory_joint_packed_openings_roundtrip_across_two_objects() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x70_67_65);
    let wide_polynomials = packed_polynomials();
    let wide = build_packed(&wide_polynomials);
    let narrow_polynomials = narrow_object_polynomials();
    let narrow = materialize_packed(&narrow_polynomials).expect("narrow object should build");
    assert_eq!(wide.packing.packed_num_vars, 5);
    assert_eq!(narrow.packing.packed_num_vars, 3);

    let (wide_prover, wide_verifier) = packed_setup(wide.packing.packed_num_vars);
    let (narrow_prover, narrow_verifier) = packed_setup(narrow.packing.packed_num_vars);
    let (wide_commitment, wide_hint) = DoryScheme::commit(&wide.polynomial, &wide_prover).unwrap();
    let (narrow_commitment, narrow_hint) =
        DoryScheme::commit(&narrow.polynomial, &narrow_prover).unwrap();

    let wide_statement = PrefixPackedStatement::new(
        wide_commitment,
        independent_claims(&wide_polynomials, &mut rng),
    );
    let narrow_statement = PrefixPackedStatement::new(
        narrow_commitment,
        independent_claims(&narrow_polynomials, &mut rng),
    );

    let mut prover_transcript = Blake2bTranscript::new(b"dory-joint-two-objects");
    let proof = prove_packed_openings::<DoryScheme, PackedId, _>(
        vec![
            PackedProverObject {
                packing: &wide.packing,
                statement: &wide_statement,
                polynomial: &wide.polynomial,
                setup: &wide_prover,
            },
            PackedProverObject {
                packing: &narrow.packing,
                statement: &narrow_statement,
                polynomial: &narrow.polynomial,
                setup: &narrow_prover,
            },
        ],
        vec![
            PackedProverGroup::singleton(0, Some(wide_hint)),
            PackedProverGroup::singleton(1, Some(narrow_hint)),
        ],
        &mut prover_transcript,
    )
    .expect("joint proof across two objects should be produced");
    assert_eq!(proof.round_polynomials.len(), 5);
    assert_eq!(proof.evaluations.len(), 2);
    assert_eq!(proof.openings.len(), 2);

    let objects = [
        PackedVerifierObject::<DoryScheme, PackedId> {
            packing: &wide.packing,
            statement: &wide_statement,
            setup: &wide_verifier,
        },
        PackedVerifierObject::<DoryScheme, PackedId> {
            packing: &narrow.packing,
            statement: &narrow_statement,
            setup: &narrow_verifier,
        },
    ];
    let mut verifier_transcript = Blake2bTranscript::new(b"dory-joint-two-objects");
    verify_packed_openings(
        &objects,
        &[
            PackedObjectGroup::singleton(0),
            PackedObjectGroup::singleton(1),
        ],
        &proof,
        &mut verifier_transcript,
    )
    .expect("joint proof across two objects should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());

    // Tamper each per-object payload: a corrupted claimed evaluation and a
    // mismatched object count must both reject.
    let mut tampered = proof.clone();
    tampered.evaluations[1] += fr(1);
    let mut transcript = Blake2bTranscript::new(b"dory-joint-two-objects");
    assert!(
        verify_packed_openings(
            &objects,
            &[
                PackedObjectGroup::singleton(0),
                PackedObjectGroup::singleton(1)
            ],
            &tampered,
            &mut transcript
        )
        .is_err(),
        "corrupted narrow-object evaluation should fail"
    );

    let mut transcript = Blake2bTranscript::new(b"dory-joint-two-objects");
    assert!(
        matches!(
            verify_packed_openings(
                &objects[..1],
                &[
                    PackedObjectGroup::singleton(0),
                    PackedObjectGroup::singleton(1)
                ],
                &proof,
                &mut transcript
            ),
            Err(OpeningsError::InvalidBatch(_))
        ),
        "object count must match the proof's per-object payloads"
    );

    // A tampered value claim on the narrow (padded) object must break the
    // joint reduction even though the wide object's claims stay honest.
    let mut lying_claims = narrow_statement.claims.clone();
    lying_claims[0].1.value += fr(1);
    let lying_statement =
        PrefixPackedStatement::new(narrow_statement.commitment.clone(), lying_claims);
    let lying_objects = [
        PackedVerifierObject::<DoryScheme, PackedId> {
            packing: &wide.packing,
            statement: &wide_statement,
            setup: &wide_verifier,
        },
        PackedVerifierObject::<DoryScheme, PackedId> {
            packing: &narrow.packing,
            statement: &lying_statement,
            setup: &narrow_verifier,
        },
    ];
    let mut transcript = Blake2bTranscript::new(b"dory-joint-two-objects");
    assert!(
        verify_packed_openings(
            &lying_objects,
            &[
                PackedObjectGroup::singleton(0),
                PackedObjectGroup::singleton(1)
            ],
            &proof,
            &mut transcript
        )
        .is_err(),
        "a lying claim on the padded object should fail the joint reduction"
    );
}
