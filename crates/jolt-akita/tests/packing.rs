#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]
#![expect(
    clippy::unwrap_used,
    reason = "benchmarks and tests unwrap successful PCS operations"
)]

mod support;

use jolt_akita::{AkitaCommitment, AkitaField, AkitaNativeBatching, AkitaScheme};
use jolt_openings::{
    prove_packed_openings, verify_packed_openings, BatchOpeningScheme, CommitmentScheme,
    EvaluationClaim, OpeningsError, PackedObjectGroup, PackedOpeningProof, PackedProverGroup,
    PackedProverObject, PackedVerifierObject, PrefixPackedStatement, PrefixPacking,
};
use jolt_poly::{MultilinearPoly, OneHotPolynomial, Polynomial};
use jolt_transcript::{Blake2bTranscript, Transcript};
use support::{
    batch_polynomials, f, layout, materialize_packed, native_statement, packed_claims,
    packed_setup, polynomial, setup_for, MaterializedPackedWitness,
};

type AkitaProverSetup = <AkitaScheme as CommitmentScheme>::ProverSetup;
type AkitaVerifierSetup = <AkitaScheme as CommitmentScheme>::VerifierSetup;
type AkitaOpeningHint = <AkitaScheme as CommitmentScheme>::OpeningHint;
type AkitaProof = <AkitaScheme as CommitmentScheme>::Proof;
type PackedStatement = PrefixPackedStatement<AkitaField, PackedId, AkitaCommitment>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum PackedId {
    Constant,
    NarrowA,
    NarrowB,
    Medium,
    Wide,
    Unused,
}

fn packed_polynomials() -> Vec<(PackedId, Polynomial<AkitaField>)> {
    vec![
        (PackedId::Wide, polynomial(12, 1)),
        (PackedId::Medium, polynomial(11, 40)),
        (PackedId::NarrowB, polynomial(10, 80)),
        (PackedId::NarrowA, polynomial(10, 120)),
        (PackedId::Constant, Polynomial::new(vec![f(200)])),
    ]
}

fn packed_point() -> Vec<AkitaField> {
    (0..14).map(|i| f(3 + 2 * i)).collect()
}

fn build_packed(
    polynomials: &[(PackedId, Polynomial<AkitaField>)],
) -> MaterializedPackedWitness<PackedId> {
    materialize_packed(polynomials).expect("packed witness should build")
}

fn statement_for(
    packed: &MaterializedPackedWitness<PackedId>,
    polynomials: &[(PackedId, Polynomial<AkitaField>)],
    commitment: AkitaCommitment,
    point: &[AkitaField],
) -> PackedStatement {
    PrefixPackedStatement::new(
        commitment,
        packed_claims(polynomials, &packed.packing, point),
    )
}

fn prove_single(
    packing: &PrefixPacking<PackedId>,
    statement: &PackedStatement,
    polynomial: &Polynomial<AkitaField>,
    setup: &AkitaProverSetup,
    hint: AkitaOpeningHint,
    label: &'static [u8],
) -> Result<PackedOpeningProof<AkitaField, AkitaProof>, OpeningsError> {
    let mut transcript = Blake2bTranscript::new(label);
    prove_packed_openings::<AkitaScheme, PackedId, _>(
        vec![PackedProverObject {
            packing,
            statement,
            polynomial,
            setup,
        }],
        vec![PackedProverGroup::singleton(0, Some(hint))],
        &mut transcript,
    )
}

fn assert_packed_verify_rejects(
    packing: &PrefixPacking<PackedId>,
    verifier_setup: &AkitaVerifierSetup,
    statement: PackedStatement,
    proof: &PackedOpeningProof<AkitaField, AkitaProof>,
    label: &'static [u8],
) {
    let mut transcript = Blake2bTranscript::new(label);
    assert!(verify_packed_openings::<AkitaScheme, PackedId, _>(
        &[PackedVerifierObject {
            packing,
            statement: &statement,
            setup: verifier_setup,
        }],
        &[PackedObjectGroup::singleton(0)],
        proof,
        &mut transcript,
    )
    .is_err());
}

#[test]
fn akita_prefix_packed_batch_roundtrips_mixed_arities() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    assert_eq!(packed.packing.packed_num_vars, 14);
    assert_eq!((&packed.packing).into_iter().count(), 5);
    assert_eq!(packed.packing[&PackedId::Wide].num_vars, 12);
    assert_eq!(packed.packing[&PackedId::Medium].num_vars, 11);

    let (prover_setup, verifier_setup) = packed_setup(packed.packing.packed_num_vars, layout(7));
    let (commitment, hint) = AkitaScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let point = packed_point();
    let statement = statement_for(&packed, &polynomials, commitment, &point);

    let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-mixed");
    let proof = prove_packed_openings::<AkitaScheme, PackedId, _>(
        vec![PackedProverObject {
            packing: &packed.packing,
            statement: &statement,
            polynomial: &packed.polynomial,
            setup: &prover_setup,
        }],
        vec![PackedProverGroup::singleton(0, Some(hint))],
        &mut prover_transcript,
    )
    .expect("Akita prefix-packed opening proof should be produced");

    let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-mixed");
    verify_packed_openings::<AkitaScheme, PackedId, _>(
        &[PackedVerifierObject {
            packing: &packed.packing,
            statement: &statement,
            setup: &verifier_setup,
        }],
        &[PackedObjectGroup::singleton(0)],
        &proof,
        &mut verifier_transcript,
    )
    .expect("Akita prefix-packed opening proof should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
}

#[test]
fn akita_prefix_packed_batch_rejects_statement_shape_errors() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.packed_num_vars, layout(7));
    let (commitment, hint) = AkitaScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let point = packed_point();
    let claims = packed_claims(&polynomials, &packed.packing, &point);

    let result = prove_single(
        &packed.packing,
        &PrefixPackedStatement::new(commitment.clone(), claims[1..].to_vec()),
        &packed.polynomial,
        &prover_setup,
        hint.clone(),
        b"akita-packed-missing-slot",
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));

    let mut unknown_id = claims.clone();
    unknown_id[0].0 = PackedId::Unused;
    let result = prove_single(
        &packed.packing,
        &PrefixPackedStatement::new(commitment.clone(), unknown_id),
        &packed.polynomial,
        &prover_setup,
        hint.clone(),
        b"akita-packed-unknown-id",
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));

    let mut duplicate_id = claims.clone();
    duplicate_id[0].0 = duplicate_id[1].0;
    let result = prove_single(
        &packed.packing,
        &PrefixPackedStatement::new(commitment.clone(), duplicate_id),
        &packed.polynomial,
        &prover_setup,
        hint.clone(),
        b"akita-packed-duplicate-id",
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));

    let mut wrong_arity = claims.clone();
    let constant = wrong_arity
        .iter_mut()
        .find(|claim| claim.0 == PackedId::Constant)
        .expect("constant claim should exist");
    constant.1 = EvaluationClaim::new(vec![f(1)], constant.1.value);
    let result = prove_single(
        &packed.packing,
        &PrefixPackedStatement::new(commitment.clone(), wrong_arity),
        &packed.polynomial,
        &prover_setup,
        hint.clone(),
        b"akita-packed-wrong-arity",
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

/// A malicious prover proving a stale value: the claim point moves but the
/// value is left at the old point's evaluation. The reduction sumcheck
/// accepts claims at independent points, so proving succeeds and the lie is
/// caught at verification.
#[test]
fn akita_prefix_packed_batch_rejects_stale_value_at_shifted_point() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.packed_num_vars, layout(7));
    let (commitment, hint) = AkitaScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let point = packed_point();
    let mut claims = packed_claims(&polynomials, &packed.packing, &point);
    let medium = claims
        .iter_mut()
        .find(|claim| claim.0 == PackedId::Medium)
        .expect("medium claim should exist");
    let mut shifted = medium.1.point.clone().into_vec();
    shifted[0] += f(1);
    medium.1 = EvaluationClaim::new(shifted, medium.1.value);
    let statement = PrefixPackedStatement::new(commitment, claims);

    let proof = prove_single(
        &packed.packing,
        &statement,
        &packed.polynomial,
        &prover_setup,
        hint,
        b"akita-packed-stale-value",
    )
    .expect("shifted-point claims are provable under the reduction sumcheck");
    assert_packed_verify_rejects(
        &packed.packing,
        &verifier_setup,
        statement,
        &proof,
        b"akita-packed-stale-value",
    );
}

#[test]
fn akita_prefix_packed_batch_rejects_tampered_verifier_inputs() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, verifier_setup) = packed_setup(packed.packing.packed_num_vars, layout(7));
    let (commitment, hint) = AkitaScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let point = packed_point();
    let claims = packed_claims(&polynomials, &packed.packing, &point);
    let statement = PrefixPackedStatement::new(commitment.clone(), claims.clone());

    let proof = prove_single(
        &packed.packing,
        &statement,
        &packed.polynomial,
        &prover_setup,
        hint,
        b"akita-packed-tamper",
    )
    .expect("Akita prefix-packed opening proof should be produced");

    let mut tampered_value = claims.clone();
    tampered_value[0].1.value += f(1);
    assert_packed_verify_rejects(
        &packed.packing,
        &verifier_setup,
        PrefixPackedStatement::new(commitment.clone(), tampered_value),
        &proof,
        b"akita-packed-tamper",
    );

    let mut wrong_point = packed_point();
    wrong_point[2] += f(1);
    let wrong_point_claims = packed_claims(&polynomials, &packed.packing, &wrong_point);
    assert_packed_verify_rejects(
        &packed.packing,
        &verifier_setup,
        PrefixPackedStatement::new(commitment.clone(), wrong_point_claims),
        &proof,
        b"akita-packed-tamper",
    );

    let mut swapped_same_arity = claims.clone();
    for claim in &mut swapped_same_arity {
        claim.0 = match claim.0 {
            PackedId::NarrowA => PackedId::NarrowB,
            PackedId::NarrowB => PackedId::NarrowA,
            id => id,
        };
    }
    assert_packed_verify_rejects(
        &packed.packing,
        &verifier_setup,
        PrefixPackedStatement::new(commitment.clone(), swapped_same_arity),
        &proof,
        b"akita-packed-tamper",
    );

    let other_packed = build_packed(&[
        (PackedId::Wide, polynomial(12, 500)),
        (PackedId::Medium, polynomial(11, 540)),
        (PackedId::NarrowB, polynomial(10, 580)),
        (PackedId::NarrowA, polynomial(10, 620)),
        (PackedId::Constant, Polynomial::new(vec![f(700)])),
    ]);
    let (other_commitment, _) =
        AkitaScheme::commit(&other_packed.polynomial, &prover_setup).unwrap();
    assert_packed_verify_rejects(
        &packed.packing,
        &verifier_setup,
        PrefixPackedStatement::new(other_commitment, claims),
        &proof,
        b"akita-packed-tamper",
    );
}

#[test]
fn akita_prefix_packed_batch_rejects_wrong_witness_dimension() {
    let polynomials = packed_polynomials();
    let packed = build_packed(&polynomials);
    let (prover_setup, _) = packed_setup(packed.packing.packed_num_vars, layout(7));
    let (commitment, hint) = AkitaScheme::commit(&packed.polynomial, &prover_setup).unwrap();
    let point = packed_point();
    let statement = statement_for(&packed, &polynomials, commitment, &point);
    let wrong_witness = polynomial(12, 900);

    let result = prove_single(
        &packed.packing,
        &statement,
        &wrong_witness,
        &prover_setup,
        hint,
        b"akita-packed-wrong-witness",
    );
    assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
}

/// Two Akita commitment objects of different widths discharged through one
/// joint reduction sumcheck — the shape of Jolt's packed final opening
/// (`OneHotTrace` plus a smaller advice/program object).
#[test]
fn akita_joint_packed_openings_roundtrip_across_two_objects() {
    let wide_polynomials = packed_polynomials();
    let wide = build_packed(&wide_polynomials);
    let narrow_polynomials = vec![
        (PackedId::Medium, polynomial(12, 300)),
        (PackedId::NarrowA, polynomial(11, 340)),
    ];
    let narrow = materialize_packed(&narrow_polynomials).expect("narrow object should build");
    assert_eq!(wide.packing.packed_num_vars, 14);
    assert_eq!(narrow.packing.packed_num_vars, 13);

    let (wide_prover, wide_verifier) = packed_setup(wide.packing.packed_num_vars, layout(7));
    let (narrow_prover, narrow_verifier) = packed_setup(narrow.packing.packed_num_vars, layout(9));
    let (wide_commitment, wide_hint) = AkitaScheme::commit(&wide.polynomial, &wide_prover).unwrap();
    let (narrow_commitment, narrow_hint) =
        AkitaScheme::commit(&narrow.polynomial, &narrow_prover).unwrap();

    let wide_statement = statement_for(&wide, &wide_polynomials, wide_commitment, &packed_point());
    let narrow_point: Vec<_> = (0..13).map(|i| f(17 + 2 * i)).collect();
    let narrow_statement = statement_for(
        &narrow,
        &narrow_polynomials,
        narrow_commitment,
        &narrow_point,
    );

    let mut prover_transcript = Blake2bTranscript::new(b"akita-joint-two-objects");
    let proof = prove_packed_openings::<AkitaScheme, PackedId, _>(
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
    .expect("joint proof across two Akita objects should be produced");

    let objects = [
        PackedVerifierObject::<AkitaScheme, PackedId> {
            packing: &wide.packing,
            statement: &wide_statement,
            setup: &wide_verifier,
        },
        PackedVerifierObject::<AkitaScheme, PackedId> {
            packing: &narrow.packing,
            statement: &narrow_statement,
            setup: &narrow_verifier,
        },
    ];
    let mut verifier_transcript = Blake2bTranscript::new(b"akita-joint-two-objects");
    verify_packed_openings(
        &objects,
        &[
            PackedObjectGroup::singleton(0),
            PackedObjectGroup::singleton(1),
        ],
        &proof,
        &mut verifier_transcript,
    )
    .expect("joint proof across two Akita objects should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());

    let mut tampered = proof.clone();
    tampered.evaluations[0] += f(1);
    let mut transcript = Blake2bTranscript::new(b"akita-joint-two-objects");
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
        "corrupted claimed evaluation should fail the joint opening"
    );
}

#[test]
fn akita_packing_and_native_batching_can_coexist_for_same_logical_claims() {
    let poly_a = polynomial(13, 1);
    let poly_b = polynomial(13, 20);
    let logical_point: Vec<_> = (0..13).map(|i| f(2 + 3 * i)).collect();
    let eval_a = poly_a.evaluate(&logical_point);
    let eval_b = poly_b.evaluate(&logical_point);

    let (native_prover_setup, native_verifier_setup) = setup_for(13, 2, layout(7));
    let (native_commitment, native_hint) = AkitaScheme::commit_group(
        &native_prover_setup,
        layout(7),
        &[poly_a.clone(), poly_b.clone()],
    )
    .expect("grouped commit should succeed");
    let native_statement = native_statement(native_commitment, &logical_point, [eval_a, eval_b]);
    let mut native_prover_transcript = Blake2bTranscript::new(b"akita-coexist-black-box");
    let native_proof = <AkitaNativeBatching as BatchOpeningScheme>::prove_batch(
        &native_prover_setup,
        native_statement.clone(),
        batch_polynomials([&poly_a, &poly_b]),
        native_hint,
        &mut native_prover_transcript,
    )
    .expect("black-box proof should be produced");
    let mut native_verifier_transcript = Blake2bTranscript::new(b"akita-coexist-black-box");
    <AkitaNativeBatching as BatchOpeningScheme>::verify_batch(
        &native_verifier_setup,
        &native_statement,
        &native_proof,
        &mut native_verifier_transcript,
    )
    .expect("black-box proof should verify");

    let packed_polynomials = vec![(PackedId::NarrowA, poly_a), (PackedId::NarrowB, poly_b)];
    let packed = build_packed(&packed_polynomials);
    let (packed_prover_setup, packed_verifier_setup) =
        packed_setup(packed.packing.packed_num_vars, layout(7));
    let (packed_commitment, packed_hint) =
        AkitaScheme::commit(&packed.polynomial, &packed_prover_setup).unwrap();
    let packed_point = {
        let prefix_len = packed
            .packing
            .prefix_challenge_len(logical_point.len())
            .expect("same arity packing should have one prefix challenge");
        let mut point = vec![f(13); prefix_len];
        point.extend_from_slice(&logical_point);
        point
    };
    let packed_statement = statement_for(
        &packed,
        &packed_polynomials,
        packed_commitment,
        &packed_point,
    );
    let mut packed_prover_transcript = Blake2bTranscript::new(b"akita-coexist-packed");
    let packed_proof = prove_packed_openings::<AkitaScheme, PackedId, _>(
        vec![PackedProverObject {
            packing: &packed.packing,
            statement: &packed_statement,
            polynomial: &packed.polynomial,
            setup: &packed_prover_setup,
        }],
        vec![PackedProverGroup::singleton(0, Some(packed_hint))],
        &mut packed_prover_transcript,
    )
    .expect("packed proof should be produced");
    let mut packed_verifier_transcript = Blake2bTranscript::new(b"akita-coexist-packed");
    verify_packed_openings::<AkitaScheme, PackedId, _>(
        &[PackedVerifierObject {
            packing: &packed.packing,
            statement: &packed_statement,
            setup: &packed_verifier_setup,
        }],
        &[PackedObjectGroup::singleton(0)],
        &packed_proof,
        &mut packed_verifier_transcript,
    )
    .expect("packed proof should verify");

    assert_eq!(
        native_prover_transcript.state(),
        native_verifier_transcript.state()
    );
    assert_eq!(
        packed_prover_transcript.state(),
        packed_verifier_transcript.state()
    );
}

/// Three K=256 one-hot member polynomials committed as one group open through
/// a single native batch proof: the packed reduction binds their shared cell
/// domain, and `open_batch`/`verify_batch` carry all member claims at the one
/// bound point.
#[test]
fn akita_grouped_one_hot_members_open_in_one_batch() {
    const K: usize = 256;
    let member_ids = [PackedId::NarrowA, PackedId::NarrowB, PackedId::Medium];
    let members: Vec<OneHotPolynomial> = (0..3u64)
        .map(|member| {
            let indices = (0..32u64)
                .map(|row| {
                    if (row + member) % 7 == 0 {
                        None
                    } else {
                        Some(((row * 11 + member * 3) % K as u64) as u8)
                    }
                })
                .collect();
            OneHotPolynomial::new(K, indices)
        })
        .collect();
    let num_vars = members[0].num_vars();
    let (prover_setup, verifier_setup) = setup_for(num_vars, member_ids.len(), layout(9));
    let (commitment, hint) = AkitaScheme::commit_one_hot_group(&prover_setup, layout(9), &members)
        .expect("one-hot group should commit");
    assert_eq!(commitment.poly_count(), member_ids.len());

    let packings: Vec<PrefixPacking<PackedId>> = member_ids
        .iter()
        .map(|id| PrefixPacking::new([(*id, num_vars)]).expect("identity packing"))
        .collect();
    let points: Vec<Vec<AkitaField>> = (0..members.len())
        .map(|member| {
            (0..num_vars)
                .map(|var| f(7 + 3 * member as u64 + var as u64))
                .collect()
        })
        .collect();
    let statements: Vec<PackedStatement> = member_ids
        .iter()
        .zip(&members)
        .zip(&points)
        .map(|((id, member), point)| {
            PrefixPackedStatement::new(
                commitment.clone(),
                vec![(
                    *id,
                    EvaluationClaim::new(
                        point.clone(),
                        MultilinearPoly::<AkitaField>::evaluate(member, point),
                    ),
                )],
            )
        })
        .collect();
    let groups = [PackedObjectGroup {
        start: 0,
        len: members.len(),
    }];

    let mut prover_transcript = Blake2bTranscript::new(b"akita-grouped-one-hot");
    let objects: Vec<PackedProverObject<'_, AkitaScheme, PackedId>> = (0..members.len())
        .map(|member| PackedProverObject {
            packing: &packings[member],
            statement: &statements[member],
            polynomial: &members[member],
            setup: &prover_setup,
        })
        .collect();
    let prover_groups = vec![PackedProverGroup {
        start: 0,
        len: members.len(),
        hint: Some(hint),
    }];
    let proof = prove_packed_openings::<AkitaScheme, PackedId, _>(
        objects,
        prover_groups,
        &mut prover_transcript,
    )
    .expect("grouped one-hot packed opening should prove");
    assert_eq!(proof.evaluations.len(), members.len());
    assert_eq!(proof.openings.len(), 1);

    let verifier_objects: Vec<PackedVerifierObject<'_, AkitaScheme, PackedId>> = (0..members.len())
        .map(|member| PackedVerifierObject {
            packing: &packings[member],
            statement: &statements[member],
            setup: &verifier_setup,
        })
        .collect();
    let mut verifier_transcript = Blake2bTranscript::new(b"akita-grouped-one-hot");
    verify_packed_openings::<AkitaScheme, PackedId, _>(
        &verifier_objects,
        &groups,
        &proof,
        &mut verifier_transcript,
    )
    .expect("grouped one-hot packed opening should verify");
    assert_eq!(prover_transcript.state(), verifier_transcript.state());

    let mut tampered = proof.clone();
    tampered.evaluations[1] += f(1);
    let mut transcript = Blake2bTranscript::new(b"akita-grouped-one-hot");
    assert!(
        verify_packed_openings::<AkitaScheme, PackedId, _>(
            &verifier_objects,
            &groups,
            &tampered,
            &mut transcript,
        )
        .is_err(),
        "a corrupted member evaluation must reject"
    );

    let mut lying = statements[2].clone();
    lying.claims[0].1.value += f(1);
    let lying_objects: Vec<PackedVerifierObject<'_, AkitaScheme, PackedId>> = (0..members.len())
        .map(|member| PackedVerifierObject {
            packing: &packings[member],
            statement: if member == 2 {
                &lying
            } else {
                &statements[member]
            },
            setup: &verifier_setup,
        })
        .collect();
    let mut transcript = Blake2bTranscript::new(b"akita-grouped-one-hot");
    assert!(
        verify_packed_openings::<AkitaScheme, PackedId, _>(
            &lying_objects,
            &groups,
            &proof,
            &mut transcript,
        )
        .is_err(),
        "a lying member claim must break the joint reduction"
    );
}
