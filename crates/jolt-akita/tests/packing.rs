#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

mod support;

use jolt_akita::{AkitaBlackBoxBatching, AkitaCommitment, AkitaField, AkitaScheme};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, OpeningsError, PackedBatch,
    PackedWitness, PrefixPackedStatement,
};
use jolt_poly::Polynomial;
use jolt_transcript::{Blake2bTranscript, Transcript};
use support::{
    batch_witness, black_box_statement, f, layout, materialize_packed, packed_claims, packed_setup,
    polynomial, run_on_large_stack, setup_for, MaterializedPackedWitness,
};

type AkitaPackedBatch = PackedBatch<AkitaScheme, PackedId>;
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
        (PackedId::Wide, polynomial(3, 1)),
        (PackedId::Medium, polynomial(2, 40)),
        (PackedId::NarrowB, polynomial(1, 80)),
        (PackedId::NarrowA, polynomial(1, 120)),
        (PackedId::Constant, Polynomial::new(vec![f(200)])),
    ]
}

fn packed_point() -> Vec<AkitaField> {
    vec![f(3), f(5), f(7), f(11), f(13)]
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

fn assert_packed_verify_rejects(
    verifier_setup: &<AkitaPackedBatch as BatchOpeningScheme>::VerifierSetup,
    statement: PackedStatement,
    proof: &<AkitaPackedBatch as BatchOpeningScheme>::Proof,
    label: &'static [u8],
) {
    let mut transcript = Blake2bTranscript::new(label);
    assert!(<AkitaPackedBatch as BatchOpeningScheme>::verify_batch(
        verifier_setup,
        statement,
        proof,
        &mut transcript,
    )
    .is_err());
}

#[test]
fn akita_prefix_packed_batch_roundtrips_mixed_arities() {
    run_on_large_stack(|| {
        let polynomials = packed_polynomials();
        let packed = build_packed(&polynomials);
        assert_eq!(packed.packing.packed_num_vars, 5);
        assert_eq!((&packed.packing).into_iter().count(), 5);
        assert_eq!(packed.packing[&PackedId::Wide].num_vars, 3);
        assert_eq!(packed.packing[&PackedId::Medium].num_vars, 2);

        let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone(), layout(7));
        let (commitment, hint) = AkitaScheme::commit(&packed.polynomial, &prover_setup.pcs);
        let point = packed_point();
        let statement = statement_for(&packed, &polynomials, commitment, &point);

        let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-mixed");
        let proof = <AkitaPackedBatch as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement.clone(),
            PackedWitness::new(&packed.polynomial, hint),
            &mut prover_transcript,
        )
        .expect("Akita prefix-packed batch proof should be produced");

        let mut verifier_transcript = Blake2bTranscript::new(b"akita-packed-mixed");
        <AkitaPackedBatch as BatchOpeningScheme>::verify_batch(
            &verifier_setup,
            statement,
            &proof,
            &mut verifier_transcript,
        )
        .expect("Akita prefix-packed batch proof should verify");
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    });
}

#[test]
fn akita_prefix_packed_batch_rejects_statement_shape_errors() {
    run_on_large_stack(|| {
        let polynomials = packed_polynomials();
        let packed = build_packed(&polynomials);
        let (prover_setup, _) = packed_setup(packed.packing.clone(), layout(7));
        let (commitment, hint) = AkitaScheme::commit(&packed.polynomial, &prover_setup.pcs);
        let point = packed_point();
        let claims = packed_claims(&polynomials, &packed.packing, &point);

        let mut transcript = Blake2bTranscript::new(b"akita-packed-missing-slot");
        let result = <AkitaPackedBatch as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            PrefixPackedStatement::new(commitment.clone(), claims[1..].to_vec()),
            PackedWitness::new(&packed.polynomial, hint.clone()),
            &mut transcript,
        );
        assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));

        let mut unknown_id = claims.clone();
        unknown_id[0].id = PackedId::Unused;
        let mut transcript = Blake2bTranscript::new(b"akita-packed-unknown-id");
        let result = <AkitaPackedBatch as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            PrefixPackedStatement::new(commitment.clone(), unknown_id),
            PackedWitness::new(&packed.polynomial, hint.clone()),
            &mut transcript,
        );
        assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));

        let mut duplicate_id = claims.clone();
        duplicate_id[0].id = duplicate_id[1].id;
        let mut transcript = Blake2bTranscript::new(b"akita-packed-duplicate-id");
        let result = <AkitaPackedBatch as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            PrefixPackedStatement::new(commitment.clone(), duplicate_id),
            PackedWitness::new(&packed.polynomial, hint.clone()),
            &mut transcript,
        );
        assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));

        let mut wrong_arity = claims.clone();
        let constant = wrong_arity
            .iter_mut()
            .find(|claim| claim.id == PackedId::Constant)
            .expect("constant claim should exist");
        constant.evaluation = EvaluationClaim::new(vec![f(1)], constant.evaluation.value);
        let mut transcript = Blake2bTranscript::new(b"akita-packed-wrong-arity");
        let result = <AkitaPackedBatch as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            PrefixPackedStatement::new(commitment.clone(), wrong_arity),
            PackedWitness::new(&packed.polynomial, hint.clone()),
            &mut transcript,
        );
        assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));

        let mut suffix_incompatible = claims;
        let medium = suffix_incompatible
            .iter_mut()
            .find(|claim| claim.id == PackedId::Medium)
            .expect("medium claim should exist");
        let mut point = medium.evaluation.point.clone().into_vec();
        point[0] += f(1);
        medium.evaluation = EvaluationClaim::new(point, medium.evaluation.value);
        let mut transcript = Blake2bTranscript::new(b"akita-packed-suffix-incompat");
        let result = <AkitaPackedBatch as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            PrefixPackedStatement::new(commitment, suffix_incompatible),
            PackedWitness::new(&packed.polynomial, hint),
            &mut transcript,
        );
        assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
    });
}

#[test]
fn akita_prefix_packed_batch_rejects_tampered_verifier_inputs() {
    run_on_large_stack(|| {
        let polynomials = packed_polynomials();
        let packed = build_packed(&polynomials);
        let (prover_setup, verifier_setup) = packed_setup(packed.packing.clone(), layout(7));
        let (commitment, hint) = AkitaScheme::commit(&packed.polynomial, &prover_setup.pcs);
        let point = packed_point();
        let claims = packed_claims(&polynomials, &packed.packing, &point);
        let statement = PrefixPackedStatement::new(commitment.clone(), claims.clone());

        let mut prover_transcript = Blake2bTranscript::new(b"akita-packed-tamper");
        let proof = <AkitaPackedBatch as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement,
            PackedWitness::new(&packed.polynomial, hint),
            &mut prover_transcript,
        )
        .expect("Akita prefix-packed batch proof should be produced");

        let mut tampered_value = claims.clone();
        tampered_value[0].evaluation.value += f(1);
        assert_packed_verify_rejects(
            &verifier_setup,
            PrefixPackedStatement::new(commitment.clone(), tampered_value),
            &proof,
            b"akita-packed-tamper",
        );

        let wrong_point = vec![f(3), f(5), f(17), f(11), f(13)];
        let wrong_point_claims = packed_claims(&polynomials, &packed.packing, &wrong_point);
        assert_packed_verify_rejects(
            &verifier_setup,
            PrefixPackedStatement::new(commitment.clone(), wrong_point_claims),
            &proof,
            b"akita-packed-tamper",
        );

        let mut swapped_same_arity = claims.clone();
        for claim in &mut swapped_same_arity {
            claim.id = match claim.id {
                PackedId::NarrowA => PackedId::NarrowB,
                PackedId::NarrowB => PackedId::NarrowA,
                id => id,
            };
        }
        assert_packed_verify_rejects(
            &verifier_setup,
            PrefixPackedStatement::new(commitment.clone(), swapped_same_arity),
            &proof,
            b"akita-packed-tamper",
        );

        let other_packed = build_packed(&[
            (PackedId::Wide, polynomial(3, 500)),
            (PackedId::Medium, polynomial(2, 540)),
            (PackedId::NarrowB, polynomial(1, 580)),
            (PackedId::NarrowA, polynomial(1, 620)),
            (PackedId::Constant, Polynomial::new(vec![f(700)])),
        ]);
        let (other_commitment, _) =
            AkitaScheme::commit(&other_packed.polynomial, &prover_setup.pcs);
        assert_packed_verify_rejects(
            &verifier_setup,
            PrefixPackedStatement::new(other_commitment, claims),
            &proof,
            b"akita-packed-tamper",
        );
    });
}

#[test]
fn akita_prefix_packed_batch_rejects_wrong_witness_dimension() {
    run_on_large_stack(|| {
        let polynomials = packed_polynomials();
        let packed = build_packed(&polynomials);
        let (prover_setup, _) = packed_setup(packed.packing.clone(), layout(7));
        let (commitment, hint) = AkitaScheme::commit(&packed.polynomial, &prover_setup.pcs);
        let point = packed_point();
        let statement = statement_for(&packed, &polynomials, commitment, &point);
        let wrong_witness = polynomial(4, 900);

        let mut transcript = Blake2bTranscript::new(b"akita-packed-wrong-witness");
        let result = <AkitaPackedBatch as BatchOpeningScheme>::prove_batch(
            &prover_setup,
            statement,
            PackedWitness::new(&wrong_witness, hint),
            &mut transcript,
        );
        assert!(matches!(result, Err(OpeningsError::InvalidBatch(_))));
    });
}

#[test]
fn akita_packing_and_black_box_batching_can_coexist_for_same_logical_claims() {
    run_on_large_stack(|| {
        let poly_a = polynomial(4, 1);
        let poly_b = polynomial(4, 20);
        let logical_point = vec![f(2), f(3), f(5), f(7)];
        let eval_a = poly_a.evaluate(&logical_point);
        let eval_b = poly_b.evaluate(&logical_point);

        let (black_box_prover_setup, black_box_verifier_setup) = setup_for(4, 2, layout(7));
        let (black_box_commitment, black_box_hint) = AkitaScheme::commit_group(
            &black_box_prover_setup,
            layout(7),
            &[poly_a.clone(), poly_b.clone()],
        )
        .expect("grouped commit should succeed");
        let black_box_statement =
            black_box_statement(black_box_commitment, &logical_point, [eval_a, eval_b]);
        let mut black_box_prover_transcript = Blake2bTranscript::new(b"akita-coexist-black-box");
        let black_box_proof = <AkitaBlackBoxBatching as BatchOpeningScheme>::prove_batch(
            &black_box_prover_setup,
            black_box_statement.clone(),
            batch_witness([&poly_a, &poly_b], black_box_hint),
            &mut black_box_prover_transcript,
        )
        .expect("black-box proof should be produced");
        let mut black_box_verifier_transcript = Blake2bTranscript::new(b"akita-coexist-black-box");
        <AkitaBlackBoxBatching as BatchOpeningScheme>::verify_batch(
            &black_box_verifier_setup,
            black_box_statement,
            &black_box_proof,
            &mut black_box_verifier_transcript,
        )
        .expect("black-box proof should verify");

        let packed_polynomials = vec![(PackedId::NarrowA, poly_a), (PackedId::NarrowB, poly_b)];
        let packed = build_packed(&packed_polynomials);
        let (packed_prover_setup, packed_verifier_setup) =
            packed_setup(packed.packing.clone(), layout(7));
        let (packed_commitment, packed_hint) =
            AkitaScheme::commit(&packed.polynomial, &packed_prover_setup.pcs);
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
        let packed_proof = <AkitaPackedBatch as BatchOpeningScheme>::prove_batch(
            &packed_prover_setup,
            packed_statement.clone(),
            PackedWitness::new(&packed.polynomial, packed_hint),
            &mut packed_prover_transcript,
        )
        .expect("packed proof should be produced");
        let mut packed_verifier_transcript = Blake2bTranscript::new(b"akita-coexist-packed");
        <AkitaPackedBatch as BatchOpeningScheme>::verify_batch(
            &packed_verifier_setup,
            packed_statement,
            &packed_proof,
            &mut packed_verifier_transcript,
        )
        .expect("packed proof should verify");

        assert_eq!(
            black_box_prover_transcript.state(),
            black_box_verifier_transcript.state()
        );
        assert_eq!(
            packed_prover_transcript.state(),
            packed_verifier_transcript.state()
        );
    });
}
