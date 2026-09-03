#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test fixtures fail immediately on proof errors"
)]

use jolt_crypto::Bn254;
use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::{HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_poly::{CompressedPoly, EqPolynomial, MultilinearPoly};
use jolt_wrapper::carry::CarryProver;
use jolt_wrapper::stream::{
    combine_packed_phases, commit_packed, commitment_prefix_challenges, prove_assembly,
    verify_assembly_with_cost, AffineForm, AssemblyMemberStatement, AssemblyStatement, Column,
    ColumnId, Commitment, CommitmentPhase, StageMember, StageMemberSpec, Term, TermContext,
    TermExporter,
};

struct CarryTerms {
    member: usize,
    column: ColumnId,
    source_point: Vec<Fr>,
}

impl TermExporter for CarryTerms {
    fn terms(&self, context: &TermContext<'_>) -> Vec<Term> {
        vec![Term {
            coefficient: context.batching_coefficients[self.member]
                * EqPolynomial::<Fr>::mle(&self.source_point, context.row_point),
            factors: vec![AffineForm {
                constant: Fr::zero(),
                weights: vec![(self.column, Fr::one())],
            }],
        }]
    }
}

#[test]
fn generic_assembly_round_trip_and_section_tampers() {
    let rows = 256;
    let columns = [
        (0..rows)
            .map(|index| Fr::from_u64((index * 7 + 3) as u64))
            .collect::<Vec<_>>(),
        (0..rows)
            .map(|index| Fr::from_u64((index * 11 + 5) as u64))
            .collect::<Vec<_>>(),
    ];
    let source_points = [
        (0..8)
            .map(|index| Fr::from_u64(index as u64 + 13))
            .collect::<Vec<_>>(),
        (0..8)
            .map(|index| Fr::from_u64(index as u64 + 29))
            .collect::<Vec<_>>(),
    ];
    let input_claims = [
        columns[0].as_slice().evaluate(&source_points[0]),
        columns[1].as_slice().evaluate(&source_points[1]),
    ];
    let setup = HyperKZGScheme::<Bn254>::setup_from_secret(
        Fr::from_u64(79),
        rows,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    let wire_columns = [Column::Fr(columns[0].clone())];
    let wire_packed = commit_packed(&wire_columns, 1, &setup).expect("wire commitment");
    let phase_challenges = commitment_prefix_challenges(
        &[83; 32],
        &[Fr::from_u64(89)],
        &[(&wire_packed.commitments, 2)],
    );
    assert_eq!(phase_challenges.len(), 2);
    let helper_columns = [Column::Fr(columns[1].clone())];
    let helper_packed = commit_packed(&helper_columns, 1, &setup).expect("helper commitment");
    let packed = combine_packed_phases(vec![wire_packed, helper_packed]).expect("combine phases");
    let mut carries = [
        CarryProver::new(&columns[0], &source_points[0], input_claims[0]).expect("carry 0"),
        CarryProver::new(&columns[1], &source_points[1], input_claims[1]).expect("carry 1"),
    ];
    let statement = AssemblyStatement {
        key_digest: [83; 32],
        public_inputs: vec![Fr::from_u64(89)],
        rows,
        column_count: 2,
        k: 1,
        members: vec![
            AssemblyMemberStatement {
                input_claim: input_claims[0],
                spec: StageMemberSpec {
                    rounds: 8,
                    degree: 5,
                    offset: 0,
                },
            },
            AssemblyMemberStatement {
                input_claim: input_claims[1],
                spec: StageMemberSpec {
                    rounds: 8,
                    degree: 2,
                    offset: 0,
                },
            },
        ],
        commitment_phases: vec![
            CommitmentPhase {
                group_count: 1,
                challenge_count: 2,
            },
            CommitmentPhase {
                group_count: 1,
                challenge_count: 0,
            },
        ],
    };
    let term_exporters = [
        CarryTerms {
            member: 0,
            column: ColumnId { group: 0, slot: 0 },
            source_point: source_points[0].clone(),
        },
        CarryTerms {
            member: 1,
            column: ColumnId { group: 1, slot: 0 },
            source_point: source_points[1].clone(),
        },
    ];
    let exporters = [
        &term_exporters[0] as &dyn TermExporter,
        &term_exporters[1] as &dyn TermExporter,
    ];
    let proof = {
        let [carry_0, carry_1] = &mut carries;
        let mut members = [
            StageMember {
                prover: carry_0,
                input_claim: input_claims[0],
                degree: 5,
                offset: 0,
            },
            StageMember {
                prover: carry_1,
                input_claim: input_claims[1],
                degree: 2,
                offset: 0,
            },
        ];
        prove_assembly(&packed, &statement, &mut members, &exporters, &setup)
            .expect("prove assembly")
    };
    let verify = |proof| verify_assembly_with_cost(proof, &statement, &exporters, &verifier_setup);
    let (results, cost) = verify(&proof).expect("verify assembly");
    assert_eq!(results.len(), 3);
    assert!(cost.pairing_pairs > 0);

    let mut commitment = proof.clone();
    commitment.commitments[0] = Commitment::new(proof.opening.com[0]);
    assert!(verify(&commitment).is_err());

    let mut stage_a = proof.clone();
    stage_a.stages[0]
        .committed_rounds
        .as_mut()
        .expect("KZG stage")
        .sum_at_zero += Fr::from_u64(1);
    assert!(verify(&stage_a).is_err());

    let mut phase_2 = proof.clone();
    phase_2.commitments[1] = Commitment::new(proof.opening.com[0]);
    assert!(verify(&phase_2).is_err());

    let mut term_evaluation = proof.clone();
    term_evaluation.term_evaluations[0] += Fr::from_u64(1);
    assert!(verify(&term_evaluation).is_err());

    let mut term_commitment = proof.clone();
    term_commitment.stages[1]
        .committed_rounds
        .as_mut()
        .expect("term KZG stage")
        .round_commitments[0] = Bn254::g1_generator();
    assert!(verify(&term_commitment).is_err());

    let mut term_sum = proof.clone();
    term_sum.stages[1]
        .committed_rounds
        .as_mut()
        .expect("term KZG stage")
        .sum_at_zero += Fr::from_u64(1);
    assert!(verify(&term_sum).is_err());

    let mut stage_b = proof.clone();
    let round = &mut stage_b.stages[2].round_polynomials.round_polynomials[0];
    let mut coefficients = round.coeffs_except_linear_term().to_vec();
    coefficients[0] += Fr::from_u64(1);
    *round = CompressedPoly::new(coefficients);
    assert!(verify(&stage_b).is_err());

    let mut reduced = proof.clone();
    reduced.reduced_claims[0] += Fr::from_u64(1);
    assert!(verify(&reduced).is_err());

    let mut opening = proof.clone();
    opening.opening.v[0][0] += Fr::from_u64(1);
    assert!(verify(&opening).is_err());
}
