#![expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test fixtures fail immediately on proof errors"
)]

use jolt_crypto::Bn254;
use jolt_field::{Fr, Ring};
use jolt_hyperkzg::{HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_poly::{CompressedPoly, MultilinearPoly};
use jolt_wrapper::carry::{carried_final, CarryProver};
use jolt_wrapper::stream::{
    AssemblyMemberStatement, AssemblyStatement, Column, Commitment, StageMember, StageMemberSpec,
};
use jolt_wrapper::wrap::{verify_wrapped, wrap};

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
        rows * 2,
        Bn254::g1_generator(),
        Bn254::g2_generator(),
    );
    let verifier_setup = HyperKZGVerifierSetup::from(&setup);
    let packed_columns = columns.iter().cloned().map(Column::Fr).collect::<Vec<_>>();
    let mut carries = [
        CarryProver::new(&columns[0], &source_points[0], input_claims[0]).expect("carry 0"),
        CarryProver::new(&columns[1], &source_points[1], input_claims[1]).expect("carry 1"),
    ];
    let statement = AssemblyStatement {
        key_digest: [83; 32],
        public_inputs: vec![Fr::from_u64(89)],
        rows,
        column_count: 2,
        k: 2,
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
        factor_columns: vec![0, 1],
    };
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
        wrap(
            &packed_columns,
            &statement,
            &mut members,
            &setup,
            |stage, claims| {
                Ok(vec![
                    carried_final(&source_points[0], &stage.point, claims[0])
                        .expect("carry 0 final"),
                    carried_final(&source_points[1], &stage.point, claims[1])
                        .expect("carry 1 final"),
                ])
            },
        )
        .expect("prove assembly")
    };
    let verify = |proof| {
        verify_wrapped(&statement, proof, &verifier_setup, |stage, claims, _| {
            Ok(vec![
                carried_final(&source_points[0], &stage.point, claims[0]).expect("carry 0 final"),
                carried_final(&source_points[1], &stage.point, claims[1]).expect("carry 1 final"),
            ])
        })
    };
    let (results, cost) = verify(&proof).expect("verify assembly");
    assert_eq!(results.len(), 2);
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

    let mut factor_claim = proof.clone();
    factor_claim.stage_claims[0][0] += Fr::from_u64(1);
    assert!(verify(&factor_claim).is_err());

    let mut stage_b = proof.clone();
    let round = &mut stage_b.stages[1].round_polynomials.round_polynomials[0];
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
