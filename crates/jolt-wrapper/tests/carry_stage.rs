#![expect(
    clippy::expect_used,
    reason = "test fixtures fail immediately on proof errors"
)]

use jolt_field::{Fr, Ring};
use jolt_poly::MultilinearPoly;
use jolt_transcript::{Keccak256Transcript, Transcript};
use jolt_wrapper::carry::{carried_final, CarryProver};
use jolt_wrapper::stream::{prove_stage, verify_stage_with, StageMember, StageMemberSpec};

#[test]
fn carries_two_claims_into_one_stage_point() {
    let evaluations_a: Vec<Fr> = (0..256)
        .map(|index| Fr::from_u64((index * 17 + 3) as u64))
        .collect();
    let evaluations_b: Vec<Fr> = (0..256)
        .map(|index| Fr::from_u64((index * index + 11) as u64))
        .collect();
    let source_a: Vec<Fr> = (0..8).map(|index| Fr::from_u64(index as u64 + 2)).collect();
    let source_b: Vec<Fr> = (0..8)
        .map(|index| Fr::from_u64(index as u64 + 19))
        .collect();
    let claim_a = evaluations_a.as_slice().evaluate(&source_a);
    let claim_b = evaluations_b.as_slice().evaluate(&source_b);
    let mut carry_a = CarryProver::new(&evaluations_a, &source_a, claim_a).expect("carry A");
    let mut carry_b = CarryProver::new(&evaluations_b, &source_b, claim_b).expect("carry B");
    let mut prover_transcript = Keccak256Transcript::<Fr>::new(b"carry-stage-test");
    let (proof, result) = {
        let mut members = [
            StageMember {
                prover: &mut carry_a,
                input_claim: claim_a,
                degree: 2,
                offset: 0,
            },
            StageMember {
                prover: &mut carry_b,
                input_claim: claim_b,
                degree: 2,
                offset: 0,
            },
        ];
        prove_stage(&mut members, &mut prover_transcript).expect("prove carries")
    };
    let final_a = carry_a.final_evaluation();
    let final_b = carry_b.final_evaluation();
    assert_eq!(final_a, evaluations_a.as_slice().evaluate(&result.point));
    assert_eq!(final_b, evaluations_b.as_slice().evaluate(&result.point));

    let mut verifier_transcript = Keccak256Transcript::<Fr>::new(b"carry-stage-test");
    let shape = [
        StageMemberSpec {
            rounds: 8,
            degree: 2,
            offset: 0,
        },
        StageMemberSpec {
            rounds: 8,
            degree: 2,
            offset: 0,
        },
    ];
    let verified = verify_stage_with(
        &proof,
        &shape,
        &[claim_a, claim_b],
        &mut verifier_transcript,
        |stage| {
            Ok(vec![
                carried_final(&source_a, &stage.point, final_a).expect("carry A final relation"),
                carried_final(&source_b, &stage.point, final_b).expect("carry B final relation"),
            ])
        },
    )
    .expect("verify carries");
    assert_eq!(verified.point, result.point);

    let mut tampered_transcript = Keccak256Transcript::<Fr>::new(b"carry-stage-test");
    assert!(verify_stage_with(
        &proof,
        &shape,
        &[claim_a, claim_b],
        &mut tampered_transcript,
        |stage| {
            Ok(vec![
                carried_final(&source_a, &stage.point, final_a + Fr::from_u64(1))
                    .expect("carry A final relation"),
                carried_final(&source_b, &stage.point, final_b).expect("carry B final relation"),
            ])
        },
    )
    .is_err());
}
