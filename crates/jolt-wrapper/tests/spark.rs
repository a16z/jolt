#![expect(
    clippy::expect_used,
    reason = "test fixtures fail immediately on proof errors"
)]

use jolt_field::{Fr, Ring};
use jolt_r1cs::ConstraintMatrices;
use jolt_transcript::{Keccak256Transcript, Transcript};
use jolt_wrapper::spark::{
    final_claim, SparkChallenges, SparkProver, SparkTables, SparkWitness, DEGREE, TOTAL_COLUMNS,
};
use jolt_wrapper::stream::{prove_stage, verify_stage_with, StageMember, StageMemberSpec};

fn matrices() -> ConstraintMatrices<Fr> {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    for row in 0..8 {
        a.push(vec![
            (row, Fr::from_u64((row + 2) as u64)),
            ((row + 3) & 7, Fr::from_u64(5)),
        ]);
        b.push(vec![
            ((row * 3 + 1) & 7, Fr::from_u64(7)),
            ((row + 3) & 7, Fr::from_u64(11)),
        ]);
        c.push(vec![
            ((row * 5 + 2) & 7, Fr::from_u64(13)),
            (row, Fr::from_u64(17)),
        ]);
    }
    ConstraintMatrices::new(8, 8, a, b, c)
}

fn challenges(entry_vars: usize) -> SparkChallenges {
    SparkChallenges {
        alpha_row: Fr::from_u64(101),
        beta_row: Fr::from_u64(19),
        alpha_col: Fr::from_u64(211),
        beta_col: Fr::from_u64(23),
        matrix_weights: [Fr::from_u64(29), Fr::from_u64(31), Fr::from_u64(37)],
        relation_weights: [
            Fr::from_u64(41),
            Fr::from_u64(43),
            Fr::from_u64(47),
            Fr::from_u64(53),
            Fr::from_u64(59),
            Fr::from_u64(61),
        ],
        tau: (0..entry_vars)
            .map(|index| Fr::from_u64(index as u64 + 67))
            .collect(),
    }
}

#[test]
fn spark_matrix_logup_round_trip_and_tampers() {
    let tables = SparkTables::new(&matrices(), 0, 8).expect("build SPARK tables");
    let rx = [Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
    let ry = [Fr::from_u64(7), Fr::from_u64(11), Fr::from_u64(13)];
    let challenges = challenges(tables.entry_vars());
    let witness = SparkWitness::new(&tables, &rx, &ry, &challenges).expect("build SPARK witness");
    assert_eq!(
        tables.columns(&witness, 256).expect("embed").len(),
        TOTAL_COLUMNS
    );

    let mut prover =
        SparkProver::new(&tables, &witness, &rx, &ry, challenges.clone()).expect("SPARK prover");
    let input_claim = prover.input_claim();
    let mut prover_transcript = Keccak256Transcript::<Fr>::new(b"spark-test");
    let (proof, result) = {
        let mut members = [StageMember {
            input_claim,
            degree: DEGREE,
            offset: 0,
            prover: &mut prover,
        }];
        prove_stage(&mut members, &mut prover_transcript).expect("prove SPARK")
    };
    let evaluations = prover.final_evaluations();
    let shape = [StageMemberSpec {
        rounds: tables.entry_vars(),
        degree: DEGREE,
        offset: 0,
    }];
    let mut verifier_transcript = Keccak256Transcript::<Fr>::new(b"spark-test");
    let _verified = verify_stage_with(
        &proof,
        &shape,
        &[input_claim],
        &mut verifier_transcript,
        |stage| {
            Ok(vec![final_claim(
                &tables,
                &rx,
                &ry,
                &challenges,
                &stage.point,
                &evaluations,
            )
            .expect("final relation")])
        },
    )
    .expect("verify SPARK");
    assert_eq!(result.output_claims.len(), 1);

    let mut bad_inverse = evaluations.clone();
    bad_inverse.witness[2] += Fr::from_u64(1);
    let mut inverse_transcript = Keccak256Transcript::<Fr>::new(b"spark-test");
    assert!(verify_stage_with(
        &proof,
        &shape,
        &[input_claim],
        &mut inverse_transcript,
        |stage| {
            Ok(vec![final_claim(
                &tables,
                &rx,
                &ry,
                &challenges,
                &stage.point,
                &bad_inverse,
            )
            .expect("tampered inverse relation")])
        },
    )
    .is_err());

    let mut bad_table = evaluations;
    bad_table.fixed[0] += Fr::from_u64(1);
    let mut table_transcript = Keccak256Transcript::<Fr>::new(b"spark-test");
    assert!(verify_stage_with(
        &proof,
        &shape,
        &[input_claim],
        &mut table_transcript,
        |stage| {
            Ok(vec![final_claim(
                &tables,
                &rx,
                &ry,
                &challenges,
                &stage.point,
                &bad_table,
            )
            .expect("tampered table relation")])
        },
    )
    .is_err());
}
