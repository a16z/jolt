use jolt_claims::protocols::jolt::geometry::bytecode::{
    read_raf_fold_stage_values, read_raf_stage_values, BytecodeReadRafStageValueInputs,
    BYTECODE_STAGE_GAMMA_COUNTS,
};
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use jolt_field::JoltField;
use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands};

#[expect(clippy::unwrap_used)]
fn check_fold<F: JoltField>() {
    let gammas = BYTECODE_STAGE_GAMMA_COUNTS.map(|n| {
        (0..n)
            .map(|i| F::from_u64(17 + i as u64))
            .collect::<Vec<_>>()
    });
    let register_point = vec![F::from_u64(11); 7];
    for len in [0, 1, 4095, 4096, 8193, 32767, 32768, 32769, 65537] {
        let bytecode = (0..len)
            .map(|i| JoltInstructionRow {
                instruction_kind: match i % 4 {
                    0 => JoltInstructionKind::ADD,
                    1 => JoltInstructionKind::SD,
                    2 => JoltInstructionKind::ADDI,
                    _ => JoltInstructionRow::default().instruction_kind,
                },
                address: 4 * i,
                operands: NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: Some(3),
                    imm: i as i128 - 4096,
                },
                ..Default::default()
            })
            .collect::<Vec<_>>();
        let weights = (0..len)
            .map(|i| -F::from_u64(i as u64 + 3))
            .collect::<Vec<_>>();
        let inputs = || BytecodeReadRafStageValueInputs {
            bytecode: &bytecode,
            register_read_write_point: &register_point,
            register_val_evaluation_point: &register_point,
            stage1_gammas: &gammas[0],
            stage2_gammas: &gammas[1],
            stage3_gammas: &gammas[2],
            stage4_gammas: &gammas[3],
            stage5_gammas: &gammas[4],
        };
        let mut expected = [F::zero(); NUM_BYTECODE_VAL_STAGES];
        for (row, weight) in read_raf_stage_values(inputs()).into_iter().zip(&weights) {
            for (sum, value) in expected.iter_mut().zip(row) {
                *sum += value * *weight;
            }
        }
        assert_eq!(
            read_raf_fold_stage_values(inputs(), &weights).unwrap(),
            expected,
            "fused len={len}"
        );
        assert!(read_raf_fold_stage_values(inputs(), &vec![F::one(); len + 1]).is_err());
    }
}

#[test]
#[expect(clippy::unwrap_used)]
fn bytecode_fold_matches_materialized_rows_one_worker() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(check_fold::<jolt_field::Fr>);
}

#[test]
#[expect(clippy::unwrap_used)]
fn bytecode_fold_matches_materialized_rows_four_workers() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(check_fold::<jolt_field::Fr>);
}
