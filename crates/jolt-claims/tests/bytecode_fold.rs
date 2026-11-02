use jolt_claims::protocols::jolt::geometry::bytecode::{
    read_raf_fold_stage_values, BytecodeReadRafStageValueInputs, BYTECODE_STAGE_GAMMA_COUNTS,
};
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use jolt_claims::protocols::jolt::geometry::dimensions::PointGeometryError;
use jolt_field::{Fr, One, Ring, Zero};
use jolt_riscv::{JoltInstructionKind as Kind, JoltInstructionRow, NormalizedOperands};

fn fold(
    bytecode: &[JoltInstructionRow],
    weights: &[Fr],
) -> Result<[Fr; NUM_BYTECODE_VAL_STAGES], PointGeometryError> {
    let mut gammas = BYTECODE_STAGE_GAMMA_COUNTS.map(|n| vec![Fr::zero(); n]);
    for stage in &mut gammas {
        stage[0] = Fr::one();
    }
    gammas[1].fill(Fr::one());
    gammas[3] = [1, 2, 4].map(Fr::from_u64).to_vec();
    let register_point = [Fr::zero(); 7];
    read_raf_fold_stage_values(
        BytecodeReadRafStageValueInputs {
            bytecode,
            register_read_write_point: &register_point,
            register_val_evaluation_point: &register_point,
            stage1_gammas: &gammas[0],
            stage2_gammas: &gammas[1],
            stage3_gammas: &gammas[2],
            stage4_gammas: &gammas[3],
            stage5_gammas: &gammas[4],
        },
        weights,
    )
}

#[test]
#[expect(clippy::unwrap_used)]
fn bytecode_fold_repeated_rows_fixed_vectors() {
    let add = JoltInstructionRow {
        instruction_kind: Kind::ADD,
        address: 4,
        operands: NormalizedOperands {
            rs1: Some(0),
            rs2: Some(0),
            rd: Some(0),
            imm: -7,
        },
        ..Default::default()
    };
    let store = JoltInstructionRow {
        instruction_kind: Kind::SD,
        address: 12,
        operands: NormalizedOperands {
            rd: None,
            imm: 5,
            ..add.operands
        },
        ..Default::default()
    };

    // At the Boolean register point x0, the base stage vectors are
    // ADD = [4, 1, -7, 1+2+4, 1], SD = [12, 0, 5, 2+4, 0].
    // Akita appends the store indicator: ADD = 0, SD = 1.
    assert_eq!(
        fold(&[], &[]).unwrap(),
        [Fr::zero(); NUM_BYTECODE_VAL_STAGES]
    );
    for len in [1, 4095, 4096, 8193, 32767, 32768, 32769, 65537] {
        let mut bytecode = vec![add; len];
        bytecode[len - 1] = store;
        let mut weights = vec![Fr::from_u64(3); len];
        weights[len - 1] = -Fr::from_u64(2);

        // Homogeneity: n identical ADD rows with weight 3 contribute
        // 3n times the fixed ADD vector; the final SD has weight -2.
        let m = 3 * (len as i128 - 1);
        #[cfg(not(feature = "akita"))]
        let expected = [4 * m - 24, m, -7 * m - 10, 7 * m - 12, m];
        #[cfg(feature = "akita")]
        let expected = [4 * m - 24, m, -7 * m - 10, 7 * m - 12, m, -2];
        assert_eq!(
            fold(&bytecode, &weights).unwrap(),
            expected.map(Fr::from_i128),
            "len={len}"
        );
    }
}

#[test]
fn bytecode_fold_rejects_mismatched_weight_lengths() {
    let bytecode = [JoltInstructionRow::default(); 2];
    for weights in [vec![Fr::one()], vec![Fr::one(); 3]] {
        assert_eq!(
            fold(&bytecode, &weights),
            Err(PointGeometryError::EvaluationDomainLengthMismatch {
                expected: bytecode.len(),
                got: weights.len(),
            })
        );
    }
}
