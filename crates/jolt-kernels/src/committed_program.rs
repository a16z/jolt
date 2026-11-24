//! Committed-program bytecode row/lane grids and padded image words.
//! Both commitment backends use one whole bytecode grid.
//! Each row uses the shared lane layout, interleaved by the trace order.

use jolt_claims::protocols::composed::geometry::committed_bytecode::is_valid_committed_program_registers;
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::{
    bytecode_total_vars, is_valid_committed_program_immediate, BYTECODE_LANE_LAYOUT,
    COMMITTED_BYTECODE_LANE_CAPACITY, INVALID_COMMITTED_PROGRAM_IMMEDIATE,
};
use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_field::JoltField;
use jolt_lookup_tables::{InstructionLookupTable, XLEN};
use jolt_riscv::{
    Flags, InstructionFlags, InterleavedBitsMarker, JoltInstruction, JoltInstructionRow,
    CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
use jolt_utils::unsafe_allocate_zero_vec;

use crate::KernelError;

/// `InstructionFlags` in discriminant order (the lane-block order
/// `jolt-claims`' `lane_weights` addresses by `flag as usize`).
const INSTRUCTION_FLAG_ORDER: [InstructionFlags; NUM_INSTRUCTION_FLAGS] = [
    InstructionFlags::LeftOperandIsPC,
    InstructionFlags::RightOperandIsImm,
    InstructionFlags::LeftOperandIsRs1Value,
    InstructionFlags::RightOperandIsRs2Value,
    InstructionFlags::Branch,
    InstructionFlags::IsNoop,
    InstructionFlags::RightOperandIsRdWriteValue,
];

fn for_each_active_lane_value<F: JoltField>(
    instruction: &JoltInstructionRow,
    mut visit: impl FnMut(usize, F),
) {
    let decoded = JoltInstruction::from(*instruction);
    let circuit_flags = decoded.circuit_flags();
    let instruction_flags = decoded.instruction_flags();
    let layout = BYTECODE_LANE_LAYOUT;

    let operands = instruction.integer_operands();
    #[cfg(all(feature = "akita", feature = "field-inline"))]
    {
        let field = instruction.field_operands();
        for (start, register) in layout
            .field_starts()
            .into_iter()
            .zip([field.rd, field.rs1, field.rs2])
        {
            if let Some(register) = register {
                visit(start + usize::from(register), F::one());
            }
        }
    }
    if let Some(register) = operands.rs1 {
        visit(layout.rs1_start + register as usize, F::one());
    }
    if let Some(register) = operands.rs2 {
        visit(layout.rs2_start + register as usize, F::one());
    }
    if let Some(register) = operands.rd {
        visit(layout.rd_start + register as usize, F::one());
    }
    let unexpanded_pc = F::from_u64(instruction.address as u64);
    if !unexpanded_pc.is_zero() {
        visit(layout.unexp_pc_idx, unexpanded_pc);
    }
    let imm = F::from_i128(instruction.operands.imm);
    if !imm.is_zero() {
        visit(layout.imm_idx, imm);
    }
    for (index, flag) in CIRCUIT_FLAGS.into_iter().enumerate() {
        if circuit_flags[flag] {
            visit(layout.circuit_start + index, F::one());
        }
    }
    for (index, flag) in INSTRUCTION_FLAG_ORDER.into_iter().enumerate() {
        if instruction_flags[flag] {
            visit(layout.instr_start + index, F::one());
        }
    }
    if let Some(table) = InstructionLookupTable::<XLEN>::lookup_table(&decoded) {
        visit(layout.lookup_start + table.index(), F::one());
    }
    if !circuit_flags.is_interleaved_operands() {
        visit(layout.raf_flag_idx, F::one());
    }
}

/// Materialize one complete bytecode row/lane grid, independent of trace length.
#[tracing::instrument(skip_all, name = "build_committed_bytecode_coeffs")]
pub fn build_committed_bytecode_coeffs<F: JoltField>(
    instructions: &[JoltInstructionRow],
    order: TracePolynomialOrder,
) -> Result<Vec<F>, KernelError<F>> {
    let num_vars =
        bytecode_total_vars(instructions.len()).map_err(|error| KernelError::InvalidGeometry {
            reason: error.to_string(),
        })?;
    for instruction in instructions {
        if !is_valid_committed_program_immediate(instruction.operands.imm) {
            return Err(KernelError::InvalidGeometry {
                reason: INVALID_COMMITTED_PROGRAM_IMMEDIATE.to_owned(),
            });
        }
        if !is_valid_committed_program_registers(instruction) {
            return Err(KernelError::InvalidGeometry {
                reason: "committed bytecode register exceeds its selector block".to_owned(),
            });
        }
    }
    let len = 1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| KernelError::InvalidGeometry {
            reason: "bytecode coefficient length overflows usize".to_owned(),
        })?;
    let mut coeffs = unsafe_allocate_zero_vec(len);
    for (row, instruction) in instructions.iter().enumerate() {
        for_each_active_lane_value::<F>(instruction, |lane, value| {
            coeffs[order.address_cycle_to_index(
                lane,
                row,
                COMMITTED_BYTECODE_LANE_CAPACITY,
                instructions.len(),
            )] += value;
        });
    }
    Ok(coeffs)
}

/// The `(lane, row)` coordinates addressed by the reduction template.
pub fn bytecode_index_to_lane_row(
    index: usize,
    row_count: usize,
    order: TracePolynomialOrder,
) -> (usize, usize) {
    order.index_to_address_cycle(index, COMMITTED_BYTECODE_LANE_CAPACITY, row_count)
}

/// The committed program-image polynomial's word vector: the RAM-remapped
/// bytecode words zero-padded to a power of two (at least 2).
pub fn program_image_words_padded(bytecode_words: &[u64]) -> Vec<u64> {
    let padded_len = bytecode_words.len().next_power_of_two().max(2);
    let mut words = bytecode_words.to_vec();
    words.resize(padded_len, 0);
    words
}

#[cfg(test)]
#[test]
fn committed_grid_rejects_registers_outside_their_blocks() {
    use jolt_field::Fr;
    use jolt_riscv::{JoltInstructionKind as Kind, NormalizedOperands};

    let assert_rejected = |instruction_kind, register| {
        let row = JoltInstructionRow {
            instruction_kind,
            operands: NormalizedOperands {
                rd: Some(register),
                rs1: Some(1),
                rs2: Some(2),
                imm: 0,
            },
            ..JoltInstructionRow::default()
        };
        assert!(matches!(
            build_committed_bytecode_coeffs::<Fr>(&[row], TracePolynomialOrder::CycleMajor),
            Err(KernelError::InvalidGeometry { .. })
        ));
    };
    assert_rejected(Kind::ADD, 128);
    #[cfg(all(feature = "akita", feature = "field-inline"))]
    assert_rejected(Kind::FIELD_MUL, 16);
}

#[cfg(all(test, feature = "akita", feature = "field-inline"))]
#[expect(
    clippy::expect_used,
    reason = "valid field selector fixtures must encode"
)]
mod tests {
    use super::*;
    use common::constants::REGISTER_COUNT;
    use jolt_akita::AkitaField as F;
    use jolt_claims::protocols::composed::geometry::committed_bytecode::FieldRegisterLaneInputs;
    use jolt_claims::protocols::field_inline::geometry::bytecode::{
        read_raf_folded_stage_values, selector_gammas, FieldInlineBytecodeReadRafStageValueInputs,
    };
    use jolt_claims::protocols::jolt::geometry::{
        bytecode::BYTECODE_STAGE_GAMMA_COUNTS,
        claim_reductions::bytecode::BytecodeLaneWeightInputs, dimensions::REGISTER_ADDRESS_BITS,
    };
    use jolt_field::{One, Ring, Zero};
    use jolt_poly::EqPolynomial;
    use jolt_riscv::{JoltInstructionKind as Kind, NormalizedOperands, FIELD_REGISTER_COUNT};

    #[test]
    fn committed_selector_blocks_match_instruction_roles() {
        let (f0, f1, f15, x31) = (Some(0), Some(1), Some(15), Some(31));
        let absent = [None; 3];
        // Entries use the encoded Rd, Rs1, Rs2 slots, followed by each register file's selectors.
        let rows = [
            (Kind::FIELD_ADD, [f0, f0, f15], absent, [f0, f0, f15]),
            (Kind::FIELD_SUB, [f15, f0, f1], absent, [f15, f0, f1]),
            (Kind::FIELD_MUL, [f15, f15, f1], absent, [f15, f15, f1]),
            (Kind::FIELD_INV, [f0, f15, None], absent, [f0, f15, None]),
            (
                Kind::FIELD_ASSERT_EQ,
                [None, f1, f0],
                absent,
                [None, f1, f0],
            ),
            (
                Kind::FIELD_ASSERT_ZERO,
                [None, f15, None],
                absent,
                [None, f15, None],
            ),
            (
                Kind::FIELD_LOAD_IMM,
                [f15, None, None],
                absent,
                [f15, None, None],
            ),
            (
                Kind::FIELD_LOAD_ACCUMULATE_FROM_REGISTER,
                [f0, x31, None],
                [None, x31, None],
                [f0, f0, None],
            ),
            (
                Kind::FIELD_LOAD_ACCUMULATE_FROM_MEMORY,
                [x31, f1, f15],
                [x31, f1, None],
                [f15, f15, None],
            ),
            (
                Kind::FIELD_ADVICE_LIMB,
                [x31, f1, f15],
                [x31, None, None],
                [f15, f1, None],
            ),
            (Kind::ADD, [x31, f1, f0], [x31, f1, f0], absent),
            (Kind::NoOp, absent, absent, absent),
        ];
        let order = TracePolynomialOrder::CycleMajor;
        let layout = BYTECODE_LANE_LAYOUT;
        for (kind, [rd, rs1, rs2], integer, field) in rows {
            let row = JoltInstructionRow {
                instruction_kind: kind,
                operands: NormalizedOperands {
                    rd,
                    rs1,
                    rs2,
                    imm: 0,
                },
                ..JoltInstructionRow::default()
            };
            let grid = build_committed_bytecode_coeffs::<F>(&[row], order).expect("one-row grid");
            for (starts, registers, width) in [
                (
                    [layout.rd_start, layout.rs1_start, layout.rs2_start],
                    integer,
                    usize::from(REGISTER_COUNT),
                ),
                (
                    layout.field_starts(),
                    field,
                    usize::from(FIELD_REGISTER_COUNT),
                ),
            ] {
                for (start, register) in starts.into_iter().zip(registers) {
                    for k in 0..width {
                        let index = order.address_cycle_to_index(
                            start + k,
                            0,
                            COMMITTED_BYTECODE_LANE_CAPACITY,
                            1,
                        );
                        assert_eq!(
                            grid[index],
                            F::from_u64(u64::from(register.map(usize::from) == Some(k))),
                            "{kind:?} lane {}",
                            start + k
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn committed_field_lane_weights_match_full_row_fold() {
        let rows = [
            JoltInstructionRow {
                instruction_kind: Kind::FIELD_LOAD_ACCUMULATE_FROM_MEMORY,
                operands: NormalizedOperands {
                    rd: Some(23),
                    rs1: Some(5),
                    rs2: Some(15),
                    imm: 8,
                },
                ..JoltInstructionRow::default()
            },
            JoltInstructionRow {
                instruction_kind: Kind::FIELD_ASSERT_EQ,
                operands: NormalizedOperands {
                    rd: None,
                    rs1: Some(0),
                    rs2: Some(15),
                    imm: 0,
                },
                ..JoltInstructionRow::default()
            },
        ];
        let rw = [2, 3, 5, 7].map(F::from_u64);
        let val = [11, 13, 17, 19].map(F::from_u64);
        let address_eq = EqPolynomial::<F>::evals(&[F::from_u64(23)], None);
        let g4 = std::array::from_fn(|i| F::from_u64(1 + i as u64));
        let g5 = std::array::from_fn(|i| F::from_u64(37 + i as u64));
        let field = read_raf_folded_stage_values(
            FieldInlineBytecodeReadRafStageValueInputs {
                bytecode: &rows,
                field_register_read_write_point: &rw,
                field_register_val_evaluation_point: &val,
                stage4_gammas: &g4,
                stage5_gammas: &g5,
            },
            &address_eq,
        );
        let ordinary = BYTECODE_STAGE_GAMMA_COUNTS.map(|n| vec![F::zero(); n]);
        let order = TracePolynomialOrder::CycleMajor;
        let grid = build_committed_bytecode_coeffs::<F>(&rows, order).expect("valid field rows");
        for eta in [F::from_u64(1), F::from_u64(29)] {
            let weights = FieldRegisterLaneInputs {
                read_write_address: &rw,
                value_address: &val,
                selector_gammas: selector_gammas(&g4, &g5),
            }
            .lane_weights(BytecodeLaneWeightInputs {
                eta,
                stage1_gammas: &ordinary[0],
                stage2_gammas: &ordinary[1],
                stage3_gammas: &ordinary[2],
                stage4_gammas: &ordinary[3],
                stage5_gammas: &ordinary[4],
                register_read_write_point: &[F::zero(); REGISTER_ADDRESS_BITS],
                register_val_evaluation_point: &[F::zero(); REGISTER_ADDRESS_BITS],
            })
            .expect("valid stage points");
            let mut actual = F::zero();
            for (row, eq) in address_eq.iter().enumerate() {
                for (lane, weight) in weights.iter().enumerate() {
                    let index = order.address_cycle_to_index(lane, row, weights.len(), rows.len());
                    actual += grid[index] * *weight * *eq;
                }
            }
            let eta6 = (0..6).fold(F::one(), |power, _| power * eta);
            assert_eq!(actual, eta6 * (field[3] + eta * field[4]));
        }
    }
}
