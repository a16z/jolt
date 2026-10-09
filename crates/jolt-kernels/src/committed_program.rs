//! Committed-program bytecode row/lane grids and padded image words.
//! Both commitment backends use one whole bytecode grid.
//! Each row uses the shared lane layout, interleaved by the trace order.

use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::{
    bytecode_total_vars, is_valid_committed_program_immediate, BYTECODE_LANE_LAYOUT,
    COMMITTED_BYTECODE_LANE_CAPACITY, INVALID_COMMITTED_PROGRAM_IMMEDIATE,
};
use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_field::JoltField;
use jolt_lookup_tables::{InstructionLookupTable, XLEN};
use jolt_riscv::instructions::Noop;
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
];

fn for_each_active_lane_value<F: JoltField>(
    instruction: &JoltInstructionRow,
    mut visit: impl FnMut(usize, F),
) {
    let decoded = JoltInstruction::try_from(*instruction)
        .unwrap_or(JoltInstruction::Noop(Noop(*instruction)));
    let circuit_flags = decoded.circuit_flags();
    let instruction_flags = decoded.instruction_flags();
    let layout = BYTECODE_LANE_LAYOUT;

    if let Some(register) = instruction.operands.rs1 {
        visit(layout.rs1_start + register as usize, F::one());
    }
    if let Some(register) = instruction.operands.rs2 {
        visit(layout.rs2_start + register as usize, F::one());
    }
    if let Some(register) = instruction.operands.rd {
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
    if instructions
        .iter()
        .any(|instruction| !is_valid_committed_program_immediate(instruction.operands.imm))
    {
        return Err(KernelError::InvalidGeometry {
            reason: INVALID_COMMITTED_PROGRAM_IMMEDIATE.to_owned(),
        });
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
