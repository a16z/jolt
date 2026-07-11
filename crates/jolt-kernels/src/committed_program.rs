//! Committed-program polynomial materialization: the per-chunk bytecode
//! coefficient grids and the padded program-image word vector, built from the
//! prover-retained full program.
//!
//! Each bytecode chunk is a `(lanes × chunk_cycles)` matrix in cycle-major
//! order (`index = lane · chunk_cycle_len + chunk_cycle`): one column per
//! bytecode row in the chunk, one lane per committed row attribute (the
//! one-hot `rs1`/`rs2`/`rd` blocks, the scalar unexpanded-PC and immediate
//! lanes, the circuit/instruction flag blocks, the lookup-table selector
//! block, and the RAF flag — [`BYTECODE_LANE_LAYOUT`]). The same grids back
//! the preprocessing-time chunk commitments, the stage-6b bytecode claim
//! reduction, and the stage-8 joint opening, so they are built here once and
//! shared.

use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::{
    is_valid_committed_bytecode_chunking_for_len, total_lanes, BYTECODE_LANE_LAYOUT,
    COMMITTED_BYTECODE_LANE_CAPACITY,
};
use jolt_field::Field;
use jolt_lookup_tables::{InstructionLookupTable, XLEN};
use jolt_riscv::instructions::Noop;
use jolt_riscv::{
    Flags, InstructionFlags, InterleavedBitsMarker, JoltInstruction, JoltInstructionRow,
    CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};

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

/// The sparse `(lane, value)` encoding of one committed bytecode row.
fn for_each_active_lane_value<F: Field>(
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

/// Build the per-chunk committed bytecode coefficient grids, cycle-major.
pub fn build_committed_bytecode_chunk_coeffs<F: Field>(
    instructions: &[JoltInstructionRow],
    chunk_count: usize,
) -> Result<Vec<Vec<F>>, KernelError<F>> {
    let bytecode_len = instructions.len();
    if !is_valid_committed_bytecode_chunking_for_len(bytecode_len, chunk_count) {
        return Err(KernelError::InvalidGeometry {
            reason: format!(
                "invalid committed bytecode chunking: {chunk_count} chunks over {bytecode_len} rows"
            ),
        });
    }
    let chunk_cycle_len = bytecode_len / chunk_count;
    let lane_capacity = COMMITTED_BYTECODE_LANE_CAPACITY;
    let mut chunk_coeffs: Vec<Vec<F>> = (0..chunk_count)
        .map(|_| vec![F::zero(); lane_capacity * chunk_cycle_len])
        .collect();

    for (cycle, instruction) in instructions.iter().enumerate() {
        let coeffs = &mut chunk_coeffs[cycle / chunk_cycle_len];
        let chunk_cycle = cycle % chunk_cycle_len;
        for_each_active_lane_value::<F>(instruction, |lane, value| {
            coeffs[lane * chunk_cycle_len + chunk_cycle] += value;
        });
    }
    Ok(chunk_coeffs)
}

/// The `(lane, cycle)` coordinates of a cycle-major chunk-grid index — the
/// pairing the reduction's lane-weight/eq template walks.
pub fn chunk_index_to_lane_cycle(index: usize, chunk_cycle_len: usize) -> (usize, usize) {
    (index / chunk_cycle_len, index % chunk_cycle_len)
}

/// The committed program-image polynomial's word vector: the RAM-remapped
/// bytecode words zero-padded to a power of two (at least 2).
pub fn program_image_words_padded(bytecode_words: &[u64]) -> Vec<u64> {
    let padded_len = bytecode_words.len().next_power_of_two().max(2);
    let mut words = bytecode_words.to_vec();
    words.resize(padded_len, 0);
    words
}

/// Sanity re-export target: the lane total the layout must fit.
pub const fn committed_total_lanes() -> usize {
    total_lanes()
}
