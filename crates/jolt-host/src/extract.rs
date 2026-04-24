//! Single-pass trace extraction: witness inputs + R1CS + instruction flags.
//!
//! [`extract_trace`] iterates the trace once, producing all three artifacts
//! the prover needs. R1CS witness construction delegates to
//! [`r1cs_cycle_witness`](crate::r1cs_witness::r1cs_cycle_witness).

use common::jolt_device::MemoryLayout;
use jolt_field::Field;
use jolt_instructions::flags::{InstructionFlags, NUM_INSTRUCTION_FLAGS};
use jolt_r1cs::constraints::rv64::*;
use jolt_witness::CycleInput;

use crate::bytecode::BytecodePreprocessing;
use crate::r1cs_witness::r1cs_cycle_witness;
use crate::CycleRow;

/// Per-cycle instruction flag polynomials for sumcheck instances.
pub struct InstructionFlagData<F> {
    pub is_noop: Vec<F>,
    pub left_is_rs1: Vec<F>,
    pub left_is_pc: Vec<F>,
    pub right_is_rs2: Vec<F>,
    pub right_is_imm: Vec<F>,
}

/// Extract witness inputs, R1CS witness, and instruction flags in one pass.
///
/// Produces `size`-length outputs, padding beyond `trace.len()` with defaults.
pub fn extract_trace<C: CycleRow, F: Field>(
    trace: &[C],
    size: usize,
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
    num_vars_padded: usize,
) -> (Vec<CycleInput>, Vec<F>, InstructionFlagData<F>) {
    let mut inputs = Vec::with_capacity(size);
    let mut r1cs = vec![F::from_u64(0); size * num_vars_padded];
    let mut flags = InstructionFlagData::new(size);

    for t in 0..size {
        let offset = t * num_vars_padded;

        if t < trace.len() {
            let cycle = &trace[t];

            if cycle.is_noop() {
                inputs.push(CycleInput::PADDING);
            } else {
                inputs.push(cycle_input(cycle, bytecode, memory_layout));
            }

            let row = r1cs_cycle_witness::<C, F>(trace, t, bytecode);
            r1cs[offset..offset + NUM_VARS_PER_CYCLE].copy_from_slice(&row);

            let iflags = cycle.instruction_flags();
            flags.set(t, &iflags);
        } else {
            inputs.push(CycleInput::PADDING);
            flags.is_noop[t] = F::from_u64(1);
            r1cs[offset + V_CONST] = F::from_u64(1);
            r1cs[offset + V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = F::from_u64(1);
            r1cs[offset + V_NEXT_IS_NOOP] = F::from_u64(1);
        }
    }

    (inputs, r1cs, flags)
}

fn cycle_input(
    cycle: &impl CycleRow,
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
) -> CycleInput {
    let rd_inc = match cycle.rd_write() {
        Some((_, pre, post)) => post as i128 - pre as i128,
        None => 0,
    };
    let ram_inc = match (cycle.ram_read_value(), cycle.ram_write_value()) {
        (Some(pre), Some(post)) => post as i128 - pre as i128,
        _ => 0,
    };
    let lowest = memory_layout.get_lowest_address();
    let ram_address = cycle.ram_access_address().map(|addr| {
        debug_assert!(
            addr >= lowest,
            "RAM address {addr:#x} below lowest {lowest:#x}"
        );
        ((addr - lowest) / 8) as u128
    });

    CycleInput {
        dense: [rd_inc, ram_inc, 0],
        one_hot: [
            Some(cycle.lookup_index()),
            Some(bytecode.get_pc(cycle) as u128),
            ram_address,
            // FR write (source index 3). Populated by a follow-up pass that
            // replays the FieldRegEvent stream through the cycle window; the
            // extract_trace path produces the zero-field-register-activity
            // value here and is overwritten downstream. See Phase 4b.
            None,
        ],
    }
}

impl<F: Field> InstructionFlagData<F> {
    fn new(size: usize) -> Self {
        Self {
            is_noop: vec![F::from_u64(0); size],
            left_is_rs1: vec![F::from_u64(0); size],
            left_is_pc: vec![F::from_u64(0); size],
            right_is_rs2: vec![F::from_u64(0); size],
            right_is_imm: vec![F::from_u64(0); size],
        }
    }

    fn set(&mut self, t: usize, iflags: &[bool; NUM_INSTRUCTION_FLAGS]) {
        self.is_noop[t] = F::from_u64(iflags[InstructionFlags::IsNoop] as u64);
        self.left_is_rs1[t] = F::from_u64(iflags[InstructionFlags::LeftOperandIsRs1Value] as u64);
        self.left_is_pc[t] = F::from_u64(iflags[InstructionFlags::LeftOperandIsPC] as u64);
        self.right_is_rs2[t] = F::from_u64(iflags[InstructionFlags::RightOperandIsRs2Value] as u64);
        self.right_is_imm[t] = F::from_u64(iflags[InstructionFlags::RightOperandIsImm] as u64);
    }
}
