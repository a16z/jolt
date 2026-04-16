//! Per-cycle R1CS witness extraction from [`CycleRow`] data.
//!
//! Produces the witness vector for one cycle, matching the variable
//! layout in [`jolt_r1cs::constraints::rv64`].

use jolt_field::Field;
use jolt_instructions::flags::{CircuitFlags, InstructionFlags, NUM_CIRCUIT_FLAGS};
use jolt_r1cs::constraints::rv64::*;

use crate::bytecode::BytecodePreprocessing;
use crate::CycleRow;

/// Per-cycle R1CS witness matching `jolt_r1cs::constraints::rv64` layout.
pub fn r1cs_cycle_witness<C: CycleRow, F: Field>(
    trace: &[C],
    t: usize,
    bytecode: &BytecodePreprocessing,
) -> [F; NUM_VARS_PER_CYCLE] {
    let cycle = &trace[t];
    let next = trace.get(t + 1);

    let mut w = [F::from_u64(0); NUM_VARS_PER_CYCLE];
    w[V_CONST] = F::from_u64(1);

    if cycle.is_noop() {
        w[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = F::from_u64(1);
        w[V_NEXT_IS_NOOP] = F::from_u64(next.is_none_or(|c| c.is_noop()) as u64);
        fill_next_fields(&mut w, next, bytecode);
        return w;
    }

    let cflags = cycle.circuit_flags();
    let iflags = cycle.instruction_flags();

    // Instruction inputs
    let left_input = if iflags[InstructionFlags::LeftOperandIsPC] {
        cycle.unexpanded_pc()
    } else if iflags[InstructionFlags::LeftOperandIsRs1Value] {
        cycle.rs1_read().map_or(0, |(_, v)| v)
    } else {
        0
    };

    let right_input: i128 = if iflags[InstructionFlags::RightOperandIsImm] {
        cycle.imm()
    } else if iflags[InstructionFlags::RightOperandIsRs2Value] {
        cycle.rs2_read().map_or(0, |(_, v)| v as i128)
    } else {
        0
    };

    w[V_LEFT_INSTRUCTION_INPUT] = F::from_u64(left_input);
    w[V_RIGHT_INSTRUCTION_INPUT] = F::from_i128(right_input);
    w[V_PRODUCT] = w[V_LEFT_INSTRUCTION_INPUT] * w[V_RIGHT_INSTRUCTION_INPUT];

    // Lookup output
    let lookup_output = cycle.lookup_output();
    w[V_LOOKUP_OUTPUT] = F::from_u64(lookup_output);

    // Lookup operands — must match R1CS constraints
    let (left_lookup, right_lookup) = lookup_operands(
        left_input,
        right_input,
        w[V_PRODUCT],
        &cflags,
        lookup_output,
    );
    w[V_LEFT_LOOKUP_OPERAND] = left_lookup;
    w[V_RIGHT_LOOKUP_OPERAND] = right_lookup;

    // Registers
    w[V_RS1_VALUE] = F::from_u64(cycle.rs1_read().map_or(0, |(_, v)| v));
    w[V_RS2_VALUE] = F::from_u64(cycle.rs2_read().map_or(0, |(_, v)| v));
    w[V_RD_WRITE_VALUE] = F::from_u64(cycle.rd_write().map_or(0, |(_, _, post)| post));

    // RAM
    w[V_RAM_ADDRESS] = F::from_u64(cycle.ram_access_address().unwrap_or(0));
    w[V_RAM_READ_VALUE] = F::from_u64(cycle.ram_read_value().unwrap_or(0));
    w[V_RAM_WRITE_VALUE] = F::from_u64(cycle.ram_write_value().unwrap_or(0));

    // PCs
    w[V_PC] = F::from_u64(bytecode.get_pc(cycle) as u64);
    w[V_UNEXPANDED_PC] = F::from_u64(cycle.unexpanded_pc());
    w[V_IMM] = F::from_i128(cycle.imm());

    // Circuit flags
    w[V_FLAG_ADD_OPERANDS] = F::from_u64(cflags[CircuitFlags::AddOperands] as u64);
    w[V_FLAG_SUBTRACT_OPERANDS] = F::from_u64(cflags[CircuitFlags::SubtractOperands] as u64);
    w[V_FLAG_MULTIPLY_OPERANDS] = F::from_u64(cflags[CircuitFlags::MultiplyOperands] as u64);
    w[V_FLAG_LOAD] = F::from_u64(cflags[CircuitFlags::Load] as u64);
    w[V_FLAG_STORE] = F::from_u64(cflags[CircuitFlags::Store] as u64);
    w[V_FLAG_JUMP] = F::from_u64(cflags[CircuitFlags::Jump] as u64);
    w[V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD] =
        F::from_u64(cflags[CircuitFlags::WriteLookupOutputToRD] as u64);
    w[V_FLAG_VIRTUAL_INSTRUCTION] = F::from_u64(cflags[CircuitFlags::VirtualInstruction] as u64);
    w[V_FLAG_ASSERT] = F::from_u64(cflags[CircuitFlags::Assert] as u64);
    w[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] =
        F::from_u64(cflags[CircuitFlags::DoNotUpdateUnexpandedPC] as u64);
    w[V_FLAG_ADVICE] = F::from_u64(cflags[CircuitFlags::Advice] as u64);
    w[V_FLAG_IS_COMPRESSED] = F::from_u64(cflags[CircuitFlags::IsCompressed] as u64);
    w[V_FLAG_IS_FIRST_IN_SEQUENCE] = F::from_u64(cflags[CircuitFlags::IsFirstInSequence] as u64);
    w[V_FLAG_IS_LAST_IN_SEQUENCE] = F::from_u64(cflags[CircuitFlags::IsLastInSequence] as u64);

    // Product factors
    w[V_BRANCH] = F::from_u64(iflags[InstructionFlags::Branch] as u64);
    w[V_SHOULD_BRANCH] = w[V_LOOKUP_OUTPUT] * w[V_BRANCH];

    // Next-cycle fields
    fill_next_fields(&mut w, next, bytecode);
    let next_is_noop = next.is_none_or(|c| c.is_noop());
    w[V_NEXT_IS_NOOP] = F::from_u64(next_is_noop as u64);
    w[V_SHOULD_JUMP] = w[V_FLAG_JUMP] * (F::from_u64(1) - w[V_NEXT_IS_NOOP]);

    w
}

fn fill_next_fields<C: CycleRow, F: Field>(
    w: &mut [F; NUM_VARS_PER_CYCLE],
    next: Option<&C>,
    bytecode: &BytecodePreprocessing,
) {
    if let Some(nc) = next {
        w[V_NEXT_PC] = F::from_u64(bytecode.get_pc(nc) as u64);
        w[V_NEXT_UNEXPANDED_PC] = F::from_u64(nc.unexpanded_pc());
        let nc_flags = nc.circuit_flags();
        w[V_NEXT_IS_VIRTUAL] = F::from_u64(nc_flags[CircuitFlags::VirtualInstruction] as u64);
        w[V_NEXT_IS_FIRST_IN_SEQUENCE] =
            F::from_u64(nc_flags[CircuitFlags::IsFirstInSequence] as u64);
    }
}

/// Lookup operands matching the R1CS constraints:
/// - Add: `(0, left + right)`
/// - Sub: `(0, left - right + 2^64)`
/// - Mul: `(0, product)`
/// - Advice: `(0, lookup_output)`
/// - Default: `(left, right)`
fn lookup_operands<F: Field>(
    left: u64,
    right: i128,
    product: F,
    cflags: &[bool; NUM_CIRCUIT_FLAGS],
    lookup_output: u64,
) -> (F, F) {
    if cflags[CircuitFlags::AddOperands] {
        (F::from_u64(0), F::from_i128(left as i128 + right))
    } else if cflags[CircuitFlags::SubtractOperands] {
        (
            F::from_u64(0),
            F::from_i128(left as i128 - right + (1i128 << 64)),
        )
    } else if cflags[CircuitFlags::MultiplyOperands] {
        (F::from_u64(0), product)
    } else if cflags[CircuitFlags::Advice] {
        (F::from_u64(0), F::from_u64(lookup_output))
    } else {
        (F::from_u64(left), F::from_i128(right))
    }
}

/// Flat R1CS witness for the entire trace.
///
/// Returns `Vec<F>` of length `trace.len() * num_vars_padded`.
pub fn build_r1cs_witness<C: CycleRow, F: Field>(
    trace: &[C],
    bytecode: &BytecodePreprocessing,
    num_vars_padded: usize,
) -> Vec<F> {
    let n = trace.len();
    let mut witness = vec![F::from_u64(0); n * num_vars_padded];
    for t in 0..n {
        let row = r1cs_cycle_witness::<C, F>(trace, t, bytecode);
        witness[t * num_vars_padded..t * num_vars_padded + NUM_VARS_PER_CYCLE]
            .copy_from_slice(&row);
    }
    witness
}
