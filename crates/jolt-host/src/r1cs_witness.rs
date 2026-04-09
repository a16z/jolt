//! Per-cycle R1CS witness extraction from [`CycleRow`] data.
//!
//! Produces the 38-element witness vector for one cycle, matching the variable
//! layout in [`jolt_r1cs::constraints::rv64`].

use jolt_field::Field;
use jolt_instructions::flags::{CircuitFlags, InstructionFlags};
use jolt_r1cs::constraints::rv64::*;

use crate::bytecode::BytecodePreprocessing;
use crate::CycleRow;

/// Extract the 38-element R1CS witness vector for cycle `t` in the trace.
///
/// Requires access to cycle `t` and optionally cycle `t+1` (for next-PC
/// and next-flags values). The witness layout matches
/// [`jolt_r1cs::constraints::rv64`].
///
/// # Arguments
///
/// - `trace` — the full execution trace
/// - `t` — index of the current cycle
/// - `bytecode` — preprocessed bytecode table (for PC mapping)
pub fn r1cs_cycle_witness<C: CycleRow, F: Field>(
    trace: &[C],
    t: usize,
    bytecode: &BytecodePreprocessing,
) -> [F; NUM_VARS_PER_CYCLE] {
    let cycle = &trace[t];
    let next = if t + 1 < trace.len() {
        Some(&trace[t + 1])
    } else {
        None
    };

    let mut w = [F::from_u64(0); NUM_VARS_PER_CYCLE];

    // V_CONST = 1
    w[V_CONST] = F::from_u64(1);

    if cycle.is_noop() {
        // No-op: DoNotUpdateUnexpandedPC = 1, all else zero.
        w[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = F::from_u64(1);
        fill_next_cycle_fields(&mut w, next, bytecode);
        return w;
    }

    let cflags = cycle.circuit_flags();
    let iflags = cycle.instruction_flags();

    // Instruction inputs
    let left_input = if iflags[InstructionFlags::LeftOperandIsPC as usize] {
        cycle.unexpanded_pc()
    } else if iflags[InstructionFlags::LeftOperandIsRs1Value as usize] {
        cycle.rs1_read().map_or(0, |(_, v)| v)
    } else {
        0
    };

    let right_input: i128 = if iflags[InstructionFlags::RightOperandIsImm as usize] {
        cycle.imm()
    } else if iflags[InstructionFlags::RightOperandIsRs2Value as usize] {
        cycle.rs2_read().map_or(0, |(_, v)| v as i128)
    } else {
        0
    };

    w[V_LEFT_INSTRUCTION_INPUT] = F::from_u64(left_input);
    w[V_RIGHT_INSTRUCTION_INPUT] = F::from_i128(right_input);

    // Product (field multiplication — matches modular semantics)
    w[V_PRODUCT] = w[V_LEFT_INSTRUCTION_INPUT] * w[V_RIGHT_INSTRUCTION_INPUT];

    // Lookup output (computed before operands — needed for Advice)
    let lookup_output = cycle.lookup_output();
    w[V_LOOKUP_OUTPUT] = F::from_u64(lookup_output);

    // Lookup operands
    let (left_lookup, right_lookup) =
        compute_lookup_operands(left_input, right_input, w[V_PRODUCT], &cflags, lookup_output);
    w[V_LEFT_LOOKUP_OPERAND] = left_lookup;
    w[V_RIGHT_LOOKUP_OPERAND] = right_lookup;

    // Registers
    let (rs1_val, rs2_val, rd_val) = (
        cycle.rs1_read().map_or(0, |(_, v)| v),
        cycle.rs2_read().map_or(0, |(_, v)| v),
        cycle.rd_write().map_or(0, |(_, _, post)| post),
    );
    w[V_RS1_VALUE] = F::from_u64(rs1_val);
    w[V_RS2_VALUE] = F::from_u64(rs2_val);
    w[V_RD_WRITE_VALUE] = F::from_u64(rd_val);

    // RAM
    let ram_addr = cycle.ram_access_address().unwrap_or(0);
    let ram_read = cycle.ram_read_value().unwrap_or(0);
    let ram_write = cycle.ram_write_value().unwrap_or(0);
    w[V_RAM_ADDRESS] = F::from_u64(ram_addr);
    w[V_RAM_READ_VALUE] = F::from_u64(ram_read);
    w[V_RAM_WRITE_VALUE] = F::from_u64(ram_write);

    // PCs
    w[V_PC] = F::from_u64(bytecode.get_pc(cycle) as u64);
    w[V_UNEXPANDED_PC] = F::from_u64(cycle.unexpanded_pc());

    // Immediate
    w[V_IMM] = F::from_i128(cycle.imm());

    // Circuit flags
    w[V_FLAG_ADD_OPERANDS] = F::from_u64(cflags[CircuitFlags::AddOperands as usize] as u64);
    w[V_FLAG_SUBTRACT_OPERANDS] =
        F::from_u64(cflags[CircuitFlags::SubtractOperands as usize] as u64);
    w[V_FLAG_MULTIPLY_OPERANDS] =
        F::from_u64(cflags[CircuitFlags::MultiplyOperands as usize] as u64);
    w[V_FLAG_LOAD] = F::from_u64(cflags[CircuitFlags::Load as usize] as u64);
    w[V_FLAG_STORE] = F::from_u64(cflags[CircuitFlags::Store as usize] as u64);
    w[V_FLAG_JUMP] = F::from_u64(cflags[CircuitFlags::Jump as usize] as u64);
    w[V_FLAG_WRITE_LOOKUP_OUTPUT_TO_RD] =
        F::from_u64(cflags[CircuitFlags::WriteLookupOutputToRD as usize] as u64);
    w[V_FLAG_VIRTUAL_INSTRUCTION] =
        F::from_u64(cflags[CircuitFlags::VirtualInstruction as usize] as u64);
    w[V_FLAG_ASSERT] = F::from_u64(cflags[CircuitFlags::Assert as usize] as u64);
    w[V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] =
        F::from_u64(cflags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] as u64);
    w[V_FLAG_ADVICE] = F::from_u64(cflags[CircuitFlags::Advice as usize] as u64);
    w[V_FLAG_IS_COMPRESSED] = F::from_u64(cflags[CircuitFlags::IsCompressed as usize] as u64);
    w[V_FLAG_IS_FIRST_IN_SEQUENCE] =
        F::from_u64(cflags[CircuitFlags::IsFirstInSequence as usize] as u64);
    w[V_FLAG_IS_LAST_IN_SEQUENCE] =
        F::from_u64(cflags[CircuitFlags::IsLastInSequence as usize] as u64);

    // Product factors
    let branch_flag = iflags[InstructionFlags::Branch as usize];
    w[V_BRANCH] = F::from_u64(branch_flag as u64);

    // ShouldBranch = LookupOutput * Branch
    w[V_SHOULD_BRANCH] = w[V_LOOKUP_OUTPUT] * w[V_BRANCH];

    // Next-cycle fields
    fill_next_cycle_fields(&mut w, next, bytecode);

    // ShouldJump = Jump * (1 - NextIsNoop)
    let next_is_noop = next.is_none_or(|c| c.is_noop());
    w[V_NEXT_IS_NOOP] = F::from_u64(next_is_noop as u64);
    w[V_SHOULD_JUMP] = w[V_FLAG_JUMP] * (F::from_u64(1) - w[V_NEXT_IS_NOOP]);

    w
}

/// Fill next-cycle PC and flag fields from the next cycle (or zeros if last).
fn fill_next_cycle_fields<C: CycleRow, F: Field>(
    w: &mut [F; NUM_VARS_PER_CYCLE],
    next: Option<&C>,
    bytecode: &BytecodePreprocessing,
) {
    if let Some(nc) = next {
        w[V_NEXT_PC] = F::from_u64(bytecode.get_pc(nc) as u64);
        w[V_NEXT_UNEXPANDED_PC] = F::from_u64(nc.unexpanded_pc());

        let nc_flags = nc.circuit_flags();
        w[V_NEXT_IS_VIRTUAL] =
            F::from_u64(nc_flags[CircuitFlags::VirtualInstruction as usize] as u64);
        w[V_NEXT_IS_FIRST_IN_SEQUENCE] =
            F::from_u64(nc_flags[CircuitFlags::IsFirstInSequence as usize] as u64);
    }
    // else: already zero-initialized
}

/// Compute lookup operands from instruction inputs and circuit flags.
///
/// The lookup operand computation matches the R1CS constraints:
/// - Add: left_lookup=0, right_lookup = left_input + right_input
/// - Sub: left_lookup=0, right_lookup = left_input - right_input + 2^64
/// - Mul: left_lookup=0, right_lookup = product
/// - Otherwise: left_lookup = left_input, right_lookup = right_input
/// - Advice: left_lookup = left_input, right_lookup is unconstrained (set to right_input)
fn compute_lookup_operands<F: Field>(
    left_input: u64,
    right_input: i128,
    product: F,
    cflags: &[bool],
    lookup_output: u64,
) -> (F, F) {
    let add = cflags[CircuitFlags::AddOperands as usize];
    let sub = cflags[CircuitFlags::SubtractOperands as usize];
    let mul = cflags[CircuitFlags::MultiplyOperands as usize];
    let advice = cflags[CircuitFlags::Advice as usize];

    if add {
        let sum = left_input as i128 + right_input;
        (F::from_u64(0), F::from_i128(sum))
    } else if sub {
        let diff = left_input as i128 - right_input + (1i128 << 64);
        (F::from_u64(0), F::from_i128(diff))
    } else if mul {
        (F::from_u64(0), product)
    } else if advice {
        // Advice: right_lookup = advice value = lookup_output
        (F::from_u64(0), F::from_u64(lookup_output))
    } else {
        (F::from_u64(left_input), F::from_i128(right_input))
    }
}

/// Build the full R1CS witness for the entire trace.
///
/// Returns a flat `Vec<F>` of length `trace.len() * num_vars_padded`, suitable for
/// passing to [`R1csSource::new`](jolt_r1cs::R1csSource::new).
pub fn build_r1cs_witness<C: CycleRow, F: Field>(
    trace: &[C],
    bytecode: &BytecodePreprocessing,
    num_vars_padded: usize,
) -> Vec<F> {
    let num_cycles = trace.len();
    let mut witness = vec![F::from_u64(0); num_cycles * num_vars_padded];

    for t in 0..num_cycles {
        let row = r1cs_cycle_witness::<C, F>(trace, t, bytecode);
        let offset = t * num_vars_padded;
        witness[offset..offset + NUM_VARS_PER_CYCLE].copy_from_slice(&row);
    }

    witness
}
