//! R1CS witness generation from execution trace cycles.
//!
//! Converts a tracer `Cycle` into a per-cycle witness vector of 38 field
//! elements matching the variable layout in [`jolt_r1cs::layout`].

use jolt_field::Field;
use jolt_instructions::flags::{CircuitFlags, InstructionFlags, NUM_CIRCUIT_FLAGS};
use jolt_r1cs::*;
use tracer::instruction::{Cycle, RAMAccess};

use crate::bytecode::BytecodePreprocessing;
use crate::cycle_data::instruction_inputs;
use crate::flags;

/// Generates a complete R1CS witness vector from a single execution cycle.
///
/// The returned vector has [`NUM_VARS_PER_CYCLE`] elements, indexed by the
/// `V_*` constants in [`jolt_r1cs::layout`].
///
/// # Arguments
///
/// * `cycle` — The current execution cycle.
/// * `next_cycle` — The next cycle (for PC continuity checks), or `None` if last.
/// * `bytecode` — Preprocessed bytecode for PC expansion.
pub fn cycle_to_witness<F: Field>(
    cycle: &Cycle,
    next_cycle: Option<&Cycle>,
    bytecode: &BytecodePreprocessing,
) -> Vec<F> {
    let mut w = vec![F::zero(); NUM_VARS_PER_CYCLE];

    w[V_CONST] = F::one();

    let instr = cycle.instruction();
    let norm = instr.normalize();

    let cflags = flags::circuit_flags(&instr);
    let iflags = flags::instruction_flags(&instr);

    // Register values
    let rs1_value = cycle.rs1_read().map_or(0u64, |(_, v)| v);
    let rs2_value = cycle.rs2_read().map_or(0u64, |(_, v)| v);
    let rd_write_value = cycle.rd_write().map_or(0u64, |(_, _, post)| post);

    w[V_RS1_VALUE] = F::from_u64(rs1_value);
    w[V_RS2_VALUE] = F::from_u64(rs2_value);
    w[V_RD_WRITE_VALUE] = F::from_u64(rd_write_value);

    // RAM access
    let (ram_addr, ram_read, ram_write) = match cycle.ram_access() {
        RAMAccess::Read(r) => (r.address, r.value, r.value),
        RAMAccess::Write(w) => (w.address, w.pre_value, w.post_value),
        RAMAccess::NoOp => (0, 0, 0),
    };
    w[V_RAM_ADDRESS] = F::from_u64(ram_addr);
    w[V_RAM_READ_VALUE] = F::from_u64(ram_read);
    w[V_RAM_WRITE_VALUE] = F::from_u64(ram_write);

    // PCs
    let unexpanded_pc = norm.address as u64;
    let pc = bytecode.get_pc(cycle);
    w[V_UNEXPANDED_PC] = F::from_u64(unexpanded_pc);
    w[V_PC] = F::from_u64(pc);

    // Next cycle state
    let (next_unexpanded_pc, next_pc, next_is_virtual, next_is_first, next_is_noop) =
        if let Some(next) = next_cycle {
            let next_instr = next.instruction();
            let next_norm = next_instr.normalize();
            let next_cflags = flags::circuit_flags(&next_instr);
            let next_iflags = flags::instruction_flags(&next_instr);
            (
                next_norm.address as u64,
                bytecode.get_pc(next),
                next_cflags[CircuitFlags::VirtualInstruction as usize],
                next_cflags[CircuitFlags::IsFirstInSequence as usize],
                next_iflags[InstructionFlags::IsNoop as usize],
            )
        } else {
            (0, 0, false, false, false)
        };

    w[V_NEXT_UNEXPANDED_PC] = F::from_u64(next_unexpanded_pc);
    w[V_NEXT_PC] = F::from_u64(next_pc);
    w[V_NEXT_IS_VIRTUAL] = bool_to_field(next_is_virtual);
    w[V_NEXT_IS_FIRST_IN_SEQUENCE] = bool_to_field(next_is_first);

    // Immediate (signed)
    let imm = norm.operands.imm;
    w[V_IMM] = F::from_i128(imm);

    // Instruction inputs: determined by InstructionFlags
    let (left_input, right_input) = instruction_inputs(cycle, &iflags, unexpanded_pc, imm);

    w[V_LEFT_INSTRUCTION_INPUT] = F::from_u64(left_input);
    w[V_RIGHT_INSTRUCTION_INPUT] = F::from_i128(right_input);

    // Product: left_input * right_input in the field
    w[V_PRODUCT] = F::from_u64(left_input) * F::from_i128(right_input);

    // Lookup operands: determined by CircuitFlags
    let add_ops = cflags[CircuitFlags::AddOperands as usize];
    let sub_ops = cflags[CircuitFlags::SubtractOperands as usize];
    let mul_ops = cflags[CircuitFlags::MultiplyOperands as usize];
    let advice = cflags[CircuitFlags::Advice as usize];

    let (left_lookup, right_lookup): (u64, F) = if add_ops {
        // RightLookup = LeftInput + RightInput
        (0, F::from_u64(left_input) + F::from_i128(right_input))
    } else if sub_ops {
        // RightLookup = LeftInput - RightInput + 2^64
        // (the +2^64 converts signed subtraction to unsigned representation)
        let sub_result =
            F::from_u64(left_input) - F::from_i128(right_input) + F::from_i128(1i128 << 64);
        (0, sub_result)
    } else if mul_ops {
        // RightLookup = Product
        (0, w[V_PRODUCT])
    } else if advice {
        // Advice instructions: right_lookup is the lookup output
        (0, F::from_u64(rd_write_value))
    } else {
        // Default (interleaved): left_lookup = left_input, right_lookup = right_input
        (left_input, F::from_i128(right_input))
    };

    w[V_LEFT_LOOKUP_OPERAND] = F::from_u64(left_lookup);
    w[V_RIGHT_LOOKUP_OPERAND] = right_lookup;

    // Lookup output
    let is_branch = iflags[InstructionFlags::Branch as usize];
    let is_jump = cflags[CircuitFlags::Jump as usize];
    let is_assert = cflags[CircuitFlags::Assert as usize];
    let write_lookup_to_rd = cflags[CircuitFlags::WriteLookupOutputToRD as usize];

    let lookup_output: F = if write_lookup_to_rd {
        F::from_u64(rd_write_value)
    } else if is_assert {
        F::one()
    } else if is_jump {
        F::from_u64(next_unexpanded_pc)
    } else if is_branch {
        // Branch taken if next PC matches branch target (PC + imm)
        let target = (unexpanded_pc as i128 + imm) as u64;
        if next_unexpanded_pc == target {
            F::one()
        } else {
            F::zero()
        }
    } else {
        F::zero()
    };

    w[V_LOOKUP_OUTPUT] = lookup_output;

    // Circuit flags → witness variables V_FLAG_*
    for i in 0..NUM_CIRCUIT_FLAGS {
        w[V_FLAG_ADD_OPERANDS + i] = bool_to_field(cflags[i]);
    }

    // Product factor variables
    w[V_BRANCH] = bool_to_field(is_branch);
    w[V_NEXT_IS_NOOP] = bool_to_field(next_is_noop);

    // Product-derived booleans (must match product constraints 20-21)
    let should_branch = lookup_output != F::zero() && is_branch;
    w[V_SHOULD_BRANCH] = bool_to_field(should_branch);

    let should_jump = is_jump && !next_is_noop;
    w[V_SHOULD_JUMP] = bool_to_field(should_jump);

    w
}

/// Generates R1CS witnesses for an entire trace.
///
/// Returns one witness vector per cycle, ready for [`interleave_witnesses`](crate::preprocessing::interleave_witnesses).
pub fn trace_to_witnesses<F: Field>(
    trace: &[Cycle],
    bytecode: &BytecodePreprocessing,
) -> Vec<Vec<F>> {
    trace
        .iter()
        .enumerate()
        .map(|(i, cycle)| {
            let next = trace.get(i + 1);
            cycle_to_witness(cycle, next, bytecode)
        })
        .collect()
}

#[inline]
fn bool_to_field<F: Field>(b: bool) -> F {
    if b {
        F::one()
    } else {
        F::zero()
    }
}

