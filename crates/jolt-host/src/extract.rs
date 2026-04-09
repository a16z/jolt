//! Unified trace extraction: single pass from [`CycleRow`] to witness + R1CS data.
//!
//! [`extract_trace`] replaces the previous dual-path extraction
//! (`cycle_to_input` + `build_r1cs_witness`) with a single iteration over the
//! trace that produces both the witness [`CycleInput`] buffer and the flat
//! R1CS witness vector.

use common::jolt_device::MemoryLayout;
use jolt_field::Field;
use jolt_instructions::flags::{CircuitFlags, InstructionFlags};
use jolt_r1cs::constraints::rv64::*;
use jolt_witness::CycleInput;

use crate::bytecode::BytecodePreprocessing;
use crate::CycleRow;

/// Per-cycle instruction flag polynomials extracted from the trace.
///
/// These flags are per-instruction-type booleans that aren't R1CS variables
/// but are needed by sumcheck instances (Shift, InstructionInput).
pub struct InstructionFlagData<F> {
    pub is_noop: Vec<F>,
    pub left_is_rs1: Vec<F>,
    pub left_is_pc: Vec<F>,
    pub right_is_rs2: Vec<F>,
    pub right_is_imm: Vec<F>,
}

/// Single-pass extraction of both witness polynomial inputs and R1CS witness data.
///
/// Iterates the trace once, producing:
/// - `Vec<CycleInput>` of length `size` (padded with `CycleInput::PADDING`)
/// - `Vec<F>` of length `size * num_vars_padded` (flat R1CS witness)
/// - `InstructionFlagData<F>` — per-cycle instruction flags for sumcheck instances
///
/// This replaces the previous two-pass approach where `cycle_to_input` and
/// `build_r1cs_witness` independently iterated the trace.
pub fn extract_trace<C: CycleRow, F: Field>(
    trace: &[C],
    size: usize,
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
    num_vars_padded: usize,
) -> (Vec<CycleInput>, Vec<F>, InstructionFlagData<F>) {
    let mut inputs = Vec::with_capacity(size);
    let mut r1cs = vec![F::from_u64(0); size * num_vars_padded];

    let mut iflag_is_noop = vec![F::from_u64(0); size];
    let mut iflag_left_is_rs1 = vec![F::from_u64(0); size];
    let mut iflag_left_is_pc = vec![F::from_u64(0); size];
    let mut iflag_right_is_rs2 = vec![F::from_u64(0); size];
    let mut iflag_right_is_imm = vec![F::from_u64(0); size];

    for t in 0..size {
        let r1cs_offset = t * num_vars_padded;

        if t >= trace.len() {
            // Padding cycle
            inputs.push(CycleInput::PADDING);
            iflag_is_noop[t] = F::from_u64(1);
            // R1CS: V_CONST=1, DoNotUpdateUnexpandedPC=1, rest zero
            r1cs[r1cs_offset + V_CONST] = F::from_u64(1);
            r1cs[r1cs_offset + V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = F::from_u64(1);
            r1cs[r1cs_offset + V_NEXT_IS_NOOP] = F::from_u64(1);
            fill_next_r1cs::<C, F>(&mut r1cs, r1cs_offset, None, bytecode);
            continue;
        }

        let cycle = &trace[t];
        let next = trace.get(t + 1);

        if cycle.is_noop() {
            inputs.push(CycleInput::PADDING);
            iflag_is_noop[t] = F::from_u64(1);
            r1cs[r1cs_offset + V_CONST] = F::from_u64(1);
            r1cs[r1cs_offset + V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = F::from_u64(1);
            r1cs[r1cs_offset + V_NEXT_IS_NOOP] =
                F::from_u64(next.is_none_or(|c| c.is_noop()) as u64);
            fill_next_r1cs::<C, F>(&mut r1cs, r1cs_offset, next, bytecode);
            continue;
        }

        // ── Shared reads from CycleRow ─────────────────────────────────

        let rd_write = cycle.rd_write();
        let rs1_read = cycle.rs1_read();
        let rs2_read = cycle.rs2_read();
        let ram_addr_raw = cycle.ram_access_address();
        let ram_read = cycle.ram_read_value();
        let ram_write = cycle.ram_write_value();
        let lookup_index = cycle.lookup_index();
        let lookup_output = cycle.lookup_output();
        let unexpanded_pc = cycle.unexpanded_pc();
        let imm = cycle.imm();
        let pc_index = bytecode.get_pc(cycle);
        let cflags = cycle.circuit_flags();
        let iflags = cycle.instruction_flags();

        // Instruction flags
        iflag_is_noop[t] = F::from_u64(iflags[InstructionFlags::IsNoop as usize] as u64);
        iflag_left_is_rs1[t] =
            F::from_u64(iflags[InstructionFlags::LeftOperandIsRs1Value as usize] as u64);
        iflag_left_is_pc[t] =
            F::from_u64(iflags[InstructionFlags::LeftOperandIsPC as usize] as u64);
        iflag_right_is_rs2[t] =
            F::from_u64(iflags[InstructionFlags::RightOperandIsRs2Value as usize] as u64);
        iflag_right_is_imm[t] =
            F::from_u64(iflags[InstructionFlags::RightOperandIsImm as usize] as u64);

        // ── CycleInput (witness polynomial data) ───────────────────────

        let rd_inc = match rd_write {
            Some((_, pre, post)) => post as i128 - pre as i128,
            None => 0,
        };

        let ram_inc = match (ram_read, ram_write) {
            (Some(pre), Some(post)) => post as i128 - pre as i128,
            _ => 0,
        };

        let ram_address = ram_addr_raw.map(|addr| {
            let lowest = memory_layout.get_lowest_address();
            debug_assert!(
                addr >= lowest,
                "unexpected RAM address {addr:#x} below lowest {lowest:#x}"
            );
            ((addr - lowest) / 8) as u128
        });

        inputs.push(CycleInput {
            dense: [rd_inc, ram_inc],
            one_hot: [Some(lookup_index), Some(pc_index as u128), ram_address],
        });

        // ── R1CS witness (per-constraint verification vector) ──────────

        let w = &mut r1cs[r1cs_offset..r1cs_offset + NUM_VARS_PER_CYCLE];
        w[V_CONST] = F::from_u64(1);

        // Instruction inputs
        let left_input = if iflags[InstructionFlags::LeftOperandIsPC as usize] {
            unexpanded_pc
        } else if iflags[InstructionFlags::LeftOperandIsRs1Value as usize] {
            rs1_read.map_or(0, |(_, v)| v)
        } else {
            0
        };

        let right_input: i128 = if iflags[InstructionFlags::RightOperandIsImm as usize] {
            imm
        } else if iflags[InstructionFlags::RightOperandIsRs2Value as usize] {
            rs2_read.map_or(0, |(_, v)| v as i128)
        } else {
            0
        };

        w[V_LEFT_INSTRUCTION_INPUT] = F::from_u64(left_input);
        w[V_RIGHT_INSTRUCTION_INPUT] = F::from_i128(right_input);
        w[V_PRODUCT] = w[V_LEFT_INSTRUCTION_INPUT] * w[V_RIGHT_INSTRUCTION_INPUT];

        // Lookup operands
        let add = cflags[CircuitFlags::AddOperands as usize];
        let sub = cflags[CircuitFlags::SubtractOperands as usize];
        let mul = cflags[CircuitFlags::MultiplyOperands as usize];

        let advice = cflags[CircuitFlags::Advice as usize];

        if add {
            let sum = left_input as i128 + right_input;
            w[V_LEFT_LOOKUP_OPERAND] = F::from_u64(0);
            w[V_RIGHT_LOOKUP_OPERAND] = F::from_i128(sum);
        } else if sub {
            let diff = left_input as i128 - right_input + (1i128 << 64);
            w[V_LEFT_LOOKUP_OPERAND] = F::from_u64(0);
            w[V_RIGHT_LOOKUP_OPERAND] = F::from_i128(diff);
        } else if mul {
            w[V_LEFT_LOOKUP_OPERAND] = F::from_u64(0);
            w[V_RIGHT_LOOKUP_OPERAND] = w[V_PRODUCT];
        } else if advice {
            // Advice: right_lookup = advice value = lookup_output
            w[V_LEFT_LOOKUP_OPERAND] = F::from_u64(0);
            w[V_RIGHT_LOOKUP_OPERAND] = F::from_u64(lookup_output);
        } else {
            w[V_LEFT_LOOKUP_OPERAND] = F::from_u64(left_input);
            w[V_RIGHT_LOOKUP_OPERAND] = F::from_i128(right_input);
        }

        w[V_LOOKUP_OUTPUT] = F::from_u64(lookup_output);

        // Registers
        w[V_RS1_VALUE] = F::from_u64(rs1_read.map_or(0, |(_, v)| v));
        w[V_RS2_VALUE] = F::from_u64(rs2_read.map_or(0, |(_, v)| v));
        w[V_RD_WRITE_VALUE] = F::from_u64(rd_write.map_or(0, |(_, _, post)| post));

        // RAM
        w[V_RAM_ADDRESS] = F::from_u64(ram_addr_raw.unwrap_or(0));
        w[V_RAM_READ_VALUE] = F::from_u64(ram_read.unwrap_or(0));
        w[V_RAM_WRITE_VALUE] = F::from_u64(ram_write.unwrap_or(0));

        // PCs
        w[V_PC] = F::from_u64(pc_index as u64);
        w[V_UNEXPANDED_PC] = F::from_u64(unexpanded_pc);

        // Immediate
        w[V_IMM] = F::from_i128(imm);

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
        w[V_SHOULD_BRANCH] = w[V_LOOKUP_OUTPUT] * w[V_BRANCH];

        // Next-cycle fields
        fill_next_r1cs::<C, F>(&mut r1cs, r1cs_offset, next, bytecode);

        let next_is_noop = next.is_none_or(|c| c.is_noop());
        r1cs[r1cs_offset + V_NEXT_IS_NOOP] = F::from_u64(next_is_noop as u64);
        r1cs[r1cs_offset + V_SHOULD_JUMP] =
            r1cs[r1cs_offset + V_FLAG_JUMP] * (F::from_u64(1) - r1cs[r1cs_offset + V_NEXT_IS_NOOP]);
    }

    let instruction_flags = InstructionFlagData {
        is_noop: iflag_is_noop,
        left_is_rs1: iflag_left_is_rs1,
        left_is_pc: iflag_left_is_pc,
        right_is_rs2: iflag_right_is_rs2,
        right_is_imm: iflag_right_is_imm,
    };

    (inputs, r1cs, instruction_flags)
}

/// Fill next-cycle PC and flag fields in the R1CS witness.
fn fill_next_r1cs<C: CycleRow, F: Field>(
    r1cs: &mut [F],
    offset: usize,
    next: Option<&C>,
    bytecode: &BytecodePreprocessing,
) {
    if let Some(nc) = next {
        r1cs[offset + V_NEXT_PC] = F::from_u64(bytecode.get_pc(nc) as u64);
        r1cs[offset + V_NEXT_UNEXPANDED_PC] = F::from_u64(nc.unexpanded_pc());

        let nc_flags = nc.circuit_flags();
        r1cs[offset + V_NEXT_IS_VIRTUAL] =
            F::from_u64(nc_flags[CircuitFlags::VirtualInstruction as usize] as u64);
        r1cs[offset + V_NEXT_IS_FIRST_IN_SEQUENCE] =
            F::from_u64(nc_flags[CircuitFlags::IsFirstInSequence as usize] as u64);
    }
}
