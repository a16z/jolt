//! Conversion from tracer [`Cycle`] to jolt-witness [`CycleData`].
//!
//! [`CycleData`] is the per-cycle input for committed polynomial witness
//! generation (Inc, Ra one-hot). This module extracts the required fields
//! from the tracer's execution trace.

use jolt_instructions::flags::{CircuitFlags, InstructionFlags};
use jolt_instructions::interleave::interleave_bits;
use tracer::instruction::{Cycle, RAMAccess};

use crate::bytecode::BytecodePreprocessing;
use crate::cycle::CycleData;
use crate::flags;

/// Converts a tracer `Cycle` into a `CycleData` for committed polynomial witness generation.
///
/// Extracts:
/// - `rd_inc`: register write increment (`post - pre`), or 0 if no write
/// - `ram_inc`: RAM write increment (`post - pre`), or 0 if no write
/// - `lookup_index`: instruction lookup index (operand encoding determined by circuit flags)
/// - `pc_index`: expanded PC from bytecode preprocessing
/// - `ram_address`: remapped RAM address, or `None` for non-memory cycles
pub fn cycle_to_cycle_data(cycle: &Cycle, bytecode: &BytecodePreprocessing) -> CycleData {
    let rd_inc = cycle
        .rd_write()
        .map_or(0i128, |(_, pre, post)| post as i128 - pre as i128);

    let ram_inc = match cycle.ram_access() {
        RAMAccess::Write(w) => w.post_value as i128 - w.pre_value as i128,
        _ => 0,
    };

    let lookup_index = compute_lookup_index(cycle);

    let pc_index = bytecode.get_pc(cycle) as u32;

    let ram_address = match cycle.ram_access() {
        RAMAccess::Read(r) => Some(r.address),
        RAMAccess::Write(w) => Some(w.address),
        RAMAccess::NoOp => None,
    };

    CycleData {
        rd_inc,
        ram_inc,
        lookup_index,
        pc_index,
        ram_address,
    }
}

/// Computes the lookup index for a cycle based on circuit and instruction flags.
///
/// The encoding depends on the operand-combination flags:
/// - **AddOperands**: `left_input + right_input` (no interleaving)
/// - **SubtractOperands**: `left_input + (2^64 - right_input)` (two's complement subtraction)
/// - **MultiplyOperands**: `left_input * right_input` (no interleaving)
/// - **Advice / NoOp**: `0` (no lookup)
/// - **Default (interleaved)**: `interleave_bits(left_input, right_input)`
fn compute_lookup_index(cycle: &Cycle) -> u128 {
    let instr = cycle.instruction();
    let norm = instr.normalize();
    let cflags = flags::circuit_flags(&instr);
    let iflags = flags::instruction_flags(&instr);

    if iflags[InstructionFlags::IsNoop] || cflags[CircuitFlags::Advice] {
        return 0;
    }

    let (left, right) = instruction_inputs(cycle, &iflags, norm.address as u64, norm.operands.imm);

    if cflags[CircuitFlags::AddOperands] {
        left as u128 + right as u64 as u128
    } else if cflags[CircuitFlags::SubtractOperands] {
        // Two's complement: left + (2^64 - right)
        left as u128 + ((1u128 << 64) - right as u64 as u128)
    } else if cflags[CircuitFlags::MultiplyOperands] {
        left as u128 * right as u64 as u128
    } else {
        // Default: interleave bits
        interleave_bits(left, right as u64)
    }
}

/// Extracts the two instruction inputs from a cycle based on instruction flags.
///
/// Returns `(left_input, right_input)` where:
/// - `left_input`: PC (if `LeftOperandIsPC`) or rs1 (if `LeftOperandIsRs1Value`) or 0
/// - `right_input`: immediate (if `RightOperandIsImm`) or rs2 (if `RightOperandIsRs2Value`) or 0
pub fn instruction_inputs(
    cycle: &Cycle,
    iflags: &[bool; jolt_instructions::flags::NUM_INSTRUCTION_FLAGS],
    unexpanded_pc: u64,
    imm: i128,
) -> (u64, i128) {
    let left = if iflags[InstructionFlags::LeftOperandIsPC] {
        unexpanded_pc
    } else if iflags[InstructionFlags::LeftOperandIsRs1Value] {
        cycle.rs1_read().map_or(0u64, |(_, v)| v)
    } else {
        0
    };

    let right = if iflags[InstructionFlags::RightOperandIsImm] {
        imm
    } else if iflags[InstructionFlags::RightOperandIsRs2Value] {
        cycle.rs2_read().map_or(0u64, |(_, v)| v) as i128
    } else {
        0
    };

    (left, right)
}

/// Converts an entire trace into `CycleData` vectors, padded to the next power of two.
pub fn trace_to_cycle_data(trace: &[Cycle], bytecode: &BytecodePreprocessing) -> Vec<CycleData> {
    let mut data: Vec<CycleData> = trace
        .iter()
        .map(|c| cycle_to_cycle_data(c, bytecode))
        .collect();

    // Pad to next power of two with PADDING cycles
    let padded_len = data.len().next_power_of_two();
    data.resize(padded_len, CycleData::PADDING);

    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracer::instruction::{
        add::ADD, format::format_r::FormatR, format::format_r::RegisterStateFormatR, mul::MUL,
        or::OR, sub::SUB, RISCVCycle,
    };

    fn make_r_cycle<T: tracer::instruction::RISCVInstruction<Format = FormatR, RAMAccess = ()>>(
        instr: T,
        rs1: u64,
        rs2: u64,
        rd_pre: u64,
        rd_post: u64,
    ) -> RISCVCycle<T> {
        RISCVCycle {
            instruction: instr,
            register_state: RegisterStateFormatR {
                rd: (rd_pre, rd_post),
                rs1,
                rs2,
            },
            ram_access: (),
        }
    }

    fn r_instr_with_regs<T: Default + AsMutFormatR>(rd: u8, rs1: u8, rs2: u8) -> T {
        let mut instr = T::default();
        instr.set_operands(FormatR { rd, rs1, rs2 });
        instr
    }

    /// Helper trait to set FormatR operands on an instruction struct.
    /// All FormatR instructions have an `operands: FormatR` field via the macro.
    trait AsMutFormatR {
        fn set_operands(&mut self, ops: FormatR);
    }

    macro_rules! impl_as_mut_format_r {
        ($($t:ty),*) => {
            $(impl AsMutFormatR for $t {
                fn set_operands(&mut self, ops: FormatR) {
                    self.operands = ops;
                }
            })*
        };
    }
    impl_as_mut_format_r!(ADD, SUB, MUL, OR);

    #[test]
    fn noop_produces_padding() {
        let prep = BytecodePreprocessing::new(&[Cycle::NoOp]);
        let cd = cycle_to_cycle_data(&Cycle::NoOp, &prep);
        assert_eq!(cd.rd_inc, 0);
        assert_eq!(cd.ram_inc, 0);
        assert_eq!(cd.lookup_index, 0);
        assert!(cd.ram_address.is_none());
    }

    #[test]
    fn noop_lookup_index_is_zero() {
        assert_eq!(compute_lookup_index(&Cycle::NoOp), 0);
    }

    #[test]
    fn add_lookup_index_is_sum() {
        // ADD uses AddOperands → lookup_index = rs1 + rs2
        let rs1 = 100u64;
        let rs2 = 200u64;
        let instr = r_instr_with_regs::<ADD>(1, 2, 3);
        let cycle = Cycle::ADD(make_r_cycle(instr, rs1, rs2, 0, rs1.wrapping_add(rs2)));

        let idx = compute_lookup_index(&cycle);
        assert_eq!(idx, rs1 as u128 + rs2 as u128);
    }

    #[test]
    fn add_lookup_index_large_values() {
        let rs1 = u64::MAX;
        let rs2 = 1u64;
        let instr = r_instr_with_regs::<ADD>(1, 2, 3);
        let cycle = Cycle::ADD(make_r_cycle(instr, rs1, rs2, 0, rs1.wrapping_add(rs2)));

        let idx = compute_lookup_index(&cycle);
        assert_eq!(idx, u64::MAX as u128 + 1);
    }

    #[test]
    fn sub_lookup_index_is_twos_complement() {
        // SUB uses SubtractOperands → lookup_index = rs1 + (2^64 - rs2)
        let rs1 = 300u64;
        let rs2 = 100u64;
        let instr = r_instr_with_regs::<SUB>(1, 2, 3);
        let cycle = Cycle::SUB(make_r_cycle(instr, rs1, rs2, 0, rs1.wrapping_sub(rs2)));

        let idx = compute_lookup_index(&cycle);
        let expected = rs1 as u128 + ((1u128 << 64) - rs2 as u128);
        assert_eq!(idx, expected);
    }

    #[test]
    fn mul_lookup_index_is_product() {
        // MUL uses MultiplyOperands → lookup_index = rs1 * rs2
        let rs1 = 7u64;
        let rs2 = 13u64;
        let instr = r_instr_with_regs::<MUL>(1, 2, 3);
        let cycle = Cycle::MUL(make_r_cycle(instr, rs1, rs2, 0, rs1.wrapping_mul(rs2)));

        let idx = compute_lookup_index(&cycle);
        assert_eq!(idx, 91); // 7 * 13
    }

    #[test]
    fn or_lookup_index_is_interleaved() {
        // OR uses default interleaved encoding → lookup_index = interleave_bits(rs1, rs2)
        let rs1 = 0xABu64;
        let rs2 = 0xCDu64;
        let instr = r_instr_with_regs::<OR>(1, 2, 3);
        let cycle = Cycle::OR(make_r_cycle(instr, rs1, rs2, 0, rs1 | rs2));

        let idx = compute_lookup_index(&cycle);
        assert_eq!(idx, interleave_bits(rs1, rs2));
    }

    #[test]
    fn rd_inc_computed_correctly() {
        let instr = r_instr_with_regs::<ADD>(1, 2, 3);
        let cycle = Cycle::ADD(make_r_cycle(instr, 100, 200, 50, 300));
        let trace = [cycle];
        let prep = BytecodePreprocessing::new(&trace);
        let cd = cycle_to_cycle_data(&cycle, &prep);
        assert_eq!(cd.rd_inc, 300i128 - 50i128);
    }

    #[test]
    fn padding_to_power_of_two() {
        let trace = vec![Cycle::NoOp; 3];
        let prep = BytecodePreprocessing::new(&trace);
        let data = trace_to_cycle_data(&trace, &prep);
        assert_eq!(data.len(), 4); // 3 → next power of two = 4
    }
}
