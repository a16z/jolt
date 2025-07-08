use std::ops::{Index, IndexMut};

use strum::EnumCount;
use strum_macros::EnumCount as EnumCountMacro;
use tracer::instruction::{RV32IMCycle, RV32IMInstruction};

use crate::utils::interleave_bits;

use super::lookup_table::LookupTables;

pub trait InstructionLookup<const WORD_SIZE: usize> {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>>;
}

pub trait LookupQuery<const WORD_SIZE: usize> {
    /// Returns a tuple of the instruction's inputs. If the instruction has only one input,
    /// one of the tuple values will be 0.
    fn to_instruction_inputs(&self) -> (u64, i64);

    /// Returns a tuple of the instruction's lookup operands. By default, these are the
    /// same as the instruction inputs returned by `to_instruction_inputs`, but in some cases
    /// (e.g. ADD, MUL) the instruction inputs are combined to form a single lookup operand.
    fn to_lookup_operands(&self) -> (u64, u64) {
        let (x, y) = self.to_instruction_inputs();
        (x, y as u64)
    }

    /// Converts this instruction's operands into a lookup index (as used in sparse-dense Shout).
    /// By default, interleaves the two bits of the two operands together.
    fn to_lookup_index(&self) -> u64 {
        let (x, y) = LookupQuery::<WORD_SIZE>::to_lookup_operands(self);
        interleave_bits(x as u32, y as u32)
    }

    /// Computes the output lookup entry for this instruction as a u64.
    fn to_lookup_output(&self) -> u64;
}

/// Boolean flags used in Jolt's R1CS constraints (`opflags` in the Jolt paper).
/// Note that the flags below deviate somewhat from those described in Appendix A.1
/// of the Jolt paper.
#[derive(Clone, Copy, Debug, PartialEq, EnumCountMacro)]
pub enum CircuitFlags {
    /// 1 if the first instruction operand is the program counter; 0 otherwise.
    LeftOperandIsPC,
    /// 1 if the second instruction operand is `imm`; 0 otherwise.
    RightOperandIsImm,
    /// 1 if the first instruction operand is RS1 value; 0 otherwise.
    LeftOperandIsRs1Value,
    /// 1 if the first instruction operand is RS2 value; 0 otherwise.
    RightOperandIsRs2Value,
    /// 1 if the first lookup operand is the sum of the two instruction operands.
    AddOperands,
    /// 1 if the first lookup operand is the difference between the two instruction operands.
    SubtractOperands,
    /// 1 if the first lookup operand is the product of the two instruction operands.
    MultiplyOperands,
    /// 1 if the instruction is a load (i.e. `LW`)
    Load,
    /// 1 if the instruction is a store (i.e. `SW`)
    Store,
    /// 1 if the instruction is a jump (i.e. `JAL`, `JALR`)
    Jump,
    /// 1 if the instruction is a branch (i.e. `BEQ`, `BNE`, etc.)
    Branch,
    /// 1 if the lookup output is to be stored in `rd` at the end of the step.
    WriteLookupOutputToRD,
    /// 1 if the instruction is "inline", as defined in Section 6.1 of the Jolt paper.
    InlineSequenceInstruction,
    /// 1 if the instruction is an assert, as defined in Section 6.1.1 of the Jolt paper.
    Assert,
    /// Used in virtual sequences; the program counter should be the same for the full sequence.
    DoNotUpdateUnexpandedPC,
    /// Is (virtual) advice instruction
    Advice,
}

pub const NUM_CIRCUIT_FLAGS: usize = CircuitFlags::COUNT;

pub trait InterleavedBitsMarker {
    fn is_interleaved_operands(&self) -> bool;
}

impl InterleavedBitsMarker for [bool; NUM_CIRCUIT_FLAGS] {
    fn is_interleaved_operands(&self) -> bool {
        !self[CircuitFlags::AddOperands]
            && !self[CircuitFlags::SubtractOperands]
            && !self[CircuitFlags::MultiplyOperands]
            && !self[CircuitFlags::Advice]
    }
}

impl Index<CircuitFlags> for [bool; NUM_CIRCUIT_FLAGS] {
    type Output = bool;
    fn index(&self, index: CircuitFlags) -> &bool {
        &self[index as usize]
    }
}

impl IndexMut<CircuitFlags> for [bool; NUM_CIRCUIT_FLAGS] {
    fn index_mut(&mut self, index: CircuitFlags) -> &mut bool {
        &mut self[index as usize]
    }
}

pub trait InstructionFlags {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS];
}

macro_rules! define_rv32im_trait_impls {
    (
        instructions: [$($instr:ident),* $(,)?]
    ) => {
        impl InstructionLookup<32> for RV32IMInstruction {
            fn lookup_table(&self) -> Option<LookupTables<32>> {
                match self {
                    RV32IMInstruction::NoOp(_) => None,
                    $(
                        RV32IMInstruction::$instr(instr) => instr.lookup_table(),
                    )*
                    RV32IMInstruction::UNIMPL => None,
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }
        }

        impl InstructionFlags for RV32IMInstruction {
            fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
                match self {
                    RV32IMInstruction::NoOp(_) => [false; NUM_CIRCUIT_FLAGS],
                    $(
                        RV32IMInstruction::$instr(instr) => instr.circuit_flags(),
                    )*
                    RV32IMInstruction::UNIMPL => [false; NUM_CIRCUIT_FLAGS],
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }
        }

        impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for RV32IMCycle {
            fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
                match self {
                    RV32IMCycle::NoOp(_) => None,
                    $(
                        RV32IMCycle::$instr(cycle) => cycle.instruction.lookup_table(),
                    )*
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }
        }

        impl<const WORD_SIZE: usize> LookupQuery<WORD_SIZE> for RV32IMCycle {
            fn to_instruction_inputs(&self) -> (u64, i64) {
                match self {
                    RV32IMCycle::NoOp(_) => (0, 0),
                    $(
                        RV32IMCycle::$instr(cycle) => LookupQuery::<WORD_SIZE>::to_instruction_inputs(cycle),
                    )*
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }

            fn to_lookup_index(&self) -> u64 {
                match self {
                    RV32IMCycle::NoOp(_) => 0,
                    $(
                        RV32IMCycle::$instr(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_index(cycle),
                    )*
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }

            fn to_lookup_operands(&self) -> (u64, u64) {
                match self {
                    RV32IMCycle::NoOp(_) => (0, 0),
                    $(
                        RV32IMCycle::$instr(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_operands(cycle),
                    )*
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }

            fn to_lookup_output(&self) -> u64 {
                match self {
                    RV32IMCycle::NoOp(_) => 0,
                    $(
                        RV32IMCycle::$instr(cycle) => LookupQuery::<WORD_SIZE>::to_lookup_output(cycle),
                    )*
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }
        }
    };
}

define_rv32im_trait_impls! {
    instructions: [
        ADD, ADDI, AND, ANDI, AUIPC, BEQ, BGE, BGEU, BLT, BLTU, BNE,
        ECALL, FENCE, JAL, JALR, LUI, LW, MUL, MULHU, OR, ORI,
        SLT, SLTI, SLTIU, SLTU, SUB, SW, XOR, XORI,
        VirtualAdvice, VirtualAssertEQ, VirtualAssertHalfwordAlignment, VirtualAssertLTE,
        VirtualAssertValidDiv0, VirtualAssertValidSignedRemainder, VirtualAssertValidUnsignedRemainder,
        VirtualMove, VirtualMovsign, VirtualMULI, VirtualPow2, VirtualPow2I,
        VirtualShiftRightBitmask, VirtualShiftRightBitmaskI, VirtualROTRI, VirtualROTLI,
        VirtualSRA, VirtualSRAI, VirtualSRL, VirtualSRLI
    ]
}

pub mod add;
pub mod addi;
pub mod and;
pub mod andi;
pub mod auipc;
pub mod beq;
pub mod bge;
pub mod bgeu;
pub mod blt;
pub mod bltu;
pub mod bne;
pub mod ecall;
pub mod fence;
pub mod jal;
pub mod jalr;
pub mod lui;
pub mod lw;
pub mod mul;
pub mod mulhu;
pub mod or;
pub mod ori;
pub mod slt;
pub mod slti;
pub mod sltiu;
pub mod sltu;
pub mod sub;
pub mod sw;
pub mod virtual_advice;
pub mod virtual_assert_eq;
pub mod virtual_assert_halfword_alignment;
pub mod virtual_assert_lte;
pub mod virtual_assert_valid_div0;
pub mod virtual_assert_valid_signed_remainder;
pub mod virtual_assert_valid_unsigned_remainder;
pub mod virtual_move;
pub mod virtual_movsign;
pub mod virtual_muli;
pub mod virtual_pow2;
pub mod virtual_pow2i;
pub mod virtual_rotli;
pub mod virtual_rotri;
pub mod virtual_shift_right_bitmask;
pub mod virtual_shift_right_bitmaski;
pub mod virtual_sra;
pub mod virtual_srai;
pub mod virtual_srl;
pub mod virtual_srli;
pub mod xor;
pub mod xori;

#[cfg(test)]
pub mod test;
