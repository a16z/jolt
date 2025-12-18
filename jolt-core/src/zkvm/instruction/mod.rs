use std::ops::{Index, IndexMut};

use allocative::Allocative;
use common::constants::XLEN;
use strum::EnumCount;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter, FromRepr};
use tracer::instruction::{Cycle, Instruction};

use crate::utils::interleave_bits;

use super::lookup_table::LookupTables;

pub trait InstructionLookup<const XLEN: usize> {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>>;
}

pub trait LookupQuery<const XLEN: usize> {
    /// Returns a tuple of the instruction's inputs. If the instruction has only one input,
    /// one of the tuple values will be 0.
    fn to_instruction_inputs(&self) -> (u64, i128);

    /// Returns a tuple of the instruction's lookup operands. By default, these are the
    /// same as the instruction inputs returned by `to_instruction_inputs`, but in some cases
    /// (e.g. ADD, MUL) the instruction inputs are combined to form a single lookup operand.
    fn to_lookup_operands(&self) -> (u64, u128) {
        let (x, y) = self.to_instruction_inputs();
        (x, (y as u64) as u128)
    }

    /// Converts this instruction's operands into a lookup index (as used in sparse-dense Shout).
    /// By default, interleaves the two bits of the two operands together.
    fn to_lookup_index(&self) -> u128 {
        let (x, y) = LookupQuery::<XLEN>::to_lookup_operands(self);
        interleave_bits(x, y as u64)
    }

    /// Computes the output lookup entry for this instruction as a u64.
    fn to_lookup_output(&self) -> u64;
}

/// Boolean flags used in Jolt's R1CS constraints (`opflags` in the Jolt paper).
/// Note that the flags below deviate somewhat from those described in Appendix A.1
/// of the Jolt paper.
#[derive(
    Clone,
    Copy,
    Debug,
    Hash,
    PartialEq,
    Eq,
    EnumCountMacro,
    EnumIter,
    PartialOrd,
    Ord,
    Allocative,
    FromRepr,
)]
#[repr(u8)]
pub enum CircuitFlags {
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
    /// 1 if the lookup output is to be stored in `rd` at the end of the step.
    WriteLookupOutputToRD,
    /// 1 if the instruction is "virtual", as defined in Section 6.1 of the Jolt paper.
    VirtualInstruction,
    /// 1 if the instruction is an assert, as defined in Section 6.1.1 of the Jolt paper.
    Assert,
    /// Used in inline sequences; the program counter should be the same for the full sequence.
    DoNotUpdateUnexpandedPC,
    /// Is (virtual) advice instruction
    Advice,
    /// Is a compressed instruction (i.e. increase UnexpandedPc by 2 only)
    IsCompressed,
    /// Is instruction the first in a virtual sequence
    IsFirstInSequence,
}

/// Boolean flags that are not part of Jolt's R1CS constraints
#[derive(
    Clone,
    Copy,
    Debug,
    Hash,
    PartialEq,
    Eq,
    EnumCountMacro,
    EnumIter,
    PartialOrd,
    Ord,
    Allocative,
    FromRepr,
)]
#[repr(u8)]
pub enum InstructionFlags {
    /// 1 if the first instruction operand is the program counter; 0 otherwise.
    LeftOperandIsPC,
    /// 1 if the second instruction operand is `imm`; 0 otherwise.
    RightOperandIsImm,
    /// 1 if the first instruction operand is RS1 value; 0 otherwise.
    LeftOperandIsRs1Value,
    /// 1 if the first instruction operand is RS2 value; 0 otherwise.
    RightOperandIsRs2Value,
    /// 1 if the instruction is a branch (i.e. `BEQ`, `BNE`, etc.)
    Branch,
    /// Is noop instruction
    IsNoop,
    /// Is Rd index 0
    IsRdNotZero,
}

pub const NUM_CIRCUIT_FLAGS: usize = CircuitFlags::COUNT;
pub const NUM_INSTRUCTION_FLAGS: usize = InstructionFlags::COUNT;

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

impl Index<InstructionFlags> for [bool; NUM_INSTRUCTION_FLAGS] {
    type Output = bool;
    fn index(&self, index: InstructionFlags) -> &bool {
        &self[index as usize]
    }
}

impl IndexMut<InstructionFlags> for [bool; NUM_INSTRUCTION_FLAGS] {
    fn index_mut(&mut self, index: InstructionFlags) -> &mut bool {
        &mut self[index as usize]
    }
}

pub trait Flags {
    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS];
    fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS];
}

macro_rules! define_rv32im_trait_impls {
    (
        instructions: [$($instr:ident),* $(,)?]
    ) => {
        impl InstructionLookup<XLEN> for Instruction {
            fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
                match self {
                    Instruction::NoOp => None,
                    $(
                        Instruction::$instr(instr) => instr.lookup_table(),
                    )*
                    Instruction::UNIMPL => None,
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }
        }

        impl Flags for Instruction {
            fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
                match self {
                    Instruction::NoOp => {
                        let mut flags = [false; NUM_CIRCUIT_FLAGS];
                        flags[CircuitFlags::DoNotUpdateUnexpandedPC] = true;
                        flags
                    },
                    $(
                        Instruction::$instr(instr) => instr.circuit_flags(),
                    )*
                    Instruction::UNIMPL => [false; NUM_CIRCUIT_FLAGS],
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }

            fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS] {
                match self {
                    Instruction::NoOp => {
                        let mut flags = [false; NUM_INSTRUCTION_FLAGS];
                        flags[InstructionFlags::IsNoop] = true;
                        flags
                    },
                    $(
                        Instruction::$instr(instr) => instr.instruction_flags(),
                    )*
                    Instruction::UNIMPL => [false; NUM_INSTRUCTION_FLAGS],
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }
        }

        impl<const XLEN: usize> InstructionLookup<XLEN> for Cycle {
            fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
                match self {
                    Cycle::NoOp => None,
                    $(
                        Cycle::$instr(cycle) => cycle.instruction.lookup_table(),
                    )*
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }
        }

        impl<const XLEN: usize> LookupQuery<XLEN> for Cycle {
            fn to_instruction_inputs(&self) -> (u64, i128) {
                match self {
                    Cycle::NoOp => (0, 0),
                    $(
                        Cycle::$instr(cycle) => LookupQuery::<XLEN>::to_instruction_inputs(cycle),
                    )*
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }

            fn to_lookup_index(&self) -> u128 {
                match self {
                    Cycle::NoOp => 0,
                    $(
                        Cycle::$instr(cycle) => LookupQuery::<XLEN>::to_lookup_index(cycle),
                    )*
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }

            fn to_lookup_operands(&self) -> (u64, u128) {
                match self {
                    Cycle::NoOp => (0, 0),
                    $(
                        Cycle::$instr(cycle) => LookupQuery::<XLEN>::to_lookup_operands(cycle),
                    )*
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }

            fn to_lookup_output(&self) -> u64 {
                match self {
                    Cycle::NoOp => 0,
                    $(
                        Cycle::$instr(cycle) => LookupQuery::<XLEN>::to_lookup_output(cycle),
                    )*
                    _ => panic!("Unexpected instruction: {:?}", self),
                }
            }
        }
    };
}

define_rv32im_trait_impls! {
    instructions: [
        ADD, ADDI, AND, ANDI, ANDN, AUIPC, BEQ, BGE, BGEU, BLT, BLTU, BNE,
        ECALL, FENCE, JAL, JALR, LUI, LD, MUL, MULHU, OR, ORI,
        SLT, SLTI, SLTIU, SLTU, SUB, SD, XOR, XORI,
        VirtualAdvice, VirtualAssertEQ, VirtualAssertHalfwordAlignment,
        VirtualAssertWordAlignment, VirtualAssertLTE,
        VirtualAssertValidDiv0, VirtualAssertValidUnsignedRemainder,
        VirtualChangeDivisor, VirtualChangeDivisorW, VirtualAssertMulUNoOverflow,
        VirtualZeroExtendWord, VirtualSignExtendWord, VirtualMovsign, VirtualMULI, VirtualPow2,
        VirtualPow2I, VirtualPow2W, VirtualPow2IW, VirtualRev8W, VirtualShiftRightBitmask, VirtualShiftRightBitmaskI,
        VirtualROTRI, VirtualROTRIW,
        VirtualSRA, VirtualSRAI, VirtualSRL, VirtualSRLI,
        VirtualXORROT32, VirtualXORROT24, VirtualXORROT16, VirtualXORROT63,
        VirtualXORROTW16, VirtualXORROTW12, VirtualXORROTW8, VirtualXORROTW7
    ]
}

pub mod add;
pub mod addi;
pub mod and;
pub mod andi;
pub mod andn;
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
pub mod ld;
pub mod lui;
pub mod mul;
pub mod mulhu;
pub mod or;
pub mod ori;
pub mod sd;
pub mod slt;
pub mod slti;
pub mod sltiu;
pub mod sltu;
pub mod sub;
pub mod virtual_advice;
pub mod virtual_assert_eq;
pub mod virtual_assert_halfword_alignment;
pub mod virtual_assert_lte;
pub mod virtual_assert_mulu_no_overflow;
pub mod virtual_assert_valid_div0;
pub mod virtual_assert_valid_unsigned_remainder;
pub mod virtual_assert_word_alignment;
pub mod virtual_change_divisor;
pub mod virtual_change_divisor_w;
pub mod virtual_movsign;
pub mod virtual_muli;
pub mod virtual_pow2;
pub mod virtual_pow2i;
pub mod virtual_pow2iw;
pub mod virtual_pow2w;
pub mod virtual_rev8w;
pub mod virtual_rotri;
pub mod virtual_rotriw;
pub mod virtual_shift_right_bitmask;
pub mod virtual_shift_right_bitmaski;
pub mod virtual_sign_extend_word;
pub mod virtual_sra;
pub mod virtual_srai;
pub mod virtual_srl;
pub mod virtual_srli;
pub mod virtual_xor_rot;
pub mod virtual_xor_rotw;
pub mod virtual_zero_extend_word;
pub mod xor;
pub mod xori;

#[cfg(test)]
pub mod test;
