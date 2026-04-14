//! Boolean flags controlling instruction behavior in R1CS constraints and witness generation.
//!
//! [`CircuitFlags`] are embedded in Jolt's R1CS constraints (the "opflags" from the Jolt paper).
//! [`InstructionFlags`] control witness generation and operand routing but are not
//! directly constrained.
//!
//! Every instruction implements the [`Flags`] trait, returning its static flag
//! configuration via [`CircuitFlagSet`] and [`InstructionFlagSet`] packed bitfields.

use std::ops::Index;

use strum::EnumCount;

/// Boolean flags used in Jolt's R1CS constraints (`opflags` in the Jolt paper).
///
/// Note: the flags below deviate somewhat from those described in Appendix A.1
/// of the Jolt paper.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, EnumCount)]
#[repr(u8)]
pub enum CircuitFlags {
    /// First lookup operand is the sum of the two instruction operands.
    AddOperands,
    /// First lookup operand is the difference of the two instruction operands.
    SubtractOperands,
    /// First lookup operand is the product of the two instruction operands.
    MultiplyOperands,
    /// Instruction is a load (e.g. `LW`).
    Load,
    /// Instruction is a store (e.g. `SW`).
    Store,
    /// Instruction is a jump (e.g. `JAL`, `JALR`).
    Jump,
    /// Lookup output is stored in `rd` at the end of the step.
    WriteLookupOutputToRD,
    /// Instruction is "virtual" (Section 6.1 of the Jolt paper).
    VirtualInstruction,
    /// Instruction is an assert (Section 6.1.1 of the Jolt paper).
    Assert,
    /// PC unchanged during inline virtual sequences.
    DoNotUpdateUnexpandedPC,
    /// Is a (virtual) advice instruction.
    Advice,
    /// Is a compressed instruction (UnexpandedPc += 2 instead of 4).
    IsCompressed,
    /// First instruction in a virtual sequence.
    IsFirstInSequence,
    /// Last instruction in a virtual sequence.
    IsLastInSequence,
}

/// Number of circuit flags.
pub const NUM_CIRCUIT_FLAGS: usize = CircuitFlags::COUNT;

/// Boolean flags that are NOT part of Jolt's R1CS constraints.
///
/// These control witness generation, operand routing, and auxiliary prover logic.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, EnumCount)]
#[repr(u8)]
pub enum InstructionFlags {
    /// First instruction operand is the program counter.
    LeftOperandIsPC,
    /// Second instruction operand is an immediate value.
    RightOperandIsImm,
    /// First instruction operand is RS1 register value.
    LeftOperandIsRs1Value,
    /// Second instruction operand is RS2 register value.
    RightOperandIsRs2Value,
    /// Instruction is a branch (e.g. `BEQ`, `BNE`).
    Branch,
    /// No-op instruction.
    IsNoop,
    /// Destination register index is nonzero.
    IsRdNotZero,
}

/// Number of instruction flags.
pub const NUM_INSTRUCTION_FLAGS: usize = InstructionFlags::COUNT;

/// Packed bitfield of [`CircuitFlags`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CircuitFlagSet(u16);

impl CircuitFlagSet {
    #[inline]
    pub fn set(self, flag: CircuitFlags) -> Self {
        Self(self.0 | (1 << flag as u16))
    }

    #[inline]
    pub fn get(self, flag: CircuitFlags) -> bool {
        self.0 & (1 << flag as u16) != 0
    }

    #[inline]
    pub fn bits(self) -> u16 {
        self.0
    }
}

impl Index<CircuitFlags> for CircuitFlagSet {
    type Output = bool;
    #[inline]
    fn index(&self, flag: CircuitFlags) -> &bool {
        if self.get(flag) {
            &true
        } else {
            &false
        }
    }
}

/// Packed bitfield of [`InstructionFlags`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct InstructionFlagSet(u8);

impl InstructionFlagSet {
    #[inline]
    pub fn set(self, flag: InstructionFlags) -> Self {
        Self(self.0 | (1 << flag as u8))
    }

    #[inline]
    pub fn get(self, flag: InstructionFlags) -> bool {
        self.0 & (1 << flag as u8) != 0
    }

    #[inline]
    pub fn bits(self) -> u8 {
        self.0
    }
}

impl Index<InstructionFlags> for InstructionFlagSet {
    type Output = bool;
    #[inline]
    fn index(&self, flag: InstructionFlags) -> &bool {
        if self.get(flag) {
            &true
        } else {
            &false
        }
    }
}

/// Static flag configuration for an instruction.
///
/// Every instruction struct implements this trait to declare which circuit
/// and instruction flags are set.
pub trait Flags {
    fn circuit_flags(&self) -> CircuitFlagSet;
    fn instruction_flags(&self) -> InstructionFlagSet;
}

/// Checks whether an instruction uses interleaved-bit operand encoding.
///
/// Instructions that combine operands (ADD, SUB, MUL) or use advice
/// set explicit operand-combination flags; all others use the default
/// interleaved-bit layout for lookup indices.
pub trait InterleavedBitsMarker {
    /// Returns `true` if neither `AddOperands`, `SubtractOperands`,
    /// `MultiplyOperands`, nor `Advice` is set.
    fn is_interleaved_operands(&self) -> bool;
}

impl InterleavedBitsMarker for CircuitFlagSet {
    #[inline]
    fn is_interleaved_operands(&self) -> bool {
        !self.get(CircuitFlags::AddOperands)
            && !self.get(CircuitFlags::SubtractOperands)
            && !self.get(CircuitFlags::MultiplyOperands)
            && !self.get(CircuitFlags::Advice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_flags_count_matches_enum() {
        assert_eq!(
            CircuitFlags::IsLastInSequence as usize + 1,
            NUM_CIRCUIT_FLAGS
        );
    }

    #[test]
    fn instruction_flags_count_matches_enum() {
        assert_eq!(
            InstructionFlags::IsRdNotZero as usize + 1,
            NUM_INSTRUCTION_FLAGS
        );
    }

    #[test]
    fn set_and_get() {
        let flags = CircuitFlagSet::default().set(CircuitFlags::Load);
        assert!(flags[CircuitFlags::Load]);
        assert!(!flags[CircuitFlags::Store]);
    }

    #[test]
    fn interleaved_default() {
        let flags = CircuitFlagSet::default();
        assert!(flags.is_interleaved_operands());
    }

    #[test]
    fn add_operands_not_interleaved() {
        let flags = CircuitFlagSet::default().set(CircuitFlags::AddOperands);
        assert!(!flags.is_interleaved_operands());
    }
}
