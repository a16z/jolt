//! Virtual arithmetic instructions used internally by the Jolt VM.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// Virtual POW2: computes `2^rs1` using the low 6 bits of `rs1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct Pow2;

impl Instruction for Pow2 {
    #[inline]
    fn name(&self) -> &'static str {
        "POW2"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        1u64 << (x & 63)
    }
}

/// Virtual POW2I: computes `2^imm` with immediate exponent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(RightOperandIsImm)]
pub struct Pow2I;

impl Instruction for Pow2I {
    #[inline]
    fn name(&self) -> &'static str {
        "POW2I"
    }

    #[inline]
    fn execute(&self, _x: u64, y: u64) -> u64 {
        1u64 << (y & 63)
    }
}

/// Virtual POW2W: computes `2^(rs1 mod 32)` for 32-bit mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value)]
pub struct Pow2W;

impl Instruction for Pow2W {
    #[inline]
    fn name(&self) -> &'static str {
        "POW2W"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        1u64 << (x & 31)
    }
}

/// Virtual POW2IW: computes `2^(imm mod 32)` for 32-bit immediate mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(RightOperandIsImm)]
pub struct Pow2IW;

impl Instruction for Pow2IW {
    #[inline]
    fn name(&self) -> &'static str {
        "POW2IW"
    }

    #[inline]
    fn execute(&self, _x: u64, y: u64) -> u64 {
        1u64 << (y & 31)
    }
}

/// Virtual MULI: multiply by immediate. `rd = rs1 * imm`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct MulI;

impl Instruction for MulI {
    #[inline]
    fn name(&self) -> &'static str {
        "MULI"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x.wrapping_mul(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow2_basic() {
        assert_eq!(Pow2.execute(0, 0), 1);
        assert_eq!(Pow2.execute(10, 0), 1024);
        assert_eq!(Pow2.execute(63, 0), 1 << 63);
    }

    #[test]
    fn pow2_masks_exponent() {
        assert_eq!(Pow2.execute(64, 0), 1);
    }

    #[test]
    fn pow2w_basic() {
        assert_eq!(Pow2W.execute(0, 0), 1);
        assert_eq!(Pow2W.execute(31, 0), 1 << 31);
    }

    #[test]
    fn pow2w_masks_to_32() {
        assert_eq!(Pow2W.execute(32, 0), 1);
    }

    #[test]
    fn immediate_pow2_variants_use_y() {
        assert_eq!(Pow2I.execute(0, 10), 1024);
        assert_eq!(Pow2IW.execute(0, 31), 1 << 31);
    }

    #[test]
    fn muli_basic() {
        assert_eq!(MulI.execute(6, 7), 42);
        assert_eq!(MulI.execute(u64::MAX, 2), u64::MAX - 1);
    }
}
