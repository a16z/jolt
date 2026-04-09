//! Virtual assertion instructions used by the Jolt VM for constraint checking.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// Virtual ASSERT_EQ: returns 1 if operands are equal, 0 otherwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AssertEq;

impl Instruction for AssertEq {
    #[inline]
    fn name(&self) -> &'static str {
        "ASSERT_EQ"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        u64::from(x == y)
    }
}

/// Virtual ASSERT_LTE: returns 1 if `x <= y` (unsigned), 0 otherwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AssertLte;

impl Instruction for AssertLte {
    #[inline]
    fn name(&self) -> &'static str {
        "ASSERT_LTE"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        u64::from(x <= y)
    }
}

/// Virtual ASSERT_VALID_DIV0: validates `(divisor, quotient)` for division-by-zero handling.
/// Returns 1 if the divisor is nonzero, or if the divisor is 0 and the quotient is MAX.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AssertValidDiv0;

impl Instruction for AssertValidDiv0 {
    #[inline]
    fn name(&self) -> &'static str {
        "ASSERT_VALID_DIV0"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        if x == 0 {
            u64::from(y == u64::MAX)
        } else {
            1
        }
    }
}

/// Virtual ASSERT_VALID_UNSIGNED_REMAINDER: validates unsigned remainder.
/// Returns 1 if divisor is 0 or remainder < divisor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AssertValidUnsignedRemainder;

impl Instruction for AssertValidUnsignedRemainder {
    #[inline]
    fn name(&self) -> &'static str {
        "ASSERT_VALID_UNSIGNED_REMAINDER"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        if y == 0 {
            1
        } else {
            u64::from(x < y)
        }
    }
}

/// Virtual ASSERT_MULU_NO_OVERFLOW: checks unsigned multiply doesn't overflow.
/// Returns 1 if the upper XLEN bits of `x * y` are all zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AssertMulUNoOverflow;

impl Instruction for AssertMulUNoOverflow {
    #[inline]
    fn name(&self) -> &'static str {
        "ASSERT_MULU_NO_OVERFLOW"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let product = (x as u128) * (y as u128);
        u64::from((product >> 64) == 0)
    }
}

/// Virtual ASSERT_WORD_ALIGNMENT: checks whether `rs1 + imm` is 4-byte aligned.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct AssertWordAlignment;

impl Instruction for AssertWordAlignment {
    #[inline]
    fn name(&self) -> &'static str {
        "ASSERT_WORD_ALIGNMENT"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        u64::from(x.wrapping_add(y).is_multiple_of(4))
    }
}

/// Virtual ASSERT_HALFWORD_ALIGNMENT: checks whether `rs1 + imm` is 2-byte aligned.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, Assert)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct AssertHalfwordAlignment;

impl Instruction for AssertHalfwordAlignment {
    #[inline]
    fn name(&self) -> &'static str {
        "ASSERT_HALFWORD_ALIGNMENT"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        u64::from(x.wrapping_add(y).is_multiple_of(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assert_eq_basic() {
        assert_eq!(AssertEq.execute(5, 5), 1);
        assert_eq!(AssertEq.execute(5, 6), 0);
    }

    #[test]
    fn assert_lte_basic() {
        assert_eq!(AssertLte.execute(3, 5), 1);
        assert_eq!(AssertLte.execute(5, 5), 1);
        assert_eq!(AssertLte.execute(6, 5), 0);
    }

    #[test]
    fn assert_valid_div0() {
        assert_eq!(AssertValidDiv0.execute(0, u64::MAX), 1);
        assert_eq!(AssertValidDiv0.execute(0, 42), 0);
        assert_eq!(AssertValidDiv0.execute(3, 42), 1);
    }

    #[test]
    fn assert_valid_unsigned_remainder() {
        assert_eq!(AssertValidUnsignedRemainder.execute(2, 5), 1);
        assert_eq!(AssertValidUnsignedRemainder.execute(5, 5), 0);
        assert_eq!(AssertValidUnsignedRemainder.execute(0, 0), 1);
    }

    #[test]
    fn assert_word_alignment() {
        assert_eq!(AssertWordAlignment.execute(0, 0), 1);
        assert_eq!(AssertWordAlignment.execute(4, 0), 1);
        assert_eq!(AssertWordAlignment.execute(2, 2), 1);
        assert_eq!(AssertWordAlignment.execute(4, 2), 0);
    }

    #[test]
    fn assert_halfword_alignment() {
        assert_eq!(AssertHalfwordAlignment.execute(0, 0), 1);
        assert_eq!(AssertHalfwordAlignment.execute(2, 0), 1);
        assert_eq!(AssertHalfwordAlignment.execute(1, 1), 1);
        assert_eq!(AssertHalfwordAlignment.execute(2, 1), 0);
    }
}
