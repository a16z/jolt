//! Virtual assertion instructions used by the Jolt VM for constraint checking.

use crate::opcodes;

define_instruction!(
    /// Virtual ASSERT_EQ: returns 1 if operands are equal, 0 otherwise.
    AssertEq, opcodes::ASSERT_EQ, "ASSERT_EQ",
    |x, y| u64::from(x == y),
    circuit: [Assert],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: Equal,
);

define_instruction!(
    /// Virtual ASSERT_LTE: returns 1 if `x <= y` (unsigned), 0 otherwise.
    AssertLte, opcodes::ASSERT_LTE, "ASSERT_LTE",
    |x, y| u64::from(x <= y),
    circuit: [Assert],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: UnsignedLessThanEqual,
);

define_instruction!(
    /// Virtual ASSERT_VALID_DIV0: validates division-by-zero result.
    /// Returns 1 if divisor is nonzero, or if divisor is 0 and quotient is MAX.
    AssertValidDiv0, opcodes::VIRTUAL_ASSERT_VALID_DIV0, "ASSERT_VALID_DIV0",
    |x, y| {
        if y == 0 { u64::from(x == u64::MAX) } else { 1 }
    },
    circuit: [Assert],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: ValidDiv0,
);

define_instruction!(
    /// Virtual ASSERT_VALID_UNSIGNED_REMAINDER: validates unsigned remainder.
    /// Returns 1 if divisor is 0 or remainder < divisor.
    AssertValidUnsignedRemainder, opcodes::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER, "ASSERT_VALID_UNSIGNED_REMAINDER",
    |x, y| {
        if y == 0 { 1 } else { u64::from(x < y) }
    },
    circuit: [Assert],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: ValidUnsignedRemainder,
);

define_instruction!(
    /// Virtual ASSERT_MULU_NO_OVERFLOW: checks unsigned multiply doesn't overflow.
    /// Returns 1 if the upper XLEN bits of `x * y` are all zero.
    AssertMulUNoOverflow, opcodes::VIRTUAL_ASSERT_MULU_NO_OVERFLOW, "ASSERT_MULU_NO_OVERFLOW",
    |x, y| {
        let product = (x as u128) * (y as u128);
        u64::from((product >> 64) == 0)
    },
    circuit: [Assert],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: MulUNoOverflow,
);

define_instruction!(
    /// Virtual ASSERT_WORD_ALIGNMENT: checks value is 4-byte aligned.
    AssertWordAlignment, opcodes::VIRTUAL_ASSERT_WORD_ALIGNMENT, "ASSERT_WORD_ALIGNMENT",
    |x, _y| u64::from(x.is_multiple_of(4)),
    circuit: [Assert],
    instruction: [LeftOperandIsRs1Value],
    table: WordAlignment,
);

define_instruction!(
    /// Virtual ASSERT_HALFWORD_ALIGNMENT: checks value is 2-byte aligned.
    AssertHalfwordAlignment, opcodes::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT, "ASSERT_HALFWORD_ALIGNMENT",
    |x, _y| u64::from(x.is_multiple_of(2)),
    circuit: [Assert],
    instruction: [LeftOperandIsRs1Value],
    table: HalfwordAlignment,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

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
        assert_eq!(AssertValidDiv0.execute(u64::MAX, 0), 1);
        assert_eq!(AssertValidDiv0.execute(42, 0), 0);
        assert_eq!(AssertValidDiv0.execute(42, 3), 1);
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
        assert_eq!(AssertWordAlignment.execute(3, 0), 0);
    }

    #[test]
    fn assert_halfword_alignment() {
        assert_eq!(AssertHalfwordAlignment.execute(0, 0), 1);
        assert_eq!(AssertHalfwordAlignment.execute(2, 0), 1);
        assert_eq!(AssertHalfwordAlignment.execute(1, 0), 0);
    }
}
