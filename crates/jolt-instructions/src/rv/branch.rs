//! RV64I conditional branch instructions.
//!
//! Each returns 1 if the branch condition is true, 0 otherwise.
//! The actual PC update is handled by the VM, not the instruction itself.

use crate::opcodes;

define_instruction!(
    /// RV64I BEQ: branch if equal. Returns 1 when `rs1 == rs2`.
    Beq, opcodes::BEQ, "BEQ",
    |x, y| u64::from(x == y),
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch],
    table: Equal,
);

define_instruction!(
    /// RV64I BNE: branch if not equal. Returns 1 when `rs1 != rs2`.
    Bne, opcodes::BNE, "BNE",
    |x, y| u64::from(x != y),
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch],
    table: NotEqual,
);

define_instruction!(
    /// RV64I BLT: branch if less than (signed).
    Blt, opcodes::BLT, "BLT",
    |x, y| u64::from((x as i64) < (y as i64)),
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch],
    table: SignedLessThan,
);

define_instruction!(
    /// RV64I BGE: branch if greater than or equal (signed).
    Bge, opcodes::BGE, "BGE",
    |x, y| u64::from((x as i64) >= (y as i64)),
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch],
    table: SignedGreaterThanEqual,
);

define_instruction!(
    /// RV64I BLTU: branch if less than (unsigned).
    BltU, opcodes::BLTU, "BLTU",
    |x, y| u64::from(x < y),
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch],
    table: UnsignedLessThan,
);

define_instruction!(
    /// RV64I BGEU: branch if greater than or equal (unsigned).
    BgeU, opcodes::BGEU, "BGEU",
    |x, y| u64::from(x >= y),
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch],
    table: UnsignedGreaterThanEqual,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn beq_bne() {
        assert_eq!(Beq.execute(5, 5), 1);
        assert_eq!(Beq.execute(5, 6), 0);
        assert_eq!(Bne.execute(5, 5), 0);
        assert_eq!(Bne.execute(5, 6), 1);
    }

    #[test]
    fn blt_bge_signed() {
        let neg1 = (-1i64) as u64;
        assert_eq!(Blt.execute(neg1, 1), 1);
        assert_eq!(Blt.execute(1, neg1), 0);
        assert_eq!(Bge.execute(neg1, 1), 0);
        assert_eq!(Bge.execute(1, neg1), 1);
        assert_eq!(Bge.execute(5, 5), 1);
    }

    #[test]
    fn bltu_bgeu_unsigned() {
        assert_eq!(BltU.execute(1, 2), 1);
        assert_eq!(BltU.execute(2, 1), 0);
        assert_eq!(BgeU.execute(2, 1), 1);
        assert_eq!(BgeU.execute(1, 2), 0);
        assert_eq!(BgeU.execute(3, 3), 1);
    }
}
