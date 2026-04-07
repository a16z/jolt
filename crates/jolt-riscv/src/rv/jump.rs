//! RV64I jump instructions.

define_instruction!(
    /// RV64I JAL: jump and link. `rd = PC + 4; PC = PC + imm`.
    /// The execute function computes the jump target `PC + imm`.
    Jal, "JAL",
    |x, y| x.wrapping_add(y),
    circuit: [AddOperands, Jump],
    instruction: [LeftOperandIsPC, RightOperandIsImm],
);

define_instruction!(
    /// RV64I JALR: jump and link register. `rd = PC + 4; PC = (rs1 + imm) & !1`.
    /// The execute function computes the jump target `(rs1 + imm) & !1`.
    Jalr, "JALR",
    |x, y| x.wrapping_add(y) & !1,
    circuit: [AddOperands, Jump],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn jal_basic() {
        assert_eq!(Jal.execute(0x1000, 0x100), 0x1100);
    }

    #[test]
    fn jal_wrapping() {
        assert_eq!(Jal.execute(u64::MAX, 1), 0);
    }

    #[test]
    fn jalr_aligns() {
        // Result should clear bit 0
        assert_eq!(Jalr.execute(0x1001, 0x100), 0x1100); // 0x1101 & !1 = 0x1100
        assert_eq!(Jalr.execute(0x1000, 0x101), 0x1100); // 0x1101 & !1 = 0x1100
    }

    #[test]
    fn jalr_even_unchanged() {
        assert_eq!(Jalr.execute(0x1000, 0x100), 0x1100);
    }
}
