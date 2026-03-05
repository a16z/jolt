//! RV64I comparison instructions that write 1 or 0 to the destination register.

use crate::macros::define_instruction;
use crate::opcodes;

define_instruction!(
    /// RV64I SLT: set if less than (signed). `rd = (rs1 < rs2) ? 1 : 0`.
    Slt, opcodes::SLT, "SLT",
    |x, y| u64::from((x as i64) < (y as i64))
);

define_instruction!(
    /// RV64I SLTI: set if less than immediate (signed).
    SltI, opcodes::SLTI, "SLTI",
    |x, y| u64::from((x as i64) < (y as i64))
);

define_instruction!(
    /// RV64I SLTU: set if less than (unsigned). `rd = (rs1 < rs2) ? 1 : 0`.
    SltU, opcodes::SLTU, "SLTU",
    |x, y| u64::from(x < y)
);

define_instruction!(
    /// RV64I SLTIU: set if less than immediate (unsigned).
    SltIU, opcodes::SLTIU, "SLTIU",
    |x, y| u64::from(x < y)
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn slt_signed() {
        assert_eq!(Slt.execute((-1i64) as u64, 1), 1);
        assert_eq!(Slt.execute(1, (-1i64) as u64), 0);
        assert_eq!(Slt.execute(5, 5), 0);
    }

    #[test]
    fn sltu_unsigned() {
        assert_eq!(SltU.execute(1, 2), 1);
        assert_eq!(SltU.execute(2, 1), 0);
        // -1 as u64 is MAX, so it's greater
        assert_eq!(SltU.execute((-1i64) as u64, 1), 0);
    }

    #[test]
    fn immediate_variants_match() {
        assert_eq!(Slt.execute(3, 5), SltI.execute(3, 5));
        assert_eq!(SltU.execute(3, 5), SltIU.execute(3, 5));
    }
}
