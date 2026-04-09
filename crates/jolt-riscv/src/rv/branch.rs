//! RV64I conditional branch instructions.
//!
//! Each returns 1 if the branch condition is true, 0 otherwise.
//! The actual PC update is handled by the VM, not the instruction itself.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// RV64I BEQ: branch if equal. Returns 1 when `rs1 == rs2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct Beq;

impl Instruction for Beq {
    #[inline]
    fn name(&self) -> &'static str {
        "BEQ"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        u64::from(x == y)
    }
}

/// RV64I BNE: branch if not equal. Returns 1 when `rs1 != rs2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct Bne;

impl Instruction for Bne {
    #[inline]
    fn name(&self) -> &'static str {
        "BNE"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        u64::from(x != y)
    }
}

/// RV64I BLT: branch if less than (signed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct Blt;

impl Instruction for Blt {
    #[inline]
    fn name(&self) -> &'static str {
        "BLT"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        u64::from((x as i64) < (y as i64))
    }
}

/// RV64I BGE: branch if greater than or equal (signed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct Bge;

impl Instruction for Bge {
    #[inline]
    fn name(&self) -> &'static str {
        "BGE"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        u64::from((x as i64) >= (y as i64))
    }
}

/// RV64I BLTU: branch if less than (unsigned).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct BltU;

impl Instruction for BltU {
    #[inline]
    fn name(&self) -> &'static str {
        "BLTU"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        u64::from(x < y)
    }
}

/// RV64I BGEU: branch if greater than or equal (unsigned).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch)]
pub struct BgeU;

impl Instruction for BgeU {
    #[inline]
    fn name(&self) -> &'static str {
        "BGEU"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        u64::from(x >= y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
