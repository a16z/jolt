//! RV64I comparison instructions that write 1 or 0 to the destination register.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// RV64I SLT: set if less than (signed). `rd = (rs1 < rs2) ? 1 : 0`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Slt;

impl Instruction for Slt {
    #[inline]
    fn name(&self) -> &'static str {
        "SLT"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        u64::from((x as i64) < (y as i64))
    }
}

/// RV64I SLTI: set if less than immediate (signed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SltI;

impl Instruction for SltI {
    #[inline]
    fn name(&self) -> &'static str {
        "SLTI"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        u64::from((x as i64) < (y as i64))
    }
}

/// RV64I SLTU: set if less than (unsigned). `rd = (rs1 < rs2) ? 1 : 0`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct SltU;

impl Instruction for SltU {
    #[inline]
    fn name(&self) -> &'static str {
        "SLTU"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        u64::from(x < y)
    }
}

/// RV64I SLTIU: set if less than immediate (unsigned).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SltIU;

impl Instruction for SltIU {
    #[inline]
    fn name(&self) -> &'static str {
        "SLTIU"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        u64::from(x < y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
