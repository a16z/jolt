//! RV64I bitwise logic instructions.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// RV64I AND: bitwise AND of two registers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct And;

impl Instruction for And {
    #[inline]
    fn name(&self) -> &'static str {
        "AND"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x & y
    }
}

/// RV64I ANDI: bitwise AND with sign-extended immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct AndI;

impl Instruction for AndI {
    #[inline]
    fn name(&self) -> &'static str {
        "ANDI"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x & y
    }
}

/// RV64I OR: bitwise OR of two registers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Or;

impl Instruction for Or {
    #[inline]
    fn name(&self) -> &'static str {
        "OR"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x | y
    }
}

/// RV64I ORI: bitwise OR with sign-extended immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct OrI;

impl Instruction for OrI {
    #[inline]
    fn name(&self) -> &'static str {
        "ORI"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x | y
    }
}

/// RV64I XOR: bitwise exclusive OR of two registers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Xor;

impl Instruction for Xor {
    #[inline]
    fn name(&self) -> &'static str {
        "XOR"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x ^ y
    }
}

/// RV64I XORI: bitwise exclusive OR with sign-extended immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct XorI;

impl Instruction for XorI {
    #[inline]
    fn name(&self) -> &'static str {
        "XORI"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x ^ y
    }
}

/// Zbb ANDN: bitwise AND-NOT. `rd = rs1 & ~rs2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Andn;

impl Instruction for Andn {
    #[inline]
    fn name(&self) -> &'static str {
        "ANDN"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x & !y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn and_basic() {
        assert_eq!(And.execute(0xFF00, 0x0FF0), 0x0F00);
        assert_eq!(And.execute(u64::MAX, 0), 0);
    }

    #[test]
    fn or_basic() {
        assert_eq!(Or.execute(0xFF00, 0x00FF), 0xFFFF);
        assert_eq!(Or.execute(0, 0), 0);
    }

    #[test]
    fn xor_basic() {
        assert_eq!(Xor.execute(0xFF, 0xFF), 0);
        assert_eq!(Xor.execute(0xFF, 0x00), 0xFF);
    }

    #[test]
    fn immediate_variants_match() {
        assert_eq!(And.execute(0xAB, 0xCD), AndI.execute(0xAB, 0xCD));
        assert_eq!(Or.execute(0xAB, 0xCD), OrI.execute(0xAB, 0xCD));
        assert_eq!(Xor.execute(0xAB, 0xCD), XorI.execute(0xAB, 0xCD));
    }

    #[test]
    fn andn_basic() {
        assert_eq!(Andn.execute(0xFF, 0x0F), 0xF0);
        assert_eq!(Andn.execute(0xFF, 0xFF), 0);
        assert_eq!(Andn.execute(0xFF, 0), 0xFF);
    }
}
