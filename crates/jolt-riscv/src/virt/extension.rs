//! Virtual sign/zero extension instructions.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// Virtual SIGN_EXTEND_WORD: sign-extends a 32-bit value to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value)]
pub struct VirtualSignExtendWord;

impl Instruction for VirtualSignExtendWord {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_SIGN_EXTEND_WORD"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        (x as i32) as i64 as u64
    }
}

/// Virtual ZERO_EXTEND_WORD: zero-extends a 32-bit value to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value)]
pub struct VirtualZeroExtendWord;

impl Instruction for VirtualZeroExtendWord {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_ZERO_EXTEND_WORD"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        x & 0xFFFF_FFFF
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_extend_negative() {
        assert_eq!(
            VirtualSignExtendWord.execute(0x8000_0000, 0),
            0xFFFF_FFFF_8000_0000
        );
    }

    #[test]
    fn sign_extend_positive() {
        assert_eq!(VirtualSignExtendWord.execute(0x7FFF_FFFF, 0), 0x7FFF_FFFF);
    }

    #[test]
    fn zero_extend() {
        assert_eq!(
            VirtualZeroExtendWord.execute(0xFFFF_FFFF_8000_0000, 0),
            0x8000_0000
        );
    }
}
