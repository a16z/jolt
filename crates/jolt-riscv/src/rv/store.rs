//! RV64I store instructions that mask a value to the appropriate width.
//!
//! In Jolt's execution model, `x` is the value to store and the instruction
//! truncates it to the target memory width.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// RV64I SB: store byte (lowest 8 bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Store)]
pub struct Sb;

impl Instruction for Sb {
    #[inline]
    fn name(&self) -> &'static str {
        "SB"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        x & 0xFF
    }
}

/// RV64I SH: store halfword (lowest 16 bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Store)]
pub struct Sh;

impl Instruction for Sh {
    #[inline]
    fn name(&self) -> &'static str {
        "SH"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        x & 0xFFFF
    }
}

/// RV64I SW: store word (lowest 32 bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Store)]
pub struct Sw;

impl Instruction for Sw {
    #[inline]
    fn name(&self) -> &'static str {
        "SW"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        x & 0xFFFF_FFFF
    }
}

/// RV64I SD: store doubleword (full 64 bits). Identity operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Store)]
pub struct Sd;

impl Instruction for Sd {
    #[inline]
    fn name(&self) -> &'static str {
        "SD"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sb_masks_to_byte() {
        assert_eq!(Sb.execute(0xDEAD_BEEF, 0), 0xEF);
    }

    #[test]
    fn sh_masks_to_halfword() {
        assert_eq!(Sh.execute(0xDEAD_BEEF, 0), 0xBEEF);
    }

    #[test]
    fn sw_masks_to_word() {
        assert_eq!(Sw.execute(0xDEAD_BEEF_CAFE_BABE, 0), 0xCAFE_BABE);
    }

    #[test]
    fn sd_identity() {
        assert_eq!(Sd.execute(0xDEAD_BEEF_CAFE_BABE, 0), 0xDEAD_BEEF_CAFE_BABE);
    }
}
