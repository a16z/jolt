//! RV64 W-suffix shift instructions operating on the lower 32 bits
//! with sign-extension of the result to 64 bits.
//! Shift amount is masked to 5 bits (0..31).
//!
//! These set `WriteLookupOutputToRD` (architectural rd-write) but have
//! no lookup table — the VM decomposes them into virtual shift sequences.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// RV64I SLLW: 32-bit shift left logical, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct SllW;

impl Instruction for SllW {
    #[inline]
    fn name(&self) -> &'static str {
        "SLLW"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        ((x as u32) << (y & 31)) as i32 as i64 as u64
    }
}

/// RV64I SLLIW: 32-bit shift left logical by immediate, sign-extended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SllIW;

impl Instruction for SllIW {
    #[inline]
    fn name(&self) -> &'static str {
        "SLLIW"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        ((x as u32) << (y & 31)) as i32 as i64 as u64
    }
}

/// RV64I SRLW: 32-bit shift right logical, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct SrlW;

impl Instruction for SrlW {
    #[inline]
    fn name(&self) -> &'static str {
        "SRLW"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        ((x as u32) >> (y & 31)) as i32 as i64 as u64
    }
}

/// RV64I SRLIW: 32-bit shift right logical by immediate, sign-extended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SrlIW;

impl Instruction for SrlIW {
    #[inline]
    fn name(&self) -> &'static str {
        "SRLIW"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        ((x as u32) >> (y & 31)) as i32 as i64 as u64
    }
}

/// RV64I SRAW: 32-bit shift right arithmetic, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct SraW;

impl Instruction for SraW {
    #[inline]
    fn name(&self) -> &'static str {
        "SRAW"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        ((x as i32) >> (y & 31)) as i64 as u64
    }
}

/// RV64I SRAIW: 32-bit shift right arithmetic by immediate, sign-extended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SraIW;

impl Instruction for SraIW {
    #[inline]
    fn name(&self) -> &'static str {
        "SRAIW"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        ((x as i32) >> (y & 31)) as i64 as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sllw_sign_extends() {
        // Shifting 1 left by 31 gives 0x8000_0000 which is negative as i32
        let result = SllW.execute(1, 31);
        assert_eq!(result, 0xFFFF_FFFF_8000_0000);
    }

    #[test]
    fn sllw_masks_to_5_bits() {
        assert_eq!(SllW.execute(1, 32), 1); // 32 & 31 = 0
    }

    #[test]
    fn srlw_basic() {
        assert_eq!(SrlW.execute(0x8000_0000, 31), 1);
    }

    #[test]
    fn sraw_sign_extends() {
        let neg = 0xFFFF_FFFF_8000_0000u64; // -2^31 sign-extended
        let result = SraW.execute(neg, 1);
        assert_eq!(result, 0xFFFF_FFFF_C000_0000); // -2^30 sign-extended
    }

    #[test]
    fn immediate_variants_match() {
        assert_eq!(SllW.execute(42, 5), SllIW.execute(42, 5));
        assert_eq!(SrlW.execute(42, 5), SrlIW.execute(42, 5));
        assert_eq!(SraW.execute(42, 5), SraIW.execute(42, 5));
    }
}
