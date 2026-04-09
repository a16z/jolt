//! Virtual shift decomposition instructions.
//!
//! The RV64 shift instructions (SRL, SRA, etc.) are decomposed into
//! virtual sequences that use specialized lookup tables for the sumcheck prover.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

#[inline]
fn shift_from_bitmask(bitmask: u64) -> u32 {
    bitmask.trailing_zeros()
}

#[inline]
fn word_shift_from_bitmask(bitmask: u64) -> u32 {
    (bitmask as u32).trailing_zeros().min(32)
}

/// Computes the bitmask for a right-shift: `((1 << (64 - shift)) - 1) << shift`.
/// Returns `u64::MAX` when `shift == 0`.
#[inline]
fn shift_right_bitmask(shift_amount: u64) -> u64 {
    let shift = shift_amount & 63;
    if shift == 0 {
        u64::MAX
    } else {
        (((1u128 << (64 - shift)) - 1) as u64) << shift
    }
}

/// Virtual SRL: logical right shift using a bitmask-encoded shift amount.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualSrl;

impl Instruction for VirtualSrl {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_SRL"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x.wrapping_shr(shift_from_bitmask(y))
    }
}

/// Virtual SRLI: logical right shift using a bitmask immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualSrli;

impl Instruction for VirtualSrli {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_SRLI"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x.wrapping_shr(shift_from_bitmask(y))
    }
}

/// Virtual SRA: arithmetic right shift using a bitmask-encoded shift amount.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualSra;

impl Instruction for VirtualSra {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_SRA"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        ((x as i64).wrapping_shr(shift_from_bitmask(y))) as u64
    }
}

/// Virtual SRAI: arithmetic right shift using a bitmask immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualSrai;

impl Instruction for VirtualSrai {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_SRAI"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        ((x as i64).wrapping_shr(shift_from_bitmask(y))) as u64
    }
}

/// Virtual SHIFT_RIGHT_BITMASK: bitmask for the shift amount stored in `rs1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualShiftRightBitmask;

impl Instruction for VirtualShiftRightBitmask {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_SHIFT_RIGHT_BITMASK"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        shift_right_bitmask(x)
    }
}

/// Virtual SHIFT_RIGHT_BITMASKI: bitmask for the shift amount stored in the immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(RightOperandIsImm)]
pub struct VirtualShiftRightBitmaski;

impl Instruction for VirtualShiftRightBitmaski {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_SHIFT_RIGHT_BITMASKI"
    }

    #[inline]
    fn execute(&self, _x: u64, y: u64) -> u64 {
        shift_right_bitmask(y)
    }
}

/// Virtual ROTRI: rotate right using a bitmask immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualRotri;

impl Instruction for VirtualRotri {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_ROTRI"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x.rotate_right(shift_from_bitmask(y))
    }
}

/// Virtual ROTRIW: 32-bit rotate right using a bitmask immediate, zero-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct VirtualRotriw;

impl Instruction for VirtualRotriw {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_ROTRIW"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        (x as u32).rotate_right(word_shift_from_bitmask(y)) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtsrl_basic() {
        assert_eq!(VirtualSrl.execute(1024, 1 << 10), 1);
        assert_eq!(VirtualSrl.execute(u64::MAX, 1 << 63), 1);
    }

    #[test]
    fn virtsra_sign_extends() {
        let neg = (-1024i64) as u64;
        assert_eq!(VirtualSra.execute(neg, 1 << 4), (-64i64) as u64);
    }

    #[test]
    fn virtshift_right_bitmask() {
        assert_eq!(VirtualShiftRightBitmask.execute(0, 0), u64::MAX);
        assert_eq!(VirtualShiftRightBitmask.execute(1, 0), u64::MAX - 1);
        assert_eq!(VirtualShiftRightBitmaski.execute(0, 1), u64::MAX - 1);
    }

    #[test]
    fn virtrotri_basic() {
        assert_eq!(VirtualRotri.execute(1, 1 << 1), 1u64 << 63);
        assert_eq!(VirtualRotri.execute(0xFF, 1 << 4), 0xF000_0000_0000_000F);
    }

    #[test]
    fn virtrotriw_zero_extends() {
        assert_eq!(VirtualRotriw.execute(0x8000_0000, 1 << 1), 0x4000_0000);
    }
}
