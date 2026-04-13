//! RV64 W-suffix arithmetic instructions operating on the lower 32 bits
//! with sign-extension of the result to 64 bits.
//!
//! Instructions that set operand-combining flags (`AddOperands`, etc.) but have
//! no lookup table are decomposed into virtual sequences by the VM.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// RV64I ADDW: 32-bit add, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct AddW;

impl Instruction for AddW {
    #[inline]
    fn name(&self) -> &'static str {
        "ADDW"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        (x as i32).wrapping_add(y as i32) as i64 as u64
    }
}

/// RV64I ADDIW: 32-bit add immediate, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct AddiW;

impl Instruction for AddiW {
    #[inline]
    fn name(&self) -> &'static str {
        "ADDIW"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        (x as i32).wrapping_add(y as i32) as i64 as u64
    }
}

/// RV64I SUBW: 32-bit subtract, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(SubtractOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct SubW;

impl Instruction for SubW {
    #[inline]
    fn name(&self) -> &'static str {
        "SUBW"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        (x as i32).wrapping_sub(y as i32) as i64 as u64
    }
}

/// RV64M MULW: 32-bit multiply, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct MulW;

impl Instruction for MulW {
    #[inline]
    fn name(&self) -> &'static str {
        "MULW"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        (x as i32).wrapping_mul(y as i32) as i64 as u64
    }
}

/// RV64M DIVW: 32-bit signed division, sign-extended to 64 bits.
///
/// Division by zero returns `u64::MAX`. Overflow (`i32::MIN / -1`) returns `i32::MIN` sign-extended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct DivW;

impl Instruction for DivW {
    fn name(&self) -> &'static str {
        "DIVW"
    }
    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let sx = x as i32;
        let sy = y as i32;
        if sy == 0 {
            u64::MAX
        } else if sx == i32::MIN && sy == -1 {
            sx as i64 as u64
        } else {
            sx.wrapping_div(sy) as i64 as u64
        }
    }
}

/// RV64M DIVUW: 32-bit unsigned division, sign-extended to 64 bits.
/// Returns `u64::MAX` on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct DivUW;

impl Instruction for DivUW {
    fn name(&self) -> &'static str {
        "DIVUW"
    }
    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let ux = x as u32;
        let uy = y as u32;
        if uy == 0 {
            u64::MAX
        } else {
            (ux / uy) as i32 as i64 as u64
        }
    }
}

/// RV64M REMW: 32-bit signed remainder, sign-extended to 64 bits.
/// Returns `x` (truncated to 32 bits, sign-extended) on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct RemW;

impl Instruction for RemW {
    fn name(&self) -> &'static str {
        "REMW"
    }
    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let sx = x as i32;
        let sy = y as i32;
        if sy == 0 {
            sx as i64 as u64
        } else if sx == i32::MIN && sy == -1 {
            0
        } else {
            sx.wrapping_rem(sy) as i64 as u64
        }
    }
}

/// RV64M REMUW: 32-bit unsigned remainder, sign-extended to 64 bits.
/// Returns `x` (truncated to 32 bits, sign-extended) on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct RemUW;

impl Instruction for RemUW {
    fn name(&self) -> &'static str {
        "REMUW"
    }
    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let ux = x as u32;
        let uy = y as u32;
        if uy == 0 {
            ux as i32 as i64 as u64
        } else {
            (ux % uy) as i32 as i64 as u64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addw_sign_extends() {
        // 0x7FFF_FFFF + 1 = 0x8000_0000 as i32 = -2147483648, sign-extended
        let result = AddW.execute(0x7FFF_FFFF, 1);
        assert_eq!(result, 0xFFFF_FFFF_8000_0000);
    }

    #[test]
    fn subw_basic() {
        assert_eq!(SubW.execute(10, 3), 7);
        assert_eq!(SubW.execute(0, 1), 0xFFFF_FFFF_FFFF_FFFF); // -1 sign-extended
    }

    #[test]
    fn mulw_basic() {
        assert_eq!(MulW.execute(6, 7), 42);
    }

    #[test]
    fn divw_by_zero() {
        assert_eq!(DivW.execute(42, 0), u64::MAX);
    }

    #[test]
    fn divw_overflow() {
        assert_eq!(
            DivW.execute(i32::MIN as u64, (-1i32) as u64),
            i32::MIN as i64 as u64
        );
    }

    #[test]
    fn divuw_basic() {
        assert_eq!(DivUW.execute(10, 3), 3);
        assert_eq!(DivUW.execute(10, 0), u64::MAX);
    }

    #[test]
    fn remw_basic() {
        assert_eq!(RemW.execute(10, 3), 1);
        assert_eq!(RemW.execute(10, 0), 10);
    }

    #[test]
    fn remuw_basic() {
        assert_eq!(RemUW.execute(10, 3), 1);
        assert_eq!(RemUW.execute(10, 0), 10);
    }

    #[test]
    fn remw_overflow() {
        assert_eq!(RemW.execute(i32::MIN as u64, (-1i32) as u64), 0);
    }
}
