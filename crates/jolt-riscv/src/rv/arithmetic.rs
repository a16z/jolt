//! RV64I/M arithmetic instructions: ADD, SUB, LUI, AUIPC, and
//! the M-extension multiply/divide family.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// RV64I ADD: `rd = rs1 + rs2` (wrapping).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Add;

impl Instruction for Add {
    #[inline]
    fn name(&self) -> &'static str {
        "ADD"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x.wrapping_add(y)
    }
}

/// RV64I ADDI: `rd = rs1 + imm` (wrapping). Immediate already decoded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct Addi;

impl Instruction for Addi {
    #[inline]
    fn name(&self) -> &'static str {
        "ADDI"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x.wrapping_add(y)
    }
}

/// RV64I SUB: `rd = rs1 - rs2` (wrapping).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(SubtractOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Sub;

impl Instruction for Sub {
    #[inline]
    fn name(&self) -> &'static str {
        "SUB"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x.wrapping_sub(y)
    }
}

/// RV64I LUI: load upper immediate. Result is the immediate value itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(RightOperandIsImm)]
pub struct Lui;

impl Instruction for Lui {
    #[inline]
    fn name(&self) -> &'static str {
        "LUI"
    }

    #[inline]
    fn execute(&self, _x: u64, y: u64) -> u64 {
        y
    }
}

/// RV64I AUIPC: add upper immediate to PC. `rd = PC + imm`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsPC, RightOperandIsImm)]
pub struct Auipc;

impl Instruction for Auipc {
    #[inline]
    fn name(&self) -> &'static str {
        "AUIPC"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x.wrapping_add(y)
    }
}

/// RV64M MUL: signed multiply, lower 64 bits of the 128-bit product.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Mul;

impl Instruction for Mul {
    #[inline]
    fn name(&self) -> &'static str {
        "MUL"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x.wrapping_mul(y)
    }
}

/// RV64M MULH: signed×signed multiply, upper 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct MulH;

impl Instruction for MulH {
    fn name(&self) -> &'static str {
        "MULH"
    }
    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let product = (x as i64 as i128).wrapping_mul(y as i64 as i128);
        (product >> 64) as u64
    }
}

/// RV64M MULHSU: signed×unsigned multiply, upper 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct MulHSU;

impl Instruction for MulHSU {
    fn name(&self) -> &'static str {
        "MULHSU"
    }
    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let product = (x as i64 as i128).wrapping_mul(y as u128 as i128);
        (product >> 64) as u64
    }
}

/// RV64M MULHU: unsigned×unsigned multiply, upper 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(MultiplyOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct MulHU;

impl Instruction for MulHU {
    fn name(&self) -> &'static str {
        "MULHU"
    }
    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let product = (x as u128).wrapping_mul(y as u128);
        (product >> 64) as u64
    }
}

/// RV64M DIV: signed division with RISC-V overflow handling.
///
/// Special cases per the RISC-V spec:
/// - Division by zero returns `u64::MAX` (all bits set, i.e. -1 unsigned).
/// - `i64::MIN / -1` returns `i64::MIN` (overflow wraps).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Div;

impl Instruction for Div {
    fn name(&self) -> &'static str {
        "DIV"
    }
    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let sx = x as i64;
        let sy = y as i64;
        if sy == 0 {
            u64::MAX
        } else if sx == i64::MIN && sy == -1 {
            sx as u64
        } else {
            sx.wrapping_div(sy) as u64
        }
    }
}

/// RV64M DIVU: unsigned division. Returns `u64::MAX` on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct DivU;

impl Instruction for DivU {
    fn name(&self) -> &'static str {
        "DIVU"
    }
    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        if y == 0 {
            u64::MAX
        } else {
            x / y
        }
    }
}

/// RV64M REM: signed remainder. Returns `x` on division by zero,
/// returns 0 when `x == i64::MIN && y == -1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Rem;

impl Instruction for Rem {
    fn name(&self) -> &'static str {
        "REM"
    }
    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let sx = x as i64;
        let sy = y as i64;
        if sy == 0 {
            x
        } else if sx == i64::MIN && sy == -1 {
            0
        } else {
            sx.wrapping_rem(sy) as u64
        }
    }
}

/// RV64M REMU: unsigned remainder. Returns `x` on division by zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct RemU;

impl Instruction for RemU {
    fn name(&self) -> &'static str {
        "REMU"
    }
    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        if y == 0 {
            x
        } else {
            x % y
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_basic() {
        assert_eq!(Add.execute(3, 5), 8);
        assert_eq!(Add.execute(u64::MAX, 1), 0); // wrapping
    }

    #[test]
    fn sub_basic() {
        assert_eq!(Sub.execute(10, 3), 7);
        assert_eq!(Sub.execute(0, 1), u64::MAX); // wrapping
    }

    #[test]
    fn mul_lower_bits() {
        assert_eq!(Mul.execute(6, 7), 42);
        assert_eq!(Mul.execute(u64::MAX, 2), u64::MAX - 1); // wrapping
    }

    #[test]
    fn mulh_upper_signed() {
        // 2^63 * 2 signed: (i64::MIN as i128) * 2 = -2^64, upper = -1
        let x = i64::MIN as u64;
        assert_eq!(MulH.execute(x, 2), u64::MAX); // -1 in two's complement

        // Small positive numbers: upper bits are 0
        assert_eq!(MulH.execute(100, 200), 0);
    }

    #[test]
    fn mulhu_upper_unsigned() {
        assert_eq!(MulHU.execute(u64::MAX, 2), 1);
        assert_eq!(MulHU.execute(100, 200), 0);
    }

    #[test]
    fn mulhsu_mixed() {
        // -1 (signed) * 2 (unsigned) = -2 as i128, upper = -1
        assert_eq!(MulHSU.execute(u64::MAX, 2), u64::MAX);
        assert_eq!(MulHSU.execute(100, 200), 0);
    }

    #[test]
    fn div_signed() {
        assert_eq!(Div.execute(20u64, 3u64), (20i64 / 3) as u64);
        // Negative: -20 / 3 = -6 (truncated toward zero)
        assert_eq!(Div.execute((-20i64) as u64, 3u64), (-6i64) as u64);
    }

    #[test]
    fn div_by_zero() {
        assert_eq!(Div.execute(42, 0), u64::MAX);
        assert_eq!(DivU.execute(42, 0), u64::MAX);
    }

    #[test]
    fn div_overflow() {
        assert_eq!(
            Div.execute(i64::MIN as u64, (-1i64) as u64),
            i64::MIN as u64
        );
    }

    #[test]
    fn rem_signed() {
        assert_eq!(Rem.execute(20, 3), (20i64 % 3) as u64);
        assert_eq!(Rem.execute((-20i64) as u64, 3), (-2i64) as u64);
    }

    #[test]
    fn rem_by_zero() {
        assert_eq!(Rem.execute(42, 0), 42);
        assert_eq!(RemU.execute(42, 0), 42);
    }

    #[test]
    fn rem_overflow() {
        assert_eq!(Rem.execute(i64::MIN as u64, (-1i64) as u64), 0);
    }

    #[test]
    fn lui_passthrough() {
        assert_eq!(Lui.execute(0xDEAD_0000, 999), 999);
    }

    #[test]
    fn auipc_add() {
        assert_eq!(Auipc.execute(0x1000, 0x2000), 0x3000);
    }
}
