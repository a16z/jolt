//! RV64 W-suffix arithmetic instructions operating on the lower 32 bits
//! with sign-extension of the result to 64 bits.

use crate::opcodes;

define_instruction!(
    /// RV64I ADDW: 32-bit add, sign-extended to 64 bits.
    AddW, opcodes::ADDW, "ADDW",
    |x, y| (x as i32).wrapping_add(y as i32) as i64 as u64,
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// RV64I ADDIW: 32-bit add immediate, sign-extended to 64 bits.
    AddiW, opcodes::ADDIW, "ADDIW",
    |x, y| (x as i32).wrapping_add(y as i32) as i64 as u64,
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

define_instruction!(
    /// RV64I SUBW: 32-bit subtract, sign-extended to 64 bits.
    SubW, opcodes::SUBW, "SUBW",
    |x, y| (x as i32).wrapping_sub(y as i32) as i64 as u64,
    circuit: [SubtractOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// RV64M MULW: 32-bit multiply, sign-extended to 64 bits.
    MulW, opcodes::MULW, "MULW",
    |x, y| (x as i32).wrapping_mul(y as i32) as i64 as u64,
    circuit: [MultiplyOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

/// RV64M DIVW: 32-bit signed division, sign-extended to 64 bits.
///
/// Division by zero returns `u64::MAX`. Overflow (`i32::MIN / -1`) returns `i32::MIN` sign-extended.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct DivW;

impl crate::Instruction for DivW {
    #[inline]
    fn opcode(&self) -> u32 {
        opcodes::DIVW
    }
    #[inline]
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
    #[inline]
    fn lookup_table(&self) -> Option<crate::LookupTableKind> {
        None
    }
}

impl crate::Flags for DivW {
    #[inline]
    fn circuit_flags(&self) -> [bool; crate::NUM_CIRCUIT_FLAGS] {
        [false; crate::NUM_CIRCUIT_FLAGS]
    }
    #[inline]
    fn instruction_flags(&self) -> [bool; crate::NUM_INSTRUCTION_FLAGS] {
        let mut flags = [false; crate::NUM_INSTRUCTION_FLAGS];
        flags[crate::InstructionFlags::LeftOperandIsRs1Value] = true;
        flags[crate::InstructionFlags::RightOperandIsRs2Value] = true;
        flags
    }
}

/// RV64M DIVUW: 32-bit unsigned division, sign-extended to 64 bits.
/// Returns `u64::MAX` on division by zero.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct DivUW;

impl crate::Instruction for DivUW {
    #[inline]
    fn opcode(&self) -> u32 {
        opcodes::DIVUW
    }
    #[inline]
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
    #[inline]
    fn lookup_table(&self) -> Option<crate::LookupTableKind> {
        None
    }
}

impl crate::Flags for DivUW {
    #[inline]
    fn circuit_flags(&self) -> [bool; crate::NUM_CIRCUIT_FLAGS] {
        [false; crate::NUM_CIRCUIT_FLAGS]
    }
    #[inline]
    fn instruction_flags(&self) -> [bool; crate::NUM_INSTRUCTION_FLAGS] {
        let mut flags = [false; crate::NUM_INSTRUCTION_FLAGS];
        flags[crate::InstructionFlags::LeftOperandIsRs1Value] = true;
        flags[crate::InstructionFlags::RightOperandIsRs2Value] = true;
        flags
    }
}

/// RV64M REMW: 32-bit signed remainder, sign-extended to 64 bits.
/// Returns `x` (truncated to 32 bits, sign-extended) on division by zero.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct RemW;

impl crate::Instruction for RemW {
    #[inline]
    fn opcode(&self) -> u32 {
        opcodes::REMW
    }
    #[inline]
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
    #[inline]
    fn lookup_table(&self) -> Option<crate::LookupTableKind> {
        None
    }
}

impl crate::Flags for RemW {
    #[inline]
    fn circuit_flags(&self) -> [bool; crate::NUM_CIRCUIT_FLAGS] {
        [false; crate::NUM_CIRCUIT_FLAGS]
    }
    #[inline]
    fn instruction_flags(&self) -> [bool; crate::NUM_INSTRUCTION_FLAGS] {
        let mut flags = [false; crate::NUM_INSTRUCTION_FLAGS];
        flags[crate::InstructionFlags::LeftOperandIsRs1Value] = true;
        flags[crate::InstructionFlags::RightOperandIsRs2Value] = true;
        flags
    }
}

/// RV64M REMUW: 32-bit unsigned remainder, sign-extended to 64 bits.
/// Returns `x` (truncated to 32 bits, sign-extended) on division by zero.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct RemUW;

impl crate::Instruction for RemUW {
    #[inline]
    fn opcode(&self) -> u32 {
        opcodes::REMUW
    }
    #[inline]
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
    #[inline]
    fn lookup_table(&self) -> Option<crate::LookupTableKind> {
        None
    }
}

impl crate::Flags for RemUW {
    #[inline]
    fn circuit_flags(&self) -> [bool; crate::NUM_CIRCUIT_FLAGS] {
        [false; crate::NUM_CIRCUIT_FLAGS]
    }
    #[inline]
    fn instruction_flags(&self) -> [bool; crate::NUM_INSTRUCTION_FLAGS] {
        let mut flags = [false; crate::NUM_INSTRUCTION_FLAGS];
        flags[crate::InstructionFlags::LeftOperandIsRs1Value] = true;
        flags[crate::InstructionFlags::RightOperandIsRs2Value] = true;
        flags
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

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
