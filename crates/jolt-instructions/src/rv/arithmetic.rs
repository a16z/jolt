//! RV64I/M arithmetic instructions: ADD, SUB, LUI, AUIPC, and
//! the M-extension multiply/divide family.

use crate::opcodes;

define_instruction!(
    /// RV64I ADD: `rd = rs1 + rs2` (wrapping).
    Add, opcodes::ADD, "ADD",
    |x, y| x.wrapping_add(y),
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: RangeCheck,
);

define_instruction!(
    /// RV64I ADDI: `rd = rs1 + imm` (wrapping). Immediate already decoded.
    Addi, opcodes::ADDI, "ADDI",
    |x, y| x.wrapping_add(y),
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: RangeCheck,
);

define_instruction!(
    /// RV64I SUB: `rd = rs1 - rs2` (wrapping).
    Sub, opcodes::SUB, "SUB",
    |x, y| x.wrapping_sub(y),
    circuit: [SubtractOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: RangeCheck,
);

define_instruction!(
    /// RV64I LUI: load upper immediate. Result is the immediate value itself.
    Lui, opcodes::LUI, "LUI",
    |x, _y| x,
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [RightOperandIsImm],
    table: RangeCheck,
);

define_instruction!(
    /// RV64I AUIPC: add upper immediate to PC. `rd = PC + imm`.
    Auipc, opcodes::AUIPC, "AUIPC",
    |x, y| x.wrapping_add(y),
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsPC, RightOperandIsImm],
    table: RangeCheck,
);

define_instruction!(
    /// RV64M MUL: signed multiply, lower 64 bits of the 128-bit product.
    Mul, opcodes::MUL, "MUL",
    |x, y| x.wrapping_mul(y),
    circuit: [MultiplyOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: RangeCheck,
);

/// RV64M MULH: signed×signed multiply, upper 64 bits.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct MulH;

impl crate::Instruction for MulH {
    #[inline]
    fn opcode(&self) -> u32 {
        opcodes::MULH
    }
    #[inline]
    fn name(&self) -> &'static str {
        "MULH"
    }
    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let product = (x as i64 as i128).wrapping_mul(y as i64 as i128);
        (product >> 64) as u64
    }
    #[inline]
    fn lookup_table(&self) -> Option<crate::LookupTableKind> {
        None
    }
}

impl crate::Flags for MulH {
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

/// RV64M MULHSU: signed×unsigned multiply, upper 64 bits.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct MulHSU;

impl crate::Instruction for MulHSU {
    #[inline]
    fn opcode(&self) -> u32 {
        opcodes::MULHSU
    }
    #[inline]
    fn name(&self) -> &'static str {
        "MULHSU"
    }
    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let product = (x as i64 as i128).wrapping_mul(y as u128 as i128);
        (product >> 64) as u64
    }
    #[inline]
    fn lookup_table(&self) -> Option<crate::LookupTableKind> {
        None
    }
}

impl crate::Flags for MulHSU {
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

/// RV64M MULHU: unsigned×unsigned multiply, upper 64 bits.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct MulHU;

impl crate::Instruction for MulHU {
    #[inline]
    fn opcode(&self) -> u32 {
        opcodes::MULHU
    }
    #[inline]
    fn name(&self) -> &'static str {
        "MULHU"
    }
    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let product = (x as u128).wrapping_mul(y as u128);
        (product >> 64) as u64
    }
    #[inline]
    fn lookup_table(&self) -> Option<crate::LookupTableKind> {
        Some(crate::LookupTableKind::UpperWord)
    }
}

impl crate::Flags for MulHU {
    #[inline]
    fn circuit_flags(&self) -> [bool; crate::NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; crate::NUM_CIRCUIT_FLAGS];
        flags[crate::CircuitFlags::MultiplyOperands] = true;
        flags[crate::CircuitFlags::WriteLookupOutputToRD] = true;
        flags
    }
    #[inline]
    fn instruction_flags(&self) -> [bool; crate::NUM_INSTRUCTION_FLAGS] {
        let mut flags = [false; crate::NUM_INSTRUCTION_FLAGS];
        flags[crate::InstructionFlags::LeftOperandIsRs1Value] = true;
        flags[crate::InstructionFlags::RightOperandIsRs2Value] = true;
        flags
    }
}

/// RV64M DIV: signed division with RISC-V overflow handling.
///
/// Special cases per the RISC-V spec:
/// - Division by zero returns `u64::MAX` (all bits set, i.e. -1 unsigned).
/// - `i64::MIN / -1` returns `i64::MIN` (overflow wraps).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct Div;

impl crate::Instruction for Div {
    #[inline]
    fn opcode(&self) -> u32 {
        opcodes::DIV
    }
    #[inline]
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
    #[inline]
    fn lookup_table(&self) -> Option<crate::LookupTableKind> {
        None
    }
}

impl crate::Flags for Div {
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

/// RV64M DIVU: unsigned division. Returns `u64::MAX` on division by zero.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct DivU;

impl crate::Instruction for DivU {
    #[inline]
    fn opcode(&self) -> u32 {
        opcodes::DIVU
    }
    #[inline]
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
    #[inline]
    fn lookup_table(&self) -> Option<crate::LookupTableKind> {
        None
    }
}

impl crate::Flags for DivU {
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

/// RV64M REM: signed remainder. Returns `x` on division by zero,
/// returns 0 when `x == i64::MIN && y == -1`.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct Rem;

impl crate::Instruction for Rem {
    #[inline]
    fn opcode(&self) -> u32 {
        opcodes::REM
    }
    #[inline]
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
    #[inline]
    fn lookup_table(&self) -> Option<crate::LookupTableKind> {
        None
    }
}

impl crate::Flags for Rem {
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

/// RV64M REMU: unsigned remainder. Returns `x` on division by zero.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct RemU;

impl crate::Instruction for RemU {
    #[inline]
    fn opcode(&self) -> u32 {
        opcodes::REMU
    }
    #[inline]
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
    #[inline]
    fn lookup_table(&self) -> Option<crate::LookupTableKind> {
        None
    }
}

impl crate::Flags for RemU {
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
        assert_eq!(Lui.execute(0xDEAD_0000, 999), 0xDEAD_0000);
    }

    #[test]
    fn auipc_add() {
        assert_eq!(Auipc.execute(0x1000, 0x2000), 0x3000);
    }
}
