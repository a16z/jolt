//! Virtual arithmetic instructions used internally by the Jolt VM.

use crate::opcodes;

define_instruction!(
    /// Virtual POW2: computes `2^y` where exponent is from lower 6 bits of `y`.
    Pow2, opcodes::POW2, "POW2",
    |_x, y| 1u64 << (y & 63),
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: Pow2,
);

define_instruction!(
    /// Virtual POW2I: computes `2^imm` with immediate exponent.
    Pow2I, opcodes::VIRTUAL_POW2I, "POW2I",
    |_x, y| 1u64 << (y & 63),
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: Pow2,
);

define_instruction!(
    /// Virtual POW2W: computes `2^(y mod 32)` for 32-bit mode.
    Pow2W, opcodes::VIRTUAL_POW2W, "POW2W",
    |_x, y| 1u64 << (y & 31),
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: Pow2W,
);

define_instruction!(
    /// Virtual POW2IW: computes `2^(imm mod 32)` for 32-bit immediate mode.
    Pow2IW, opcodes::VIRTUAL_POW2IW, "POW2IW",
    |_x, y| 1u64 << (y & 31),
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: Pow2W,
);

define_instruction!(
    /// Virtual MULI: multiply by immediate. `rd = rs1 * imm`.
    MulI, opcodes::VIRTUAL_MULI, "MULI",
    |x, y| x.wrapping_mul(y),
    circuit: [MultiplyOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: RangeCheck,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn pow2_basic() {
        assert_eq!(Pow2.execute(0, 0), 1);
        assert_eq!(Pow2.execute(0, 10), 1024);
        assert_eq!(Pow2.execute(0, 63), 1 << 63);
    }

    #[test]
    fn pow2_masks_exponent() {
        assert_eq!(Pow2.execute(0, 64), 1);
    }

    #[test]
    fn pow2w_basic() {
        assert_eq!(Pow2W.execute(0, 0), 1);
        assert_eq!(Pow2W.execute(0, 31), 1 << 31);
    }

    #[test]
    fn pow2w_masks_to_32() {
        assert_eq!(Pow2W.execute(0, 32), 1);
    }

    #[test]
    fn muli_basic() {
        assert_eq!(MulI.execute(6, 7), 42);
        assert_eq!(MulI.execute(u64::MAX, 2), u64::MAX - 1);
    }
}
