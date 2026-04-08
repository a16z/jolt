//! Virtual arithmetic instructions used internally by the Jolt VM.

define_instruction!(
    /// Virtual POW2: computes `2^rs1` using the low 6 bits of `rs1`.
    Pow2, "POW2",
    |x, _y| 1u64 << (x & 63),
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

define_instruction!(
    /// Virtual POW2I: computes `2^imm` with immediate exponent.
    Pow2I, "POW2I",
    |_x, y| 1u64 << (y & 63),
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [RightOperandIsImm],
);

define_instruction!(
    /// Virtual POW2W: computes `2^(rs1 mod 32)` for 32-bit mode.
    Pow2W, "POW2W",
    |x, _y| 1u64 << (x & 31),
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value],
);

define_instruction!(
    /// Virtual POW2IW: computes `2^(imm mod 32)` for 32-bit immediate mode.
    Pow2IW, "POW2IW",
    |_x, y| 1u64 << (y & 31),
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [RightOperandIsImm],
);

define_instruction!(
    /// Virtual MULI: multiply by immediate. `rd = rs1 * imm`.
    MulI, "MULI",
    |x, y| x.wrapping_mul(y),
    circuit: [MultiplyOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn pow2_basic() {
        assert_eq!(Pow2.execute(0, 0), 1);
        assert_eq!(Pow2.execute(10, 0), 1024);
        assert_eq!(Pow2.execute(63, 0), 1 << 63);
    }

    #[test]
    fn pow2_masks_exponent() {
        assert_eq!(Pow2.execute(64, 0), 1);
    }

    #[test]
    fn pow2w_basic() {
        assert_eq!(Pow2W.execute(0, 0), 1);
        assert_eq!(Pow2W.execute(31, 0), 1 << 31);
    }

    #[test]
    fn pow2w_masks_to_32() {
        assert_eq!(Pow2W.execute(32, 0), 1);
    }

    #[test]
    fn immediate_pow2_variants_use_y() {
        assert_eq!(Pow2I.execute(0, 10), 1024);
        assert_eq!(Pow2IW.execute(0, 31), 1 << 31);
    }

    #[test]
    fn muli_basic() {
        assert_eq!(MulI.execute(6, 7), 42);
        assert_eq!(MulI.execute(u64::MAX, 2), u64::MAX - 1);
    }
}
