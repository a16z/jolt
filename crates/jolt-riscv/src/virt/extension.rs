//! Virtual sign/zero extension instructions.

define_instruction!(
    /// Virtual SIGN_EXTEND_WORD: sign-extends a 32-bit value to 64 bits.
    VirtualSignExtendWord, "VIRTUAL_SIGN_EXTEND_WORD",
    |x, _y| (x as i32) as i64 as u64,
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value],
);

define_instruction!(
    /// Virtual ZERO_EXTEND_WORD: zero-extends a 32-bit value to 64 bits.
    VirtualZeroExtendWord, "VIRTUAL_ZERO_EXTEND_WORD",
    |x, _y| x & 0xFFFF_FFFF,
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value],
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

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
