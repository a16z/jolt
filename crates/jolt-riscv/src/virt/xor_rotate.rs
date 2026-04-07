//! Virtual XOR-rotate instructions for SHA hash functions.

define_instruction!(
    /// Virtual XOR then rotate right by 32 bits.
    VirtualXorRot32, "VIRTUAL_XORROT32",
    |x, y| (x ^ y).rotate_right(32),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// Virtual XOR then rotate right by 24 bits.
    VirtualXorRot24, "VIRTUAL_XORROT24",
    |x, y| (x ^ y).rotate_right(24),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// Virtual XOR then rotate right by 16 bits.
    VirtualXorRot16, "VIRTUAL_XORROT16",
    |x, y| (x ^ y).rotate_right(16),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// Virtual XOR then rotate right by 63 bits.
    VirtualXorRot63, "VIRTUAL_XORROT63",
    |x, y| (x ^ y).rotate_right(63),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// Virtual XOR then rotate right word (32-bit) by 16 bits.
    VirtualXorRotW16, "VIRTUAL_XORROTW16",
    |x, y| {
        let val = (x as u32) ^ (y as u32);
        val.rotate_right(16) as u64
    },
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// Virtual XOR then rotate right word by 12 bits.
    VirtualXorRotW12, "VIRTUAL_XORROTW12",
    |x, y| {
        let val = (x as u32) ^ (y as u32);
        val.rotate_right(12) as u64
    },
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// Virtual XOR then rotate right word by 8 bits.
    VirtualXorRotW8, "VIRTUAL_XORROTW8",
    |x, y| {
        let val = (x as u32) ^ (y as u32);
        val.rotate_right(8) as u64
    },
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// Virtual XOR then rotate right word by 7 bits.
    VirtualXorRotW7, "VIRTUAL_XORROTW7",
    |x, y| {
        let val = (x as u32) ^ (y as u32);
        val.rotate_right(7) as u64
    },
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn xor_rot32() {
        assert_eq!(VirtualXorRot32.execute(0xFF, 0), 0xFF_0000_0000);
    }

    #[test]
    fn xor_rot63() {
        assert_eq!(VirtualXorRot63.execute(1, 0), 2);
    }
}
