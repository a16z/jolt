//! Virtual XOR-rotate instructions for SHA hash functions.

use crate::opcodes;

define_instruction!(
    /// Virtual XOR then rotate right by 32 bits.
    VirtualXorRot32, opcodes::VIRTUAL_XORROT32, "VIRTUAL_XORROT32",
    |x, y| (x ^ y).rotate_right(32),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: VirtualXORROT32,
);

define_instruction!(
    /// Virtual XOR then rotate right by 24 bits.
    VirtualXorRot24, opcodes::VIRTUAL_XORROT24, "VIRTUAL_XORROT24",
    |x, y| (x ^ y).rotate_right(24),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: VirtualXORROT24,
);

define_instruction!(
    /// Virtual XOR then rotate right by 16 bits.
    VirtualXorRot16, opcodes::VIRTUAL_XORROT16, "VIRTUAL_XORROT16",
    |x, y| (x ^ y).rotate_right(16),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: VirtualXORROT16,
);

define_instruction!(
    /// Virtual XOR then rotate right by 63 bits.
    VirtualXorRot63, opcodes::VIRTUAL_XORROT63, "VIRTUAL_XORROT63",
    |x, y| (x ^ y).rotate_right(63),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: VirtualXORROT63,
);

define_instruction!(
    /// Virtual XOR then rotate right word (32-bit) by 16 bits.
    VirtualXorRotW16, opcodes::VIRTUAL_XORROTW16, "VIRTUAL_XORROTW16",
    |x, y| {
        let val = (x as u32) ^ (y as u32);
        val.rotate_right(16) as i32 as i64 as u64
    },
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: VirtualXORROTW16,
);

define_instruction!(
    /// Virtual XOR then rotate right word by 12 bits.
    VirtualXorRotW12, opcodes::VIRTUAL_XORROTW12, "VIRTUAL_XORROTW12",
    |x, y| {
        let val = (x as u32) ^ (y as u32);
        val.rotate_right(12) as i32 as i64 as u64
    },
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: VirtualXORROTW12,
);

define_instruction!(
    /// Virtual XOR then rotate right word by 8 bits.
    VirtualXorRotW8, opcodes::VIRTUAL_XORROTW8, "VIRTUAL_XORROTW8",
    |x, y| {
        let val = (x as u32) ^ (y as u32);
        val.rotate_right(8) as i32 as i64 as u64
    },
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: VirtualXORROTW8,
);

define_instruction!(
    /// Virtual XOR then rotate right word by 7 bits.
    VirtualXorRotW7, opcodes::VIRTUAL_XORROTW7, "VIRTUAL_XORROTW7",
    |x, y| {
        let val = (x as u32) ^ (y as u32);
        val.rotate_right(7) as i32 as i64 as u64
    },
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: VirtualXORROTW7,
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
