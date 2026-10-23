use crate::jolt_instruction;

jolt_instruction!(
    /// Word window mask: `0xFFFFFFFF << (32·ea_2)` where `ea_2` is bit 2 of the
    /// effective address `rs1 + imm`. Produces the byte mask of the word at offset
    /// `ea mod 8` within its containing doubleword. Bits 0-1 are ignored: the
    /// sequence's `VirtualLoadW`/`VirtualStoreW` rows reject an `ea` that
    /// is not word-aligned.
    WindowMaskW,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
