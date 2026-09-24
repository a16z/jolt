use crate::jolt_instruction;

jolt_instruction!(
    /// Word window mask: `0xFFFFFFFF << (32·ea_2)` where `ea_2` is bit 2 of the
    /// effective address `rs1 + imm`. Produces the byte mask of the word at offset
    /// `ea mod 8` within its containing doubleword (word alignment asserted
    /// separately by the surrounding sequence).
    WindowMaskW,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
