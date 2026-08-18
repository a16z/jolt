use crate::jolt_instruction;

jolt_instruction!(
    /// Byte window mask: `0xFF << (8·(ea mod 8))` where `ea` is the effective
    /// address `rs1 + imm`. Produces the byte mask of the byte at offset
    /// `ea mod 8` within its containing doubleword.
    WindowMaskB,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
