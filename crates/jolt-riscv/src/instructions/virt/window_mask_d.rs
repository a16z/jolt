use crate::jolt_instruction;

jolt_instruction!(
    /// Doubleword window mask: `!0 << (8·(ea mod 8))` where `ea` is the
    /// effective address `rs1 + imm`. Sets the bytes from offset `ea mod 8`
    /// through the end of the containing doubleword, where a misaligned access
    /// splits between its two doublewords.
    WindowMaskD,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
