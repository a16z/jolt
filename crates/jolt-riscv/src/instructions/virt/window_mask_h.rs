use crate::jolt_instruction;

jolt_instruction!(
    /// Halfword window mask: `0xFFFF << (8·(ea mod 8 & !1))` where `ea` is the
    /// effective address `rs1 + imm`. Produces the byte mask of the halfword at
    /// offset `ea mod 8` within its containing doubleword (bit 0 ignored;
    /// halfword alignment asserted separately by the surrounding sequence).
    WindowMaskH,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
