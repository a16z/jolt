use crate::jolt_instruction;

jolt_instruction!(
    /// Halfword window mask: `0xFFFF << (8·(ea mod 8 & !1))` where `ea` is the
    /// effective address `rs1 + imm`. Produces the byte mask of the halfword at
    /// offset `ea mod 8` within its containing doubleword. Bit 0 is ignored:
    /// the sequence's `VirtualLoadH`/`VirtualStoreH` rows reject an odd
    /// `ea`.
    WindowMaskH,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
