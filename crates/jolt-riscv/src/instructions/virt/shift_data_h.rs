use crate::jolt_instruction;

jolt_instruction!(
    /// Halfword store-data shifter: `(value & 0xFFFF) << (8·(ea mod 8 & !1))`,
    /// where `value` is `rs1` and the effective address is `ea = rs2 + imm`
    /// (offset mask 6). Moves the store halfword into its lane within the
    /// containing doubleword. Both right-operand flags are set, so the right
    /// instruction input is the sum `rs2 + imm`.
    ShiftDataH,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value, RightOperandIsImm]
);
