use crate::jolt_instruction;

jolt_instruction!(
    /// Word store-data shifter: `(value & 0xFFFFFFFF) << (8·(ea mod 8 & !3))`,
    /// where `value` is `rs1` and the effective address is `ea = rs2 + imm`
    /// (offset mask 4). Moves the store word into its lane within the
    /// containing doubleword. Both right-operand flags are set, so the right
    /// instruction input is the sum `rs2 + imm`.
    ShiftDataW,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value, RightOperandIsImm]
);
