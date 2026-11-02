use crate::jolt_instruction;

jolt_instruction!(
    /// Byte store-data shifter: `(value & 0xFF) << (8·(ea mod 8))`, where
    /// `value` is `rs1` and the effective address is `ea = rs2 + imm` (offset
    /// mask 7). Moves the store byte into its lane within the containing
    /// doubleword. Both right-operand flags are set, so the right instruction
    /// input is the sum `rs2 + imm`.
    ShiftDataB,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value, RightOperandIsImm]
);
