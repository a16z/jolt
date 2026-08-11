use crate::jolt_instruction;

jolt_instruction!(
    /// Word store-data shifter: `(rs2 & 0xFFFFFFFF) << (8·(ea mod 8 & !3))`
    /// where `rs2` is the store value (carried in this instruction's `rs1`
    /// slot) and `ea` is the effective address (in the `rs2` slot, offset
    /// mask 4). Moves the store word into its lane within the containing
    /// doubleword.
    ShiftDataW,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
