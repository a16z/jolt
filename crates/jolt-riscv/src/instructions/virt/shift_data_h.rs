use crate::jolt_instruction;

jolt_instruction!(
    /// Halfword store-data shifter: `(rs2 & 0xFFFF) << (8·(ea mod 8 & !1))`
    /// where `rs2` is the store value (carried in this instruction's `rs1`
    /// slot) and `ea` is the effective address (in the `rs2` slot, offset
    /// mask 6). Moves the store halfword into its lane within the containing
    /// doubleword.
    ShiftDataH,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
