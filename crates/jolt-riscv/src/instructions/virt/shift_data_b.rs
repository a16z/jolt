use crate::jolt_instruction;

jolt_instruction!(
    /// Byte store-data shifter: `(rs2 & 0xFF) << (8·(ea mod 8))` where `rs2`
    /// is the store value (carried in this instruction's `rs1` slot) and `ea`
    /// is the effective address (in the `rs2` slot, offset mask 7). Moves the
    /// store byte into its lane within the containing doubleword.
    ShiftDataB,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
