use crate::jolt_instruction;

jolt_instruction!(
    /// Jolt ADDC: `rd = rs1 + rs2 + prev_aux`, where `prev_aux` comes from the previous row.
    Addc,
    circuit flags: [AddOperands, UsePreviousAux, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
