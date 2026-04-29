use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SLLI: shift left logical by immediate. Immediate already masked.
    SllI,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
