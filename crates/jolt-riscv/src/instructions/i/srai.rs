use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SRAI: shift right arithmetic by immediate.
    SraI,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
