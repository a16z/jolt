use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SRLIW: 32-bit shift right logical by immediate, sign-extended.
    SrlIW,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
