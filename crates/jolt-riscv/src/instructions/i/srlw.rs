use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SRLW: 32-bit shift right logical, sign-extended to 64 bits.
    SrlW,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
