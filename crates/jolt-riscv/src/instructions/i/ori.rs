use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I ORI: bitwise OR with sign-extended immediate.
    OrI,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
