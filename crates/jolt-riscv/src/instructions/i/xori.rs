use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I XORI: bitwise exclusive OR with sign-extended immediate.
    XorI,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
