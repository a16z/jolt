use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I ANDI: bitwise AND with sign-extended immediate.
    AndI,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
