use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I AND: bitwise AND of two registers.
    And,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
