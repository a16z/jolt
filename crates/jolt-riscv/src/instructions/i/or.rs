use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I OR: bitwise OR of two registers.
    Or,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
