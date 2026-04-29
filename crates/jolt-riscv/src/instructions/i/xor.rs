use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I XOR: bitwise exclusive OR of two registers.
    Xor,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
