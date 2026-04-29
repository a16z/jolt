use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SRA: shift right arithmetic. Preserves sign bit.
    Sra,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
