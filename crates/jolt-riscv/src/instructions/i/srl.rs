use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SRL: shift right logical. Shift amount from lower 6 bits of `y`.
    Srl,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
