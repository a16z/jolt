use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SLL: shift left logical. Shift amount from lower 6 bits of `y`.
    Sll,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
