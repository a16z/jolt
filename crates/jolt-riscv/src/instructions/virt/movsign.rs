use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual MOVSIGN: returns all-ones if `x` is negative (signed), otherwise zero.
    MovSign,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
