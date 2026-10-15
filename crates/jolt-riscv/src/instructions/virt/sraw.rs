use crate::jolt_instruction;

jolt_instruction!(
    /// Word-width arithmetic right shift using a bitmask register.
    VirtualSraw,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
