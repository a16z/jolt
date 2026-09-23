use crate::jolt_instruction;

jolt_instruction!(
    /// Word-width logical right shift using a bitmask register.
    VirtualSrlw,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
