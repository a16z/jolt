use crate::jolt_instruction;

jolt_instruction!(
    /// Word-width arithmetic right shift using a bitmask immediate.
    VirtualSraiw,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
