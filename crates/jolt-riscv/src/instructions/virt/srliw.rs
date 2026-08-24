use crate::jolt_instruction;

jolt_instruction!(
    /// Word-width logical right shift using a bitmask immediate.
    VirtualSrliw,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
