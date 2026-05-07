use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual ROTRI: rotate right using a bitmask immediate.
    VirtualRotri,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
