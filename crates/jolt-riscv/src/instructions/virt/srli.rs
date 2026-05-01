use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual SRLI: logical right shift using a bitmask immediate.
    VirtualSrli,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
