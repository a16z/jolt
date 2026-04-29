use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual SRL: logical right shift using a bitmask-encoded shift amount.
    VirtualSrl,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
