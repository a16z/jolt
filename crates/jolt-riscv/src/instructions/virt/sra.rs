use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual SRA: arithmetic right shift using a bitmask-encoded shift amount.
    VirtualSra,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
