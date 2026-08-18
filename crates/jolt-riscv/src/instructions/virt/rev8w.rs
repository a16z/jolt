use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual REV8W: byte-reverse each 32-bit word.
    VirtualRev8W,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value]
);
