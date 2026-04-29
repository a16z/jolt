use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual REV8W: byte-reverse within the lower 32 bits.
    VirtualRev8W,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value]
);
