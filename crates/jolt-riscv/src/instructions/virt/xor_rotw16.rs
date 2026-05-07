use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual XOR then rotate right word (32-bit) by 16 bits.
    VirtualXorRotW16,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
