use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual XOR then rotate right word by 8 bits.
    VirtualXorRotW8,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
