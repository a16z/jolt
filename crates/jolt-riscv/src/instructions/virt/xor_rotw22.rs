use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual XOR then rotate right word by 22 bits.
    VirtualXorRotW22,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
