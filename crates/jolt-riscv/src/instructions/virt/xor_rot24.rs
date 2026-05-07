use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual XOR then rotate right by 24 bits.
    VirtualXorRot24,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
