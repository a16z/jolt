use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual XOR with the second operand rotated left by one bit.
    VirtualXorRotL1,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
