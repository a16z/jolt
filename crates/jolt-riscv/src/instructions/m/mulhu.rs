use crate::jolt_instruction;

jolt_instruction!(
    /// RV64M MULHU: unsigned×unsigned multiply, upper 64 bits.
    MulHU,
    circuit flags: [MultiplyOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
