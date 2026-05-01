use crate::jolt_instruction;

jolt_instruction!(
    /// RV64M MUL: signed multiply, lower 64 bits of the 128-bit product.
    Mul,
    circuit flags: [MultiplyOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
