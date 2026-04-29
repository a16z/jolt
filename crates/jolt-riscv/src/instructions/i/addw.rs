use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I ADDW: 32-bit add, sign-extended to 64 bits.
    AddW,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
