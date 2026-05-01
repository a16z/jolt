use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual MULI: multiply by immediate. `rd = rs1 * imm`.
    MulI,
    circuit flags: [MultiplyOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
