use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SUB: `rd = rs1 - rs2` (wrapping).
    Sub,
    circuit flags: [SubtractOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
