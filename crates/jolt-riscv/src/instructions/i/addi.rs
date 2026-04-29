use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I ADDI: `rd = rs1 + imm` (wrapping). Immediate already decoded.
    Addi,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
