use crate::jolt_instruction;

jolt_instruction!(
    /// Aligned containing-doubleword address: `(rs1 + imm) & !7`, the fused
    /// ADDI + ANDI(-8) of the sub-word memory sequences. The lookup index is
    /// the unwrapped sum `rs1 + imm`; the table drops the carry bit and
    /// clears bits 0-2.
    AlignAddr,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
