use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I JALR: jump and link register. `rd = PC + 4; PC = (rs1 + imm) & !1`.
    Jalr,
    circuit flags: [AddOperands, Jump],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
