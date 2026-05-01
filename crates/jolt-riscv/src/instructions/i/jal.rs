use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I JAL: jump and link. `rd = PC + 4; PC = PC + imm`.
    Jal,
    circuit flags: [AddOperands, Jump],
    instruction flags: [LeftOperandIsPC, RightOperandIsImm]
);
