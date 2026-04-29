use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I AUIPC: add upper immediate to PC. `rd = PC + imm`.
    Auipc,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsPC, RightOperandIsImm]
);
