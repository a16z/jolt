use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I BNE: branch if not equal. Returns 1 when `rs1 != rs2`.
    Bne,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch]
);
