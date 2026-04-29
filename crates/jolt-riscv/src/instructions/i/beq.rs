use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I BEQ: branch if equal. Returns 1 when `rs1 == rs2`.
    Beq,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch]
);
