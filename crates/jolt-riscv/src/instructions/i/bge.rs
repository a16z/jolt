use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I BGE: branch if greater than or equal (signed).
    Bge,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch]
);
