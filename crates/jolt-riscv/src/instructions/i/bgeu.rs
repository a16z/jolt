use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I BGEU: branch if greater than or equal (unsigned).
    BgeU,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch]
);
