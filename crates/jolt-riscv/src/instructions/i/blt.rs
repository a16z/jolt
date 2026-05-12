use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I BLT: branch if less than (signed).
    Blt,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch]
);
