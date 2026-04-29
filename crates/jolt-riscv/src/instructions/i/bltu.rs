use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I BLTU: branch if less than (unsigned).
    BltU,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value, Branch]
);
