use crate::jolt_instruction;

jolt_instruction!(
    /// RV64M REMU: unsigned remainder. Returns `x` on division by zero.
    RemU,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
