use crate::jolt_instruction;

jolt_instruction!(
    /// RV64M DIVU: unsigned division. Returns `u64::MAX` on division by zero.
    DivU,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
