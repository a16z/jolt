use crate::jolt_instruction;

jolt_instruction!(
    /// RV64M REM: signed remainder. Returns `x` on division by zero,
    /// returns 0 when `x == i64::MIN && y == -1`.
    Rem,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
