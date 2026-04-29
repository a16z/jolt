use crate::jolt_instruction;

jolt_instruction!(
    /// RV64M DIVW: 32-bit signed division, sign-extended to 64 bits.
    ///
    /// Division by zero returns `u64::MAX`. Overflow (`i32::MIN / -1`) returns `i32::MIN` sign-extended.
    DivW,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
