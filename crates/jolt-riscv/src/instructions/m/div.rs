use crate::jolt_instruction;

jolt_instruction!(
    /// RV64M DIV: signed division with RISC-V overflow handling.
    ///
    /// Special cases per the RISC-V spec:
    /// - Division by zero returns `u64::MAX` (all bits set, i.e. -1 unsigned).
    /// - `i64::MIN / -1` returns `i64::MIN` (overflow wraps).
    Div,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
