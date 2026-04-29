use crate::jolt_instruction;

jolt_instruction!(
    /// RV64M MULHSU: signed×unsigned multiply, upper 64 bits.
    MulHSU,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
