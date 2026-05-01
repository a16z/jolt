use crate::jolt_instruction;

jolt_instruction!(
    /// RV64M MULH: signed×signed multiply, upper 64 bits.
    MulH,
    circuit flags: [],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
