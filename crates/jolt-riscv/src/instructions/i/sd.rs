use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SD: store doubleword (full 64 bits). Identity operation.
    Sd,
    circuit flags: [AddOperands, Store, Assert],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
