use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I LD: load doubleword (64 bits). Identity operation.
    Ld,
    circuit flags: [AddOperands, Load, Assert],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
