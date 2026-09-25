use crate::jolt_instruction;

jolt_instruction!(
    /// Fold an ordinary x-register word into a field accumulator:
    /// `field_rd = field_rd · 2^64 + x_rs1`.
    FieldLoadAccumulateFromRegister,
    circuit flags: [FieldLoadAccumulateFromRegister],
    instruction flags: []
);
