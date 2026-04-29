use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SD: store doubleword (full 64 bits). Identity operation.
    Sd,
    circuit flags: [Store],
    instruction flags: []
);
