use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SB: store byte (lowest 8 bits).
    Sb,
    circuit flags: [Store],
    instruction flags: []
);
