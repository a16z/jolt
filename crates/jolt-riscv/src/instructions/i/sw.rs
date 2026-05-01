use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SW: store word (lowest 32 bits).
    Sw,
    circuit flags: [Store],
    instruction flags: []
);
