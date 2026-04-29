use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SH: store halfword (lowest 16 bits).
    Sh,
    circuit flags: [Store],
    instruction flags: []
);
