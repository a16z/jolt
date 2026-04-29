use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I EBREAK: breakpoint trap.
    Ebreak,
    circuit flags: [],
    instruction flags: []
);
