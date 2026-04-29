use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I LW: load word (32 bits), sign-extended to 64 bits.
    Lw,
    circuit flags: [Load],
    instruction flags: []
);
