use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I LWU: load word, zero-extended to 64 bits.
    Lwu,
    circuit flags: [Load],
    instruction flags: []
);
