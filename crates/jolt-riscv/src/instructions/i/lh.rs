use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I LH: load halfword (16 bits), sign-extended to 64 bits.
    Lh,
    circuit flags: [Load],
    instruction flags: []
);
