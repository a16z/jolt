use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I LBU: load byte, zero-extended to 64 bits.
    Lbu,
    circuit flags: [Load],
    instruction flags: []
);
