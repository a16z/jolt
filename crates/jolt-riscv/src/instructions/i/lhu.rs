use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I LHU: load halfword, zero-extended to 64 bits.
    Lhu,
    circuit flags: [Load],
    instruction flags: []
);
