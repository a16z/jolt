use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I LB: load byte, sign-extended to 64 bits.
    Lb,
    circuit flags: [Load],
    instruction flags: []
);
