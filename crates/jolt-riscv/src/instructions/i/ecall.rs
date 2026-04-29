use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I ECALL: environment call (syscall).
    Ecall,
    circuit flags: [],
    instruction flags: []
);
