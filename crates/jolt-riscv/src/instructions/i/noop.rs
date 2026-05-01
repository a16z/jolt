use crate::jolt_instruction;

jolt_instruction!(
    /// No-operation pseudo-instruction.
    Noop,
    circuit flags: [],
    instruction flags: [IsNoop]
);
