use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I FENCE: memory ordering fence.
    Fence,
    circuit flags: [],
    instruction flags: []
);
