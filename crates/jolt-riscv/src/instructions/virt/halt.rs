use crate::jolt_instruction;

jolt_instruction!(
    /// `jolt.success`: ends a run successfully. A table-free jump (lookup output 0, so it targets
    /// address 0, where only padding sits) that writes its link to `rd`.
    Success,
    circuit flags: [Jump],
    instruction flags: []
);

jolt_instruction!(
    /// Jolt's failure halt, the one-row lowering of `EBREAK`: a table-free jump that writes its
    /// link to `rd`.
    Panic,
    circuit flags: [Jump],
    instruction flags: []
);
