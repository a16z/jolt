use crate::jolt_instruction;

jolt_instruction!(
    /// Load a 64-bit word into integer register `rd` using the ordinary RV64
    /// load constraints, and update field register `rs2` to `2^64 · rs2 + word`.
    FieldLoadAccumulateFromMemory,
    circuit flags: [FieldLoadAccumulateFromMemory, Load],
    instruction flags: []
);
