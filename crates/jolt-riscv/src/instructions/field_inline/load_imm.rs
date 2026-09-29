use crate::jolt_instruction;

jolt_instruction!(
    /// Load an immediate field value into a field register.
    FieldLoadImm,
    circuit flags: [FieldLoadImm],
    instruction flags: []
);
