use crate::jolt_instruction;

jolt_instruction!(
    /// Field-inline equality assertion over two field registers.
    FieldAssertEq,
    circuit flags: [FieldAssertEq],
    instruction flags: []
);
