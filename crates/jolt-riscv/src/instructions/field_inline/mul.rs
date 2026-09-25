use crate::jolt_instruction;

jolt_instruction!(
    /// Field-inline multiply: `field_rd = field_rs1 * field_rs2`.
    FieldMul,
    circuit flags: [FieldMul],
    instruction flags: []
);
