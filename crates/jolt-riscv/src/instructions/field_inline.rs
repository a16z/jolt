use crate::jolt_instruction;

jolt_instruction!(
    /// Field-inline add: `field_rd = field_rs1 + field_rs2`.
    FieldAdd
);

jolt_instruction!(
    /// Field-inline subtract: `field_rd = field_rs1 - field_rs2`.
    FieldSub
);

jolt_instruction!(
    /// Field-inline multiply: `field_rd = field_rs1 * field_rs2`.
    FieldMul
);

jolt_instruction!(
    /// Field-inline inverse: `field_rd = field_rs1^-1`.
    FieldInv
);

jolt_instruction!(
    /// Field-inline equality assertion over two field registers.
    FieldAssertEq
);

jolt_instruction!(
    /// Fold an ordinary x-register word into a field accumulator:
    /// `field_rd = field_rd · 2^64 + x_rs1`.
    FieldLoadAccumulateFromRegister
);

jolt_instruction!(
    /// Assert that a field register is zero without modifying either register file.
    FieldAssertZero
);

jolt_instruction!(
    /// Load an immediate field value into a field register.
    FieldLoadImm
);

jolt_instruction!(
    /// Load a 64-bit word from memory and fold it in: field register
    /// `rs2` becomes `2^64 · rs2 + word`, the word again an `LD` into the
    /// scratch x-register `rd`.
    FieldLoadAccumulateFromMemory,
    circuit flags: [Load],
    instruction flags: []
);

jolt_instruction!(
    /// Supply a range-checked advice limb in x-register `rd`, with
    /// `field_rs1 = rd + 2^64 · field_rs2` in the proof field. Canonical integer
    /// readout requires a guest range check after the final store.
    FieldAdviceLimb,
    circuit flags: [Advice, WriteLookupOutputToRD],
    instruction flags: []
);
