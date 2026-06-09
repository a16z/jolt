use crate::jolt_instruction;

jolt_instruction!(
    /// Field-inline add: `frd = fr1 + fr2`.
    FieldAdd
);

jolt_instruction!(
    /// Field-inline subtract: `frd = fr1 - fr2`.
    FieldSub
);

jolt_instruction!(
    /// Field-inline multiply: `frd = fr1 * fr2`.
    FieldMul
);

jolt_instruction!(
    /// Field-inline inverse: `frd = fr1^-1`.
    FieldInv
);

jolt_instruction!(
    /// Field-inline equality assertion over two field registers.
    FieldAssertEq
);

jolt_instruction!(
    /// Bridge an ordinary x-register value into a field register.
    FieldLoadFromX
);

jolt_instruction!(
    /// Bridge a field-register value into an ordinary x-register.
    FieldStoreToX
);

jolt_instruction!(
    /// Load an immediate field value into a field register.
    FieldLoadImm
);
