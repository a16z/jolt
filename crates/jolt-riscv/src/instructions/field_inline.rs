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
    /// Bridge a field-register value into an ordinary x-register. The write
    /// is range-bound through the instruction lookup like `VirtualAdvice`:
    /// the rd value is the (non-interleaved) `RangeCheck` lookup operand and
    /// the FR bridge rows pin both the operand and the write to
    /// `FieldRs1Value`, so the store is satisfiable only when the field value
    /// fits in 64 bits (`jolt-r1cs` `field_constraints`).
    FieldStoreToX,
    circuit flags: [Advice, WriteLookupOutputToRD],
    instruction flags: []
);

jolt_instruction!(
    /// Load an immediate field value into a field register.
    FieldLoadImm
);
