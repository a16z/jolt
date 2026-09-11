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

jolt_instruction!(
    /// Load a 64-bit word from memory into a field register: an `LD` into the
    /// scratch x-register `rd` (the ordinary load rows bind the word) whose
    /// loaded value the FR row copies into field register `rs2`.
    FieldLoadWord,
    circuit flags: [Load],
    instruction flags: []
);

jolt_instruction!(
    /// Load the low word of a two-limb operand and fold it in: field register
    /// `rs2` becomes `2^64 · rs2 + word`, the word again an `LD` into the
    /// scratch x-register `rd`.
    FieldLoadWordHi,
    circuit flags: [Load],
    instruction flags: []
);

jolt_instruction!(
    /// Supply a range-checked advice limb in x-register `rd`, with
    /// `frs1 = rd + 2^64 · frs2` in the proof field. Canonical integer
    /// readout requires a guest range check after the final store.
    FieldAdviceLimb,
    circuit flags: [Advice, WriteLookupOutputToRD],
    instruction flags: []
);
