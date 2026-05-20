use crate::jolt_instruction;

jolt_instruction!(
    /// BN254 Fr FieldAssertEq: assert FReg[frs1] == FReg[frs2]; no write.
    FieldAssertEq,
    circuit flags: [IsFieldAssertEq],
    instruction flags: []
);
