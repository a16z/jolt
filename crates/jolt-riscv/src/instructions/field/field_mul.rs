use crate::jolt_instruction;

jolt_instruction!(
    /// BN254 Fr FieldMul: `FReg[frd] = FReg[frs1] · FReg[frs2]`.
    FieldMul,
    circuit flags: [IsFieldMul],
    instruction flags: []
);
