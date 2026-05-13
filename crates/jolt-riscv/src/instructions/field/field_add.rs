use crate::jolt_instruction;

jolt_instruction!(
    /// BN254 Fr FieldAdd: `FReg[frd] = FReg[frs1] + FReg[frs2]`.
    FieldAdd,
    circuit flags: [IsFieldAdd],
    instruction flags: []
);
