use crate::jolt_instruction;

jolt_instruction!(
    /// BN254 Fr FieldSub: `FReg[frd] = FReg[frs1] − FReg[frs2]`.
    FieldSub,
    circuit flags: [IsFieldSub],
    instruction flags: []
);
