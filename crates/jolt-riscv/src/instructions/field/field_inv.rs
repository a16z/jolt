use crate::jolt_instruction;

jolt_instruction!(
    /// BN254 Fr FieldInv: `FReg[frd] = FReg[frs1]⁻¹`.
    FieldInv,
    circuit flags: [IsFieldInv],
    instruction flags: []
);
