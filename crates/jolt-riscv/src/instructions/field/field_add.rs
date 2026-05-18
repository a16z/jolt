use crate::jolt_instruction;

jolt_instruction!(
    /// BN254 Fr FieldAdd: `FReg[frd] = FReg[frs1] + FReg[frs2]`.
    ///
    /// Decoded from funct7=0x40 + funct3=0x03.
    FieldAdd,
    circuit flags: [IsFieldAdd],
    instruction flags: []
);
