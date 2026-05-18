use crate::jolt_instruction;

jolt_instruction!(
    /// BN254 Fr FieldSub: `FReg[frd] = FReg[frs1] − FReg[frs2]`.
    ///
    /// Decoded from funct7=0x40 + funct3=0x05.
    FieldSub,
    circuit flags: [IsFieldSub],
    instruction flags: []
);
