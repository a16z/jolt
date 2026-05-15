use crate::jolt_instruction;

jolt_instruction!(
    /// BN254 Fr FieldSLL64: integer→field shift-left-64 on-ramp.
    /// `FReg[frd] = XReg[rs1] · 2^64` (lands in limb 1).
    FieldSLL64,
    circuit flags: [IsFieldSLL64],
    instruction flags: []
);
