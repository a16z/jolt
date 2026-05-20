use crate::jolt_instruction;

jolt_instruction!(
    /// BN254 Fr FieldSLL128: integer→field shift-left-128 on-ramp.
    /// `FReg[frd] = XReg[rs1] · 2^128` (lands in limb 2).
    FieldSLL128,
    circuit flags: [IsFieldSLL128],
    instruction flags: []
);
