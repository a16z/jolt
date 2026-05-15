use crate::jolt_instruction;

jolt_instruction!(
    /// BN254 Fr FieldSLL192: integer→field shift-left-192 on-ramp.
    /// `FReg[frd] = XReg[rs1] · 2^192` (lands in limb 3).
    FieldSLL192,
    circuit flags: [IsFieldSLL192],
    instruction flags: []
);
