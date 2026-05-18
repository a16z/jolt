use crate::jolt_instruction;

jolt_instruction!(
    /// BN254 Fr FieldInv: `FReg[frd] = FReg[frs1]⁻¹`.
    ///
    /// Decoded from funct7=0x40 + funct3=0x04. FINV(0) is unsatisfiable —
    /// the SDK guards via `Fr::inverse() -> Option<Fr>`, and the tracer
    /// panics if execute() reaches that branch with frs1 = 0.
    FieldInv,
    circuit flags: [IsFieldInv],
    instruction flags: []
);
