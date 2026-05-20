use crate::jolt_instruction;

jolt_instruction!(
    /// BN254 Fr FieldMul: `FReg[frd] = FReg[frs1] · FReg[frs2]`.
    ///
    /// One of four ops decoded from the `FieldOp` opcode (custom-0, funct7=0x40)
    /// by funct3=0x02. The funct3 lives in the encoded instruction word; the
    /// tracer dispatches `(funct7, funct3) → JoltInstructionKind` in
    /// `From<tracer::FieldOp> for NormalizedInstruction`.
    FieldMul,
    circuit flags: [IsFieldMul],
    instruction flags: []
);
