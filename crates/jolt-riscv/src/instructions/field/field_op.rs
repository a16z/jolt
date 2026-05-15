use crate::jolt_instruction;

jolt_instruction!(
    /// BN254 Fr FieldOp: arithmetic over BN254 Fr (FMUL/FADD/FSUB/FINV).
    ///
    /// The specific arithmetic variant is selected at runtime by the encoded
    /// funct3 (0x02=FMUL, 0x03=FADD, 0x04=FINV, 0x05=FSUB). Because the
    /// `NormalizedInstruction` row does not carry funct3, the static circuit
    /// flag set here is empty; the appropriate `IsFieldMul/Add/Sub/Inv` flag
    /// is populated downstream by the FR witness-gen path (Phase 2).
    FieldOp
);
