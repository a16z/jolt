use crate::jolt_instruction;

jolt_instruction!(
    /// BN254 Fr FieldMov: integer→field on-ramp.
    /// `FReg[frd] = [XReg[rs1] as u64, 0, 0, 0]` (low limb).
    FieldMov,
    circuit flags: [IsFieldMov],
    instruction flags: []
);
