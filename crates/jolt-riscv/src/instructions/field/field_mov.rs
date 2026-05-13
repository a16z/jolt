use crate::jolt_instruction;

jolt_instruction!(
    /// Integer→field FieldMov: `FReg[frd] = XReg[rs1] as Fr`.
    FieldMov,
    circuit flags: [IsFieldMov],
    instruction flags: [LeftOperandIsRs1Value]
);
