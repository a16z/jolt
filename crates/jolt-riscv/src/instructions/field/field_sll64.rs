use crate::jolt_instruction;

jolt_instruction!(
    /// Integer→field FieldSLL64: `FReg[frd] = XReg[rs1] · 2⁶⁴`.
    FieldSLL64,
    circuit flags: [IsFieldSLL64],
    instruction flags: [LeftOperandIsRs1Value]
);
