use crate::jolt_instruction;

jolt_instruction!(
    /// Integer→field FieldSLL128: `FReg[frd] = XReg[rs1] · 2¹²⁸`.
    FieldSLL128,
    circuit flags: [IsFieldSLL128],
    instruction flags: [LeftOperandIsRs1Value]
);
