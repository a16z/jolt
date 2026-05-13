use crate::jolt_instruction;

jolt_instruction!(
    /// Integer→field FieldSLL192: `FReg[frd] = XReg[rs1] · 2¹⁹²`.
    FieldSLL192,
    circuit flags: [IsFieldSLL192],
    instruction flags: [LeftOperandIsRs1Value]
);
