use crate::jolt_instruction;

jolt_instruction!(
    /// Floor-addressed doubleword load for byte accesses: `rd` receives the
    /// doubleword containing `ea = rs1 + imm`.
    ///
    /// The lookup slot carries `AddressOffsetB(ea) = ea & 7`, which R1CS
    /// constraint 0 subtracts from `ea` to form `RamAddress`.
    LoadB,
    circuit flags: [Load, AddOperands],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
