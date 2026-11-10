use crate::jolt_instruction;

jolt_instruction!(
    /// Floor-addressed doubleword store for byte accesses: `rs2` is written to
    /// the doubleword containing `ea = rs1 + imm`.
    ///
    /// The lookup slot carries `AddressOffsetB(ea) = ea & 7`, which R1CS
    /// constraint 0 subtracts from `ea` to form `RamAddress`.
    StoreB,
    circuit flags: [Store, AddOperands],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
