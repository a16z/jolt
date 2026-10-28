use crate::jolt_instruction;

jolt_instruction!(
    /// Floor-addressed doubleword load for halfword accesses: `rd` receives
    /// the doubleword containing `ea = rs1 + imm`.
    ///
    /// The lookup slot carries `AddressOffsetH(ea) = ea & 6`, which R1CS
    /// constraint 0 subtracts from `ea` to form `RamAddress`. An odd `ea`
    /// leaves bit 0 set in `RamAddress`, which RAM RAF evaluation rejects.
    LoadH,
    circuit flags: [Load, AddOperands],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
