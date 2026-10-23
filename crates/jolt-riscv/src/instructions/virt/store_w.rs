use crate::jolt_instruction;

jolt_instruction!(
    /// Floor-addressed doubleword store for word accesses: `rs2` is written to
    /// the doubleword containing `ea = rs1 + imm`.
    ///
    /// The lookup slot carries `AddressOffsetW(ea) = ea & 4`, which R1CS
    /// constraint 0 subtracts from `ea` to form `RamAddress`. An `ea` that is
    /// not a multiple of 4 leaves bit 0 or 1 set in `RamAddress`, which RAM RAF
    /// evaluation rejects.
    StoreW,
    circuit flags: [Store, AddOperands],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
