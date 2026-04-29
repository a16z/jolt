use crate::jolt_instruction;

jolt_instruction!(
    /// Zbb ANDN: bitwise AND-NOT. `rd = rs1 & ~rs2`.
    Andn,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
