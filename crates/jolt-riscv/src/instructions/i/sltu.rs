use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SLTU: set if less than (unsigned). `rd = (rs1 < rs2) ? 1 : 0`.
    SltU,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
