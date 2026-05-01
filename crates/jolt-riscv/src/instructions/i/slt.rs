use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SLT: set if less than (signed). `rd = (rs1 < rs2) ? 1 : 0`.
    Slt,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
