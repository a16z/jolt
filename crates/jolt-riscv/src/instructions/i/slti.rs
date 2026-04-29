use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SLTI: set if less than immediate (signed).
    SltI,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
