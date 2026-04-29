use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SLTIU: set if less than immediate (unsigned).
    SltIU,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
