use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SRAIW: 32-bit shift right arithmetic by immediate, sign-extended.
    SraIW,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
