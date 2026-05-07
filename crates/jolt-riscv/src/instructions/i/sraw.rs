use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SRAW: 32-bit shift right arithmetic, sign-extended to 64 bits.
    SraW,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
