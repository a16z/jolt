use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SLLW: 32-bit shift left logical, sign-extended to 64 bits.
    SllW,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
