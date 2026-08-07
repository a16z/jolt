use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SLLIW: 32-bit shift left logical by immediate, sign-extended.
    SllIW,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
