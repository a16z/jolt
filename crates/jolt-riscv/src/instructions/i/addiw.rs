use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I ADDIW: 32-bit add immediate, sign-extended to 64 bits.
    AddiW,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
