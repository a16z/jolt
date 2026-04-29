use crate::jolt_instruction;

jolt_instruction!(
    /// RV64I SUBW: 32-bit subtract, sign-extended to 64 bits.
    SubW,
    circuit flags: [SubtractOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
