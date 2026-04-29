use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual SIGN_EXTEND_WORD: sign-extends a 32-bit value to 64 bits.
    VirtualSignExtendWord,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value]
);
