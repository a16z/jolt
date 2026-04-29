use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual ZERO_EXTEND_WORD: zero-extends a 32-bit value to 64 bits.
    VirtualZeroExtendWord,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value]
);
