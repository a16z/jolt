use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual CHANGE_DIVISOR: transforms divisor for signed division overflow.
    /// Returns the divisor unchanged, unless dividend == MIN && divisor == -1,
    /// in which case returns 1 to avoid overflow.
    VirtualChangeDivisor,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
