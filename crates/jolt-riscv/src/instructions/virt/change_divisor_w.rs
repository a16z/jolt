use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual CHANGE_DIVISOR_W: 32-bit version of change divisor.
    VirtualChangeDivisorW,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
