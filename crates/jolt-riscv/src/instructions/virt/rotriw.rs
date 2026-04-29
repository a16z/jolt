use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual ROTRIW: 32-bit rotate right using a bitmask immediate, zero-extended to 64 bits.
    VirtualRotriw,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsImm]
);
