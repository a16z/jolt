use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual SHIFT_RIGHT_BITMASK: bitmask for the shift amount stored in `rs1`.
    VirtualShiftRightBitmask,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value]
);
