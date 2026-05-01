use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual SHIFT_RIGHT_BITMASKI: bitmask for the shift amount stored in the immediate.
    VirtualShiftRightBitmaski,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [RightOperandIsImm]
);
