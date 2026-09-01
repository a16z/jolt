use crate::jolt_instruction;

jolt_instruction!(
    /// Word-width shift bitmask for the amount stored in `rs1`.
    VirtualShiftRightBitmaskW,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value]
);
