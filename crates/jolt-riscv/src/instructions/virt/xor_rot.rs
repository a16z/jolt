use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual XOR then rotate right by the immediate, one of
    /// [`XOR_ROT_ROTATIONS`].
    VirtualXorRot,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);

/// Rotations with a lookup table: BLAKE2b's four and Keccak-f[1600]'s
/// twenty-three distinct nonzero rho offsets, as rotate-right amounts.
pub const XOR_ROT_ROTATIONS: [u32; 27] = [
    2, 3, 8, 9, 16, 19, 20, 21, 23, 24, 25, 28, 32, 36, 37, 39, 43, 44, 46, 49, 50, 54, 56, 58, 61,
    62, 63,
];
