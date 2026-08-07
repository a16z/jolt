//! Fused XOR-then-rotate instructions for the Keccak-f[1600] rho step.
//!
//! One instruction per distinct rho rotation amount (expressed as a right
//! rotation, `64 - rho_left_rotation`), mirroring the Blake
//! `VirtualXorRot*` family. Together with `VirtualXorRot63` these cover
//! all 24 nonzero Keccak rotations, letting the keccak256 inline fuse its
//! theta-apply XOR into each rho rotation row.

use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual XOR then rotate right by 2 bits (Keccak-f rho fusion).
    VirtualXorRot2,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 3 bits (Keccak-f rho fusion).
    VirtualXorRot3,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 8 bits (Keccak-f rho fusion).
    VirtualXorRot8,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 9 bits (Keccak-f rho fusion).
    VirtualXorRot9,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 19 bits (Keccak-f rho fusion).
    VirtualXorRot19,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 20 bits (Keccak-f rho fusion).
    VirtualXorRot20,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 21 bits (Keccak-f rho fusion).
    VirtualXorRot21,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 23 bits (Keccak-f rho fusion).
    VirtualXorRot23,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 25 bits (Keccak-f rho fusion).
    VirtualXorRot25,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 28 bits (Keccak-f rho fusion).
    VirtualXorRot28,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 36 bits (Keccak-f rho fusion).
    VirtualXorRot36,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 37 bits (Keccak-f rho fusion).
    VirtualXorRot37,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 39 bits (Keccak-f rho fusion).
    VirtualXorRot39,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 43 bits (Keccak-f rho fusion).
    VirtualXorRot43,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 44 bits (Keccak-f rho fusion).
    VirtualXorRot44,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 46 bits (Keccak-f rho fusion).
    VirtualXorRot46,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 49 bits (Keccak-f rho fusion).
    VirtualXorRot49,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 50 bits (Keccak-f rho fusion).
    VirtualXorRot50,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 54 bits (Keccak-f rho fusion).
    VirtualXorRot54,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 56 bits (Keccak-f rho fusion).
    VirtualXorRot56,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 58 bits (Keccak-f rho fusion).
    VirtualXorRot58,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 61 bits (Keccak-f rho fusion).
    VirtualXorRot61,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
jolt_instruction!(
    /// Virtual XOR then rotate right by 62 bits (Keccak-f rho fusion).
    VirtualXorRot62,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
