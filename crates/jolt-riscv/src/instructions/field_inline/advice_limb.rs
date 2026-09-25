use crate::jolt_instruction;

jolt_instruction!(
    /// Supply a range-checked advice limb in x-register `rd`, with
    /// `field_rs1 = rd + 2^64 · field_rs2` in the proof field. Canonical integer
    /// readout requires a zero assertion on the final quotient and an integer
    /// check that the reconstructed value is below the field modulus.
    FieldAdviceLimb,
    circuit flags: [Advice, WriteLookupOutputToRD],
    instruction flags: []
);
