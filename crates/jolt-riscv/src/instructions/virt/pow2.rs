use crate::jolt_instruction;

jolt_instruction!(
    /// Virtual POW2: computes `2^rs1` using the low 6 bits of `rs1`.
    Pow2,
    circuit flags: [AddOperands, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value]
);
