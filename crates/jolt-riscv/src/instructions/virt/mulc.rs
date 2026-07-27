use crate::jolt_instruction;

jolt_instruction!(
    /// Jolt MULC: `rd = low_u64(rs1 * rs2 + prev_aux)`, where `prev_aux` comes from the previous row.
    Mulc,
    circuit flags: [MultiplyOperands, UsePreviousAux, WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
