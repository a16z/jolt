use crate::jolt_instruction;

jolt_instruction!(
    /// Extracts and sign-extends the lane selected by the mask in `rs2`.
    VirtualLaneExtractS,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
