use crate::jolt_instruction;

jolt_instruction!(
    /// Sign-extending parallel bit extract: packs `rs1`'s bits at `rs2`'s set
    /// positions and sign-extends by the extracted window's top bit. With `rs2`
    /// a contiguous window mask, this is the sign-extended lane of `rs1` at the
    /// mask's byte offset (the fused extract for sub-word loads).
    PextSigned,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
