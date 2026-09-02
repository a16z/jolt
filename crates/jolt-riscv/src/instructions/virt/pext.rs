use crate::jolt_instruction;

jolt_instruction!(
    /// Parallel bit extract: packs `rs1`'s bits at `rs2`'s set positions,
    /// zero-extended. With `rs2` a contiguous window mask, this is the
    /// zero-extended lane of `rs1` at the mask's byte offset (the fused
    /// extract for unsigned sub-word loads).
    Pext,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
