use crate::jolt_instruction;

jolt_instruction!(
    /// Unsigned byte from `rs1` at byte offset `(rs2 + imm) & 7`.
    /// Both right sources bind the effective address, including signed imm.
    ExtractByteUnsigned,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value, RightOperandIsImm]
);
