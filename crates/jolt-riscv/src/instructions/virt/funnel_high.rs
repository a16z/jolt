use crate::jolt_instruction;

jolt_instruction!(
    /// High half of a funnel shift: with `rs2 = !0 << s`, writes
    /// `(rs1 mod 2^s) << (64 − s)`, and 0 when `s = 0`. A misaligned load
    /// combines it with `Pext(lo, rs2) = lo >> s` to gather the bytes that
    /// spill into the next doubleword.
    FunnelHigh,
    circuit flags: [WriteLookupOutputToRD],
    instruction flags: [LeftOperandIsRs1Value, RightOperandIsRs2Value]
);
