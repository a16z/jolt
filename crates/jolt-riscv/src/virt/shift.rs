//! Virtual shift decomposition instructions.
//!
//! The RV64 shift instructions (SRL, SRA, etc.) are decomposed into
//! virtual sequences that use specialized lookup tables for the sumcheck prover.

/// Computes the bitmask for a right-shift: `((1 << (64 - shift)) - 1) << shift`.
/// Returns `u64::MAX` when `shift == 0`.
#[inline]
fn shift_right_bitmask(shift_amount: u64) -> u64 {
    let shift = shift_amount & 63;
    if shift == 0 {
        u64::MAX
    } else {
        (((1u128 << (64 - shift)) - 1) as u64) << shift
    }
}

define_instruction!(
    /// Virtual SRL: logical right shift decomposition.
    VirtualSrl, "VIRTUAL_SRL",
    |x, y| x >> (y & 63),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: VirtualSRL,
);

define_instruction!(
    /// Virtual SRLI: logical right shift by immediate decomposition.
    VirtualSrli, "VIRTUAL_SRLI",
    |x, y| x >> (y & 63),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: VirtualSRL,
);

define_instruction!(
    /// Virtual SRA: arithmetic right shift decomposition.
    VirtualSra, "VIRTUAL_SRA",
    |x, y| ((x as i64) >> (y & 63)) as u64,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: VirtualSRA,
);

define_instruction!(
    /// Virtual SRAI: arithmetic right shift by immediate decomposition.
    VirtualSrai, "VIRTUAL_SRAI",
    |x, y| ((x as i64) >> (y & 63)) as u64,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: VirtualSRA,
);

define_instruction!(
    /// Virtual SHIFT_RIGHT_BITMASK: bitmask for right-shift amount.
    VirtualShiftRightBitmask, "VIRTUAL_SHIFT_RIGHT_BITMASK",
    |_x, y| shift_right_bitmask(y),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: ShiftRightBitmask,
);

define_instruction!(
    /// Virtual SHIFT_RIGHT_BITMASKI: bitmask for right-shift by immediate.
    VirtualShiftRightBitmaski, "VIRTUAL_SHIFT_RIGHT_BITMASKI",
    |_x, y| shift_right_bitmask(y),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: ShiftRightBitmask,
);

define_instruction!(
    /// Virtual ROTRI: rotate right by immediate.
    VirtualRotri, "VIRTUAL_ROTRI",
    |x, y| x.rotate_right((y & 63) as u32),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: VirtualROTR,
);

define_instruction!(
    /// Virtual ROTRIW: 32-bit rotate right by immediate, sign-extended.
    VirtualRotriw, "VIRTUAL_ROTRIW",
    |x, y| (x as u32).rotate_right((y & 31) as u32) as u64,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: VirtualROTRW,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn virtsrl_basic() {
        assert_eq!(VirtualSrl.execute(1024, 10), 1);
        assert_eq!(VirtualSrl.execute(u64::MAX, 63), 1);
    }

    #[test]
    fn virtsra_sign_extends() {
        let neg = (-1024i64) as u64;
        assert_eq!(VirtualSra.execute(neg, 4), (-64i64) as u64);
    }

    #[test]
    fn virtshift_right_bitmask() {
        assert_eq!(VirtualShiftRightBitmask.execute(0, 0), u64::MAX);
        assert_eq!(VirtualShiftRightBitmask.execute(0, 1), u64::MAX - 1);
    }

    #[test]
    fn virtrotri_basic() {
        assert_eq!(VirtualRotri.execute(1, 1), 1u64 << 63);
        assert_eq!(VirtualRotri.execute(0xFF, 4), 0xF000_0000_0000_000F);
    }
}
