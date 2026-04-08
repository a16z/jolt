//! Virtual shift decomposition instructions.
//!
//! The RV64 shift instructions (SRL, SRA, etc.) are decomposed into
//! virtual sequences that use specialized lookup tables for the sumcheck prover.

#[inline]
fn shift_from_bitmask(bitmask: u64) -> u32 {
    bitmask.trailing_zeros()
}

#[inline]
fn word_shift_from_bitmask(bitmask: u64) -> u32 {
    (bitmask as u32).trailing_zeros().min(32)
}

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
    /// Virtual SRL: logical right shift using a bitmask-encoded shift amount.
    VirtualSrl, "VIRTUAL_SRL",
    |x, y| x.wrapping_shr(shift_from_bitmask(y)),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// Virtual SRLI: logical right shift using a bitmask immediate.
    VirtualSrli, "VIRTUAL_SRLI",
    |x, y| x.wrapping_shr(shift_from_bitmask(y)),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

define_instruction!(
    /// Virtual SRA: arithmetic right shift using a bitmask-encoded shift amount.
    VirtualSra, "VIRTUAL_SRA",
    |x, y| ((x as i64).wrapping_shr(shift_from_bitmask(y))) as u64,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
);

define_instruction!(
    /// Virtual SRAI: arithmetic right shift using a bitmask immediate.
    VirtualSrai, "VIRTUAL_SRAI",
    |x, y| ((x as i64).wrapping_shr(shift_from_bitmask(y))) as u64,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

define_instruction!(
    /// Virtual SHIFT_RIGHT_BITMASK: bitmask for the shift amount stored in `rs1`.
    VirtualShiftRightBitmask, "VIRTUAL_SHIFT_RIGHT_BITMASK",
    |x, _y| shift_right_bitmask(x),
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

define_instruction!(
    /// Virtual SHIFT_RIGHT_BITMASKI: bitmask for the shift amount stored in the immediate.
    VirtualShiftRightBitmaski, "VIRTUAL_SHIFT_RIGHT_BITMASKI",
    |_x, y| shift_right_bitmask(y),
    circuit: [AddOperands, WriteLookupOutputToRD],
    instruction: [RightOperandIsImm],
);

define_instruction!(
    /// Virtual ROTRI: rotate right using a bitmask immediate.
    VirtualRotri, "VIRTUAL_ROTRI",
    |x, y| x.rotate_right(shift_from_bitmask(y)),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

define_instruction!(
    /// Virtual ROTRIW: 32-bit rotate right using a bitmask immediate, zero-extended to 64 bits.
    VirtualRotriw, "VIRTUAL_ROTRIW",
    |x, y| (x as u32).rotate_right(word_shift_from_bitmask(y)) as u64,
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn virtsrl_basic() {
        assert_eq!(VirtualSrl.execute(1024, 1 << 10), 1);
        assert_eq!(VirtualSrl.execute(u64::MAX, 1 << 63), 1);
    }

    #[test]
    fn virtsra_sign_extends() {
        let neg = (-1024i64) as u64;
        assert_eq!(VirtualSra.execute(neg, 1 << 4), (-64i64) as u64);
    }

    #[test]
    fn virtshift_right_bitmask() {
        assert_eq!(VirtualShiftRightBitmask.execute(0, 0), u64::MAX);
        assert_eq!(VirtualShiftRightBitmask.execute(1, 0), u64::MAX - 1);
        assert_eq!(VirtualShiftRightBitmaski.execute(0, 1), u64::MAX - 1);
    }

    #[test]
    fn virtrotri_basic() {
        assert_eq!(VirtualRotri.execute(1, 1 << 1), 1u64 << 63);
        assert_eq!(VirtualRotri.execute(0xFF, 1 << 4), 0xF000_0000_0000_000F);
    }

    #[test]
    fn virtrotriw_zero_extends() {
        assert_eq!(VirtualRotriw.execute(0x8000_0000, 1 << 1), 0x4000_0000);
    }
}
