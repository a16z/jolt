//! Virtual division-related instructions.

use crate::opcodes;

/// Returns 1 if this is the signed division overflow case (MIN / -1), else returns the divisor.
#[inline]
fn change_divisor_64(dividend: u64, divisor: u64) -> u64 {
    if (dividend as i64) == i64::MIN && (divisor as i64) == -1 {
        1
    } else {
        divisor
    }
}

/// 32-bit version of [`change_divisor_64`].
#[inline]
fn change_divisor_32(dividend: u64, divisor: u64) -> u64 {
    if dividend as u32 == i32::MIN as u32 && divisor as u32 == u32::MAX {
        1
    } else {
        divisor
    }
}

define_instruction!(
    /// Virtual CHANGE_DIVISOR: transforms divisor for signed division overflow.
    /// Returns the divisor unchanged, unless dividend == MIN && divisor == -1,
    /// in which case returns 1 to avoid overflow.
    VirtualChangeDivisor, opcodes::VIRTUAL_CHANGE_DIVISOR, "VIRTUAL_CHANGE_DIVISOR",
    |x, y| change_divisor_64(x, y),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: VirtualChangeDivisor,
);

define_instruction!(
    /// Virtual CHANGE_DIVISOR_W: 32-bit version of change divisor.
    VirtualChangeDivisorW, opcodes::VIRTUAL_CHANGE_DIVISOR_W, "VIRTUAL_CHANGE_DIVISOR_W",
    |x, y| change_divisor_32(x, y),
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsRs2Value],
    table: VirtualChangeDivisorW,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn change_divisor_normal() {
        assert_eq!(VirtualChangeDivisor.execute(10, 3), 3);
    }

    #[test]
    fn change_divisor_overflow() {
        assert_eq!(
            VirtualChangeDivisor.execute(i64::MIN as u64, (-1i64) as u64),
            1
        );
    }

    #[test]
    fn change_divisor_w_normal() {
        assert_eq!(VirtualChangeDivisorW.execute(10, 3), 3);
    }

    #[test]
    fn change_divisor_w_overflow() {
        assert_eq!(
            VirtualChangeDivisorW.execute(i32::MIN as u64, (-1i32) as u64),
            1
        );
    }
}
