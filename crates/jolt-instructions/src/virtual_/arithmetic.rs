//! Virtual arithmetic instructions used internally by the Jolt VM.

use crate::macros::define_instruction;
use crate::opcodes;

define_instruction!(
    /// Virtual POW2: computes `2^y` where the exponent is taken from the lower 6 bits of `y`.
    Pow2, opcodes::POW2, "POW2",
    |_x, y| 1u64 << (y & 63)
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn pow2_basic() {
        assert_eq!(Pow2.execute(0, 0), 1);
        assert_eq!(Pow2.execute(0, 10), 1024);
        assert_eq!(Pow2.execute(0, 63), 1 << 63);
    }

    #[test]
    fn pow2_masks_exponent() {
        assert_eq!(Pow2.execute(0, 64), 1); // 64 & 63 = 0
    }
}
