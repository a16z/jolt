//! Virtual bitwise instructions used internally by the Jolt VM.

use crate::macros::define_instruction;
use crate::opcodes;

define_instruction!(
    /// Virtual MOVSIGN: conditional move based on the sign bit of `x`.
    ///
    /// Returns `y` if `x` is negative (as signed i64), otherwise returns 0.
    /// Used in the Jolt VM for sign-dependent conditional logic.
    MovSign, opcodes::MOVSIGN, "MOVSIGN",
    |x, y| if (x as i64) < 0 { y } else { 0 }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn movsign_negative() {
        let neg = (-1i64) as u64;
        assert_eq!(MovSign.execute(neg, 42), 42);
    }

    #[test]
    fn movsign_positive() {
        assert_eq!(MovSign.execute(1, 42), 0);
    }

    #[test]
    fn movsign_zero() {
        assert_eq!(MovSign.execute(0, 42), 0);
    }

    #[test]
    fn movsign_min() {
        let min = i64::MIN as u64;
        assert_eq!(MovSign.execute(min, 99), 99);
    }
}
