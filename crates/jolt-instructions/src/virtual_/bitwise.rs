//! Virtual bitwise instructions used internally by the Jolt VM.

use crate::opcodes;

define_instruction!(
    /// Virtual MOVSIGN: returns all-ones if `x` is negative (signed), otherwise zero.
    MovSign, opcodes::MOVSIGN, "MOVSIGN",
    |x, _y| if (x as i64) < 0 { u64::MAX } else { 0 },
    circuit: [WriteLookupOutputToRD],
    instruction: [LeftOperandIsRs1Value, RightOperandIsImm],
    table: Movsign,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn movsign_negative() {
        let neg = (-1i64) as u64;
        assert_eq!(MovSign.execute(neg, 42), u64::MAX);
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
        assert_eq!(MovSign.execute(min, 99), u64::MAX);
    }
}
