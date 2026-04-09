//! Virtual bitwise instructions used internally by the Jolt VM.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// Virtual MOVSIGN: returns all-ones if `x` is negative (signed), otherwise zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct MovSign;

impl Instruction for MovSign {
    #[inline]
    fn name(&self) -> &'static str {
        "MOVSIGN"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        if (x as i64) < 0 {
            u64::MAX
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
