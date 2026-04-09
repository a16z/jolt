//! Virtual byte manipulation instructions.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// Virtual REV8W: byte-reverse within the lower 32 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(AddOperands, WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value)]
pub struct VirtualRev8W;

impl Instruction for VirtualRev8W {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_REV8W"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        let w = x as u32;
        w.swap_bytes() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rev8w_basic() {
        assert_eq!(VirtualRev8W.execute(0x0102_0304, 0), 0x0403_0201);
    }

    #[test]
    fn rev8w_zero() {
        assert_eq!(VirtualRev8W.execute(0, 0), 0);
    }
}
