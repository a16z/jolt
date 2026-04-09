//! Virtual XOR-rotate instructions for SHA hash functions.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// Virtual XOR then rotate right by 32 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRot32;

impl Instruction for VirtualXorRot32 {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_XORROT32"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        (x ^ y).rotate_right(32)
    }
}

/// Virtual XOR then rotate right by 24 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRot24;

impl Instruction for VirtualXorRot24 {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_XORROT24"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        (x ^ y).rotate_right(24)
    }
}

/// Virtual XOR then rotate right by 16 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRot16;

impl Instruction for VirtualXorRot16 {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_XORROT16"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        (x ^ y).rotate_right(16)
    }
}

/// Virtual XOR then rotate right by 63 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRot63;

impl Instruction for VirtualXorRot63 {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_XORROT63"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        (x ^ y).rotate_right(63)
    }
}

/// Virtual XOR then rotate right word (32-bit) by 16 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRotW16;

impl Instruction for VirtualXorRotW16 {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_XORROTW16"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let val = (x as u32) ^ (y as u32);
        val.rotate_right(16) as u64
    }
}

/// Virtual XOR then rotate right word by 12 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRotW12;

impl Instruction for VirtualXorRotW12 {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_XORROTW12"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let val = (x as u32) ^ (y as u32);
        val.rotate_right(12) as u64
    }
}

/// Virtual XOR then rotate right word by 8 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRotW8;

impl Instruction for VirtualXorRotW8 {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_XORROTW8"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let val = (x as u32) ^ (y as u32);
        val.rotate_right(8) as u64
    }
}

/// Virtual XOR then rotate right word by 7 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct VirtualXorRotW7;

impl Instruction for VirtualXorRotW7 {
    #[inline]
    fn name(&self) -> &'static str {
        "VIRTUAL_XORROTW7"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        let val = (x as u32) ^ (y as u32);
        val.rotate_right(7) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_rot32() {
        assert_eq!(VirtualXorRot32.execute(0xFF, 0), 0xFF_0000_0000);
    }

    #[test]
    fn xor_rot63() {
        assert_eq!(VirtualXorRot63.execute(1, 0), 2);
    }
}
