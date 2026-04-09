//! RV64I load instructions that extract and extend bytes from a memory word.
//!
//! In Jolt's execution model, `x` contains the loaded value from memory
//! and the instruction performs sign/zero extension to 64 bits.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// RV64I LB: load byte, sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lb;

impl Instruction for Lb {
    #[inline]
    fn name(&self) -> &'static str {
        "LB"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        (x as i8) as i64 as u64
    }
}

/// RV64I LBU: load byte, zero-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lbu;

impl Instruction for Lbu {
    #[inline]
    fn name(&self) -> &'static str {
        "LBU"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        x & 0xFF
    }
}

/// RV64I LH: load halfword (16 bits), sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lh;

impl Instruction for Lh {
    #[inline]
    fn name(&self) -> &'static str {
        "LH"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        (x as i16) as i64 as u64
    }
}

/// RV64I LHU: load halfword, zero-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lhu;

impl Instruction for Lhu {
    #[inline]
    fn name(&self) -> &'static str {
        "LHU"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        x & 0xFFFF
    }
}

/// RV64I LW: load word (32 bits), sign-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lw;

impl Instruction for Lw {
    #[inline]
    fn name(&self) -> &'static str {
        "LW"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        (x as i32) as i64 as u64
    }
}

/// RV64I LWU: load word, zero-extended to 64 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Lwu;

impl Instruction for Lwu {
    #[inline]
    fn name(&self) -> &'static str {
        "LWU"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        x & 0xFFFF_FFFF
    }
}

/// RV64I LD: load doubleword (64 bits). Identity operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(Load)]
pub struct Ld;

impl Instruction for Ld {
    #[inline]
    fn name(&self) -> &'static str {
        "LD"
    }

    #[inline]
    fn execute(&self, x: u64, _y: u64) -> u64 {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lb_sign_extends() {
        assert_eq!(Lb.execute(0x80, 0), 0xFFFF_FFFF_FFFF_FF80); // -128
        assert_eq!(Lb.execute(0x7F, 0), 0x7F); // +127
    }

    #[test]
    fn lbu_zero_extends() {
        assert_eq!(Lbu.execute(0x80, 0), 0x80);
        assert_eq!(Lbu.execute(0xFF_FF, 0), 0xFF);
    }

    #[test]
    fn lh_sign_extends() {
        assert_eq!(Lh.execute(0x8000, 0), 0xFFFF_FFFF_FFFF_8000);
        assert_eq!(Lh.execute(0x7FFF, 0), 0x7FFF);
    }

    #[test]
    fn lhu_zero_extends() {
        assert_eq!(Lhu.execute(0x8000, 0), 0x8000);
    }

    #[test]
    fn lw_sign_extends() {
        assert_eq!(Lw.execute(0x8000_0000, 0), 0xFFFF_FFFF_8000_0000);
        assert_eq!(Lw.execute(0x7FFF_FFFF, 0), 0x7FFF_FFFF);
    }

    #[test]
    fn lwu_zero_extends() {
        assert_eq!(Lwu.execute(0x8000_0000, 0), 0x8000_0000);
    }

    #[test]
    fn ld_identity() {
        assert_eq!(Ld.execute(0xDEAD_BEEF_CAFE_BABE, 0), 0xDEAD_BEEF_CAFE_BABE);
    }
}
