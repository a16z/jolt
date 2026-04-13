//! RV64I shift instructions operating on full 64-bit values.
//! Shift amount is masked to 6 bits (0..63) per the RISC-V spec.
//!
//! These set `WriteLookupOutputToRD` (architectural rd-write) but have
//! no lookup table — the VM decomposes them into virtual shift sequences.

use jolt_riscv_derive::Flags;
use serde::{Deserialize, Serialize};

use crate::Instruction;

/// RV64I SLL: shift left logical. Shift amount from lower 6 bits of `y`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Sll;

impl Instruction for Sll {
    #[inline]
    fn name(&self) -> &'static str {
        "SLL"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x << (y & 63)
    }
}

/// RV64I SLLI: shift left logical by immediate. Immediate already masked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SllI;

impl Instruction for SllI {
    #[inline]
    fn name(&self) -> &'static str {
        "SLLI"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x << (y & 63)
    }
}

/// RV64I SRL: shift right logical. Shift amount from lower 6 bits of `y`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Srl;

impl Instruction for Srl {
    #[inline]
    fn name(&self) -> &'static str {
        "SRL"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x >> (y & 63)
    }
}

/// RV64I SRLI: shift right logical by immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SrlI;

impl Instruction for SrlI {
    #[inline]
    fn name(&self) -> &'static str {
        "SRLI"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        x >> (y & 63)
    }
}

/// RV64I SRA: shift right arithmetic. Preserves sign bit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsRs2Value)]
pub struct Sra;

impl Instruction for Sra {
    #[inline]
    fn name(&self) -> &'static str {
        "SRA"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        ((x as i64) >> (y & 63)) as u64
    }
}

/// RV64I SRAI: shift right arithmetic by immediate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, Flags)]
#[circuit(WriteLookupOutputToRD)]
#[instruction(LeftOperandIsRs1Value, RightOperandIsImm)]
pub struct SraI;

impl Instruction for SraI {
    #[inline]
    fn name(&self) -> &'static str {
        "SRAI"
    }

    #[inline]
    fn execute(&self, x: u64, y: u64) -> u64 {
        ((x as i64) >> (y & 63)) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sll_basic() {
        assert_eq!(Sll.execute(1, 10), 1024);
        assert_eq!(Sll.execute(1, 63), 1 << 63);
    }

    #[test]
    fn sll_masks_shift_amount() {
        // Shift by 64 should wrap to shift by 0
        assert_eq!(Sll.execute(1, 64), 1);
    }

    #[test]
    fn srl_basic() {
        assert_eq!(Srl.execute(1024, 10), 1);
        assert_eq!(Srl.execute(u64::MAX, 63), 1);
    }

    #[test]
    fn sra_sign_extends() {
        let neg = (-1024i64) as u64;
        let result = Sra.execute(neg, 4);
        assert_eq!(result, (-64i64) as u64);
    }

    #[test]
    fn sra_positive() {
        assert_eq!(Sra.execute(1024, 4), 64);
    }

    #[test]
    fn immediate_variants_match() {
        assert_eq!(Sll.execute(42, 5), SllI.execute(42, 5));
        assert_eq!(Srl.execute(42, 5), SrlI.execute(42, 5));
        assert_eq!(Sra.execute(42, 5), SraI.execute(42, 5));
    }
}
