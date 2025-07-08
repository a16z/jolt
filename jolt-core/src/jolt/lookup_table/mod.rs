use and::AndTable;
use enum_dispatch::enum_dispatch;
use equal::EqualTable;
use halfword_alignment::HalfwordAlignmentTable;
use movsign::MovsignTable;
use not_equal::NotEqualTable;
use or::OrTable;
use pow2::Pow2Table;
use prefixes::PrefixEval;
use range_check::RangeCheckTable;
use serde::{Deserialize, Serialize};
use shift_right_bitmask::ShiftRightBitmaskTable;
use signed_greater_than_equal::SignedGreaterThanEqualTable;
use signed_less_than::SignedLessThanTable;
use std::marker::Sync;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};
use suffixes::{SuffixEval, Suffixes};
use unsigned_greater_than_equal::UnsignedGreaterThanEqualTable;
use unsigned_less_than::UnsignedLessThanTable;
use unsigned_less_than_equal::UnsignedLessThanEqualTable;
use upper_word::UpperWordTable;
use valid_div0::ValidDiv0Table;
use valid_signed_remainder::ValidSignedRemainderTable;
use valid_unsigned_remainder::ValidUnsignedRemainderTable;
use virtual_rotl::VirtualRotlTable;
use virtual_rotr::VirtualRotrTable;
use virtual_sra::VirtualSRATable;
use virtual_srl::VirtualSRLTable;
use xor::XorTable;

use crate::field::JoltField;
use derive_more::From;
use std::fmt::Debug;

#[enum_dispatch]
pub trait JoltLookupTable: Clone + Debug + Send + Sync + Serialize {
    /// Materializes the entire lookup table for this instruction (assuming an 8-bit word size).
    #[cfg(test)]
    fn materialize(&self) -> Vec<u64> {
        (0..1 << 16).map(|i| self.materialize_entry(i)).collect()
    }

    /// Materialize the entry at the given `index` in the lookup table for this instruction.
    fn materialize_entry(&self, index: u64) -> u64;

    /// Evaluates the MLE of this lookup table on the given point `r`.
    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F;
}

pub trait PrefixSuffixDecomposition<const WORD_SIZE: usize>: JoltLookupTable + Default {
    fn suffixes(&self) -> Vec<Suffixes>;
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F;
    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u64 {
        rand::RngCore::next_u64(rng)
    }
}

pub mod prefixes;
pub mod suffixes;

pub mod and;
pub mod equal;
pub mod halfword_alignment;
pub mod movsign;
pub mod not_equal;
pub mod or;
pub mod pow2;
pub mod range_check;
pub mod shift_right_bitmask;
pub mod signed_greater_than_equal;
pub mod signed_less_than;
pub mod sub;
pub mod unsigned_greater_than_equal;
pub mod unsigned_less_than;
pub mod unsigned_less_than_equal;
pub mod upper_word;
pub mod valid_div0;
pub mod valid_signed_remainder;
pub mod valid_unsigned_remainder;
pub mod virtual_rotl;
pub mod virtual_rotr;
pub mod virtual_sra;
pub mod virtual_srl;
pub mod xor;

#[cfg(test)]
pub mod test;

#[derive(Copy, Clone, Debug, From, Serialize, Deserialize, EnumIter, EnumCountMacro)]
#[repr(u8)]
pub enum LookupTables<const WORD_SIZE: usize> {
    RangeCheck(RangeCheckTable<WORD_SIZE>),
    And(AndTable<WORD_SIZE>),
    Or(OrTable<WORD_SIZE>),
    Xor(XorTable<WORD_SIZE>),
    Equal(EqualTable<WORD_SIZE>),
    SignedGreaterThanEqual(SignedGreaterThanEqualTable<WORD_SIZE>),
    UnsignedGreaterThanEqual(UnsignedGreaterThanEqualTable<WORD_SIZE>),
    NotEqual(NotEqualTable<WORD_SIZE>),
    SignedLessThan(SignedLessThanTable<WORD_SIZE>),
    UnsignedLessThan(UnsignedLessThanTable<WORD_SIZE>),
    Movsign(MovsignTable<WORD_SIZE>),
    UpperWord(UpperWordTable<WORD_SIZE>),
    LessThanEqual(UnsignedLessThanEqualTable<WORD_SIZE>),
    ValidSignedRemainder(ValidSignedRemainderTable<WORD_SIZE>),
    ValidUnsignedRemainder(ValidUnsignedRemainderTable<WORD_SIZE>),
    ValidDiv0(ValidDiv0Table<WORD_SIZE>),
    HalfwordAlignment(HalfwordAlignmentTable<WORD_SIZE>),
    Pow2(Pow2Table<WORD_SIZE>),
    ShiftRightBitmask(ShiftRightBitmaskTable<WORD_SIZE>),
    VirtualSRL(VirtualSRLTable<WORD_SIZE>),
    VirtualSRA(VirtualSRATable<WORD_SIZE>),
    VirtualROTLI(VirtualRotlTable<WORD_SIZE>),
    VirtualROTRI(VirtualRotrTable<WORD_SIZE>),
}

impl<const WORD_SIZE: usize> LookupTables<WORD_SIZE> {
    pub fn enum_index(table: &Self) -> usize {
        // Discriminant: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
        let byte = unsafe { *(table as *const Self as *const u8) };
        byte as usize
    }

    #[cfg(test)]
    pub fn materialize(&self) -> Vec<u64> {
        match self {
            LookupTables::RangeCheck(table) => table.materialize(),
            LookupTables::And(table) => table.materialize(),
            LookupTables::Or(table) => table.materialize(),
            LookupTables::Xor(table) => table.materialize(),
            LookupTables::Equal(table) => table.materialize(),
            LookupTables::SignedGreaterThanEqual(table) => table.materialize(),
            LookupTables::UnsignedGreaterThanEqual(table) => table.materialize(),
            LookupTables::NotEqual(table) => table.materialize(),
            LookupTables::SignedLessThan(table) => table.materialize(),
            LookupTables::UnsignedLessThan(table) => table.materialize(),
            LookupTables::Movsign(table) => table.materialize(),
            LookupTables::UpperWord(table) => table.materialize(),
            LookupTables::LessThanEqual(table) => table.materialize(),
            LookupTables::ValidSignedRemainder(table) => table.materialize(),
            LookupTables::ValidUnsignedRemainder(table) => table.materialize(),
            LookupTables::ValidDiv0(table) => table.materialize(),
            LookupTables::HalfwordAlignment(table) => table.materialize(),
            LookupTables::Pow2(table) => table.materialize(),
            LookupTables::ShiftRightBitmask(table) => table.materialize(),
            LookupTables::VirtualSRL(table) => table.materialize(),
            LookupTables::VirtualSRA(table) => table.materialize(),
            LookupTables::VirtualROTLI(table) => table.materialize(),
            LookupTables::VirtualROTRI(table) => table.materialize(),
        }
    }

    pub fn materialize_entry(&self, index: u64) -> u64 {
        match self {
            LookupTables::RangeCheck(table) => table.materialize_entry(index),
            LookupTables::And(table) => table.materialize_entry(index),
            LookupTables::Or(table) => table.materialize_entry(index),
            LookupTables::Xor(table) => table.materialize_entry(index),
            LookupTables::Equal(table) => table.materialize_entry(index),
            LookupTables::SignedGreaterThanEqual(table) => table.materialize_entry(index),
            LookupTables::UnsignedGreaterThanEqual(table) => table.materialize_entry(index),
            LookupTables::NotEqual(table) => table.materialize_entry(index),
            LookupTables::SignedLessThan(table) => table.materialize_entry(index),
            LookupTables::UnsignedLessThan(table) => table.materialize_entry(index),
            LookupTables::Movsign(table) => table.materialize_entry(index),
            LookupTables::UpperWord(table) => table.materialize_entry(index),
            LookupTables::LessThanEqual(table) => table.materialize_entry(index),
            LookupTables::ValidSignedRemainder(table) => table.materialize_entry(index),
            LookupTables::ValidUnsignedRemainder(table) => table.materialize_entry(index),
            LookupTables::ValidDiv0(table) => table.materialize_entry(index),
            LookupTables::HalfwordAlignment(table) => table.materialize_entry(index),
            LookupTables::Pow2(table) => table.materialize_entry(index),
            LookupTables::ShiftRightBitmask(table) => table.materialize_entry(index),
            LookupTables::VirtualSRL(table) => table.materialize_entry(index),
            LookupTables::VirtualSRA(table) => table.materialize_entry(index),
            LookupTables::VirtualROTLI(table) => table.materialize_entry(index),
            LookupTables::VirtualROTRI(table) => table.materialize_entry(index),
        }
    }

    pub fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        match self {
            LookupTables::RangeCheck(table) => table.evaluate_mle(r),
            LookupTables::And(table) => table.evaluate_mle(r),
            LookupTables::Or(table) => table.evaluate_mle(r),
            LookupTables::Xor(table) => table.evaluate_mle(r),
            LookupTables::Equal(table) => table.evaluate_mle(r),
            LookupTables::SignedGreaterThanEqual(table) => table.evaluate_mle(r),
            LookupTables::UnsignedGreaterThanEqual(table) => table.evaluate_mle(r),
            LookupTables::NotEqual(table) => table.evaluate_mle(r),
            LookupTables::SignedLessThan(table) => table.evaluate_mle(r),
            LookupTables::UnsignedLessThan(table) => table.evaluate_mle(r),
            LookupTables::Movsign(table) => table.evaluate_mle(r),
            LookupTables::UpperWord(table) => table.evaluate_mle(r),
            LookupTables::LessThanEqual(table) => table.evaluate_mle(r),
            LookupTables::ValidSignedRemainder(table) => table.evaluate_mle(r),
            LookupTables::ValidUnsignedRemainder(table) => table.evaluate_mle(r),
            LookupTables::ValidDiv0(table) => table.evaluate_mle(r),
            LookupTables::HalfwordAlignment(table) => table.evaluate_mle(r),
            LookupTables::Pow2(table) => table.evaluate_mle(r),
            LookupTables::ShiftRightBitmask(table) => table.evaluate_mle(r),
            LookupTables::VirtualSRL(table) => table.evaluate_mle(r),
            LookupTables::VirtualSRA(table) => table.evaluate_mle(r),
            LookupTables::VirtualROTLI(table) => table.evaluate_mle(r),
            LookupTables::VirtualROTRI(table) => table.evaluate_mle(r),
        }
    }

    pub fn suffixes(&self) -> Vec<Suffixes> {
        match self {
            LookupTables::RangeCheck(table) => table.suffixes(),
            LookupTables::And(table) => table.suffixes(),
            LookupTables::Or(table) => table.suffixes(),
            LookupTables::Xor(table) => table.suffixes(),
            LookupTables::Equal(table) => table.suffixes(),
            LookupTables::SignedGreaterThanEqual(table) => table.suffixes(),
            LookupTables::UnsignedGreaterThanEqual(table) => table.suffixes(),
            LookupTables::NotEqual(table) => table.suffixes(),
            LookupTables::SignedLessThan(table) => table.suffixes(),
            LookupTables::UnsignedLessThan(table) => table.suffixes(),
            LookupTables::Movsign(table) => table.suffixes(),
            LookupTables::UpperWord(table) => table.suffixes(),
            LookupTables::LessThanEqual(table) => table.suffixes(),
            LookupTables::ValidSignedRemainder(table) => table.suffixes(),
            LookupTables::ValidUnsignedRemainder(table) => table.suffixes(),
            LookupTables::ValidDiv0(table) => table.suffixes(),
            LookupTables::HalfwordAlignment(table) => table.suffixes(),
            LookupTables::Pow2(table) => table.suffixes(),
            LookupTables::ShiftRightBitmask(table) => table.suffixes(),
            LookupTables::VirtualSRL(table) => table.suffixes(),
            LookupTables::VirtualSRA(table) => table.suffixes(),
            LookupTables::VirtualROTLI(table) => table.suffixes(),
            LookupTables::VirtualROTRI(table) => table.suffixes(),
        }
    }

    pub fn combine<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        match self {
            LookupTables::RangeCheck(table) => table.combine(prefixes, suffixes),
            LookupTables::And(table) => table.combine(prefixes, suffixes),
            LookupTables::Or(table) => table.combine(prefixes, suffixes),
            LookupTables::Xor(table) => table.combine(prefixes, suffixes),
            LookupTables::Equal(table) => table.combine(prefixes, suffixes),
            LookupTables::SignedGreaterThanEqual(table) => table.combine(prefixes, suffixes),
            LookupTables::UnsignedGreaterThanEqual(table) => table.combine(prefixes, suffixes),
            LookupTables::NotEqual(table) => table.combine(prefixes, suffixes),
            LookupTables::SignedLessThan(table) => table.combine(prefixes, suffixes),
            LookupTables::UnsignedLessThan(table) => table.combine(prefixes, suffixes),
            LookupTables::Movsign(table) => table.combine(prefixes, suffixes),
            LookupTables::UpperWord(table) => table.combine(prefixes, suffixes),
            LookupTables::LessThanEqual(table) => table.combine(prefixes, suffixes),
            LookupTables::ValidSignedRemainder(table) => table.combine(prefixes, suffixes),
            LookupTables::ValidUnsignedRemainder(table) => table.combine(prefixes, suffixes),
            LookupTables::ValidDiv0(table) => table.combine(prefixes, suffixes),
            LookupTables::HalfwordAlignment(table) => table.combine(prefixes, suffixes),
            LookupTables::Pow2(table) => table.combine(prefixes, suffixes),
            LookupTables::ShiftRightBitmask(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualSRL(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualSRA(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualROTLI(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualROTRI(table) => table.combine(prefixes, suffixes),
        }
    }
}
