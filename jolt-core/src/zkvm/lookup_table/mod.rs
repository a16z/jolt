use and::AndTable;
use andn::AndnTable;
use equal::EqualTable;
use halfword_alignment::HalfwordAlignmentTable;
use lower_half_word::LowerHalfWordTable;
use movsign::MovsignTable;
use mulu_no_overflow::MulUNoOverflowTable;
use not_equal::NotEqualTable;
use or::OrTable;
use pow2::Pow2Table;
use pow2_w::Pow2WTable;
use prefixes::PrefixEval;
use range_check::RangeCheckTable;
use range_check_aligned::RangeCheckAlignedTable;
use serde::{Deserialize, Serialize};
use shift_right_bitmask::ShiftRightBitmaskTable;
use sign_extend_half_word::SignExtendHalfWordTable;
use signed_greater_than_equal::SignedGreaterThanEqualTable;
use signed_less_than::SignedLessThanTable;
use std::marker::Sync;
use strum::EnumCount;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter, IntoStaticStr};
use suffixes::{SuffixEval, Suffixes};
use unsigned_greater_than_equal::UnsignedGreaterThanEqualTable;
use unsigned_less_than::UnsignedLessThanTable;
use unsigned_less_than_equal::UnsignedLessThanEqualTable;
use upper_word::UpperWordTable;
use valid_div0::ValidDiv0Table;
use valid_signed_remainder::ValidSignedRemainderTable;
use valid_unsigned_remainder::ValidUnsignedRemainderTable;
use virtual_change_divisor::VirtualChangeDivisorTable;
use virtual_change_divisor_w::VirtualChangeDivisorWTable;
use virtual_rev8w::VirtualRev8WTable;
use virtual_rotr::VirtualRotrTable;
use virtual_rotrw::VirtualRotrWTable;
use virtual_sra::VirtualSRATable;
use virtual_srl::VirtualSRLTable;
use virtual_xor_rot::VirtualXORROTTable;
use virtual_xor_rotw::VirtualXORROTWTable;
use word_alignment::WordAlignmentTable;
use xor::XorTable;

use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use derive_more::From;
use std::fmt::Debug;

pub trait JoltLookupTable: Clone + Debug + Send + Sync + Serialize {
    /// Materializes the entire lookup table for this instruction (assuming an 8-bit word size).
    #[cfg(test)]
    fn materialize(&self) -> Vec<u64> {
        (0..1 << 16)
            .map(|i| self.materialize_entry(i as u128))
            .collect()
    }

    /// Materialize the entry at the given `index` in the lookup table for this instruction.
    fn materialize_entry(&self, index: u128) -> u64;

    /// Evaluates the MLE of this lookup table on the given point `r`.
    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>;
}

pub trait PrefixSuffixDecomposition<const XLEN: usize>: JoltLookupTable + Default {
    fn suffixes(&self) -> Vec<Suffixes>;
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F;
    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u128 {
        rand::Rng::gen(rng)
    }
}

pub mod prefixes;
pub mod suffixes;

pub mod and;
pub mod andn;
pub mod equal;
pub mod halfword_alignment;
pub mod lower_half_word;
pub mod movsign;
pub mod mulu_no_overflow;
pub mod not_equal;
pub mod or;
pub mod pow2;
pub mod pow2_w;
pub mod range_check;
pub mod range_check_aligned;
pub mod shift_right_bitmask;
pub mod sign_extend_half_word;
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
pub mod virtual_change_divisor;
pub mod virtual_change_divisor_w;
pub mod virtual_rev8w;
pub mod virtual_rotr;
pub mod virtual_rotrw;
pub mod virtual_sra;
pub mod virtual_srl;
pub mod virtual_xor_rot;
pub mod virtual_xor_rotw;
pub mod word_alignment;
pub mod xor;

#[cfg(test)]
pub mod test;

pub const NUM_LOOKUP_TABLES: usize = LookupTables::<32>::COUNT;

#[derive(
    Copy, Clone, Debug, From, Serialize, Deserialize, EnumIter, EnumCountMacro, IntoStaticStr,
)]
#[repr(u8)]
pub enum LookupTables<const XLEN: usize> {
    RangeCheck(RangeCheckTable<XLEN>),
    RangeCheckAligned(RangeCheckAlignedTable<XLEN>),
    And(AndTable<XLEN>),
    Andn(AndnTable<XLEN>),
    Or(OrTable<XLEN>),
    Xor(XorTable<XLEN>),
    Equal(EqualTable<XLEN>),
    SignedGreaterThanEqual(SignedGreaterThanEqualTable<XLEN>),
    UnsignedGreaterThanEqual(UnsignedGreaterThanEqualTable<XLEN>),
    NotEqual(NotEqualTable<XLEN>),
    SignedLessThan(SignedLessThanTable<XLEN>),
    UnsignedLessThan(UnsignedLessThanTable<XLEN>),
    Movsign(MovsignTable<XLEN>),
    UpperWord(UpperWordTable<XLEN>),
    LessThanEqual(UnsignedLessThanEqualTable<XLEN>),
    ValidSignedRemainder(ValidSignedRemainderTable<XLEN>),
    ValidUnsignedRemainder(ValidUnsignedRemainderTable<XLEN>),
    ValidDiv0(ValidDiv0Table<XLEN>),
    HalfwordAlignment(HalfwordAlignmentTable<XLEN>),
    WordAlignment(WordAlignmentTable<XLEN>),
    LowerHalfWord(LowerHalfWordTable<XLEN>),
    SignExtendHalfWord(SignExtendHalfWordTable<XLEN>),
    Pow2(Pow2Table<XLEN>),
    Pow2W(Pow2WTable<XLEN>),
    ShiftRightBitmask(ShiftRightBitmaskTable<XLEN>),
    VirtualRev8W(VirtualRev8WTable<XLEN>),
    VirtualSRL(VirtualSRLTable<XLEN>),
    VirtualSRA(VirtualSRATable<XLEN>),
    VirtualROTR(VirtualRotrTable<XLEN>),
    VirtualROTRW(VirtualRotrWTable<XLEN>),
    VirtualChangeDivisor(VirtualChangeDivisorTable<XLEN>),
    VirtualChangeDivisorW(VirtualChangeDivisorWTable<XLEN>),
    MulUNoOverflow(MulUNoOverflowTable<XLEN>),
    VirtualXORROT32(VirtualXORROTTable<XLEN, 32>),
    VirtualXORROT24(VirtualXORROTTable<XLEN, 24>),
    VirtualXORROT16(VirtualXORROTTable<XLEN, 16>),
    VirtualXORROT63(VirtualXORROTTable<XLEN, 63>),
    VirtualXORROTW16(VirtualXORROTWTable<XLEN, 16>),
    VirtualXORROTW12(VirtualXORROTWTable<XLEN, 12>),
    VirtualXORROTW8(VirtualXORROTWTable<XLEN, 8>),
    VirtualXORROTW7(VirtualXORROTWTable<XLEN, 7>),
}

impl<const XLEN: usize> LookupTables<XLEN> {
    pub fn enum_index(table: &Self) -> usize {
        // Discriminant: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
        let byte = unsafe { *(table as *const Self as *const u8) };
        byte as usize
    }

    #[cfg(test)]
    pub fn materialize(&self) -> Vec<u64> {
        match self {
            LookupTables::RangeCheck(table) => table.materialize(),
            LookupTables::RangeCheckAligned(table) => table.materialize(),
            LookupTables::And(table) => table.materialize(),
            LookupTables::Andn(table) => table.materialize(),
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
            LookupTables::WordAlignment(table) => table.materialize(),
            LookupTables::LowerHalfWord(table) => table.materialize(),
            LookupTables::SignExtendHalfWord(table) => table.materialize(),
            LookupTables::Pow2(table) => table.materialize(),
            LookupTables::Pow2W(table) => table.materialize(),
            LookupTables::ShiftRightBitmask(table) => table.materialize(),
            LookupTables::VirtualRev8W(table) => table.materialize(),
            LookupTables::VirtualSRL(table) => table.materialize(),
            LookupTables::VirtualSRA(table) => table.materialize(),
            LookupTables::VirtualROTR(table) => table.materialize(),
            LookupTables::VirtualROTRW(table) => table.materialize(),
            LookupTables::VirtualChangeDivisor(table) => table.materialize(),
            LookupTables::VirtualChangeDivisorW(table) => table.materialize(),
            LookupTables::MulUNoOverflow(table) => table.materialize(),
            LookupTables::VirtualXORROT32(table) => table.materialize(),
            LookupTables::VirtualXORROT24(table) => table.materialize(),
            LookupTables::VirtualXORROT16(table) => table.materialize(),
            LookupTables::VirtualXORROT63(table) => table.materialize(),
            LookupTables::VirtualXORROTW7(table) => table.materialize(),
            LookupTables::VirtualXORROTW8(table) => table.materialize(),
            LookupTables::VirtualXORROTW12(table) => table.materialize(),
            LookupTables::VirtualXORROTW16(table) => table.materialize(),
        }
    }

    pub fn materialize_entry(&self, index: u128) -> u64 {
        match self {
            LookupTables::RangeCheck(table) => table.materialize_entry(index),
            LookupTables::RangeCheckAligned(table) => table.materialize_entry(index),
            LookupTables::And(table) => table.materialize_entry(index),
            LookupTables::Andn(table) => table.materialize_entry(index),
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
            LookupTables::WordAlignment(table) => table.materialize_entry(index),
            LookupTables::LowerHalfWord(table) => table.materialize_entry(index),
            LookupTables::SignExtendHalfWord(table) => table.materialize_entry(index),
            LookupTables::Pow2(table) => table.materialize_entry(index),
            LookupTables::Pow2W(table) => table.materialize_entry(index),
            LookupTables::ShiftRightBitmask(table) => table.materialize_entry(index),
            LookupTables::VirtualRev8W(table) => table.materialize_entry(index),
            LookupTables::VirtualSRL(table) => table.materialize_entry(index),
            LookupTables::VirtualSRA(table) => table.materialize_entry(index),
            LookupTables::VirtualROTR(table) => table.materialize_entry(index),
            LookupTables::VirtualROTRW(table) => table.materialize_entry(index),
            LookupTables::VirtualChangeDivisor(table) => table.materialize_entry(index),
            LookupTables::VirtualChangeDivisorW(table) => table.materialize_entry(index),
            LookupTables::MulUNoOverflow(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT32(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT24(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT16(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT63(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROTW7(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROTW8(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROTW12(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROTW16(table) => table.materialize_entry(index),
        }
    }

    pub fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        match self {
            LookupTables::RangeCheck(table) => table.evaluate_mle(r),
            LookupTables::RangeCheckAligned(table) => table.evaluate_mle(r),
            LookupTables::And(table) => table.evaluate_mle(r),
            LookupTables::Andn(table) => table.evaluate_mle(r),
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
            LookupTables::WordAlignment(table) => table.evaluate_mle(r),
            LookupTables::LowerHalfWord(table) => table.evaluate_mle(r),
            LookupTables::SignExtendHalfWord(table) => table.evaluate_mle(r),
            LookupTables::Pow2(table) => table.evaluate_mle(r),
            LookupTables::Pow2W(table) => table.evaluate_mle(r),
            LookupTables::ShiftRightBitmask(table) => table.evaluate_mle(r),
            LookupTables::VirtualRev8W(table) => table.evaluate_mle(r),
            LookupTables::VirtualSRL(table) => table.evaluate_mle(r),
            LookupTables::VirtualSRA(table) => table.evaluate_mle(r),
            LookupTables::VirtualROTR(table) => table.evaluate_mle(r),
            LookupTables::VirtualROTRW(table) => table.evaluate_mle(r),
            LookupTables::VirtualChangeDivisor(table) => table.evaluate_mle(r),
            LookupTables::VirtualChangeDivisorW(table) => table.evaluate_mle(r),
            LookupTables::MulUNoOverflow(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT32(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT24(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT16(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT63(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROTW7(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROTW8(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROTW12(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROTW16(table) => table.evaluate_mle(r),
        }
    }

    pub fn suffixes(&self) -> Vec<Suffixes> {
        match self {
            LookupTables::RangeCheck(table) => table.suffixes(),
            LookupTables::RangeCheckAligned(table) => table.suffixes(),
            LookupTables::And(table) => table.suffixes(),
            LookupTables::Andn(table) => table.suffixes(),
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
            LookupTables::WordAlignment(table) => table.suffixes(),
            LookupTables::LowerHalfWord(table) => table.suffixes(),
            LookupTables::SignExtendHalfWord(table) => table.suffixes(),
            LookupTables::Pow2(table) => table.suffixes(),
            LookupTables::Pow2W(table) => table.suffixes(),
            LookupTables::ShiftRightBitmask(table) => table.suffixes(),
            LookupTables::VirtualRev8W(table) => table.suffixes(),
            LookupTables::VirtualSRL(table) => table.suffixes(),
            LookupTables::VirtualSRA(table) => table.suffixes(),
            LookupTables::VirtualROTR(table) => table.suffixes(),
            LookupTables::VirtualROTRW(table) => table.suffixes(),
            LookupTables::VirtualChangeDivisor(table) => table.suffixes(),
            LookupTables::VirtualChangeDivisorW(table) => table.suffixes(),
            LookupTables::MulUNoOverflow(table) => table.suffixes(),
            LookupTables::VirtualXORROT32(table) => table.suffixes(),
            LookupTables::VirtualXORROT24(table) => table.suffixes(),
            LookupTables::VirtualXORROT16(table) => table.suffixes(),
            LookupTables::VirtualXORROT63(table) => table.suffixes(),
            LookupTables::VirtualXORROTW7(table) => table.suffixes(),
            LookupTables::VirtualXORROTW8(table) => table.suffixes(),
            LookupTables::VirtualXORROTW12(table) => table.suffixes(),
            LookupTables::VirtualXORROTW16(table) => table.suffixes(),
        }
    }

    pub fn combine<F: JoltField>(
        &self,
        prefixes: &[PrefixEval<F>],
        suffixes: &[SuffixEval<F>],
    ) -> F {
        match self {
            LookupTables::RangeCheck(table) => table.combine(prefixes, suffixes),
            LookupTables::RangeCheckAligned(table) => table.combine(prefixes, suffixes),
            LookupTables::And(table) => table.combine(prefixes, suffixes),
            LookupTables::Andn(table) => table.combine(prefixes, suffixes),
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
            LookupTables::WordAlignment(table) => table.combine(prefixes, suffixes),
            LookupTables::LowerHalfWord(table) => table.combine(prefixes, suffixes),
            LookupTables::SignExtendHalfWord(table) => table.combine(prefixes, suffixes),
            LookupTables::Pow2(table) => table.combine(prefixes, suffixes),
            LookupTables::Pow2W(table) => table.combine(prefixes, suffixes),
            LookupTables::ShiftRightBitmask(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualRev8W(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualSRL(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualSRA(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualROTR(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualROTRW(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualChangeDivisor(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualChangeDivisorW(table) => table.combine(prefixes, suffixes),
            LookupTables::MulUNoOverflow(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT32(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT24(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT16(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT63(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROTW7(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROTW8(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROTW12(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROTW16(table) => table.combine(prefixes, suffixes),
        }
    }
}
