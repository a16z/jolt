use align_addr::AlignAddrTable;
use and::AndTable;
use andn::AndnTable;
use equal::EqualTable;
use halfword_alignment::HalfwordAlignmentTable;
use lower_half_word::LowerHalfWordTable;
use movsign::MovsignTable;
use mulu_no_overflow::MulUNoOverflowTable;
use not_equal::NotEqualTable;
use or::OrTable;
use pext::PextTable;
use pext_signed::PextSignedTable;
use pow2::Pow2Table;
use pow2_w::Pow2WTable;
use prefixes::PrefixEval;
use range_check::RangeCheckTable;
use range_check_aligned::RangeCheckAlignedTable;
use serde::{Deserialize, Serialize};
use shift_data_b::ShiftDataBTable;
use shift_data_h::ShiftDataHTable;
use shift_data_w::ShiftDataWTable;
use shift_right_bitmask::ShiftRightBitmaskTable;
use shift_right_bitmask_w::ShiftRightBitmaskWTable;
use sign_extend_word::SignExtendWordTable;
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
use valid_unsigned_remainder::ValidUnsignedRemainderTable;
use virtual_negate_if::VirtualNegateIfTable;
use virtual_rev8w::VirtualRev8WTable;
use virtual_rotr::VirtualRotrTable;
use virtual_rotrw::VirtualRotrWTable;
use virtual_sra::VirtualSRATable;
use virtual_sraw::VirtualSRAWTable;
use virtual_srl::VirtualSRLTable;
use virtual_srlw::VirtualSRLWTable;
use virtual_xor_rot::VirtualXORROTTable;
use virtual_xor_rotl1::VirtualXORROTL1Table;
use virtual_xor_rotw::VirtualXORROTWTable;
use window_mask_b::WindowMaskBTable;
use window_mask_h::WindowMaskHTable;
use window_mask_w::WindowMaskWTable;
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

pub mod align_addr;
pub mod and;
pub mod andn;
pub mod equal;
pub mod halfword_alignment;
pub mod lower_half_word;
pub mod movsign;
pub mod mulu_no_overflow;
pub mod not_equal;
pub mod or;
pub mod pext;
pub mod pext_signed;
pub mod pow2;
pub mod pow2_w;
pub mod range_check;
pub mod range_check_aligned;
pub mod shift_data_b;
pub mod shift_data_h;
pub mod shift_data_w;
pub mod shift_right_bitmask;
pub mod shift_right_bitmask_w;
pub mod sign_extend_word;
pub mod signed_greater_than_equal;
pub mod signed_less_than;
pub mod sub;
pub mod unsigned_greater_than_equal;
pub mod unsigned_less_than;
pub mod unsigned_less_than_equal;
pub mod upper_word;
pub mod valid_div0;
pub mod valid_unsigned_remainder;
pub mod virtual_negate_if;
pub mod virtual_rev8w;
pub mod virtual_rotr;
pub mod virtual_rotrw;
pub mod virtual_sra;
pub mod virtual_sraw;
pub mod virtual_srl;
pub mod virtual_srlw;
pub mod virtual_xor_rot;
pub mod virtual_xor_rotl1;
pub mod virtual_xor_rotw;
pub mod window_mask_b;
pub mod window_mask_h;
pub mod window_mask_w;
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
    ValidUnsignedRemainder(ValidUnsignedRemainderTable<XLEN>),
    ValidDiv0(ValidDiv0Table<XLEN>),
    HalfwordAlignment(HalfwordAlignmentTable<XLEN>),
    WordAlignment(WordAlignmentTable<XLEN>),
    LowerHalfWord(LowerHalfWordTable<XLEN>),
    SignExtendWord(SignExtendWordTable<XLEN>),
    Pow2(Pow2Table<XLEN>),
    Pow2W(Pow2WTable<XLEN>),
    ShiftRightBitmask(ShiftRightBitmaskTable<XLEN>),
    VirtualRev8W(VirtualRev8WTable<XLEN>),
    VirtualSRL(VirtualSRLTable<XLEN>),
    VirtualSRA(VirtualSRATable<XLEN>),
    VirtualROTR(VirtualRotrTable<XLEN>),
    VirtualROTRW(VirtualRotrWTable<XLEN>),
    VirtualNegateIf(VirtualNegateIfTable<XLEN>),
    MulUNoOverflow(MulUNoOverflowTable<XLEN>),
    VirtualXORROT32(VirtualXORROTTable<XLEN, 32>),
    VirtualXORROT24(VirtualXORROTTable<XLEN, 24>),
    VirtualXORROT16(VirtualXORROTTable<XLEN, 16>),
    VirtualXORROT63(VirtualXORROTTable<XLEN, 63>),
    VirtualXORROTW16(VirtualXORROTWTable<XLEN, 16>),
    VirtualXORROTW12(VirtualXORROTWTable<XLEN, 12>),
    VirtualXORROTW8(VirtualXORROTWTable<XLEN, 8>),
    VirtualXORROTW7(VirtualXORROTWTable<XLEN, 7>),
    WindowMaskW(WindowMaskWTable<XLEN>),
    PextSigned(PextSignedTable<XLEN>),
    VirtualXORROTW22(VirtualXORROTWTable<XLEN, 22>),
    VirtualXORROTW19(VirtualXORROTWTable<XLEN, 19>),
    VirtualXORROTW6(VirtualXORROTWTable<XLEN, 6>),
    ShiftRightBitmaskW(ShiftRightBitmaskWTable<XLEN>),
    VirtualSRLW(VirtualSRLWTable<XLEN>),
    VirtualSRAW(VirtualSRAWTable<XLEN>),
    Pext(PextTable<XLEN>),
    WindowMaskB(WindowMaskBTable<XLEN>),
    WindowMaskH(WindowMaskHTable<XLEN>),
    AlignAddr(AlignAddrTable<XLEN>),
    ShiftDataB(ShiftDataBTable<XLEN>),
    ShiftDataH(ShiftDataHTable<XLEN>),
    ShiftDataW(ShiftDataWTable<XLEN>),
    VirtualXORROTL1(VirtualXORROTL1Table<XLEN>),
    VirtualXORROT2(VirtualXORROTTable<XLEN, 2>),
    VirtualXORROT3(VirtualXORROTTable<XLEN, 3>),
    VirtualXORROT8(VirtualXORROTTable<XLEN, 8>),
    VirtualXORROT9(VirtualXORROTTable<XLEN, 9>),
    VirtualXORROT19(VirtualXORROTTable<XLEN, 19>),
    VirtualXORROT20(VirtualXORROTTable<XLEN, 20>),
    VirtualXORROT21(VirtualXORROTTable<XLEN, 21>),
    VirtualXORROT23(VirtualXORROTTable<XLEN, 23>),
    VirtualXORROT25(VirtualXORROTTable<XLEN, 25>),
    VirtualXORROT28(VirtualXORROTTable<XLEN, 28>),
    VirtualXORROT36(VirtualXORROTTable<XLEN, 36>),
    VirtualXORROT37(VirtualXORROTTable<XLEN, 37>),
    VirtualXORROT39(VirtualXORROTTable<XLEN, 39>),
    VirtualXORROT43(VirtualXORROTTable<XLEN, 43>),
    VirtualXORROT44(VirtualXORROTTable<XLEN, 44>),
    VirtualXORROT46(VirtualXORROTTable<XLEN, 46>),
    VirtualXORROT49(VirtualXORROTTable<XLEN, 49>),
    VirtualXORROT50(VirtualXORROTTable<XLEN, 50>),
    VirtualXORROT54(VirtualXORROTTable<XLEN, 54>),
    VirtualXORROT56(VirtualXORROTTable<XLEN, 56>),
    VirtualXORROT58(VirtualXORROTTable<XLEN, 58>),
    VirtualXORROT61(VirtualXORROTTable<XLEN, 61>),
    VirtualXORROT62(VirtualXORROTTable<XLEN, 62>),
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
            LookupTables::ValidUnsignedRemainder(table) => table.materialize(),
            LookupTables::ValidDiv0(table) => table.materialize(),
            LookupTables::HalfwordAlignment(table) => table.materialize(),
            LookupTables::WordAlignment(table) => table.materialize(),
            LookupTables::LowerHalfWord(table) => table.materialize(),
            LookupTables::SignExtendWord(table) => table.materialize(),
            LookupTables::Pow2(table) => table.materialize(),
            LookupTables::Pow2W(table) => table.materialize(),
            LookupTables::ShiftRightBitmask(table) => table.materialize(),
            LookupTables::VirtualRev8W(table) => table.materialize(),
            LookupTables::VirtualSRL(table) => table.materialize(),
            LookupTables::VirtualSRA(table) => table.materialize(),
            LookupTables::VirtualROTR(table) => table.materialize(),
            LookupTables::VirtualROTRW(table) => table.materialize(),
            LookupTables::VirtualNegateIf(table) => table.materialize(),
            LookupTables::MulUNoOverflow(table) => table.materialize(),
            LookupTables::VirtualXORROT32(table) => table.materialize(),
            LookupTables::VirtualXORROT24(table) => table.materialize(),
            LookupTables::VirtualXORROT16(table) => table.materialize(),
            LookupTables::VirtualXORROT63(table) => table.materialize(),
            LookupTables::VirtualXORROTW7(table) => table.materialize(),
            LookupTables::VirtualXORROT2(table) => table.materialize(),
            LookupTables::VirtualXORROT3(table) => table.materialize(),
            LookupTables::VirtualXORROT8(table) => table.materialize(),
            LookupTables::VirtualXORROT9(table) => table.materialize(),
            LookupTables::VirtualXORROT19(table) => table.materialize(),
            LookupTables::VirtualXORROT20(table) => table.materialize(),
            LookupTables::VirtualXORROT21(table) => table.materialize(),
            LookupTables::VirtualXORROT23(table) => table.materialize(),
            LookupTables::VirtualXORROT25(table) => table.materialize(),
            LookupTables::VirtualXORROT28(table) => table.materialize(),
            LookupTables::VirtualXORROT36(table) => table.materialize(),
            LookupTables::VirtualXORROT37(table) => table.materialize(),
            LookupTables::VirtualXORROT39(table) => table.materialize(),
            LookupTables::VirtualXORROT43(table) => table.materialize(),
            LookupTables::VirtualXORROT44(table) => table.materialize(),
            LookupTables::VirtualXORROT46(table) => table.materialize(),
            LookupTables::VirtualXORROT49(table) => table.materialize(),
            LookupTables::VirtualXORROT50(table) => table.materialize(),
            LookupTables::VirtualXORROT54(table) => table.materialize(),
            LookupTables::VirtualXORROT56(table) => table.materialize(),
            LookupTables::VirtualXORROT58(table) => table.materialize(),
            LookupTables::VirtualXORROT61(table) => table.materialize(),
            LookupTables::VirtualXORROT62(table) => table.materialize(),
            LookupTables::VirtualXORROTW8(table) => table.materialize(),
            LookupTables::VirtualXORROTW12(table) => table.materialize(),
            LookupTables::VirtualXORROTW16(table) => table.materialize(),
            LookupTables::WindowMaskW(table) => table.materialize(),
            LookupTables::PextSigned(table) => table.materialize(),
            LookupTables::VirtualXORROTW22(table) => table.materialize(),
            LookupTables::VirtualXORROTW19(table) => table.materialize(),
            LookupTables::VirtualXORROTW6(table) => table.materialize(),
            LookupTables::ShiftRightBitmaskW(table) => table.materialize(),
            LookupTables::VirtualSRLW(table) => table.materialize(),
            LookupTables::VirtualSRAW(table) => table.materialize(),
            LookupTables::Pext(table) => table.materialize(),
            LookupTables::WindowMaskB(table) => table.materialize(),
            LookupTables::WindowMaskH(table) => table.materialize(),
            LookupTables::AlignAddr(table) => table.materialize(),
            LookupTables::ShiftDataB(table) => table.materialize(),
            LookupTables::ShiftDataH(table) => table.materialize(),
            LookupTables::ShiftDataW(table) => table.materialize(),
            LookupTables::VirtualXORROTL1(table) => table.materialize(),
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
            LookupTables::ValidUnsignedRemainder(table) => table.materialize_entry(index),
            LookupTables::ValidDiv0(table) => table.materialize_entry(index),
            LookupTables::HalfwordAlignment(table) => table.materialize_entry(index),
            LookupTables::WordAlignment(table) => table.materialize_entry(index),
            LookupTables::LowerHalfWord(table) => table.materialize_entry(index),
            LookupTables::SignExtendWord(table) => table.materialize_entry(index),
            LookupTables::Pow2(table) => table.materialize_entry(index),
            LookupTables::Pow2W(table) => table.materialize_entry(index),
            LookupTables::ShiftRightBitmask(table) => table.materialize_entry(index),
            LookupTables::VirtualRev8W(table) => table.materialize_entry(index),
            LookupTables::VirtualSRL(table) => table.materialize_entry(index),
            LookupTables::VirtualSRA(table) => table.materialize_entry(index),
            LookupTables::VirtualROTR(table) => table.materialize_entry(index),
            LookupTables::VirtualROTRW(table) => table.materialize_entry(index),
            LookupTables::VirtualNegateIf(table) => table.materialize_entry(index),
            LookupTables::MulUNoOverflow(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT32(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT24(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT16(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT63(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROTW7(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT2(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT3(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT8(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT9(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT19(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT20(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT21(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT23(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT25(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT28(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT36(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT37(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT39(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT43(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT44(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT46(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT49(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT50(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT54(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT56(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT58(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT61(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROT62(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROTW8(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROTW12(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROTW16(table) => table.materialize_entry(index),
            LookupTables::WindowMaskW(table) => table.materialize_entry(index),
            LookupTables::PextSigned(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROTW22(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROTW19(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROTW6(table) => table.materialize_entry(index),
            LookupTables::ShiftRightBitmaskW(table) => table.materialize_entry(index),
            LookupTables::VirtualSRLW(table) => table.materialize_entry(index),
            LookupTables::VirtualSRAW(table) => table.materialize_entry(index),
            LookupTables::Pext(table) => table.materialize_entry(index),
            LookupTables::WindowMaskB(table) => table.materialize_entry(index),
            LookupTables::WindowMaskH(table) => table.materialize_entry(index),
            LookupTables::AlignAddr(table) => table.materialize_entry(index),
            LookupTables::ShiftDataB(table) => table.materialize_entry(index),
            LookupTables::ShiftDataH(table) => table.materialize_entry(index),
            LookupTables::ShiftDataW(table) => table.materialize_entry(index),
            LookupTables::VirtualXORROTL1(table) => table.materialize_entry(index),
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
            LookupTables::ValidUnsignedRemainder(table) => table.evaluate_mle(r),
            LookupTables::ValidDiv0(table) => table.evaluate_mle(r),
            LookupTables::HalfwordAlignment(table) => table.evaluate_mle(r),
            LookupTables::WordAlignment(table) => table.evaluate_mle(r),
            LookupTables::LowerHalfWord(table) => table.evaluate_mle(r),
            LookupTables::SignExtendWord(table) => table.evaluate_mle(r),
            LookupTables::Pow2(table) => table.evaluate_mle(r),
            LookupTables::Pow2W(table) => table.evaluate_mle(r),
            LookupTables::ShiftRightBitmask(table) => table.evaluate_mle(r),
            LookupTables::VirtualRev8W(table) => table.evaluate_mle(r),
            LookupTables::VirtualSRL(table) => table.evaluate_mle(r),
            LookupTables::VirtualSRA(table) => table.evaluate_mle(r),
            LookupTables::VirtualROTR(table) => table.evaluate_mle(r),
            LookupTables::VirtualROTRW(table) => table.evaluate_mle(r),
            LookupTables::VirtualNegateIf(table) => table.evaluate_mle(r),
            LookupTables::MulUNoOverflow(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT32(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT24(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT16(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT63(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROTW7(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT2(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT3(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT8(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT9(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT19(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT20(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT21(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT23(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT25(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT28(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT36(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT37(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT39(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT43(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT44(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT46(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT49(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT50(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT54(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT56(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT58(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT61(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROT62(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROTW8(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROTW12(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROTW16(table) => table.evaluate_mle(r),
            LookupTables::WindowMaskW(table) => table.evaluate_mle(r),
            LookupTables::PextSigned(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROTW22(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROTW19(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROTW6(table) => table.evaluate_mle(r),
            LookupTables::ShiftRightBitmaskW(table) => table.evaluate_mle(r),
            LookupTables::VirtualSRLW(table) => table.evaluate_mle(r),
            LookupTables::VirtualSRAW(table) => table.evaluate_mle(r),
            LookupTables::Pext(table) => table.evaluate_mle(r),
            LookupTables::WindowMaskB(table) => table.evaluate_mle(r),
            LookupTables::WindowMaskH(table) => table.evaluate_mle(r),
            LookupTables::AlignAddr(table) => table.evaluate_mle(r),
            LookupTables::ShiftDataB(table) => table.evaluate_mle(r),
            LookupTables::ShiftDataH(table) => table.evaluate_mle(r),
            LookupTables::ShiftDataW(table) => table.evaluate_mle(r),
            LookupTables::VirtualXORROTL1(table) => table.evaluate_mle(r),
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
            LookupTables::ValidUnsignedRemainder(table) => table.suffixes(),
            LookupTables::ValidDiv0(table) => table.suffixes(),
            LookupTables::HalfwordAlignment(table) => table.suffixes(),
            LookupTables::WordAlignment(table) => table.suffixes(),
            LookupTables::LowerHalfWord(table) => table.suffixes(),
            LookupTables::SignExtendWord(table) => table.suffixes(),
            LookupTables::Pow2(table) => table.suffixes(),
            LookupTables::Pow2W(table) => table.suffixes(),
            LookupTables::ShiftRightBitmask(table) => table.suffixes(),
            LookupTables::VirtualRev8W(table) => table.suffixes(),
            LookupTables::VirtualSRL(table) => table.suffixes(),
            LookupTables::VirtualSRA(table) => table.suffixes(),
            LookupTables::VirtualROTR(table) => table.suffixes(),
            LookupTables::VirtualROTRW(table) => table.suffixes(),
            LookupTables::VirtualNegateIf(table) => table.suffixes(),
            LookupTables::MulUNoOverflow(table) => table.suffixes(),
            LookupTables::VirtualXORROT32(table) => table.suffixes(),
            LookupTables::VirtualXORROT24(table) => table.suffixes(),
            LookupTables::VirtualXORROT16(table) => table.suffixes(),
            LookupTables::VirtualXORROT63(table) => table.suffixes(),
            LookupTables::VirtualXORROTW7(table) => table.suffixes(),
            LookupTables::VirtualXORROT2(table) => table.suffixes(),
            LookupTables::VirtualXORROT3(table) => table.suffixes(),
            LookupTables::VirtualXORROT8(table) => table.suffixes(),
            LookupTables::VirtualXORROT9(table) => table.suffixes(),
            LookupTables::VirtualXORROT19(table) => table.suffixes(),
            LookupTables::VirtualXORROT20(table) => table.suffixes(),
            LookupTables::VirtualXORROT21(table) => table.suffixes(),
            LookupTables::VirtualXORROT23(table) => table.suffixes(),
            LookupTables::VirtualXORROT25(table) => table.suffixes(),
            LookupTables::VirtualXORROT28(table) => table.suffixes(),
            LookupTables::VirtualXORROT36(table) => table.suffixes(),
            LookupTables::VirtualXORROT37(table) => table.suffixes(),
            LookupTables::VirtualXORROT39(table) => table.suffixes(),
            LookupTables::VirtualXORROT43(table) => table.suffixes(),
            LookupTables::VirtualXORROT44(table) => table.suffixes(),
            LookupTables::VirtualXORROT46(table) => table.suffixes(),
            LookupTables::VirtualXORROT49(table) => table.suffixes(),
            LookupTables::VirtualXORROT50(table) => table.suffixes(),
            LookupTables::VirtualXORROT54(table) => table.suffixes(),
            LookupTables::VirtualXORROT56(table) => table.suffixes(),
            LookupTables::VirtualXORROT58(table) => table.suffixes(),
            LookupTables::VirtualXORROT61(table) => table.suffixes(),
            LookupTables::VirtualXORROT62(table) => table.suffixes(),
            LookupTables::VirtualXORROTW8(table) => table.suffixes(),
            LookupTables::VirtualXORROTW12(table) => table.suffixes(),
            LookupTables::VirtualXORROTW16(table) => table.suffixes(),
            LookupTables::WindowMaskW(table) => table.suffixes(),
            LookupTables::PextSigned(table) => table.suffixes(),
            LookupTables::VirtualXORROTW22(table) => table.suffixes(),
            LookupTables::VirtualXORROTW19(table) => table.suffixes(),
            LookupTables::VirtualXORROTW6(table) => table.suffixes(),
            LookupTables::ShiftRightBitmaskW(table) => table.suffixes(),
            LookupTables::VirtualSRLW(table) => table.suffixes(),
            LookupTables::VirtualSRAW(table) => table.suffixes(),
            LookupTables::Pext(table) => table.suffixes(),
            LookupTables::WindowMaskB(table) => table.suffixes(),
            LookupTables::WindowMaskH(table) => table.suffixes(),
            LookupTables::AlignAddr(table) => table.suffixes(),
            LookupTables::ShiftDataB(table) => table.suffixes(),
            LookupTables::ShiftDataH(table) => table.suffixes(),
            LookupTables::ShiftDataW(table) => table.suffixes(),
            LookupTables::VirtualXORROTL1(table) => table.suffixes(),
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
            LookupTables::ValidUnsignedRemainder(table) => table.combine(prefixes, suffixes),
            LookupTables::ValidDiv0(table) => table.combine(prefixes, suffixes),
            LookupTables::HalfwordAlignment(table) => table.combine(prefixes, suffixes),
            LookupTables::WordAlignment(table) => table.combine(prefixes, suffixes),
            LookupTables::LowerHalfWord(table) => table.combine(prefixes, suffixes),
            LookupTables::SignExtendWord(table) => table.combine(prefixes, suffixes),
            LookupTables::Pow2(table) => table.combine(prefixes, suffixes),
            LookupTables::Pow2W(table) => table.combine(prefixes, suffixes),
            LookupTables::ShiftRightBitmask(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualRev8W(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualSRL(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualSRA(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualROTR(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualROTRW(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualNegateIf(table) => table.combine(prefixes, suffixes),
            LookupTables::MulUNoOverflow(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT32(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT24(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT16(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT63(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROTW7(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT2(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT3(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT8(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT9(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT19(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT20(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT21(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT23(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT25(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT28(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT36(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT37(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT39(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT43(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT44(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT46(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT49(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT50(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT54(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT56(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT58(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT61(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROT62(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROTW8(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROTW12(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROTW16(table) => table.combine(prefixes, suffixes),
            LookupTables::WindowMaskW(table) => table.combine(prefixes, suffixes),
            LookupTables::PextSigned(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROTW22(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROTW19(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROTW6(table) => table.combine(prefixes, suffixes),
            LookupTables::ShiftRightBitmaskW(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualSRLW(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualSRAW(table) => table.combine(prefixes, suffixes),
            LookupTables::Pext(table) => table.combine(prefixes, suffixes),
            LookupTables::WindowMaskB(table) => table.combine(prefixes, suffixes),
            LookupTables::WindowMaskH(table) => table.combine(prefixes, suffixes),
            LookupTables::AlignAddr(table) => table.combine(prefixes, suffixes),
            LookupTables::ShiftDataB(table) => table.combine(prefixes, suffixes),
            LookupTables::ShiftDataH(table) => table.combine(prefixes, suffixes),
            LookupTables::ShiftDataW(table) => table.combine(prefixes, suffixes),
            LookupTables::VirtualXORROTL1(table) => table.combine(prefixes, suffixes),
        }
    }
}
