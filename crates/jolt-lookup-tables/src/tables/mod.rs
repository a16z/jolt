//! Lookup table definitions for Jolt instruction decomposition.
//!
//! Each instruction that participates in the sumcheck-based lookup argument
//! maps to exactly one [`LookupTableKind`]. Concrete table implementations
//! provide [`materialize_entry`](crate::LookupTable::materialize_entry) for
//! preprocessing and [`evaluate_mle`](crate::LookupTable::evaluate_mle) for
//! the sumcheck verifier.
//!
//! All tables are generic over `const XLEN: usize`. The supported word sizes
//! are `XLEN = 64` (production) and `XLEN = 8` (full-hypercube tests).

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::traits::LookupTable;

pub mod and;
pub mod andn;
pub mod equal;
pub mod halfword_alignment;
pub mod lower_half_word;
pub mod mulu_no_overflow;
pub mod not_equal;
pub mod or;
pub mod pow2;
pub mod pow2_w;
pub mod prefixes;
pub mod range_check;
pub mod range_check_aligned;
pub mod shift_right_bitmask;
pub mod sign_extend_half_word;
pub mod sign_mask;
pub mod signed_greater_than_equal;
pub mod signed_less_than;
pub mod suffixes;
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

pub use prefixes::{PrefixEval, Prefixes};
pub use suffixes::{SuffixEval, Suffixes};

use and::AndTable;
use andn::AndnTable;
use equal::EqualTable;
use halfword_alignment::HalfwordAlignmentTable;
use lower_half_word::LowerHalfWordTable;
use mulu_no_overflow::MulUNoOverflowTable;
use not_equal::NotEqualTable;
use or::OrTable;
use pow2::Pow2Table;
use pow2_w::Pow2WTable;
use range_check::RangeCheckTable;
use range_check_aligned::RangeCheckAlignedTable;
use shift_right_bitmask::ShiftRightBitmaskTable;
use sign_extend_half_word::SignExtendHalfWordTable;
use sign_mask::SignMaskTable;
use signed_greater_than_equal::SignedGreaterThanEqualTable;
use signed_less_than::SignedLessThanTable;
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
use virtual_rotr::VirtualROTRTable;
use virtual_rotrw::VirtualROTRWTable;
use virtual_sra::VirtualSRATable;
use virtual_srl::VirtualSRLTable;
use virtual_xor_rot::VirtualXORROTTable;
use virtual_xor_rotw::VirtualXORROTWTable;
use word_alignment::WordAlignmentTable;
use xor::XorTable;

/// Identifies a lookup table type at a given word size.
///
/// Each variant carries the corresponding zero-sized table marker. Instructions
/// declare which table they use via
/// [`InstructionLookupTable::lookup_table`](crate::InstructionLookupTable::lookup_table).
#[expect(clippy::unsafe_derive_deserialize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, strum::EnumCount)]
#[repr(u8)]
pub enum LookupTableKind<const XLEN: usize> {
    RangeCheck(RangeCheckTable<XLEN>),
    RangeCheckAligned(RangeCheckAlignedTable<XLEN>),
    And(AndTable<XLEN>),
    Andn(AndnTable<XLEN>),
    Or(OrTable<XLEN>),
    Xor(XorTable<XLEN>),
    Equal(EqualTable<XLEN>),
    NotEqual(NotEqualTable<XLEN>),
    SignedLessThan(SignedLessThanTable<XLEN>),
    UnsignedLessThan(UnsignedLessThanTable<XLEN>),
    SignedGreaterThanEqual(SignedGreaterThanEqualTable<XLEN>),
    UnsignedGreaterThanEqual(UnsignedGreaterThanEqualTable<XLEN>),
    UnsignedLessThanEqual(UnsignedLessThanEqualTable<XLEN>),
    UpperWord(UpperWordTable<XLEN>),
    LowerHalfWord(LowerHalfWordTable<XLEN>),
    SignExtendHalfWord(SignExtendHalfWordTable<XLEN>),
    SignMask(SignMaskTable<XLEN>),
    Pow2(Pow2Table<XLEN>),
    Pow2W(Pow2WTable<XLEN>),
    ShiftRightBitmask(ShiftRightBitmaskTable<XLEN>),
    VirtualSRL(VirtualSRLTable<XLEN>),
    VirtualSRA(VirtualSRATable<XLEN>),
    VirtualROTR(VirtualROTRTable<XLEN>),
    VirtualROTRW(VirtualROTRWTable<XLEN>),
    ValidDiv0(ValidDiv0Table<XLEN>),
    ValidUnsignedRemainder(ValidUnsignedRemainderTable<XLEN>),
    ValidSignedRemainder(ValidSignedRemainderTable<XLEN>),
    VirtualChangeDivisor(VirtualChangeDivisorTable<XLEN>),
    VirtualChangeDivisorW(VirtualChangeDivisorWTable<XLEN>),
    HalfwordAlignment(HalfwordAlignmentTable<XLEN>),
    WordAlignment(WordAlignmentTable<XLEN>),
    MulUNoOverflow(MulUNoOverflowTable<XLEN>),
    VirtualRev8W(VirtualRev8WTable<XLEN>),
    VirtualXORROT32(VirtualXORROTTable<XLEN, 32>),
    VirtualXORROT24(VirtualXORROTTable<XLEN, 24>),
    VirtualXORROT16(VirtualXORROTTable<XLEN, 16>),
    VirtualXORROT63(VirtualXORROTTable<XLEN, 63>),
    VirtualXORROTW16(VirtualXORROTWTable<XLEN, 16>),
    VirtualXORROTW12(VirtualXORROTWTable<XLEN, 12>),
    VirtualXORROTW8(VirtualXORROTWTable<XLEN, 8>),
    VirtualXORROTW7(VirtualXORROTWTable<XLEN, 7>),
}

impl<const XLEN: usize> LookupTableKind<XLEN> {
    /// Returns the discriminant as a `usize`, suitable for array indexing.
    #[inline]
    pub fn index(&self) -> usize {
        // SAFETY: `LookupTableKind` is `#[repr(u8)]`, so its first byte is the
        // discriminant. See:
        // https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
        let byte = unsafe { *std::ptr::from_ref::<Self>(self).cast::<u8>() };
        byte as usize
    }

    pub fn materialize_entry(&self, index: u128) -> u64 {
        match self {
            Self::RangeCheck(t) => t.materialize_entry(index),
            Self::RangeCheckAligned(t) => t.materialize_entry(index),
            Self::And(t) => t.materialize_entry(index),
            Self::Andn(t) => t.materialize_entry(index),
            Self::Or(t) => t.materialize_entry(index),
            Self::Xor(t) => t.materialize_entry(index),
            Self::Equal(t) => t.materialize_entry(index),
            Self::NotEqual(t) => t.materialize_entry(index),
            Self::SignedLessThan(t) => t.materialize_entry(index),
            Self::UnsignedLessThan(t) => t.materialize_entry(index),
            Self::SignedGreaterThanEqual(t) => t.materialize_entry(index),
            Self::UnsignedGreaterThanEqual(t) => t.materialize_entry(index),
            Self::UnsignedLessThanEqual(t) => t.materialize_entry(index),
            Self::UpperWord(t) => t.materialize_entry(index),
            Self::LowerHalfWord(t) => t.materialize_entry(index),
            Self::SignExtendHalfWord(t) => t.materialize_entry(index),
            Self::SignMask(t) => t.materialize_entry(index),
            Self::Pow2(t) => t.materialize_entry(index),
            Self::Pow2W(t) => t.materialize_entry(index),
            Self::ShiftRightBitmask(t) => t.materialize_entry(index),
            Self::VirtualSRL(t) => t.materialize_entry(index),
            Self::VirtualSRA(t) => t.materialize_entry(index),
            Self::VirtualROTR(t) => t.materialize_entry(index),
            Self::VirtualROTRW(t) => t.materialize_entry(index),
            Self::ValidDiv0(t) => t.materialize_entry(index),
            Self::ValidUnsignedRemainder(t) => t.materialize_entry(index),
            Self::ValidSignedRemainder(t) => t.materialize_entry(index),
            Self::VirtualChangeDivisor(t) => t.materialize_entry(index),
            Self::VirtualChangeDivisorW(t) => t.materialize_entry(index),
            Self::HalfwordAlignment(t) => t.materialize_entry(index),
            Self::WordAlignment(t) => t.materialize_entry(index),
            Self::MulUNoOverflow(t) => t.materialize_entry(index),
            Self::VirtualRev8W(t) => t.materialize_entry(index),
            Self::VirtualXORROT32(t) => t.materialize_entry(index),
            Self::VirtualXORROT24(t) => t.materialize_entry(index),
            Self::VirtualXORROT16(t) => t.materialize_entry(index),
            Self::VirtualXORROT63(t) => t.materialize_entry(index),
            Self::VirtualXORROTW16(t) => t.materialize_entry(index),
            Self::VirtualXORROTW12(t) => t.materialize_entry(index),
            Self::VirtualXORROTW8(t) => t.materialize_entry(index),
            Self::VirtualXORROTW7(t) => t.materialize_entry(index),
        }
    }

    pub fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        match self {
            Self::RangeCheck(t) => t.evaluate_mle(r),
            Self::RangeCheckAligned(t) => t.evaluate_mle(r),
            Self::And(t) => t.evaluate_mle(r),
            Self::Andn(t) => t.evaluate_mle(r),
            Self::Or(t) => t.evaluate_mle(r),
            Self::Xor(t) => t.evaluate_mle(r),
            Self::Equal(t) => t.evaluate_mle(r),
            Self::NotEqual(t) => t.evaluate_mle(r),
            Self::SignedLessThan(t) => t.evaluate_mle(r),
            Self::UnsignedLessThan(t) => t.evaluate_mle(r),
            Self::SignedGreaterThanEqual(t) => t.evaluate_mle(r),
            Self::UnsignedGreaterThanEqual(t) => t.evaluate_mle(r),
            Self::UnsignedLessThanEqual(t) => t.evaluate_mle(r),
            Self::UpperWord(t) => t.evaluate_mle(r),
            Self::LowerHalfWord(t) => t.evaluate_mle(r),
            Self::SignExtendHalfWord(t) => t.evaluate_mle(r),
            Self::SignMask(t) => t.evaluate_mle(r),
            Self::Pow2(t) => t.evaluate_mle(r),
            Self::Pow2W(t) => t.evaluate_mle(r),
            Self::ShiftRightBitmask(t) => t.evaluate_mle(r),
            Self::VirtualSRL(t) => t.evaluate_mle(r),
            Self::VirtualSRA(t) => t.evaluate_mle(r),
            Self::VirtualROTR(t) => t.evaluate_mle(r),
            Self::VirtualROTRW(t) => t.evaluate_mle(r),
            Self::ValidDiv0(t) => t.evaluate_mle(r),
            Self::ValidUnsignedRemainder(t) => t.evaluate_mle(r),
            Self::ValidSignedRemainder(t) => t.evaluate_mle(r),
            Self::VirtualChangeDivisor(t) => t.evaluate_mle(r),
            Self::VirtualChangeDivisorW(t) => t.evaluate_mle(r),
            Self::HalfwordAlignment(t) => t.evaluate_mle(r),
            Self::WordAlignment(t) => t.evaluate_mle(r),
            Self::MulUNoOverflow(t) => t.evaluate_mle(r),
            Self::VirtualRev8W(t) => t.evaluate_mle(r),
            Self::VirtualXORROT32(t) => t.evaluate_mle(r),
            Self::VirtualXORROT24(t) => t.evaluate_mle(r),
            Self::VirtualXORROT16(t) => t.evaluate_mle(r),
            Self::VirtualXORROT63(t) => t.evaluate_mle(r),
            Self::VirtualXORROTW16(t) => t.evaluate_mle(r),
            Self::VirtualXORROTW12(t) => t.evaluate_mle(r),
            Self::VirtualXORROTW8(t) => t.evaluate_mle(r),
            Self::VirtualXORROTW7(t) => t.evaluate_mle(r),
        }
    }

    pub fn suffixes(&self) -> &'static [Suffixes] {
        match self {
            Self::RangeCheck(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::RangeCheckAligned(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::And(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::Andn(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::Or(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::Xor(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::Equal(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::NotEqual(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::SignedLessThan(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::UnsignedLessThan(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::SignedGreaterThanEqual(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::UnsignedGreaterThanEqual(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::UnsignedLessThanEqual(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::UpperWord(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::LowerHalfWord(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::SignExtendHalfWord(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::SignMask(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::Pow2(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::Pow2W(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::ShiftRightBitmask(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualSRL(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualSRA(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualROTR(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualROTRW(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::ValidDiv0(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::ValidUnsignedRemainder(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::ValidSignedRemainder(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualChangeDivisor(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualChangeDivisorW(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::HalfwordAlignment(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::WordAlignment(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::MulUNoOverflow(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualRev8W(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualXORROT32(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualXORROT24(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualXORROT16(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualXORROT63(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualXORROTW16(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualXORROTW12(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualXORROTW8(t) => PrefixSuffixDecomposition::suffixes(t),
            Self::VirtualXORROTW7(t) => PrefixSuffixDecomposition::suffixes(t),
        }
    }

    pub fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        match self {
            Self::RangeCheck(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::RangeCheckAligned(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::And(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::Andn(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::Or(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::Xor(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::Equal(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::NotEqual(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::SignedLessThan(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::UnsignedLessThan(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::SignedGreaterThanEqual(t) => {
                PrefixSuffixDecomposition::combine(t, prefixes, suffixes)
            }
            Self::UnsignedGreaterThanEqual(t) => {
                PrefixSuffixDecomposition::combine(t, prefixes, suffixes)
            }
            Self::UnsignedLessThanEqual(t) => {
                PrefixSuffixDecomposition::combine(t, prefixes, suffixes)
            }
            Self::UpperWord(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::LowerHalfWord(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::SignExtendHalfWord(t) => {
                PrefixSuffixDecomposition::combine(t, prefixes, suffixes)
            }
            Self::SignMask(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::Pow2(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::Pow2W(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::ShiftRightBitmask(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::VirtualSRL(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::VirtualSRA(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::VirtualROTR(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::VirtualROTRW(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::ValidDiv0(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::ValidUnsignedRemainder(t) => {
                PrefixSuffixDecomposition::combine(t, prefixes, suffixes)
            }
            Self::ValidSignedRemainder(t) => {
                PrefixSuffixDecomposition::combine(t, prefixes, suffixes)
            }
            Self::VirtualChangeDivisor(t) => {
                PrefixSuffixDecomposition::combine(t, prefixes, suffixes)
            }
            Self::VirtualChangeDivisorW(t) => {
                PrefixSuffixDecomposition::combine(t, prefixes, suffixes)
            }
            Self::HalfwordAlignment(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::WordAlignment(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::MulUNoOverflow(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::VirtualRev8W(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::VirtualXORROT32(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::VirtualXORROT24(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::VirtualXORROT16(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::VirtualXORROT63(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::VirtualXORROTW16(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::VirtualXORROTW12(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::VirtualXORROTW8(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
            Self::VirtualXORROTW7(t) => PrefixSuffixDecomposition::combine(t, prefixes, suffixes),
        }
    }
}

/// Prefix/suffix decomposition for sub-linear MLE evaluation.
///
/// Each lookup table decomposes its MLE as:
/// ```text
/// table_mle(r) = Σ_i prefix_i(r_high) · suffix_i(r_low)
/// ```
///
/// where the sum is over a small number of prefix-suffix pairs.
/// This enables the sumcheck prover to avoid materializing the entire table.
pub trait PrefixSuffixDecomposition<const XLEN: usize>: crate::LookupTable + Default {
    /// The suffix types used in this table's decomposition.
    fn suffixes(&self) -> &'static [Suffixes];

    /// Recombine evaluated prefix and suffix values into the table's MLE evaluation.
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F;

    /// Generate a random lookup index for testing.
    ///
    /// The default returns a uniform random `u128` masked to `2 * XLEN` bits.
    /// Tables with constrained input domains (e.g., shift/rotate tables that expect
    /// bitmask-shaped right operands) should override this.
    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u128 {
        let raw: u128 = rand::Rng::gen(rng);
        if XLEN == 64 {
            raw
        } else {
            raw & ((1u128 << (2 * XLEN)) - 1)
        }
    }
}

#[cfg(test)]
pub(crate) mod test_utils;
