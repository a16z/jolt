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
use sign_mask::SignMaskTable as MovsignTable;
use signed_greater_than_equal::SignedGreaterThanEqualTable;
use signed_less_than::SignedLessThanTable;
use unsigned_greater_than_equal::UnsignedGreaterThanEqualTable;
use unsigned_less_than::UnsignedLessThanTable;
use unsigned_less_than_equal::UnsignedLessThanEqualTable;
use upper_word::UpperWordTable;
use valid_div0::ValidDiv0Table;
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
    SignExtendHalfWord(SignExtendHalfWordTable<XLEN>),
    Pow2(Pow2Table<XLEN>),
    Pow2W(Pow2WTable<XLEN>),
    ShiftRightBitmask(ShiftRightBitmaskTable<XLEN>),
    VirtualRev8W(VirtualRev8WTable<XLEN>),
    VirtualSRL(VirtualSRLTable<XLEN>),
    VirtualSRA(VirtualSRATable<XLEN>),
    VirtualROTR(VirtualROTRTable<XLEN>),
    VirtualROTRW(VirtualROTRWTable<XLEN>),
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

/// Dispatches a method call to the inner table for every
/// [`LookupTableKind`] variant, binding the inner table to `$t` and
/// evaluating `$expr`. Variants are listed once here so that
/// [`LookupTableKind`]'s dispatch methods stay a single line each.
macro_rules! dispatch {
    ($self:expr, $t:ident => $expr:expr) => {
        match $self {
            Self::RangeCheck($t) => $expr,
            Self::RangeCheckAligned($t) => $expr,
            Self::And($t) => $expr,
            Self::Andn($t) => $expr,
            Self::Or($t) => $expr,
            Self::Xor($t) => $expr,
            Self::Equal($t) => $expr,
            Self::SignedGreaterThanEqual($t) => $expr,
            Self::UnsignedGreaterThanEqual($t) => $expr,
            Self::NotEqual($t) => $expr,
            Self::SignedLessThan($t) => $expr,
            Self::UnsignedLessThan($t) => $expr,
            Self::Movsign($t) => $expr,
            Self::UpperWord($t) => $expr,
            Self::LessThanEqual($t) => $expr,
            Self::ValidUnsignedRemainder($t) => $expr,
            Self::ValidDiv0($t) => $expr,
            Self::HalfwordAlignment($t) => $expr,
            Self::WordAlignment($t) => $expr,
            Self::LowerHalfWord($t) => $expr,
            Self::SignExtendHalfWord($t) => $expr,
            Self::Pow2($t) => $expr,
            Self::Pow2W($t) => $expr,
            Self::ShiftRightBitmask($t) => $expr,
            Self::VirtualRev8W($t) => $expr,
            Self::VirtualSRL($t) => $expr,
            Self::VirtualSRA($t) => $expr,
            Self::VirtualROTR($t) => $expr,
            Self::VirtualROTRW($t) => $expr,
            Self::VirtualChangeDivisor($t) => $expr,
            Self::VirtualChangeDivisorW($t) => $expr,
            Self::MulUNoOverflow($t) => $expr,
            Self::VirtualXORROT32($t) => $expr,
            Self::VirtualXORROT24($t) => $expr,
            Self::VirtualXORROT16($t) => $expr,
            Self::VirtualXORROT63($t) => $expr,
            Self::VirtualXORROTW16($t) => $expr,
            Self::VirtualXORROTW12($t) => $expr,
            Self::VirtualXORROTW8($t) => $expr,
            Self::VirtualXORROTW7($t) => $expr,
        }
    };
}

impl<const XLEN: usize> LookupTableKind<XLEN> {
    /// Returns the canonical Jolt lookup table list in `jolt-core` order.
    pub fn all() -> [Self; 40] {
        [
            Self::RangeCheck(Default::default()),
            Self::RangeCheckAligned(Default::default()),
            Self::And(Default::default()),
            Self::Andn(Default::default()),
            Self::Or(Default::default()),
            Self::Xor(Default::default()),
            Self::Equal(Default::default()),
            Self::SignedGreaterThanEqual(Default::default()),
            Self::UnsignedGreaterThanEqual(Default::default()),
            Self::NotEqual(Default::default()),
            Self::SignedLessThan(Default::default()),
            Self::UnsignedLessThan(Default::default()),
            Self::Movsign(Default::default()),
            Self::UpperWord(Default::default()),
            Self::LessThanEqual(Default::default()),
            Self::ValidUnsignedRemainder(Default::default()),
            Self::ValidDiv0(Default::default()),
            Self::HalfwordAlignment(Default::default()),
            Self::WordAlignment(Default::default()),
            Self::LowerHalfWord(Default::default()),
            Self::SignExtendHalfWord(Default::default()),
            Self::Pow2(Default::default()),
            Self::Pow2W(Default::default()),
            Self::ShiftRightBitmask(Default::default()),
            Self::VirtualRev8W(Default::default()),
            Self::VirtualSRL(Default::default()),
            Self::VirtualSRA(Default::default()),
            Self::VirtualROTR(Default::default()),
            Self::VirtualROTRW(Default::default()),
            Self::VirtualChangeDivisor(Default::default()),
            Self::VirtualChangeDivisorW(Default::default()),
            Self::MulUNoOverflow(Default::default()),
            Self::VirtualXORROT32(Default::default()),
            Self::VirtualXORROT24(Default::default()),
            Self::VirtualXORROT16(Default::default()),
            Self::VirtualXORROT63(Default::default()),
            Self::VirtualXORROTW16(Default::default()),
            Self::VirtualXORROTW12(Default::default()),
            Self::VirtualXORROTW8(Default::default()),
            Self::VirtualXORROTW7(Default::default()),
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::RangeCheck(_) => "RangeCheck",
            Self::RangeCheckAligned(_) => "RangeCheckAligned",
            Self::And(_) => "And",
            Self::Andn(_) => "Andn",
            Self::Or(_) => "Or",
            Self::Xor(_) => "Xor",
            Self::Equal(_) => "Equal",
            Self::SignedGreaterThanEqual(_) => "SignedGreaterThanEqual",
            Self::UnsignedGreaterThanEqual(_) => "UnsignedGreaterThanEqual",
            Self::NotEqual(_) => "NotEqual",
            Self::SignedLessThan(_) => "SignedLessThan",
            Self::UnsignedLessThan(_) => "UnsignedLessThan",
            Self::Movsign(_) => "Movsign",
            Self::UpperWord(_) => "UpperWord",
            Self::LessThanEqual(_) => "LessThanEqual",
            Self::ValidUnsignedRemainder(_) => "ValidUnsignedRemainder",
            Self::ValidDiv0(_) => "ValidDiv0",
            Self::HalfwordAlignment(_) => "HalfwordAlignment",
            Self::WordAlignment(_) => "WordAlignment",
            Self::LowerHalfWord(_) => "LowerHalfWord",
            Self::SignExtendHalfWord(_) => "SignExtendHalfWord",
            Self::Pow2(_) => "Pow2",
            Self::Pow2W(_) => "Pow2W",
            Self::ShiftRightBitmask(_) => "ShiftRightBitmask",
            Self::VirtualRev8W(_) => "VirtualRev8W",
            Self::VirtualSRL(_) => "VirtualSRL",
            Self::VirtualSRA(_) => "VirtualSRA",
            Self::VirtualROTR(_) => "VirtualROTR",
            Self::VirtualROTRW(_) => "VirtualROTRW",
            Self::VirtualChangeDivisor(_) => "VirtualChangeDivisor",
            Self::VirtualChangeDivisorW(_) => "VirtualChangeDivisorW",
            Self::MulUNoOverflow(_) => "MulUNoOverflow",
            Self::VirtualXORROT32(_) => "VirtualXORROT32",
            Self::VirtualXORROT24(_) => "VirtualXORROT24",
            Self::VirtualXORROT16(_) => "VirtualXORROT16",
            Self::VirtualXORROT63(_) => "VirtualXORROT63",
            Self::VirtualXORROTW16(_) => "VirtualXORROTW16",
            Self::VirtualXORROTW12(_) => "VirtualXORROTW12",
            Self::VirtualXORROTW8(_) => "VirtualXORROTW8",
            Self::VirtualXORROTW7(_) => "VirtualXORROTW7",
        }
    }

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
        dispatch!(self, t => t.materialize_entry(index))
    }

    pub fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        dispatch!(self, t => t.evaluate_mle(r))
    }

    pub fn suffixes(&self) -> &'static [Suffixes] {
        dispatch!(self, t => PrefixSuffixDecomposition::suffixes(t))
    }

    pub fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        dispatch!(self, t => PrefixSuffixDecomposition::combine(t, prefixes, suffixes))
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
