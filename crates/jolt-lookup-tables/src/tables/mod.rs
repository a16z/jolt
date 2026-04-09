//! Lookup table definitions for Jolt instruction decomposition.
//!
//! Each instruction that participates in the sumcheck-based lookup argument
//! maps to exactly one [`LookupTableKind`]. Concrete table implementations
//! provide [`materialize_entry`](crate::LookupTable::materialize_entry) for
//! preprocessing and [`evaluate_mle`](crate::LookupTable::evaluate_mle) for
//! the sumcheck verifier.
//!
//! The prefix/suffix sparse-dense decomposition enables sub-linear MLE
//! evaluation during the sumcheck prover's inner loop.

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

/// Identifies a lookup table type.
///
/// Each variant corresponds to a concrete table with its own
/// [`LookupTable`] implementation. Instructions
/// declare which table they use via [`InstructionLookupTable::lookup_table()`](crate::InstructionLookupTable::lookup_table).
///
/// The enum is `#[repr(u8)]` for compact serialization and efficient
/// discriminant extraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, strum::EnumCount)]
#[repr(u8)]
pub enum LookupTableKind {
    // Arithmetic
    /// Identity/range-check: extracts the lower XLEN bits.
    /// Used by ADD, SUB, MUL, ADDI, JAL, and other combined-operand instructions.
    RangeCheck,
    /// Range check with LSB alignment (clears bit 0). Used by JALR.
    RangeCheckAligned,

    // Bitwise
    /// Bitwise AND. Used by AND, ANDI.
    And,
    /// Bitwise AND-NOT (x & !y). Used by ANDN (Zbb extension).
    Andn,
    /// Bitwise OR. Used by OR, ORI.
    Or,
    /// Bitwise XOR. Used by XOR, XORI.
    Xor,

    // Comparison
    /// Equality check: returns 1 if x == y. Used by BEQ.
    Equal,
    /// Not-equal: returns 1 if x != y. Used by BNE.
    NotEqual,
    /// Signed less-than. Used by BLT, SLT, SLTI.
    SignedLessThan,
    /// Unsigned less-than. Used by BLTU, SLTU, SLTIU.
    UnsignedLessThan,
    /// Signed greater-than-or-equal. Used by BGE.
    SignedGreaterThanEqual,
    /// Unsigned greater-than-or-equal. Used by BGEU.
    UnsignedGreaterThanEqual,
    /// Unsigned less-than-or-equal.
    UnsignedLessThanEqual,

    // Word extraction
    /// Extract upper XLEN bits of a 2*XLEN-bit value. Used by MULHU.
    UpperWord,
    /// Extract lower half-word (XLEN/2 bits).
    LowerHalfWord,
    /// Sign-extend half-word to full word.
    SignExtendHalfWord,

    // Sign/conditional
    /// Sign-bit conditional: returns all-ones if MSB set, else zero. Used by MOVSIGN.
    SignMask,

    // Power of 2
    /// 2^(index mod XLEN). Used by POW2, POW2I.
    Pow2,
    /// 2^(index mod 32). Used by POW2W, POW2IW.
    Pow2W,

    // Shift
    /// Bitmask for right-shift: `((1 << (XLEN - shift)) - 1) << shift`.
    ShiftRightBitmask,
    /// Logical right shift (virtual decomposition). Used by SRL, SRLI.
    VirtualSRL,
    /// Arithmetic right shift (virtual decomposition). Used by SRA, SRAI.
    VirtualSRA,
    /// Rotate right. Used by ROTRI.
    VirtualROTR,
    /// Rotate right word (32-bit). Used by ROTRIW.
    VirtualROTRW,

    // Division validation
    /// Division-by-zero validity check. Used by ASSERT_VALID_DIV0.
    ValidDiv0,
    /// Unsigned remainder validity (remainder < divisor or divisor == 0).
    ValidUnsignedRemainder,
    /// Signed remainder validity.
    ValidSignedRemainder,
    /// Divisor transform for signed div overflow. Used by CHANGE_DIVISOR.
    VirtualChangeDivisor,
    /// Divisor transform (32-bit). Used by CHANGE_DIVISOR_W.
    VirtualChangeDivisorW,

    // Alignment
    /// Halfword alignment check (divisible by 2).
    HalfwordAlignment,
    /// Word alignment check (divisible by 4).
    WordAlignment,

    // Multiply overflow
    /// Unsigned multiply no-overflow check. Used by ASSERT_MULU_NO_OVERFLOW.
    MulUNoOverflow,

    // Byte manipulation
    /// Byte-reverse within word. Used by REV8W.
    VirtualRev8W,

    // XOR-rotate (SHA)
    /// XOR then rotate right by 32 bits.
    VirtualXORROT32,
    /// XOR then rotate right by 24 bits.
    VirtualXORROT24,
    /// XOR then rotate right by 16 bits.
    VirtualXORROT16,
    /// XOR then rotate right by 63 bits.
    VirtualXORROT63,
    /// XOR then rotate right word by 16 bits.
    VirtualXORROTW16,
    /// XOR then rotate right word by 12 bits.
    VirtualXORROTW12,
    /// XOR then rotate right word by 8 bits.
    VirtualXORROTW8,
    /// XOR then rotate right word by 7 bits.
    VirtualXORROTW7,
}

impl LookupTableKind {
    /// Returns the discriminant as a `usize`, suitable for array indexing.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn materialize_entry(&self, index: u128) -> u64 {
        match self {
            Self::RangeCheck => range_check::RangeCheckTable.materialize_entry(index),
            Self::RangeCheckAligned => {
                range_check_aligned::RangeCheckAlignedTable.materialize_entry(index)
            }
            Self::And => and::AndTable.materialize_entry(index),
            Self::Andn => andn::AndnTable.materialize_entry(index),
            Self::Or => or::OrTable.materialize_entry(index),
            Self::Xor => xor::XorTable.materialize_entry(index),
            Self::Equal => equal::EqualTable.materialize_entry(index),
            Self::NotEqual => not_equal::NotEqualTable.materialize_entry(index),
            Self::SignedLessThan => signed_less_than::SignedLessThanTable.materialize_entry(index),
            Self::UnsignedLessThan => {
                unsigned_less_than::UnsignedLessThanTable.materialize_entry(index)
            }
            Self::SignedGreaterThanEqual => {
                signed_greater_than_equal::SignedGreaterThanEqualTable.materialize_entry(index)
            }
            Self::UnsignedGreaterThanEqual => {
                unsigned_greater_than_equal::UnsignedGreaterThanEqualTable.materialize_entry(index)
            }
            Self::UnsignedLessThanEqual => {
                unsigned_less_than_equal::UnsignedLessThanEqualTable.materialize_entry(index)
            }
            Self::UpperWord => upper_word::UpperWordTable.materialize_entry(index),
            Self::LowerHalfWord => lower_half_word::LowerHalfWordTable.materialize_entry(index),
            Self::SignExtendHalfWord => {
                sign_extend_half_word::SignExtendHalfWordTable.materialize_entry(index)
            }
            Self::SignMask => sign_mask::SignMaskTable.materialize_entry(index),
            Self::Pow2 => pow2::Pow2Table.materialize_entry(index),
            Self::Pow2W => pow2_w::Pow2WTable.materialize_entry(index),
            Self::ShiftRightBitmask => {
                shift_right_bitmask::ShiftRightBitmaskTable.materialize_entry(index)
            }
            Self::VirtualSRL => virtual_srl::VirtualSRLTable.materialize_entry(index),
            Self::VirtualSRA => virtual_sra::VirtualSRATable.materialize_entry(index),
            Self::VirtualROTR => virtual_rotr::VirtualROTRTable.materialize_entry(index),
            Self::VirtualROTRW => virtual_rotrw::VirtualROTRWTable.materialize_entry(index),
            Self::ValidDiv0 => valid_div0::ValidDiv0Table.materialize_entry(index),
            Self::ValidUnsignedRemainder => {
                valid_unsigned_remainder::ValidUnsignedRemainderTable.materialize_entry(index)
            }
            Self::ValidSignedRemainder => {
                valid_signed_remainder::ValidSignedRemainderTable.materialize_entry(index)
            }
            Self::VirtualChangeDivisor => {
                virtual_change_divisor::VirtualChangeDivisorTable.materialize_entry(index)
            }
            Self::VirtualChangeDivisorW => {
                virtual_change_divisor_w::VirtualChangeDivisorWTable.materialize_entry(index)
            }
            Self::HalfwordAlignment => {
                halfword_alignment::HalfwordAlignmentTable.materialize_entry(index)
            }
            Self::WordAlignment => word_alignment::WordAlignmentTable.materialize_entry(index),
            Self::MulUNoOverflow => mulu_no_overflow::MulUNoOverflowTable.materialize_entry(index),
            Self::VirtualRev8W => virtual_rev8w::VirtualRev8WTable.materialize_entry(index),
            Self::VirtualXORROT32 => {
                virtual_xor_rot::VirtualXORROTTable::<32>.materialize_entry(index)
            }
            Self::VirtualXORROT24 => {
                virtual_xor_rot::VirtualXORROTTable::<24>.materialize_entry(index)
            }
            Self::VirtualXORROT16 => {
                virtual_xor_rot::VirtualXORROTTable::<16>.materialize_entry(index)
            }
            Self::VirtualXORROT63 => {
                virtual_xor_rot::VirtualXORROTTable::<63>.materialize_entry(index)
            }
            Self::VirtualXORROTW16 => {
                virtual_xor_rotw::VirtualXORROTWTable::<16>.materialize_entry(index)
            }
            Self::VirtualXORROTW12 => {
                virtual_xor_rotw::VirtualXORROTWTable::<12>.materialize_entry(index)
            }
            Self::VirtualXORROTW8 => {
                virtual_xor_rotw::VirtualXORROTWTable::<8>.materialize_entry(index)
            }
            Self::VirtualXORROTW7 => {
                virtual_xor_rotw::VirtualXORROTWTable::<7>.materialize_entry(index)
            }
        }
    }

    pub fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        match self {
            Self::RangeCheck => range_check::RangeCheckTable.evaluate_mle(r),
            Self::RangeCheckAligned => range_check_aligned::RangeCheckAlignedTable.evaluate_mle(r),
            Self::And => and::AndTable.evaluate_mle(r),
            Self::Andn => andn::AndnTable.evaluate_mle(r),
            Self::Or => or::OrTable.evaluate_mle(r),
            Self::Xor => xor::XorTable.evaluate_mle(r),
            Self::Equal => equal::EqualTable.evaluate_mle(r),
            Self::NotEqual => not_equal::NotEqualTable.evaluate_mle(r),
            Self::SignedLessThan => signed_less_than::SignedLessThanTable.evaluate_mle(r),
            Self::UnsignedLessThan => unsigned_less_than::UnsignedLessThanTable.evaluate_mle(r),
            Self::SignedGreaterThanEqual => {
                signed_greater_than_equal::SignedGreaterThanEqualTable.evaluate_mle(r)
            }
            Self::UnsignedGreaterThanEqual => {
                unsigned_greater_than_equal::UnsignedGreaterThanEqualTable.evaluate_mle(r)
            }
            Self::UnsignedLessThanEqual => {
                unsigned_less_than_equal::UnsignedLessThanEqualTable.evaluate_mle(r)
            }
            Self::UpperWord => upper_word::UpperWordTable.evaluate_mle(r),
            Self::LowerHalfWord => lower_half_word::LowerHalfWordTable.evaluate_mle(r),
            Self::SignExtendHalfWord => {
                sign_extend_half_word::SignExtendHalfWordTable.evaluate_mle(r)
            }
            Self::SignMask => sign_mask::SignMaskTable.evaluate_mle(r),
            Self::Pow2 => pow2::Pow2Table.evaluate_mle(r),
            Self::Pow2W => pow2_w::Pow2WTable.evaluate_mle(r),
            Self::ShiftRightBitmask => shift_right_bitmask::ShiftRightBitmaskTable.evaluate_mle(r),
            Self::VirtualSRL => virtual_srl::VirtualSRLTable.evaluate_mle(r),
            Self::VirtualSRA => virtual_sra::VirtualSRATable.evaluate_mle(r),
            Self::VirtualROTR => virtual_rotr::VirtualROTRTable.evaluate_mle(r),
            Self::VirtualROTRW => virtual_rotrw::VirtualROTRWTable.evaluate_mle(r),
            Self::ValidDiv0 => valid_div0::ValidDiv0Table.evaluate_mle(r),
            Self::ValidUnsignedRemainder => {
                valid_unsigned_remainder::ValidUnsignedRemainderTable.evaluate_mle(r)
            }
            Self::ValidSignedRemainder => {
                valid_signed_remainder::ValidSignedRemainderTable.evaluate_mle(r)
            }
            Self::VirtualChangeDivisor => {
                virtual_change_divisor::VirtualChangeDivisorTable.evaluate_mle(r)
            }
            Self::VirtualChangeDivisorW => {
                virtual_change_divisor_w::VirtualChangeDivisorWTable.evaluate_mle(r)
            }
            Self::HalfwordAlignment => halfword_alignment::HalfwordAlignmentTable.evaluate_mle(r),
            Self::WordAlignment => word_alignment::WordAlignmentTable.evaluate_mle(r),
            Self::MulUNoOverflow => mulu_no_overflow::MulUNoOverflowTable.evaluate_mle(r),
            Self::VirtualRev8W => virtual_rev8w::VirtualRev8WTable.evaluate_mle(r),
            Self::VirtualXORROT32 => virtual_xor_rot::VirtualXORROTTable::<32>.evaluate_mle(r),
            Self::VirtualXORROT24 => virtual_xor_rot::VirtualXORROTTable::<24>.evaluate_mle(r),
            Self::VirtualXORROT16 => virtual_xor_rot::VirtualXORROTTable::<16>.evaluate_mle(r),
            Self::VirtualXORROT63 => virtual_xor_rot::VirtualXORROTTable::<63>.evaluate_mle(r),
            Self::VirtualXORROTW16 => virtual_xor_rotw::VirtualXORROTWTable::<16>.evaluate_mle(r),
            Self::VirtualXORROTW12 => virtual_xor_rotw::VirtualXORROTWTable::<12>.evaluate_mle(r),
            Self::VirtualXORROTW8 => virtual_xor_rotw::VirtualXORROTWTable::<8>.evaluate_mle(r),
            Self::VirtualXORROTW7 => virtual_xor_rotw::VirtualXORROTWTable::<7>.evaluate_mle(r),
        }
    }

    pub fn suffixes(&self) -> &'static [Suffixes] {
        match self {
            Self::RangeCheck => PrefixSuffixDecomposition::suffixes(&range_check::RangeCheckTable),
            Self::RangeCheckAligned => {
                PrefixSuffixDecomposition::suffixes(&range_check_aligned::RangeCheckAlignedTable)
            }
            Self::And => PrefixSuffixDecomposition::suffixes(&and::AndTable),
            Self::Andn => PrefixSuffixDecomposition::suffixes(&andn::AndnTable),
            Self::Or => PrefixSuffixDecomposition::suffixes(&or::OrTable),
            Self::Xor => PrefixSuffixDecomposition::suffixes(&xor::XorTable),
            Self::Equal => PrefixSuffixDecomposition::suffixes(&equal::EqualTable),
            Self::NotEqual => PrefixSuffixDecomposition::suffixes(&not_equal::NotEqualTable),
            Self::SignedLessThan => {
                PrefixSuffixDecomposition::suffixes(&signed_less_than::SignedLessThanTable)
            }
            Self::UnsignedLessThan => {
                PrefixSuffixDecomposition::suffixes(&unsigned_less_than::UnsignedLessThanTable)
            }
            Self::SignedGreaterThanEqual => PrefixSuffixDecomposition::suffixes(
                &signed_greater_than_equal::SignedGreaterThanEqualTable,
            ),
            Self::UnsignedGreaterThanEqual => PrefixSuffixDecomposition::suffixes(
                &unsigned_greater_than_equal::UnsignedGreaterThanEqualTable,
            ),
            Self::UnsignedLessThanEqual => PrefixSuffixDecomposition::suffixes(
                &unsigned_less_than_equal::UnsignedLessThanEqualTable,
            ),
            Self::UpperWord => PrefixSuffixDecomposition::suffixes(&upper_word::UpperWordTable),
            Self::LowerHalfWord => {
                PrefixSuffixDecomposition::suffixes(&lower_half_word::LowerHalfWordTable)
            }
            Self::SignExtendHalfWord => {
                PrefixSuffixDecomposition::suffixes(&sign_extend_half_word::SignExtendHalfWordTable)
            }
            Self::SignMask => PrefixSuffixDecomposition::suffixes(&sign_mask::SignMaskTable),
            Self::Pow2 => PrefixSuffixDecomposition::suffixes(&pow2::Pow2Table),
            Self::Pow2W => PrefixSuffixDecomposition::suffixes(&pow2_w::Pow2WTable),
            Self::ShiftRightBitmask => {
                PrefixSuffixDecomposition::suffixes(&shift_right_bitmask::ShiftRightBitmaskTable)
            }
            Self::VirtualSRL => PrefixSuffixDecomposition::suffixes(&virtual_srl::VirtualSRLTable),
            Self::VirtualSRA => PrefixSuffixDecomposition::suffixes(&virtual_sra::VirtualSRATable),
            Self::VirtualROTR => {
                PrefixSuffixDecomposition::suffixes(&virtual_rotr::VirtualROTRTable)
            }
            Self::VirtualROTRW => {
                PrefixSuffixDecomposition::suffixes(&virtual_rotrw::VirtualROTRWTable)
            }
            Self::ValidDiv0 => PrefixSuffixDecomposition::suffixes(&valid_div0::ValidDiv0Table),
            Self::ValidUnsignedRemainder => PrefixSuffixDecomposition::suffixes(
                &valid_unsigned_remainder::ValidUnsignedRemainderTable,
            ),
            Self::ValidSignedRemainder => PrefixSuffixDecomposition::suffixes(
                &valid_signed_remainder::ValidSignedRemainderTable,
            ),
            Self::VirtualChangeDivisor => PrefixSuffixDecomposition::suffixes(
                &virtual_change_divisor::VirtualChangeDivisorTable,
            ),
            Self::VirtualChangeDivisorW => PrefixSuffixDecomposition::suffixes(
                &virtual_change_divisor_w::VirtualChangeDivisorWTable,
            ),
            Self::HalfwordAlignment => {
                PrefixSuffixDecomposition::suffixes(&halfword_alignment::HalfwordAlignmentTable)
            }
            Self::WordAlignment => {
                PrefixSuffixDecomposition::suffixes(&word_alignment::WordAlignmentTable)
            }
            Self::MulUNoOverflow => {
                PrefixSuffixDecomposition::suffixes(&mulu_no_overflow::MulUNoOverflowTable)
            }
            Self::VirtualRev8W => {
                PrefixSuffixDecomposition::suffixes(&virtual_rev8w::VirtualRev8WTable)
            }
            Self::VirtualXORROT32 => {
                PrefixSuffixDecomposition::suffixes(&virtual_xor_rot::VirtualXORROTTable::<32>)
            }
            Self::VirtualXORROT24 => {
                PrefixSuffixDecomposition::suffixes(&virtual_xor_rot::VirtualXORROTTable::<24>)
            }
            Self::VirtualXORROT16 => {
                PrefixSuffixDecomposition::suffixes(&virtual_xor_rot::VirtualXORROTTable::<16>)
            }
            Self::VirtualXORROT63 => {
                PrefixSuffixDecomposition::suffixes(&virtual_xor_rot::VirtualXORROTTable::<63>)
            }
            Self::VirtualXORROTW16 => {
                PrefixSuffixDecomposition::suffixes(&virtual_xor_rotw::VirtualXORROTWTable::<16>)
            }
            Self::VirtualXORROTW12 => {
                PrefixSuffixDecomposition::suffixes(&virtual_xor_rotw::VirtualXORROTWTable::<12>)
            }
            Self::VirtualXORROTW8 => {
                PrefixSuffixDecomposition::suffixes(&virtual_xor_rotw::VirtualXORROTWTable::<8>)
            }
            Self::VirtualXORROTW7 => {
                PrefixSuffixDecomposition::suffixes(&virtual_xor_rotw::VirtualXORROTWTable::<7>)
            }
        }
    }

    pub fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        match self {
            Self::RangeCheck => PrefixSuffixDecomposition::combine(
                &range_check::RangeCheckTable,
                prefixes,
                suffixes,
            ),
            Self::RangeCheckAligned => PrefixSuffixDecomposition::combine(
                &range_check_aligned::RangeCheckAlignedTable,
                prefixes,
                suffixes,
            ),
            Self::And => PrefixSuffixDecomposition::combine(&and::AndTable, prefixes, suffixes),
            Self::Andn => PrefixSuffixDecomposition::combine(&andn::AndnTable, prefixes, suffixes),
            Self::Or => PrefixSuffixDecomposition::combine(&or::OrTable, prefixes, suffixes),
            Self::Xor => PrefixSuffixDecomposition::combine(&xor::XorTable, prefixes, suffixes),
            Self::Equal => {
                PrefixSuffixDecomposition::combine(&equal::EqualTable, prefixes, suffixes)
            }
            Self::NotEqual => {
                PrefixSuffixDecomposition::combine(&not_equal::NotEqualTable, prefixes, suffixes)
            }
            Self::SignedLessThan => PrefixSuffixDecomposition::combine(
                &signed_less_than::SignedLessThanTable,
                prefixes,
                suffixes,
            ),
            Self::UnsignedLessThan => PrefixSuffixDecomposition::combine(
                &unsigned_less_than::UnsignedLessThanTable,
                prefixes,
                suffixes,
            ),
            Self::SignedGreaterThanEqual => PrefixSuffixDecomposition::combine(
                &signed_greater_than_equal::SignedGreaterThanEqualTable,
                prefixes,
                suffixes,
            ),
            Self::UnsignedGreaterThanEqual => PrefixSuffixDecomposition::combine(
                &unsigned_greater_than_equal::UnsignedGreaterThanEqualTable,
                prefixes,
                suffixes,
            ),
            Self::UnsignedLessThanEqual => PrefixSuffixDecomposition::combine(
                &unsigned_less_than_equal::UnsignedLessThanEqualTable,
                prefixes,
                suffixes,
            ),
            Self::UpperWord => {
                PrefixSuffixDecomposition::combine(&upper_word::UpperWordTable, prefixes, suffixes)
            }
            Self::LowerHalfWord => PrefixSuffixDecomposition::combine(
                &lower_half_word::LowerHalfWordTable,
                prefixes,
                suffixes,
            ),
            Self::SignExtendHalfWord => PrefixSuffixDecomposition::combine(
                &sign_extend_half_word::SignExtendHalfWordTable,
                prefixes,
                suffixes,
            ),
            Self::SignMask => {
                PrefixSuffixDecomposition::combine(&sign_mask::SignMaskTable, prefixes, suffixes)
            }
            Self::Pow2 => PrefixSuffixDecomposition::combine(&pow2::Pow2Table, prefixes, suffixes),
            Self::Pow2W => {
                PrefixSuffixDecomposition::combine(&pow2_w::Pow2WTable, prefixes, suffixes)
            }
            Self::ShiftRightBitmask => PrefixSuffixDecomposition::combine(
                &shift_right_bitmask::ShiftRightBitmaskTable,
                prefixes,
                suffixes,
            ),
            Self::VirtualSRL => PrefixSuffixDecomposition::combine(
                &virtual_srl::VirtualSRLTable,
                prefixes,
                suffixes,
            ),
            Self::VirtualSRA => PrefixSuffixDecomposition::combine(
                &virtual_sra::VirtualSRATable,
                prefixes,
                suffixes,
            ),
            Self::VirtualROTR => PrefixSuffixDecomposition::combine(
                &virtual_rotr::VirtualROTRTable,
                prefixes,
                suffixes,
            ),
            Self::VirtualROTRW => PrefixSuffixDecomposition::combine(
                &virtual_rotrw::VirtualROTRWTable,
                prefixes,
                suffixes,
            ),
            Self::ValidDiv0 => {
                PrefixSuffixDecomposition::combine(&valid_div0::ValidDiv0Table, prefixes, suffixes)
            }
            Self::ValidUnsignedRemainder => PrefixSuffixDecomposition::combine(
                &valid_unsigned_remainder::ValidUnsignedRemainderTable,
                prefixes,
                suffixes,
            ),
            Self::ValidSignedRemainder => PrefixSuffixDecomposition::combine(
                &valid_signed_remainder::ValidSignedRemainderTable,
                prefixes,
                suffixes,
            ),
            Self::VirtualChangeDivisor => PrefixSuffixDecomposition::combine(
                &virtual_change_divisor::VirtualChangeDivisorTable,
                prefixes,
                suffixes,
            ),
            Self::VirtualChangeDivisorW => PrefixSuffixDecomposition::combine(
                &virtual_change_divisor_w::VirtualChangeDivisorWTable,
                prefixes,
                suffixes,
            ),
            Self::HalfwordAlignment => PrefixSuffixDecomposition::combine(
                &halfword_alignment::HalfwordAlignmentTable,
                prefixes,
                suffixes,
            ),
            Self::WordAlignment => PrefixSuffixDecomposition::combine(
                &word_alignment::WordAlignmentTable,
                prefixes,
                suffixes,
            ),
            Self::MulUNoOverflow => PrefixSuffixDecomposition::combine(
                &mulu_no_overflow::MulUNoOverflowTable,
                prefixes,
                suffixes,
            ),
            Self::VirtualRev8W => PrefixSuffixDecomposition::combine(
                &virtual_rev8w::VirtualRev8WTable,
                prefixes,
                suffixes,
            ),
            Self::VirtualXORROT32 => PrefixSuffixDecomposition::combine(
                &virtual_xor_rot::VirtualXORROTTable::<32>,
                prefixes,
                suffixes,
            ),
            Self::VirtualXORROT24 => PrefixSuffixDecomposition::combine(
                &virtual_xor_rot::VirtualXORROTTable::<24>,
                prefixes,
                suffixes,
            ),
            Self::VirtualXORROT16 => PrefixSuffixDecomposition::combine(
                &virtual_xor_rot::VirtualXORROTTable::<16>,
                prefixes,
                suffixes,
            ),
            Self::VirtualXORROT63 => PrefixSuffixDecomposition::combine(
                &virtual_xor_rot::VirtualXORROTTable::<63>,
                prefixes,
                suffixes,
            ),
            Self::VirtualXORROTW16 => PrefixSuffixDecomposition::combine(
                &virtual_xor_rotw::VirtualXORROTWTable::<16>,
                prefixes,
                suffixes,
            ),
            Self::VirtualXORROTW12 => PrefixSuffixDecomposition::combine(
                &virtual_xor_rotw::VirtualXORROTWTable::<12>,
                prefixes,
                suffixes,
            ),
            Self::VirtualXORROTW8 => PrefixSuffixDecomposition::combine(
                &virtual_xor_rotw::VirtualXORROTWTable::<8>,
                prefixes,
                suffixes,
            ),
            Self::VirtualXORROTW7 => PrefixSuffixDecomposition::combine(
                &virtual_xor_rotw::VirtualXORROTWTable::<7>,
                prefixes,
                suffixes,
            ),
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
pub trait PrefixSuffixDecomposition: crate::LookupTable + Default {
    /// The suffix types used in this table's decomposition.
    fn suffixes(&self) -> &'static [Suffixes];

    /// Recombine evaluated prefix and suffix values into the table's MLE evaluation.
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F;

    /// Generate a random lookup index for testing.
    ///
    /// The default returns a uniform random u128. Tables with constrained input
    /// domains (e.g., shift/rotate tables that expect bitmask-shaped right operands)
    /// should override this to produce valid test inputs.
    #[cfg(test)]
    fn random_lookup_index(rng: &mut rand::rngs::StdRng) -> u128 {
        rand::Rng::gen(rng)
    }
}

#[cfg(test)]
pub(crate) mod test_utils;
