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
pub mod movsign;
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
/// declare which table they use via [`Instruction::lookup_table()`](crate::Instruction::lookup_table).
///
/// The enum is `#[repr(u8)]` for compact serialization and efficient
/// discriminant extraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    Movsign,

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
    /// Total number of distinct lookup table types.
    pub const COUNT: usize = 40;

    /// Returns the discriminant as a `usize`, suitable for array indexing.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
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
pub trait PrefixSuffixDecomposition<const XLEN: usize>: crate::LookupTable<XLEN> + Default {
    /// The suffix types used in this table's decomposition.
    fn suffixes(&self) -> Vec<Suffixes>;

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

#[cfg(test)]
mod lookup_table_tests;

/// Runtime dispatch wrapper over all concrete lookup tables.
///
/// Each variant corresponds 1:1 to a [`LookupTableKind`] and delegates to the
/// concrete ZST table type. The `XLEN` const generic selects the word size
/// (8 for tests, 64 for production).
///
/// Construct from a [`LookupTableKind`] via [`From`]:
/// ```ignore
/// let table = LookupTables::<64>::from(LookupTableKind::And);
/// assert_eq!(table.materialize_entry(0b11), 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LookupTables<const XLEN: usize> {
    RangeCheck,
    RangeCheckAligned,
    And,
    Andn,
    Or,
    Xor,
    Equal,
    NotEqual,
    SignedLessThan,
    UnsignedLessThan,
    SignedGreaterThanEqual,
    UnsignedGreaterThanEqual,
    UnsignedLessThanEqual,
    UpperWord,
    LowerHalfWord,
    SignExtendHalfWord,
    Movsign,
    Pow2,
    Pow2W,
    ShiftRightBitmask,
    VirtualSRL,
    VirtualSRA,
    VirtualROTR,
    VirtualROTRW,
    ValidDiv0,
    ValidUnsignedRemainder,
    ValidSignedRemainder,
    VirtualChangeDivisor,
    VirtualChangeDivisorW,
    HalfwordAlignment,
    WordAlignment,
    MulUNoOverflow,
    VirtualRev8W,
    VirtualXORROT32,
    VirtualXORROT24,
    VirtualXORROT16,
    VirtualXORROT63,
    VirtualXORROTW16,
    VirtualXORROTW12,
    VirtualXORROTW8,
    VirtualXORROTW7,
}

/// Dispatches a method call to the concrete table type for each variant.
macro_rules! dispatch_table {
    ($self:expr, |$t:ident| $body:expr) => {
        match $self {
            Self::RangeCheck => { let $t = range_check::RangeCheckTable::<XLEN>; $body }
            Self::RangeCheckAligned => { let $t = range_check_aligned::RangeCheckAlignedTable::<XLEN>; $body }
            Self::And => { let $t = and::AndTable::<XLEN>; $body }
            Self::Andn => { let $t = andn::AndnTable::<XLEN>; $body }
            Self::Or => { let $t = or::OrTable::<XLEN>; $body }
            Self::Xor => { let $t = xor::XorTable::<XLEN>; $body }
            Self::Equal => { let $t = equal::EqualTable::<XLEN>; $body }
            Self::NotEqual => { let $t = not_equal::NotEqualTable::<XLEN>; $body }
            Self::SignedLessThan => { let $t = signed_less_than::SignedLessThanTable::<XLEN>; $body }
            Self::UnsignedLessThan => { let $t = unsigned_less_than::UnsignedLessThanTable::<XLEN>; $body }
            Self::SignedGreaterThanEqual => { let $t = signed_greater_than_equal::SignedGreaterThanEqualTable::<XLEN>; $body }
            Self::UnsignedGreaterThanEqual => { let $t = unsigned_greater_than_equal::UnsignedGreaterThanEqualTable::<XLEN>; $body }
            Self::UnsignedLessThanEqual => { let $t = unsigned_less_than_equal::UnsignedLessThanEqualTable::<XLEN>; $body }
            Self::UpperWord => { let $t = upper_word::UpperWordTable::<XLEN>; $body }
            Self::LowerHalfWord => { let $t = lower_half_word::LowerHalfWordTable::<XLEN>; $body }
            Self::SignExtendHalfWord => { let $t = sign_extend_half_word::SignExtendHalfWordTable::<XLEN>; $body }
            Self::Movsign => { let $t = movsign::MovsignTable::<XLEN>; $body }
            Self::Pow2 => { let $t = pow2::Pow2Table::<XLEN>; $body }
            Self::Pow2W => { let $t = pow2_w::Pow2WTable::<XLEN>; $body }
            Self::ShiftRightBitmask => { let $t = shift_right_bitmask::ShiftRightBitmaskTable::<XLEN>; $body }
            Self::VirtualSRL => { let $t = virtual_srl::VirtualSRLTable::<XLEN>; $body }
            Self::VirtualSRA => { let $t = virtual_sra::VirtualSRATable::<XLEN>; $body }
            Self::VirtualROTR => { let $t = virtual_rotr::VirtualRotrTable::<XLEN>; $body }
            Self::VirtualROTRW => { let $t = virtual_rotrw::VirtualRotrWTable::<XLEN>; $body }
            Self::ValidDiv0 => { let $t = valid_div0::ValidDiv0Table::<XLEN>; $body }
            Self::ValidUnsignedRemainder => { let $t = valid_unsigned_remainder::ValidUnsignedRemainderTable::<XLEN>; $body }
            Self::ValidSignedRemainder => { let $t = valid_signed_remainder::ValidSignedRemainderTable::<XLEN>; $body }
            Self::VirtualChangeDivisor => { let $t = virtual_change_divisor::VirtualChangeDivisorTable::<XLEN>; $body }
            Self::VirtualChangeDivisorW => { let $t = virtual_change_divisor_w::VirtualChangeDivisorWTable::<XLEN>; $body }
            Self::HalfwordAlignment => { let $t = halfword_alignment::HalfwordAlignmentTable::<XLEN>; $body }
            Self::WordAlignment => { let $t = word_alignment::WordAlignmentTable::<XLEN>; $body }
            Self::MulUNoOverflow => { let $t = mulu_no_overflow::MulUNoOverflowTable::<XLEN>; $body }
            Self::VirtualRev8W => { let $t = virtual_rev8w::VirtualRev8WTable::<XLEN>; $body }
            Self::VirtualXORROT32 => { let $t = virtual_xor_rot::VirtualXORROTTable::<XLEN, 32>; $body }
            Self::VirtualXORROT24 => { let $t = virtual_xor_rot::VirtualXORROTTable::<XLEN, 24>; $body }
            Self::VirtualXORROT16 => { let $t = virtual_xor_rot::VirtualXORROTTable::<XLEN, 16>; $body }
            Self::VirtualXORROT63 => { let $t = virtual_xor_rot::VirtualXORROTTable::<XLEN, 63>; $body }
            Self::VirtualXORROTW16 => { let $t = virtual_xor_rotw::VirtualXORROTWTable::<XLEN, 16>; $body }
            Self::VirtualXORROTW12 => { let $t = virtual_xor_rotw::VirtualXORROTWTable::<XLEN, 12>; $body }
            Self::VirtualXORROTW8 => { let $t = virtual_xor_rotw::VirtualXORROTWTable::<XLEN, 8>; $body }
            Self::VirtualXORROTW7 => { let $t = virtual_xor_rotw::VirtualXORROTWTable::<XLEN, 7>; $body }
        }
    };
}

impl<const XLEN: usize> LookupTables<XLEN> {
    /// Returns the corresponding [`LookupTableKind`] identifier.
    #[inline]
    pub fn kind(self) -> LookupTableKind {
        match self {
            Self::RangeCheck => LookupTableKind::RangeCheck,
            Self::RangeCheckAligned => LookupTableKind::RangeCheckAligned,
            Self::And => LookupTableKind::And,
            Self::Andn => LookupTableKind::Andn,
            Self::Or => LookupTableKind::Or,
            Self::Xor => LookupTableKind::Xor,
            Self::Equal => LookupTableKind::Equal,
            Self::NotEqual => LookupTableKind::NotEqual,
            Self::SignedLessThan => LookupTableKind::SignedLessThan,
            Self::UnsignedLessThan => LookupTableKind::UnsignedLessThan,
            Self::SignedGreaterThanEqual => LookupTableKind::SignedGreaterThanEqual,
            Self::UnsignedGreaterThanEqual => LookupTableKind::UnsignedGreaterThanEqual,
            Self::UnsignedLessThanEqual => LookupTableKind::UnsignedLessThanEqual,
            Self::UpperWord => LookupTableKind::UpperWord,
            Self::LowerHalfWord => LookupTableKind::LowerHalfWord,
            Self::SignExtendHalfWord => LookupTableKind::SignExtendHalfWord,
            Self::Movsign => LookupTableKind::Movsign,
            Self::Pow2 => LookupTableKind::Pow2,
            Self::Pow2W => LookupTableKind::Pow2W,
            Self::ShiftRightBitmask => LookupTableKind::ShiftRightBitmask,
            Self::VirtualSRL => LookupTableKind::VirtualSRL,
            Self::VirtualSRA => LookupTableKind::VirtualSRA,
            Self::VirtualROTR => LookupTableKind::VirtualROTR,
            Self::VirtualROTRW => LookupTableKind::VirtualROTRW,
            Self::ValidDiv0 => LookupTableKind::ValidDiv0,
            Self::ValidUnsignedRemainder => LookupTableKind::ValidUnsignedRemainder,
            Self::ValidSignedRemainder => LookupTableKind::ValidSignedRemainder,
            Self::VirtualChangeDivisor => LookupTableKind::VirtualChangeDivisor,
            Self::VirtualChangeDivisorW => LookupTableKind::VirtualChangeDivisorW,
            Self::HalfwordAlignment => LookupTableKind::HalfwordAlignment,
            Self::WordAlignment => LookupTableKind::WordAlignment,
            Self::MulUNoOverflow => LookupTableKind::MulUNoOverflow,
            Self::VirtualRev8W => LookupTableKind::VirtualRev8W,
            Self::VirtualXORROT32 => LookupTableKind::VirtualXORROT32,
            Self::VirtualXORROT24 => LookupTableKind::VirtualXORROT24,
            Self::VirtualXORROT16 => LookupTableKind::VirtualXORROT16,
            Self::VirtualXORROT63 => LookupTableKind::VirtualXORROT63,
            Self::VirtualXORROTW16 => LookupTableKind::VirtualXORROTW16,
            Self::VirtualXORROTW12 => LookupTableKind::VirtualXORROTW12,
            Self::VirtualXORROTW8 => LookupTableKind::VirtualXORROTW8,
            Self::VirtualXORROTW7 => LookupTableKind::VirtualXORROTW7,
        }
    }

    /// The suffix types used in this table's prefix/suffix decomposition.
    pub fn suffixes(&self) -> Vec<Suffixes> {
        dispatch_table!(self, |t| PrefixSuffixDecomposition::<XLEN>::suffixes(&t))
    }

    /// Recombine evaluated prefix and suffix values into the table's MLE evaluation.
    pub fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        dispatch_table!(self, |t| PrefixSuffixDecomposition::<XLEN>::combine(&t, prefixes, suffixes))
    }
}

impl<const XLEN: usize> LookupTable<XLEN> for LookupTables<XLEN> {
    #[inline]
    fn materialize_entry(&self, index: u128) -> u64 {
        dispatch_table!(self, |t| t.materialize_entry(index))
    }

    #[inline]
    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        dispatch_table!(self, |t| t.evaluate_mle(r))
    }
}

impl<const XLEN: usize> From<LookupTableKind> for LookupTables<XLEN> {
    #[inline]
    fn from(kind: LookupTableKind) -> Self {
        match kind {
            LookupTableKind::RangeCheck => Self::RangeCheck,
            LookupTableKind::RangeCheckAligned => Self::RangeCheckAligned,
            LookupTableKind::And => Self::And,
            LookupTableKind::Andn => Self::Andn,
            LookupTableKind::Or => Self::Or,
            LookupTableKind::Xor => Self::Xor,
            LookupTableKind::Equal => Self::Equal,
            LookupTableKind::NotEqual => Self::NotEqual,
            LookupTableKind::SignedLessThan => Self::SignedLessThan,
            LookupTableKind::UnsignedLessThan => Self::UnsignedLessThan,
            LookupTableKind::SignedGreaterThanEqual => Self::SignedGreaterThanEqual,
            LookupTableKind::UnsignedGreaterThanEqual => Self::UnsignedGreaterThanEqual,
            LookupTableKind::UnsignedLessThanEqual => Self::UnsignedLessThanEqual,
            LookupTableKind::UpperWord => Self::UpperWord,
            LookupTableKind::LowerHalfWord => Self::LowerHalfWord,
            LookupTableKind::SignExtendHalfWord => Self::SignExtendHalfWord,
            LookupTableKind::Movsign => Self::Movsign,
            LookupTableKind::Pow2 => Self::Pow2,
            LookupTableKind::Pow2W => Self::Pow2W,
            LookupTableKind::ShiftRightBitmask => Self::ShiftRightBitmask,
            LookupTableKind::VirtualSRL => Self::VirtualSRL,
            LookupTableKind::VirtualSRA => Self::VirtualSRA,
            LookupTableKind::VirtualROTR => Self::VirtualROTR,
            LookupTableKind::VirtualROTRW => Self::VirtualROTRW,
            LookupTableKind::ValidDiv0 => Self::ValidDiv0,
            LookupTableKind::ValidUnsignedRemainder => Self::ValidUnsignedRemainder,
            LookupTableKind::ValidSignedRemainder => Self::ValidSignedRemainder,
            LookupTableKind::VirtualChangeDivisor => Self::VirtualChangeDivisor,
            LookupTableKind::VirtualChangeDivisorW => Self::VirtualChangeDivisorW,
            LookupTableKind::HalfwordAlignment => Self::HalfwordAlignment,
            LookupTableKind::WordAlignment => Self::WordAlignment,
            LookupTableKind::MulUNoOverflow => Self::MulUNoOverflow,
            LookupTableKind::VirtualRev8W => Self::VirtualRev8W,
            LookupTableKind::VirtualXORROT32 => Self::VirtualXORROT32,
            LookupTableKind::VirtualXORROT24 => Self::VirtualXORROT24,
            LookupTableKind::VirtualXORROT16 => Self::VirtualXORROT16,
            LookupTableKind::VirtualXORROT63 => Self::VirtualXORROT63,
            LookupTableKind::VirtualXORROTW16 => Self::VirtualXORROTW16,
            LookupTableKind::VirtualXORROTW12 => Self::VirtualXORROTW12,
            LookupTableKind::VirtualXORROTW8 => Self::VirtualXORROTW8,
            LookupTableKind::VirtualXORROTW7 => Self::VirtualXORROTW7,
        }
    }
}

impl<const XLEN: usize> From<LookupTables<XLEN>> for LookupTableKind {
    #[inline]
    fn from(table: LookupTables<XLEN>) -> Self {
        table.kind()
    }
}
