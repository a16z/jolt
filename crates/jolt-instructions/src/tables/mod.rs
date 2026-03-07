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
pub mod suffixes;
pub mod upper_word;
pub mod valid_div0;
pub mod valid_signed_remainder;
pub mod valid_unsigned_remainder;
pub mod virtual_change_divisor_w;
pub mod virtual_rev8w;
pub mod virtual_xor_rot;
pub mod virtual_xor_rotw;
pub mod word_alignment;
pub mod xor;

pub use prefixes::{PrefixEval, Prefixes};
pub use suffixes::{SuffixEval, Suffixes};

/// Identifies a lookup table type.
///
/// Each variant corresponds to a concrete table with its own
/// [`LookupTable`](crate::LookupTable) implementation. Instructions
/// declare which table they use via [`Instruction::lookup_table()`](crate::Instruction::lookup_table).
///
/// The enum is `#[repr(u8)]` for compact serialization and efficient
/// discriminant extraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum LookupTableKind {
    //Arithmetic
    /// Identity/range-check: extracts the lower XLEN bits.
    /// Used by ADD, SUB, MUL, ADDI, JAL, and other combined-operand instructions.
    RangeCheck,
    /// Range check with LSB alignment (clears bit 0). Used by JALR.
    RangeCheckAligned,

    //Bitwise
    /// Bitwise AND. Used by AND, ANDI.
    And,
    /// Bitwise AND-NOT (x & !y). Used by ANDN (Zbb extension).
    Andn,
    /// Bitwise OR. Used by OR, ORI.
    Or,
    /// Bitwise XOR. Used by XOR, XORI.
    Xor,

    //Comparison
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

    //Word extraction
    /// Extract upper XLEN bits of a 2*XLEN-bit value. Used by MULHU.
    UpperWord,
    /// Extract lower half-word (XLEN/2 bits).
    LowerHalfWord,
    /// Sign-extend half-word to full word.
    SignExtendHalfWord,

    //Sign/conditional
    /// Sign-bit conditional: returns all-ones if MSB set, else zero. Used by MOVSIGN.
    Movsign,

    //Power of 2
    /// 2^(index mod XLEN). Used by POW2, POW2I.
    Pow2,
    /// 2^(index mod 32). Used by POW2W, POW2IW.
    Pow2W,

    //Shift
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

    //Division validation
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

    //Alignment
    /// Halfword alignment check (divisible by 2).
    HalfwordAlignment,
    /// Word alignment check (divisible by 4).
    WordAlignment,

    //Multiply overflow
    /// Unsigned multiply no-overflow check. Used by ASSERT_MULU_NO_OVERFLOW.
    MulUNoOverflow,

    //Byte manipulation
    /// Byte-reverse within word. Used by REV8W.
    VirtualRev8W,

    //XOR-rotate (SHA)
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
}
