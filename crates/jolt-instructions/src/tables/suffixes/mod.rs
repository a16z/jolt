//! Suffix polynomial evaluations for the sparse-dense decomposition.
//!
//! Each suffix computes a function over the "unbound" low-order bits of a
//! lookup index during the sumcheck protocol. Suffixes evaluate to `u64`
//! values (not field elements), making them cheap to compute and
//! field-independent.
//!
//! The decomposition works as: `table_mle(r) = Σ prefix_i(r_high) · suffix_i(b_low)`,
//! where `b_low` ranges over the Boolean hypercube.

use crate::lookup_bits::LookupBits;

mod and;
mod change_divisor;
mod change_divisor_w;
mod div_by_zero;
mod eq;
mod gt;
mod left_is_zero;
mod left_shift;
mod left_shift_w;
mod left_shift_w_helper;
mod lower_half_word;
mod lower_word;
mod lsb;
mod lt;
mod notand;
mod one;
mod or;
mod overflow_bits_zero;
mod pow2;
mod pow2_w;
mod rev8w;
mod right_is_zero;
mod right_operand;
mod right_operand_w;
mod right_shift;
mod right_shift_helper;
mod right_shift_padding;
mod right_shift_w;
mod right_shift_w_helper;
mod sign_extension;
mod sign_extension_right_operand;
mod sign_extension_upper_half;
mod two_lsb;
mod upper_word;
mod xor;
mod xor_rot;
mod xor_rotw;

use and::AndSuffix;
use change_divisor::ChangeDivisorSuffix;
use change_divisor_w::ChangeDivisorWSuffix;
use div_by_zero::DivByZeroSuffix;
use eq::EqSuffix;
use gt::GreaterThanSuffix;
use left_is_zero::LeftOperandIsZeroSuffix;
use left_shift::LeftShiftSuffix;
use left_shift_w::LeftShiftWSuffix;
use left_shift_w_helper::LeftShiftWHelperSuffix;
use lower_half_word::LowerHalfWordSuffix;
use lower_word::LowerWordSuffix;
use lsb::LsbSuffix;
use lt::LessThanSuffix;
use notand::NotAndSuffix;
use one::OneSuffix;
use or::OrSuffix;
use overflow_bits_zero::OverflowBitsZeroSuffix;
use pow2::Pow2Suffix;
use pow2_w::Pow2WSuffix;
use rev8w::Rev8WSuffix;
use right_is_zero::RightOperandIsZeroSuffix;
use right_operand::RightOperandSuffix;
use right_operand_w::RightOperandWSuffix;
use right_shift::RightShiftSuffix;
use right_shift_helper::RightShiftHelperSuffix;
use right_shift_padding::RightShiftPaddingSuffix;
use right_shift_w::RightShiftWSuffix;
use right_shift_w_helper::RightShiftWHelperSuffix;
use sign_extension::SignExtensionSuffix;
use sign_extension_right_operand::SignExtensionRightOperandSuffix;
use sign_extension_upper_half::SignExtensionUpperHalfSuffix;
use two_lsb::TwoLsbSuffix;
use upper_word::UpperWordSuffix;
use xor::XorSuffix;
use xor_rot::XorRotSuffix;
use xor_rotw::XorRotWSuffix;

use jolt_field::Field;

/// A suffix polynomial: evaluates on unbound Boolean variables during sumcheck.
///
/// Suffixes return `u64` values (not field elements) to avoid unnecessary
/// field arithmetic when the result is a small integer.
pub trait SparseDenseSuffix: 'static + Sync {
    /// Evaluate this suffix's MLE on bitvector `b`, where `b.len()` variables
    /// are set to Boolean values.
    fn suffix_mle(b: LookupBits) -> u64;
}

/// Type alias for suffix evaluations promoted to field elements.
pub type SuffixEval<F> = F;

/// All suffix types used by Jolt's lookup tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Suffixes {
    One,
    And,
    NotAnd,
    Xor,
    Or,
    RightOperand,
    RightOperandW,
    ChangeDivisor,
    ChangeDivisorW,
    UpperWord,
    LowerWord,
    LowerHalfWord,
    LessThan,
    GreaterThan,
    Eq,
    LeftOperandIsZero,
    RightOperandIsZero,
    Lsb,
    DivByZero,
    Pow2,
    Pow2W,
    Rev8W,
    RightShiftPadding,
    RightShift,
    RightShiftHelper,
    SignExtension,
    LeftShift,
    TwoLsb,
    SignExtensionUpperHalf,
    SignExtensionRightOperand,
    RightShiftW,
    RightShiftWHelper,
    LeftShiftWHelper,
    LeftShiftW,
    OverflowBitsZero,
    XorRot16,
    XorRot24,
    XorRot32,
    XorRot63,
    XorRotW16,
    XorRotW12,
    XorRotW8,
    XorRotW7,
}

/// Total number of suffix variants.
pub const NUM_SUFFIXES: usize = 43;

impl Suffixes {
    /// Returns `true` if this suffix's output is guaranteed to be in {0, 1}.
    ///
    /// This enables micro-optimizations in the sumcheck prover that avoid
    /// multiplying by 1 (directly adding the unreduced field element instead).
    #[inline(always)]
    pub fn is_01_valued(&self) -> bool {
        matches!(
            self,
            Suffixes::One
                | Suffixes::Eq
                | Suffixes::LessThan
                | Suffixes::GreaterThan
                | Suffixes::LeftOperandIsZero
                | Suffixes::RightOperandIsZero
                | Suffixes::Lsb
                | Suffixes::TwoLsb
                | Suffixes::DivByZero
                | Suffixes::OverflowBitsZero
                | Suffixes::ChangeDivisor
                | Suffixes::ChangeDivisorW
        )
    }

    /// Evaluate this suffix's MLE on bitvector `b`.
    pub fn suffix_mle<const XLEN: usize>(&self, b: LookupBits) -> u64 {
        match self {
            Suffixes::One => OneSuffix::suffix_mle(b),
            Suffixes::And => AndSuffix::suffix_mle(b),
            Suffixes::NotAnd => NotAndSuffix::suffix_mle(b),
            Suffixes::Or => OrSuffix::suffix_mle(b),
            Suffixes::Xor => XorSuffix::suffix_mle(b),
            Suffixes::RightOperand => RightOperandSuffix::suffix_mle(b),
            Suffixes::RightOperandW => RightOperandWSuffix::suffix_mle(b),
            Suffixes::ChangeDivisor => ChangeDivisorSuffix::suffix_mle(b),
            Suffixes::ChangeDivisorW => ChangeDivisorWSuffix::<XLEN>::suffix_mle(b),
            Suffixes::UpperWord => UpperWordSuffix::<XLEN>::suffix_mle(b),
            Suffixes::LowerWord => LowerWordSuffix::<XLEN>::suffix_mle(b),
            Suffixes::LowerHalfWord => LowerHalfWordSuffix::<XLEN>::suffix_mle(b),
            Suffixes::LessThan => LessThanSuffix::suffix_mle(b),
            Suffixes::GreaterThan => GreaterThanSuffix::suffix_mle(b),
            Suffixes::Eq => EqSuffix::suffix_mle(b),
            Suffixes::LeftOperandIsZero => LeftOperandIsZeroSuffix::suffix_mle(b),
            Suffixes::RightOperandIsZero => RightOperandIsZeroSuffix::suffix_mle(b),
            Suffixes::Lsb => LsbSuffix::suffix_mle(b),
            Suffixes::DivByZero => DivByZeroSuffix::suffix_mle(b),
            Suffixes::Pow2 => Pow2Suffix::<XLEN>::suffix_mle(b),
            Suffixes::Pow2W => Pow2WSuffix::<XLEN>::suffix_mle(b),
            Suffixes::Rev8W => Rev8WSuffix::suffix_mle(b),
            Suffixes::RightShiftPadding => RightShiftPaddingSuffix::<XLEN>::suffix_mle(b),
            Suffixes::RightShift => RightShiftSuffix::suffix_mle(b),
            Suffixes::RightShiftHelper => RightShiftHelperSuffix::suffix_mle(b),
            Suffixes::SignExtension => SignExtensionSuffix::<XLEN>::suffix_mle(b),
            Suffixes::LeftShift => LeftShiftSuffix::suffix_mle(b),
            Suffixes::TwoLsb => TwoLsbSuffix::suffix_mle(b),
            Suffixes::SignExtensionUpperHalf => SignExtensionUpperHalfSuffix::<XLEN>::suffix_mle(b),
            Suffixes::SignExtensionRightOperand => {
                SignExtensionRightOperandSuffix::<XLEN>::suffix_mle(b)
            }
            Suffixes::RightShiftW => RightShiftWSuffix::<XLEN>::suffix_mle(b),
            Suffixes::RightShiftWHelper => RightShiftWHelperSuffix::<XLEN>::suffix_mle(b),
            Suffixes::LeftShiftWHelper => LeftShiftWHelperSuffix::suffix_mle(b),
            Suffixes::LeftShiftW => LeftShiftWSuffix::<XLEN>::suffix_mle(b),
            Suffixes::OverflowBitsZero => OverflowBitsZeroSuffix::<XLEN>::suffix_mle(b),
            Suffixes::XorRot16 => XorRotSuffix::<16>::suffix_mle(b),
            Suffixes::XorRot24 => XorRotSuffix::<24>::suffix_mle(b),
            Suffixes::XorRot32 => XorRotSuffix::<32>::suffix_mle(b),
            Suffixes::XorRot63 => XorRotSuffix::<63>::suffix_mle(b),
            Suffixes::XorRotW7 => XorRotWSuffix::<7>::suffix_mle(b),
            Suffixes::XorRotW8 => XorRotWSuffix::<8>::suffix_mle(b),
            Suffixes::XorRotW12 => XorRotWSuffix::<12>::suffix_mle(b),
            Suffixes::XorRotW16 => XorRotWSuffix::<16>::suffix_mle(b),
        }
    }

    /// Evaluate and promote to a field element.
    #[inline]
    pub fn evaluate<const XLEN: usize, F: Field>(&self, b: LookupBits) -> SuffixEval<F> {
        F::from_u64(self.suffix_mle::<XLEN>(b))
    }
}
