use crate::zkvm::lookup_table::suffixes::change_divisor::ChangeDivisorSuffix;
use crate::zkvm::lookup_table::suffixes::left_shift::LeftShiftSuffix;
use crate::zkvm::lookup_table::suffixes::right_operand::RightOperandSuffix;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};
use div_by_zero::DivByZeroSuffix;
use eq::EqSuffix;
use gt::GreaterThanSuffix;
use left_is_zero::LeftOperandIsZeroSuffix;
use lsb::LsbSuffix;
use lt::LessThanSuffix;
use num_derive::FromPrimitive;
use or::OrSuffix;
use pow2::Pow2Suffix;
use pow2_w::Pow2WSuffix;
use right_is_zero::RightOperandIsZeroSuffix;
use right_shift::RightShiftSuffix;
use right_shift_helper::RightShiftHelperSuffix;
use right_shift_padding::RightShiftPaddingSuffix;
use sign_extension::SignExtensionSuffix;
use sign_extension_upper_half::SignExtensionUpperHalfSuffix;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use and::AndSuffix;
use lower_half_word::LowerHalfWordSuffix;
use lower_word::LowerWordSuffix;
use notand::NotAndSuffix;
use one::OneSuffix;
use two_lsb::TwoLsbSuffix;
use upper_word::UpperWordSuffix;
use xor::XorSuffix;

pub mod and;
pub mod change_divisor;
pub mod div_by_zero;
pub mod eq;
pub mod gt;
pub mod left_is_zero;
pub mod left_shift;
pub mod lower_half_word;
pub mod lower_word;
pub mod lsb;
pub mod lt;
pub mod notand;
pub mod one;
pub mod or;
pub mod pow2;
pub mod pow2_w;
pub mod right_is_zero;
pub mod right_operand;
pub mod right_shift;
pub mod right_shift_helper;
pub mod right_shift_padding;
pub mod sign_extension;
pub mod sign_extension_upper_half;
pub mod two_lsb;
pub mod upper_word;
pub mod xor;

pub trait SparseDenseSuffix: 'static + Sync {
    /// Evaluates the MLE for this suffix on the bitvector `b`, where
    /// `b` represents `b.len()` variables, each assuming a Boolean value.
    fn suffix_mle(b: LookupBits) -> u64;
}

/// An enum containing all suffixes used by Jolt's instruction lookup tables.
#[repr(u8)]
#[derive(EnumCountMacro, EnumIter, FromPrimitive)]
pub enum Suffixes {
    One,
    And,
    NotAnd,
    Xor,
    Or,
    RightOperand,
    ChangeDivisor,
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
    RightShiftPadding,
    RightShift,
    RightShiftHelper,
    SignExtension,
    LeftShift,
    TwoLsb,
    SignExtensionUpperHalf,
}

pub type SuffixEval<F: JoltField> = F;

impl Suffixes {
    /// Evaluates the MLE for this suffix on the bitvector `b`, where
    /// `b` represents `b.len()` variables, each assuming a Boolean value.
    pub fn suffix_mle<const WORD_SIZE: usize>(&self, b: LookupBits) -> u64 {
        match self {
            Suffixes::One => OneSuffix::suffix_mle(b),
            Suffixes::And => AndSuffix::suffix_mle(b),
            Suffixes::NotAnd => NotAndSuffix::suffix_mle(b),
            Suffixes::Or => OrSuffix::suffix_mle(b),
            Suffixes::Xor => XorSuffix::suffix_mle(b),
            Suffixes::RightOperand => RightOperandSuffix::suffix_mle(b),
            Suffixes::ChangeDivisor => ChangeDivisorSuffix::suffix_mle(b),
            Suffixes::UpperWord => UpperWordSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::LowerWord => LowerWordSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::LowerHalfWord => LowerHalfWordSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::LessThan => LessThanSuffix::suffix_mle(b),
            Suffixes::GreaterThan => GreaterThanSuffix::suffix_mle(b),
            Suffixes::Eq => EqSuffix::suffix_mle(b),
            Suffixes::LeftOperandIsZero => LeftOperandIsZeroSuffix::suffix_mle(b),
            Suffixes::RightOperandIsZero => RightOperandIsZeroSuffix::suffix_mle(b),
            Suffixes::Lsb => LsbSuffix::suffix_mle(b),
            Suffixes::DivByZero => DivByZeroSuffix::suffix_mle(b),
            Suffixes::Pow2 => Pow2Suffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::Pow2W => Pow2WSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::RightShiftPadding => RightShiftPaddingSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::RightShift => RightShiftSuffix::suffix_mle(b),
            Suffixes::RightShiftHelper => RightShiftHelperSuffix::suffix_mle(b),
            Suffixes::SignExtension => SignExtensionSuffix::<WORD_SIZE>::suffix_mle(b),
            Suffixes::LeftShift => LeftShiftSuffix::suffix_mle(b),
            Suffixes::TwoLsb => TwoLsbSuffix::suffix_mle(b),
            Suffixes::SignExtensionUpperHalf => {
                SignExtensionUpperHalfSuffix::<WORD_SIZE>::suffix_mle(b)
            }
        }
    }
}
