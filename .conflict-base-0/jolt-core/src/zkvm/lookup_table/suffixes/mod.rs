use crate::zkvm::lookup_table::suffixes::change_divisor::ChangeDivisorSuffix;
use crate::zkvm::lookup_table::suffixes::change_divisor_w::ChangeDivisorWSuffix;
use crate::zkvm::lookup_table::suffixes::left_shift::LeftShiftSuffix;
use crate::zkvm::lookup_table::suffixes::left_shift_w::LeftShiftWSuffix;
use crate::zkvm::lookup_table::suffixes::left_shift_w_helper::LeftShiftWHelperSuffix;
use crate::zkvm::lookup_table::suffixes::right_operand::RightOperandSuffix;
use crate::zkvm::lookup_table::suffixes::right_operand_w::RightOperandWSuffix;
use crate::zkvm::lookup_table::suffixes::sign_extension_right_operand::SignExtensionRightOperandSuffix;
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
use rev8w::Rev8W;
use right_is_zero::RightOperandIsZeroSuffix;
use right_shift::RightShiftSuffix;
use right_shift_helper::RightShiftHelperSuffix;
use right_shift_padding::RightShiftPaddingSuffix;
use right_shift_w::RightShiftWSuffix;
use right_shift_w_helper::RightShiftWHelperSuffix;
use sign_extension::SignExtensionSuffix;
use sign_extension_upper_half::SignExtensionUpperHalfSuffix;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use and::AndSuffix;
use lower_half_word::LowerHalfWordSuffix;
use lower_word::LowerWordSuffix;
use notand::NotAndSuffix;
use one::OneSuffix;
use overflow_bits_zero::OverflowBitsZeroSuffix;
use two_lsb::TwoLsbSuffix;
use upper_word::UpperWordSuffix;
use xor::XorSuffix;
use xor_rot::XorRotSuffix;
use xor_rotw::XorRotWSuffix;

pub mod and;
pub mod change_divisor;
pub mod change_divisor_w;
pub mod div_by_zero;
pub mod eq;
pub mod gt;
pub mod left_is_zero;
pub mod left_shift;
pub mod left_shift_w;
pub mod left_shift_w_helper;
pub mod lower_half_word;
pub mod lower_word;
pub mod lsb;
pub mod lt;
pub mod notand;
pub mod one;
pub mod or;
pub mod overflow_bits_zero;
pub mod pow2;
pub mod pow2_w;
pub mod rev8w;
pub mod right_is_zero;
pub mod right_operand;
pub mod right_operand_w;
pub mod right_shift;
pub mod right_shift_helper;
pub mod right_shift_padding;
pub mod right_shift_w;
pub mod right_shift_w_helper;
pub mod sign_extension;
pub mod sign_extension_right_operand;
pub mod sign_extension_upper_half;
pub mod two_lsb;
pub mod upper_word;
pub mod xor;
pub mod xor_rot;
pub mod xor_rotw;

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

pub type SuffixEval<F: JoltField> = F;

impl Suffixes {
    /// Returns true iff this suffix's `suffix_mle` output is guaranteed to be in {0, 1}.
    ///
    /// This is used for micro-optimizations that avoid calling `mul_u64_unreduced(1)`
    /// and instead directly add the unreduced field element when the suffix evaluates to 1.
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

    /// Evaluates the MLE for this suffix on the bitvector `b`, where
    /// `b` represents `b.len()` variables, each assuming a Boolean value.
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
            Suffixes::Rev8W => Rev8W::suffix_mle(b),
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
            Suffixes::RightShiftW => RightShiftWSuffix::suffix_mle(b),
            Suffixes::RightShiftWHelper => RightShiftWHelperSuffix::suffix_mle(b),
            Suffixes::LeftShiftWHelper => LeftShiftWHelperSuffix::suffix_mle(b),
            Suffixes::LeftShiftW => LeftShiftWSuffix::suffix_mle(b),
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
}
