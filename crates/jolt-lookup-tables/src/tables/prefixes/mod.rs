//! Prefix polynomial evaluations for the sparse-dense decomposition.
//!
//! Each prefix captures the "contribution" of high-order bound variables
//! to a lookup table's MLE during sumcheck. Prefixes are evaluated at
//! binary points to materialize a dense polynomial, which is then bound
//! using standard polynomial operations during sumcheck rounds.
//!
//! Checkpoints accumulate prefix values across phases. They are initialized
//! via [`SparseDensePrefix::default_checkpoint`] and updated by the consumer
//! at phase boundaries (the bound polynomial's final scalar becomes the new
//! checkpoint).

pub mod and;
pub mod andn;
pub mod change_divisor;
pub mod change_divisor_w;
pub mod div_by_zero;
pub mod eq;
pub mod left_is_zero;
pub mod left_operand_msb;
pub mod left_shift;
pub mod left_shift_helper;
pub mod left_shift_w;
pub mod left_shift_w_helper;
pub mod lower_half_word;
pub mod lower_word;
pub mod lsb;
pub mod lt;
pub mod negative_divisor_equals_remainder;
pub mod negative_divisor_greater_than_remainder;
pub mod negative_divisor_zero_remainder;
pub mod or;
pub mod overflow_bits_zero;
pub mod positive_remainder_equals_divisor;
pub mod positive_remainder_less_than_divisor;
pub mod pow2;
pub mod pow2_w;
pub mod rev8w;
pub mod right_is_zero;
pub mod right_operand;
pub mod right_operand_msb;
pub mod right_operand_w;
pub mod right_shift;
pub mod right_shift_w;
pub mod sign_extension;
pub mod sign_extension_right_operand;
pub mod sign_extension_upper_half;
pub mod two_lsb;
pub mod upper_word;
pub mod xor;
pub mod xor_rot;
pub mod xor_rotw;

use jolt_field::Field;
use std::fmt::Display;
use std::ops::Index;

use crate::lookup_bits::LookupBits;

/// A prefix polynomial evaluated at binary points during materialization.
///
/// Implementations provide:
/// - `default_checkpoint()`: the initial checkpoint value before any phases
/// - `evaluate()`: the prefix value at a binary point, given accumulated
///   checkpoints from previous phases
pub trait SparseDensePrefix<F: Field>: 'static + Sync {
    /// Default checkpoint value for this prefix before any phases have run.
    fn default_checkpoint() -> F;

    /// Evaluate this prefix at binary point `b`, given accumulated checkpoints
    /// from previous phases and the number of remaining suffix variables.
    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F;
}

/// Wrapper for prefix polynomial evaluations, used for type safety.
#[derive(Clone, Copy)]
pub struct PrefixEval<F>(pub(crate) F);

impl<F: Display> Display for PrefixEval<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<F> From<F> for PrefixEval<F> {
    fn from(value: F) -> Self {
        Self(value)
    }
}

impl<F> Index<Prefixes> for &[PrefixEval<F>] {
    type Output = F;

    #[expect(clippy::unwrap_used)]
    fn index(&self, prefix: Prefixes) -> &Self::Output {
        let index = prefix as usize;
        &self.get(index).unwrap().0
    }
}

/// All prefix types used by Jolt's lookup tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::EnumCount, strum::VariantArray)]
#[repr(u8)]
pub enum Prefixes {
    LowerWord,
    LowerHalfWord,
    UpperWord,
    Eq,
    And,
    Andn,
    Or,
    Xor,
    LessThan,
    LeftOperandIsZero,
    RightOperandIsZero,
    LeftOperandMsb,
    RightOperandMsb,
    DivByZero,
    PositiveRemainderEqualsDivisor,
    PositiveRemainderLessThanDivisor,
    NegativeDivisorZeroRemainder,
    NegativeDivisorEqualsRemainder,
    NegativeDivisorGreaterThanRemainder,
    Lsb,
    Pow2,
    Pow2W,
    Rev8W,
    RightShift,
    SignExtension,
    LeftShift,
    LeftShiftHelper,
    TwoLsb,
    SignExtensionUpperHalf,
    ChangeDivisor,
    ChangeDivisorW,
    RightOperand,
    RightOperandW,
    SignExtensionRightOperand,
    RightShiftW,
    LeftShiftWHelper,
    LeftShiftW,
    OverflowBitsZero,
    XorRot16,
    XorRot24,
    XorRot32,
    XorRot63,
    XorRotW7,
    XorRotW8,
    XorRotW12,
    XorRotW16,
}

/// Total number of prefix variants.
pub const NUM_PREFIXES: usize = <Prefixes as strum::EnumCount>::COUNT;

/// All prefix variants in discriminant order.
pub const ALL_PREFIXES: &[Prefixes] = <Prefixes as strum::VariantArray>::VARIANTS;

/// Dispatches a `SparseDensePrefix` method call to the concrete type for each `Prefixes` variant.
macro_rules! dispatch_prefix {
    ($self:expr, $method:ident) => {
        dispatch_prefix!($self, $method,)
    };
    ($self:expr, $method:ident, $($args:expr),* $(,)?) => {
        match $self {
            Prefixes::LowerWord => lower_word::LowerWordPrefix::$method($($args),*),
            Prefixes::LowerHalfWord => lower_half_word::LowerHalfWordPrefix::$method($($args),*),
            Prefixes::UpperWord => upper_word::UpperWordPrefix::$method($($args),*),
            Prefixes::Eq => eq::EqPrefix::$method($($args),*),
            Prefixes::And => and::AndPrefix::$method($($args),*),
            Prefixes::Andn => andn::AndnPrefix::$method($($args),*),
            Prefixes::Or => or::OrPrefix::$method($($args),*),
            Prefixes::Xor => xor::XorPrefix::$method($($args),*),
            Prefixes::LessThan => lt::LessThanPrefix::$method($($args),*),
            Prefixes::LeftOperandIsZero => left_is_zero::LeftOperandIsZeroPrefix::$method($($args),*),
            Prefixes::RightOperandIsZero => right_is_zero::RightOperandIsZeroPrefix::$method($($args),*),
            Prefixes::LeftOperandMsb => left_operand_msb::LeftOperandMsbPrefix::$method($($args),*),
            Prefixes::RightOperandMsb => right_operand_msb::RightOperandMsbPrefix::$method($($args),*),
            Prefixes::DivByZero => div_by_zero::DivByZeroPrefix::$method($($args),*),
            Prefixes::PositiveRemainderEqualsDivisor => positive_remainder_equals_divisor::PositiveRemainderEqualsDivisorPrefix::$method($($args),*),
            Prefixes::PositiveRemainderLessThanDivisor => positive_remainder_less_than_divisor::PositiveRemainderLessThanDivisorPrefix::$method($($args),*),
            Prefixes::NegativeDivisorZeroRemainder => negative_divisor_zero_remainder::NegativeDivisorZeroRemainderPrefix::$method($($args),*),
            Prefixes::NegativeDivisorEqualsRemainder => negative_divisor_equals_remainder::NegativeDivisorEqualsRemainderPrefix::$method($($args),*),
            Prefixes::NegativeDivisorGreaterThanRemainder => negative_divisor_greater_than_remainder::NegativeDivisorGreaterThanRemainderPrefix::$method($($args),*),
            Prefixes::Lsb => lsb::LsbPrefix::$method($($args),*),
            Prefixes::Pow2 => pow2::Pow2Prefix::$method($($args),*),
            Prefixes::Pow2W => pow2_w::Pow2WPrefix::$method($($args),*),
            Prefixes::Rev8W => rev8w::Rev8WPrefix::$method($($args),*),
            Prefixes::RightShift => right_shift::RightShiftPrefix::$method($($args),*),
            Prefixes::SignExtension => sign_extension::SignExtensionPrefix::$method($($args),*),
            Prefixes::LeftShift => left_shift::LeftShiftPrefix::$method($($args),*),
            Prefixes::LeftShiftHelper => left_shift_helper::LeftShiftHelperPrefix::$method($($args),*),
            Prefixes::TwoLsb => two_lsb::TwoLsbPrefix::$method($($args),*),
            Prefixes::SignExtensionUpperHalf => sign_extension_upper_half::SignExtensionUpperHalfPrefix::$method($($args),*),
            Prefixes::ChangeDivisor => change_divisor::ChangeDivisorPrefix::$method($($args),*),
            Prefixes::ChangeDivisorW => change_divisor_w::ChangeDivisorWPrefix::$method($($args),*),
            Prefixes::RightOperand => right_operand::RightOperandPrefix::$method($($args),*),
            Prefixes::RightOperandW => right_operand_w::RightOperandWPrefix::$method($($args),*),
            Prefixes::SignExtensionRightOperand => sign_extension_right_operand::SignExtensionRightOperandPrefix::$method($($args),*),
            Prefixes::RightShiftW => right_shift_w::RightShiftWPrefix::$method($($args),*),
            Prefixes::LeftShiftWHelper => left_shift_w_helper::LeftShiftWHelperPrefix::$method($($args),*),
            Prefixes::LeftShiftW => left_shift_w::LeftShiftWPrefix::$method($($args),*),
            Prefixes::OverflowBitsZero => overflow_bits_zero::OverflowBitsZeroPrefix::$method($($args),*),
            Prefixes::XorRot16 => xor_rot::XorRotPrefix::<16>::$method($($args),*),
            Prefixes::XorRot24 => xor_rot::XorRotPrefix::<24>::$method($($args),*),
            Prefixes::XorRot32 => xor_rot::XorRotPrefix::<32>::$method($($args),*),
            Prefixes::XorRot63 => xor_rot::XorRotPrefix::<63>::$method($($args),*),
            Prefixes::XorRotW7 => xor_rotw::XorRotWPrefix::<7>::$method($($args),*),
            Prefixes::XorRotW8 => xor_rotw::XorRotWPrefix::<8>::$method($($args),*),
            Prefixes::XorRotW12 => xor_rotw::XorRotWPrefix::<12>::$method($($args),*),
            Prefixes::XorRotW16 => xor_rotw::XorRotWPrefix::<16>::$method($($args),*),
        }
    };
}

impl Prefixes {
    /// Return the default checkpoint value for this prefix variant.
    pub fn default_checkpoint<F: Field>(&self) -> PrefixEval<F> {
        PrefixEval(dispatch_prefix!(self, default_checkpoint))
    }

    /// Evaluate this prefix at binary point `b`.
    pub fn evaluate<F: Field>(
        &self,
        checkpoints: &[PrefixEval<F>],
        b: LookupBits,
        suffix_len: usize,
    ) -> PrefixEval<F> {
        PrefixEval(dispatch_prefix!(self, evaluate, checkpoints, b, suffix_len))
    }
}
