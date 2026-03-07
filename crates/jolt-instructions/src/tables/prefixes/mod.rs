//! Prefix polynomial evaluations for the sparse-dense decomposition.
//!
//! Each prefix captures the "contribution" of high-order bound variables
//! to a lookup table's MLE during sumcheck. Prefixes are field-valued
//! (unlike suffixes which are `u64`), and maintain checkpoints that are
//! updated every two sumcheck rounds.

pub mod and;
pub mod andn;
pub mod change_divisor;
pub mod change_divisor_w;
pub mod div_by_zero;
pub mod eq;
pub mod left_is_zero;
pub mod left_msb;
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
pub mod right_msb;
pub mod right_operand;
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

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;

/// A prefix polynomial: evaluates bound high-order variables during sumcheck.
///
/// The challenge type `C` supports smaller-than-field challenge values
/// for performance (e.g., 128-bit challenges with a 254-bit field).
pub trait SparseDensePrefix<F: Field>: 'static + Sync {
    /// Evaluate the prefix MLE incorporating the checkpoint, current variable `c`,
    /// and unbound variables `b`.
    ///
    /// - On odd rounds (`j` odd): `r_x` is `Some(challenge)` from the previous round.
    /// - On even rounds (`j` even): `r_x` is `None`; `c` is the current x-variable.
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>;

    /// Update the checkpoint after binding two variables (`r_x`, `r_y`).
    ///
    /// Called every two sumcheck rounds. May depend on other prefix checkpoints.
    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>;
}

/// Wrapper for prefix polynomial evaluations, used for type safety.
#[derive(Clone, Copy)]
pub struct PrefixEval<F>(pub(crate) F);

/// Cached prefix evaluation after each pair of address-binding rounds.
pub type PrefixCheckpoint<F> = PrefixEval<Option<F>>;

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

impl<F> PrefixCheckpoint<F> {
    /// Unwrap the checkpoint, panicking if it hasn't been initialized.
    pub fn unwrap(self) -> PrefixEval<F> {
        self.0.unwrap().into()
    }

    /// Returns the inner value if set, or the provided default.
    pub fn unwrap_or(self, default: F) -> F {
        self.0.unwrap_or(default)
    }
}

impl<F> Index<Prefixes> for &[PrefixEval<F>] {
    type Output = F;

    fn index(&self, prefix: Prefixes) -> &Self::Output {
        let index = prefix as usize;
        &self.get(index).unwrap().0
    }
}

/// All prefix types used by Jolt's lookup tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
pub const NUM_PREFIXES: usize = 45;

impl Prefixes {
    /// Evaluate the prefix MLE for this variant.
    pub fn prefix_mle<const XLEN: usize, F, C>(
        &self,
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> PrefixEval<F>
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        use self::and::AndPrefix;
        use self::andn::AndnPrefix;
        use self::change_divisor::ChangeDivisorPrefix;
        use self::change_divisor_w::ChangeDivisorWPrefix;
        use self::div_by_zero::DivByZeroPrefix;
        use self::eq::EqPrefix;
        use self::left_is_zero::LeftOperandIsZeroPrefix;
        use self::left_msb::LeftMsbPrefix;
        use self::left_shift::LeftShiftPrefix;
        use self::left_shift_helper::LeftShiftHelperPrefix;
        use self::left_shift_w::LeftShiftWPrefix;
        use self::left_shift_w_helper::LeftShiftWHelperPrefix;
        use self::lower_half_word::LowerHalfWordPrefix;
        use self::lower_word::LowerWordPrefix;
        use self::lsb::LsbPrefix;
        use self::lt::LessThanPrefix;
        use self::negative_divisor_equals_remainder::NegativeDivisorEqualsRemainderPrefix;
        use self::negative_divisor_greater_than_remainder::NegativeDivisorGreaterThanRemainderPrefix;
        use self::negative_divisor_zero_remainder::NegativeDivisorZeroRemainderPrefix;
        use self::or::OrPrefix;
        use self::overflow_bits_zero::OverflowBitsZeroPrefix;
        use self::positive_remainder_equals_divisor::PositiveRemainderEqualsDivisorPrefix;
        use self::positive_remainder_less_than_divisor::PositiveRemainderLessThanDivisorPrefix;
        use self::pow2::Pow2Prefix;
        use self::pow2_w::Pow2WPrefix;
        use self::rev8w::Rev8WPrefix;
        use self::right_is_zero::RightOperandIsZeroPrefix;
        use self::right_msb::RightMsbPrefix;
        use self::right_operand::RightOperandPrefix;
        use self::right_operand_w::RightOperandWPrefix;
        use self::right_shift::RightShiftPrefix;
        use self::right_shift_w::RightShiftWPrefix;
        use self::sign_extension::SignExtensionPrefix;
        use self::sign_extension_right_operand::SignExtensionRightOperandPrefix;
        use self::sign_extension_upper_half::SignExtensionUpperHalfPrefix;
        use self::two_lsb::TwoLsbPrefix;
        use self::upper_word::UpperWordPrefix;
        use self::xor::XorPrefix;
        use self::xor_rot::XorRotPrefix;
        use self::xor_rotw::XorRotWPrefix;

        let eval = match self {
            Prefixes::LowerWord => LowerWordPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::LowerHalfWord => {
                LowerHalfWordPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::UpperWord => {
                UpperWordPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::And => AndPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Andn => AndnPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Or => OrPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Xor => XorPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Eq => EqPrefix::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::LessThan => LessThanPrefix::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::LeftOperandIsZero => {
                LeftOperandIsZeroPrefix::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::RightOperandIsZero => {
                RightOperandIsZeroPrefix::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::LeftOperandMsb => LeftMsbPrefix::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::RightOperandMsb => RightMsbPrefix::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::DivByZero => DivByZeroPrefix::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::PositiveRemainderEqualsDivisor => {
                PositiveRemainderEqualsDivisorPrefix::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::PositiveRemainderLessThanDivisor => {
                PositiveRemainderLessThanDivisorPrefix::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::NegativeDivisorZeroRemainder => {
                NegativeDivisorZeroRemainderPrefix::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::NegativeDivisorEqualsRemainder => {
                NegativeDivisorEqualsRemainderPrefix::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::NegativeDivisorGreaterThanRemainder => {
                NegativeDivisorGreaterThanRemainderPrefix::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::Lsb => LsbPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Pow2 => Pow2Prefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Pow2W => Pow2WPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Rev8W => Rev8WPrefix::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::RightShift => RightShiftPrefix::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::SignExtension => {
                SignExtensionPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::LeftShift => {
                LeftShiftPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::LeftShiftHelper => {
                LeftShiftHelperPrefix::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::TwoLsb => TwoLsbPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::SignExtensionUpperHalf => {
                SignExtensionUpperHalfPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::ChangeDivisor => {
                ChangeDivisorPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::ChangeDivisorW => {
                ChangeDivisorWPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::RightOperand => {
                RightOperandPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::RightOperandW => {
                RightOperandWPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::SignExtensionRightOperand => {
                SignExtensionRightOperandPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::RightShiftW => {
                RightShiftWPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::LeftShiftWHelper => {
                LeftShiftWHelperPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::LeftShiftW => {
                LeftShiftWPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::OverflowBitsZero => {
                OverflowBitsZeroPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::XorRot16 => {
                XorRotPrefix::<XLEN, 16>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::XorRot24 => {
                XorRotPrefix::<XLEN, 24>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::XorRot32 => {
                XorRotPrefix::<XLEN, 32>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::XorRot63 => {
                XorRotPrefix::<XLEN, 63>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::XorRotW7 => {
                XorRotWPrefix::<XLEN, 7>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::XorRotW8 => {
                XorRotWPrefix::<XLEN, 8>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::XorRotW12 => {
                XorRotWPrefix::<XLEN, 12>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::XorRotW16 => {
                XorRotWPrefix::<XLEN, 16>::prefix_mle(checkpoints, r_x, c, b, j)
            }
        };
        PrefixEval(eval)
    }

    /// Update the checkpoint for this prefix variant.
    fn update_prefix_checkpoint<const XLEN: usize, F, C>(
        &self,
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        use self::and::AndPrefix;
        use self::andn::AndnPrefix;
        use self::change_divisor::ChangeDivisorPrefix;
        use self::change_divisor_w::ChangeDivisorWPrefix;
        use self::div_by_zero::DivByZeroPrefix;
        use self::eq::EqPrefix;
        use self::left_is_zero::LeftOperandIsZeroPrefix;
        use self::left_msb::LeftMsbPrefix;
        use self::left_shift::LeftShiftPrefix;
        use self::left_shift_helper::LeftShiftHelperPrefix;
        use self::left_shift_w::LeftShiftWPrefix;
        use self::left_shift_w_helper::LeftShiftWHelperPrefix;
        use self::lower_half_word::LowerHalfWordPrefix;
        use self::lower_word::LowerWordPrefix;
        use self::lsb::LsbPrefix;
        use self::lt::LessThanPrefix;
        use self::negative_divisor_equals_remainder::NegativeDivisorEqualsRemainderPrefix;
        use self::negative_divisor_greater_than_remainder::NegativeDivisorGreaterThanRemainderPrefix;
        use self::negative_divisor_zero_remainder::NegativeDivisorZeroRemainderPrefix;
        use self::or::OrPrefix;
        use self::overflow_bits_zero::OverflowBitsZeroPrefix;
        use self::positive_remainder_equals_divisor::PositiveRemainderEqualsDivisorPrefix;
        use self::positive_remainder_less_than_divisor::PositiveRemainderLessThanDivisorPrefix;
        use self::pow2::Pow2Prefix;
        use self::pow2_w::Pow2WPrefix;
        use self::rev8w::Rev8WPrefix;
        use self::right_is_zero::RightOperandIsZeroPrefix;
        use self::right_msb::RightMsbPrefix;
        use self::right_operand::RightOperandPrefix;
        use self::right_operand_w::RightOperandWPrefix;
        use self::right_shift::RightShiftPrefix;
        use self::right_shift_w::RightShiftWPrefix;
        use self::sign_extension::SignExtensionPrefix;
        use self::sign_extension_right_operand::SignExtensionRightOperandPrefix;
        use self::sign_extension_upper_half::SignExtensionUpperHalfPrefix;
        use self::two_lsb::TwoLsbPrefix;
        use self::upper_word::UpperWordPrefix;
        use self::xor::XorPrefix;
        use self::xor_rot::XorRotPrefix;
        use self::xor_rotw::XorRotWPrefix;

        match self {
            Prefixes::LowerWord => LowerWordPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::LowerHalfWord => LowerHalfWordPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::UpperWord => UpperWordPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::And => AndPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::Andn => AndnPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::Or => OrPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::Xor => XorPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::Eq => {
                EqPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::LessThan => {
                LessThanPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::LeftOperandIsZero => {
                LeftOperandIsZeroPrefix::update_prefix_checkpoint(
                    checkpoints, r_x, r_y, j, suffix_len,
                )
            }
            Prefixes::RightOperandIsZero => {
                RightOperandIsZeroPrefix::update_prefix_checkpoint(
                    checkpoints, r_x, r_y, j, suffix_len,
                )
            }
            Prefixes::LeftOperandMsb => {
                LeftMsbPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::RightOperandMsb => {
                RightMsbPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::DivByZero => {
                DivByZeroPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::PositiveRemainderEqualsDivisor => {
                PositiveRemainderEqualsDivisorPrefix::update_prefix_checkpoint(
                    checkpoints, r_x, r_y, j, suffix_len,
                )
            }
            Prefixes::PositiveRemainderLessThanDivisor => {
                PositiveRemainderLessThanDivisorPrefix::update_prefix_checkpoint(
                    checkpoints, r_x, r_y, j, suffix_len,
                )
            }
            Prefixes::NegativeDivisorZeroRemainder => {
                NegativeDivisorZeroRemainderPrefix::update_prefix_checkpoint(
                    checkpoints, r_x, r_y, j, suffix_len,
                )
            }
            Prefixes::NegativeDivisorEqualsRemainder => {
                NegativeDivisorEqualsRemainderPrefix::update_prefix_checkpoint(
                    checkpoints, r_x, r_y, j, suffix_len,
                )
            }
            Prefixes::NegativeDivisorGreaterThanRemainder => {
                NegativeDivisorGreaterThanRemainderPrefix::update_prefix_checkpoint(
                    checkpoints, r_x, r_y, j, suffix_len,
                )
            }
            Prefixes::Lsb => LsbPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::Pow2 => Pow2Prefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::Pow2W => Pow2WPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::Rev8W => Rev8WPrefix::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::RightShift => {
                RightShiftPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::SignExtension => SignExtensionPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::LeftShift => LeftShiftPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::LeftShiftHelper => {
                LeftShiftHelperPrefix::update_prefix_checkpoint(
                    checkpoints, r_x, r_y, j, suffix_len,
                )
            }
            Prefixes::TwoLsb => TwoLsbPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::SignExtensionUpperHalf => {
                SignExtensionUpperHalfPrefix::<XLEN>::update_prefix_checkpoint(
                    checkpoints, r_x, r_y, j, suffix_len,
                )
            }
            Prefixes::ChangeDivisor => ChangeDivisorPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::ChangeDivisorW => ChangeDivisorWPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::RightOperand => RightOperandPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::RightOperandW => RightOperandWPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::SignExtensionRightOperand => {
                SignExtensionRightOperandPrefix::<XLEN>::update_prefix_checkpoint(
                    checkpoints, r_x, r_y, j, suffix_len,
                )
            }
            Prefixes::RightShiftW => RightShiftWPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::LeftShiftWHelper => {
                LeftShiftWHelperPrefix::<XLEN>::update_prefix_checkpoint(
                    checkpoints, r_x, r_y, j, suffix_len,
                )
            }
            Prefixes::LeftShiftW => LeftShiftWPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::OverflowBitsZero => {
                OverflowBitsZeroPrefix::<XLEN>::update_prefix_checkpoint(
                    checkpoints, r_x, r_y, j, suffix_len,
                )
            }
            Prefixes::XorRot16 => XorRotPrefix::<XLEN, 16>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::XorRot24 => XorRotPrefix::<XLEN, 24>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::XorRot32 => XorRotPrefix::<XLEN, 32>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::XorRot63 => XorRotPrefix::<XLEN, 63>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::XorRotW7 => XorRotWPrefix::<XLEN, 7>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::XorRotW8 => XorRotWPrefix::<XLEN, 8>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::XorRotW12 => XorRotWPrefix::<XLEN, 12>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
            Prefixes::XorRotW16 => XorRotWPrefix::<XLEN, 16>::update_prefix_checkpoint(
                checkpoints, r_x, r_y, j, suffix_len,
            ),
        }
    }

    /// Update all prefix checkpoints after binding two variables.
    pub fn update_checkpoints<const XLEN: usize, F, C>(
        checkpoints: &mut [PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        suffix_len: usize,
    ) where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(checkpoints.len(), NUM_PREFIXES);
        let previous_checkpoints: Vec<_> = checkpoints.to_vec();
        for (index, checkpoint) in checkpoints.iter_mut().enumerate() {
            // SAFETY: repr(u8) enum with NUM_PREFIXES variants starting at 0
            let prefix: Prefixes = unsafe { std::mem::transmute(index as u8) };
            *checkpoint = prefix.update_prefix_checkpoint::<XLEN, F, C>(
                &previous_checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            );
        }
    }
}
