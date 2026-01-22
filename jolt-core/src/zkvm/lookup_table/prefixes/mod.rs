use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};
use allocative::Allocative;
use lsb::LsbPrefix;
use negative_divisor_equals_remainder::NegativeDivisorEqualsRemainderPrefix;
use negative_divisor_greater_than_remainder::NegativeDivisorGreaterThanRemainderPrefix;
use negative_divisor_zero_remainder::NegativeDivisorZeroRemainderPrefix;
use num_derive::FromPrimitive;
use positive_remainder_equals_divisor::PositiveRemainderEqualsDivisorPrefix;
use positive_remainder_less_than_divisor::PositiveRemainderLessThanDivisorPrefix;
use pow2::Pow2Prefix;
use pow2_w::Pow2WPrefix;
use rayon::prelude::*;
use rev8w::Rev8WPrefix;
use right_shift::RightShiftPrefix;
use right_shift_w::RightShiftWPrefix;
use sign_extension::SignExtensionPrefix;
use sign_extension_right_operand::SignExtensionRightOperandPrefix;
use sign_extension_upper_half::SignExtensionUpperHalfPrefix;
use std::{fmt::Display, ops::Index};
use strum::EnumCount;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use and::AndPrefix;
use andn::AndnPrefix;
use change_divisor::ChangeDivisorPrefix;
use change_divisor_w::ChangeDivisorWPrefix;
use div_by_zero::DivByZeroPrefix;
use eq::EqPrefix;
use left_is_zero::LeftOperandIsZeroPrefix;
use left_msb::LeftMsbPrefix;
use left_shift::LeftShiftPrefix;
use left_shift_helper::LeftShiftHelperPrefix;
use left_shift_w::LeftShiftWPrefix;
use left_shift_w_helper::LeftShiftWHelperPrefix;
use lower_half_word::LowerHalfWordPrefix;
use lower_word::LowerWordPrefix;
use lt::LessThanPrefix;
use num::FromPrimitive;
use or::OrPrefix;
use overflow_bits_zero::OverflowBitsZeroPrefix;
use right_is_zero::RightOperandIsZeroPrefix;
use right_msb::RightMsbPrefix;
use right_operand::RightOperandPrefix;
use right_operand_w::RightOperandWPrefix;
use two_lsb::TwoLsbPrefix;
use upper_word::UpperWordPrefix;
use xor::XorPrefix;
use xor_rot::XorRotPrefix;
use xor_rotw::XorRotWPrefix;

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

pub trait SparseDensePrefix<F: JoltField>: 'static + Sync {
    /// Evalautes the MLE for this prefix:
    /// - prefix(r, r_x, c, b)   if j is odd
    /// - prefix(r, c, b)        if j is even
    ///
    /// where the prefix checkpoint captures the "contribution" of
    /// `r` to this evaluation.
    ///
    /// `r` (and potentially `r_x`) capture the variables of the prefix
    /// that have been bound in the previous rounds of sumcheck.
    /// To compute the current round's prover message, we're fixing the
    /// current variable to `c`.
    /// The remaining variables of the prefix are captured by `b`. We sum
    /// over these variables as they range over the Boolean hypercube, so
    /// they can be represented by a single bitvector.
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>;
    /// Every two rounds of sumcheck, we update the "checkpoint" value for each
    /// prefix, incorporating the two random challenges `r_x` and `r_y` received
    /// since the last update.
    /// `j` is the sumcheck round index.
    /// A checkpoint update may depend on the values of the other prefix checkpoints,
    /// so we pass in all such `checkpoints` to this function.
    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>;
}

/// An enum containing all prefixes used by Jolt's instruction lookup tables.
#[repr(u8)]
#[derive(EnumCountMacro, EnumIter, FromPrimitive)]
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

#[derive(Clone, Copy, Allocative)]
/// Wrapper for prefix polynomial evaluations, used for type safety in prefix operations.
pub struct PrefixEval<F>(F);
/// Optional prefix evaluation cached after each pair of address-binding rounds (r_x, r_y).
pub type PrefixCheckpoint<F: JoltField> = PrefixEval<Option<F>>;

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
    pub fn unwrap(self) -> PrefixEval<F> {
        self.0.unwrap().into()
    }
}

impl<F> Index<Prefixes> for &[PrefixEval<F>] {
    type Output = F;

    fn index(&self, prefix: Prefixes) -> &Self::Output {
        let index = prefix as usize;
        &self.get(index).unwrap().0
    }
}

impl Prefixes {
    /// Evalautes the MLE for this prefix:
    /// - prefix(r, r_x, c, b)   if j is odd
    /// - prefix(r, c, b)        if j is even
    ///
    /// where the prefix checkpoint captures the "contribution" of
    /// `r` to this evaluation.
    ///
    /// `r` (and potentially `r_x`) capture the variables of the prefix
    /// that have been bound in the previous rounds of sumcheck.
    /// To compute the current round's prover message, we're fixing the
    /// current variable to `c`.
    /// The remaining variables of the prefix are captured by `b`. We sum
    /// over these variables as they range over the Boolean hypercube, so
    /// they can be represented by a single bitvector.
    pub fn prefix_mle<const XLEN: usize, F, C>(
        &self,
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> PrefixEval<F>
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        let eval = match self {
            Prefixes::LowerWord => LowerWordPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::LowerHalfWord => {
                LowerHalfWordPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::UpperWord => UpperWordPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::And => AndPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Andn => AndnPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Or => OrPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Xor => XorPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::XorRot16 => XorRotPrefix::<XLEN, 16>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::XorRot24 => XorRotPrefix::<XLEN, 24>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::XorRot32 => XorRotPrefix::<XLEN, 32>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::XorRot63 => XorRotPrefix::<XLEN, 63>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::XorRotW7 => XorRotWPrefix::<XLEN, 7>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::XorRotW8 => XorRotWPrefix::<XLEN, 8>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::XorRotW12 => XorRotWPrefix::<XLEN, 12>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::XorRotW16 => XorRotWPrefix::<XLEN, 16>::prefix_mle(checkpoints, r_x, c, b, j),
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
            Prefixes::LeftShift => LeftShiftPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
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
            Prefixes::RightOperand => {
                RightOperandPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::ChangeDivisorW => {
                ChangeDivisorWPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
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
            Prefixes::LeftShiftW => LeftShiftWPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::OverflowBitsZero => {
                OverflowBitsZeroPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
        };
        PrefixEval(eval)
    }

    /// Every two rounds of sumcheck, we update the "checkpoint" value for each
    /// prefix, incorporating the two random challenges `r_x` and `r_y` received
    /// since the last update.
    /// This function updates all the prefix checkpoints.
    #[tracing::instrument(skip_all)]
    pub fn update_checkpoints<const XLEN: usize, F, C>(
        checkpoints: &mut [PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        suffix_len: usize,
    ) where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(checkpoints.len(), Self::COUNT);
        let previous_checkpoints = checkpoints.to_vec();
        checkpoints
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, new_checkpoint)| {
                let prefix: Self = FromPrimitive::from_u8(index as u8).unwrap();
                *new_checkpoint = prefix.update_prefix_checkpoint::<XLEN, F, C>(
                    &previous_checkpoints,
                    r_x,
                    r_y,
                    j,
                    suffix_len,
                );
            });
    }

    /// Every two rounds of sumcheck, we update the "checkpoint" value for each
    /// prefix, incorporating the two random challenges `r_x` and `r_y` received
    /// since the last update.
    /// `j` is the sumcheck round index.
    /// A checkpoint update may depend on the values of the other prefix checkpoints,
    /// so we pass in all such `checkpoints` to this function.
    fn update_prefix_checkpoint<const XLEN: usize, F, C>(
        &self,
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        match self {
            Prefixes::LowerWord => LowerWordPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::LowerHalfWord => LowerHalfWordPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::UpperWord => UpperWordPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::And => {
                AndPrefix::<XLEN>::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::Andn => {
                AndnPrefix::<XLEN>::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::Or => {
                OrPrefix::<XLEN>::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::Xor => {
                XorPrefix::<XLEN>::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::XorRot16 => XorRotPrefix::<XLEN, 16>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::XorRot24 => XorRotPrefix::<XLEN, 24>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::XorRot32 => XorRotPrefix::<XLEN, 32>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::XorRot63 => XorRotPrefix::<XLEN, 63>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::XorRotW7 => XorRotWPrefix::<XLEN, 7>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::XorRotW8 => XorRotWPrefix::<XLEN, 8>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::XorRotW12 => XorRotWPrefix::<XLEN, 12>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::XorRotW16 => XorRotWPrefix::<XLEN, 16>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::Eq => {
                EqPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::LessThan => {
                LessThanPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::LeftOperandIsZero => LeftOperandIsZeroPrefix::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::RightOperandIsZero => RightOperandIsZeroPrefix::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
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
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                    suffix_len,
                )
            }
            Prefixes::PositiveRemainderLessThanDivisor => {
                PositiveRemainderLessThanDivisorPrefix::update_prefix_checkpoint(
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                    suffix_len,
                )
            }
            Prefixes::NegativeDivisorZeroRemainder => {
                NegativeDivisorZeroRemainderPrefix::update_prefix_checkpoint(
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                    suffix_len,
                )
            }
            Prefixes::NegativeDivisorEqualsRemainder => {
                NegativeDivisorEqualsRemainderPrefix::update_prefix_checkpoint(
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                    suffix_len,
                )
            }
            Prefixes::NegativeDivisorGreaterThanRemainder => {
                NegativeDivisorGreaterThanRemainderPrefix::update_prefix_checkpoint(
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                    suffix_len,
                )
            }
            Prefixes::Lsb => {
                LsbPrefix::<XLEN>::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::Pow2 => {
                Pow2Prefix::<XLEN>::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::Pow2W => {
                Pow2WPrefix::<XLEN>::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::Rev8W => {
                Rev8WPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::RightShift => {
                RightShiftPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::SignExtension => SignExtensionPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::LeftShift => LeftShiftPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::LeftShiftHelper => LeftShiftHelperPrefix::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::TwoLsb => {
                TwoLsbPrefix::<XLEN>::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::SignExtensionUpperHalf => {
                SignExtensionUpperHalfPrefix::<XLEN>::update_prefix_checkpoint(
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                    suffix_len,
                )
            }
            Prefixes::ChangeDivisor => ChangeDivisorPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::RightOperand => RightOperandPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::ChangeDivisorW => ChangeDivisorWPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::RightOperandW => RightOperandWPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::SignExtensionRightOperand => {
                SignExtensionRightOperandPrefix::<XLEN>::update_prefix_checkpoint(
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                    suffix_len,
                )
            }
            Prefixes::RightShiftW => RightShiftWPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::LeftShiftWHelper => LeftShiftWHelperPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::LeftShiftW => LeftShiftWPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::OverflowBitsZero => OverflowBitsZeroPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
        }
    }
}
