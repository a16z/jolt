use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};
use lsb::LsbPrefix;
use negative_divisor_equals_remainder::NegativeDivisorEqualsRemainderPrefix;
use negative_divisor_greater_than_remainder::NegativeDivisorGreaterThanRemainderPrefix;
use negative_divisor_zero_remainder::NegativeDivisorZeroRemainderPrefix;
use num_derive::FromPrimitive;
use positive_remainder_equals_divisor::PositiveRemainderEqualsDivisorPrefix;
use positive_remainder_less_than_divisor::PositiveRemainderLessThanDivisorPrefix;
use pow2::Pow2Prefix;
use rayon::prelude::*;
use right_shift_padding::RightShiftPaddingPrefix;
use std::{fmt::Display, ops::Index};
use strum::EnumCount;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use and::AndPrefix;
use div_by_zero::DivByZeroPrefix;
use eq::EqPrefix;
use left_is_zero::LeftOperandIsZeroPrefix;
use left_msb::LeftMsbPrefix;
use lower_word::LowerWordPrefix;
use lt::LessThanPrefix;
use num::FromPrimitive;
use or::OrPrefix;
use right_is_zero::RightOperandIsZeroPrefix;
use right_msb::RightMsbPrefix;
use upper_word::UpperWordPrefix;
use xor::XorPrefix;

pub mod and;
pub mod div_by_zero;
pub mod eq;
pub mod left_is_zero;
pub mod left_msb;
pub mod lower_word;
pub mod lsb;
pub mod lt;
pub mod negative_divisor_equals_remainder;
pub mod negative_divisor_greater_than_remainder;
pub mod negative_divisor_zero_remainder;
pub mod or;
pub mod positive_remainder_equals_divisor;
pub mod positive_remainder_less_than_divisor;
pub mod pow2;
pub mod right_is_zero;
pub mod right_msb;
pub mod right_shift_padding;
pub mod upper_word;
pub mod xor;

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
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F;

    /// Every two rounds of sumcheck, we update the "checkpoint" value for each
    /// prefix, incorporating the two random challenges `r_x` and `r_y` received
    /// since the last update.
    /// `j` is the sumcheck round index.
    /// A checkpoint update may depend on the values of the other prefix checkpoints,
    /// so we pass in all such `checkpoints` to this function.
    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F>;
}

/// An enum containing all prefixes used by Jolt's instruction lookup tables.
#[repr(u8)]
#[derive(EnumCountMacro, EnumIter, FromPrimitive)]
pub enum Prefixes {
    LowerWord,
    UpperWord,
    Eq,
    And,
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
    RightShiftPadding,
}

#[derive(Clone, Copy)]
pub struct PrefixEval<F>(F);
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
    pub fn prefix_mle<const WORD_SIZE: usize, F: JoltField>(
        &self,
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> PrefixEval<F> {
        let eval = match self {
            Prefixes::LowerWord => {
                LowerWordPrefix::<WORD_SIZE>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::UpperWord => {
                UpperWordPrefix::<WORD_SIZE>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::And => AndPrefix::<WORD_SIZE>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Or => OrPrefix::<WORD_SIZE>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Xor => XorPrefix::<WORD_SIZE>::prefix_mle(checkpoints, r_x, c, b, j),
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
            Prefixes::Lsb => LsbPrefix::<WORD_SIZE>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Pow2 => Pow2Prefix::<WORD_SIZE>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::RightShiftPadding => {
                RightShiftPaddingPrefix::<WORD_SIZE>::prefix_mle(checkpoints, r_x, c, b, j)
            }
        };
        PrefixEval(eval)
    }

    /// Every two rounds of sumcheck, we update the "checkpoint" value for each
    /// prefix, incorporating the two random challenges `r_x` and `r_y` received
    /// since the last update.
    /// This function updates all the prefix checkpoints.
    #[tracing::instrument(skip_all)]
    pub fn update_checkpoints<const WORD_SIZE: usize, F: JoltField>(
        checkpoints: &mut [PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) {
        debug_assert_eq!(checkpoints.len(), Self::COUNT);
        let previous_checkpoints = checkpoints.to_vec();
        checkpoints
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, new_checkpoint)| {
                let prefix: Self = FromPrimitive::from_u8(index as u8).unwrap();
                *new_checkpoint = prefix.update_prefix_checkpoint::<WORD_SIZE, F>(
                    &previous_checkpoints,
                    r_x,
                    r_y,
                    j,
                );
            });
    }

    /// Every two rounds of sumcheck, we update the "checkpoint" value for each
    /// prefix, incorporating the two random challenges `r_x` and `r_y` received
    /// since the last update.
    /// `j` is the sumcheck round index.
    /// A checkpoint update may depend on the values of the other prefix checkpoints,
    /// so we pass in all such `checkpoints` to this function.
    fn update_prefix_checkpoint<const WORD_SIZE: usize, F: JoltField>(
        &self,
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        match self {
            Prefixes::LowerWord => {
                LowerWordPrefix::<WORD_SIZE>::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::UpperWord => {
                UpperWordPrefix::<WORD_SIZE>::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::And => {
                AndPrefix::<WORD_SIZE>::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::Or => {
                OrPrefix::<WORD_SIZE>::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::Xor => {
                XorPrefix::<WORD_SIZE>::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::Eq => EqPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j),
            Prefixes::LessThan => {
                LessThanPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::LeftOperandIsZero => {
                LeftOperandIsZeroPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::RightOperandIsZero => {
                RightOperandIsZeroPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::LeftOperandMsb => {
                LeftMsbPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::RightOperandMsb => {
                RightMsbPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::DivByZero => {
                DivByZeroPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::PositiveRemainderEqualsDivisor => {
                PositiveRemainderEqualsDivisorPrefix::update_prefix_checkpoint(
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                )
            }
            Prefixes::PositiveRemainderLessThanDivisor => {
                PositiveRemainderLessThanDivisorPrefix::update_prefix_checkpoint(
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                )
            }
            Prefixes::NegativeDivisorZeroRemainder => {
                NegativeDivisorZeroRemainderPrefix::update_prefix_checkpoint(
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                )
            }
            Prefixes::NegativeDivisorEqualsRemainder => {
                NegativeDivisorEqualsRemainderPrefix::update_prefix_checkpoint(
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                )
            }
            Prefixes::NegativeDivisorGreaterThanRemainder => {
                NegativeDivisorGreaterThanRemainderPrefix::update_prefix_checkpoint(
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                )
            }
            Prefixes::Lsb => {
                LsbPrefix::<WORD_SIZE>::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::Pow2 => {
                Pow2Prefix::<WORD_SIZE>::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::RightShiftPadding => {
                RightShiftPaddingPrefix::<WORD_SIZE>::update_prefix_checkpoint(
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                )
            }
        }
    }
}
