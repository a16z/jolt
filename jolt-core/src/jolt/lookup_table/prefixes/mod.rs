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
use right_shift::RightShiftPrefix;
use sign_extension::SignExtensionPrefix;
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
use rotr::RotrPrefix;
use rotr_helper::RotrHelperPrefix;
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
pub mod right_shift;
pub mod rotr;
pub mod rotr_helper;
pub mod sign_extension;
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
    RightShift,
    SignExtension,
    Rotr,
    RotrHelper,
}

#[derive(Clone, Copy)]
pub struct PrefixEval<F>(F);
#[derive(Clone, Copy)]
pub enum PrefixCheckpoint<F> {
    Default(PrefixEval<F>),
    Rotr {
        prod_one_plus_y: F,
        sum_x_y_prod: F,
        second_sum: F,
    },
    RotrHelper {
        prod_one_minus_y: F,
        sum_contributions: F,
    },
    None,
}

impl<F: Display> Display for PrefixEval<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<F> PrefixCheckpoint<F> {
    /// Returns the field element if this is a default checkpoint.
    /// Panics otherwise
    pub fn unwrap(self) -> F {
        match self {
            PrefixCheckpoint::Default(e) => e.0,
            _ => panic!("invalid prefix checkpoint"),
        }
    }

    /// Returns the field element if this is a default checkpoint.
    /// Returns `default` otherwise
    pub fn unwrap_or(self, default: F) -> F {
        match self {
            PrefixCheckpoint::Default(e) => e.0,
            _ => default,
        }
    }
}

impl<F> From<Option<F>> for PrefixCheckpoint<F> {
    fn from(value: Option<F>) -> Self {
        match value {
            Some(value) => PrefixCheckpoint::Default(PrefixEval(value)),
            None => PrefixCheckpoint::None,
        }
    }
}

impl<F: JoltField> Into<PrefixEval<F>> for PrefixCheckpoint<F> {
    fn into(self) -> PrefixEval<F> {
        match self {
            PrefixCheckpoint::Default(e) => e,
            PrefixCheckpoint::Rotr {
                sum_x_y_prod,
                second_sum,
                ..
            } => PrefixEval(sum_x_y_prod + second_sum),
            PrefixCheckpoint::RotrHelper {
                sum_contributions, ..
            } => PrefixEval(sum_contributions),
            PrefixCheckpoint::None => panic!("invalid prefix checkpoint"),
        }
    }
}

impl<F> Index<Prefixes> for &[PrefixCheckpoint<F>] {
    type Output = PrefixCheckpoint<F>;

    fn index(&self, prefix: Prefixes) -> &Self::Output {
        let index = prefix as usize;
        self.get(index).unwrap()
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
            Prefixes::RightShift => RightShiftPrefix::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::SignExtension => {
                SignExtensionPrefix::<WORD_SIZE>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::Rotr => RotrPrefix::<WORD_SIZE>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::RotrHelper => {
                RotrHelperPrefix::<WORD_SIZE>::prefix_mle(checkpoints, r_x, c, b, j)
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
            Prefixes::RightShift => {
                RightShiftPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::SignExtension => {
                SignExtensionPrefix::<WORD_SIZE>::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::Rotr => {
                RotrPrefix::<WORD_SIZE>::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
            Prefixes::RotrHelper => {
                RotrHelperPrefix::<WORD_SIZE>::update_prefix_checkpoint(checkpoints, r_x, r_y, j)
            }
        }
    }
}
