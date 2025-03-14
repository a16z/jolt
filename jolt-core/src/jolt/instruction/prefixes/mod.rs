use crate::{field::JoltField, subprotocols::sparse_dense_shout::LookupBits};
use negative_divisor_equals_remainder::NegativeDivisorEqualsRemainderPrefix;
use negative_divisor_greater_than_remainder::NegativeDivisorGreaterThanRemainderPrefix;
use negative_divisor_zero_remainder::NegativeDivisorZeroRemainderPrefix;
use num_derive::FromPrimitive;
use positive_remainder_equals_divisor::PositiveRemainderEqualsDivisorPrefix;
use positive_remainder_less_than_divisor::PositiveRemainderLessThanDivisorPrefix;
use rayon::prelude::*;
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
pub mod lt;
pub mod negative_divisor_equals_remainder;
pub mod negative_divisor_greater_than_remainder;
pub mod negative_divisor_zero_remainder;
pub mod or;
pub mod positive_remainder_equals_divisor;
pub mod positive_remainder_less_than_divisor;
pub mod right_is_zero;
pub mod right_msb;
pub mod upper_word;
pub mod xor;

pub trait SparseDensePrefix<F: JoltField>: 'static + Sync {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F;

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F>;
}

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
        };
        PrefixEval(eval)
    }

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
        }
    }
}
