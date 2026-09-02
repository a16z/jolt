use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};
use allocative::Allocative;
use lsb::LsbPrefix;
use num_derive::FromPrimitive;
use offset_scale::OffsetScalePrefix;
use pow2::Pow2Prefix;
use pow2_offset::Pow2OffsetPrefix;
use pow2_w::Pow2WPrefix;
use rayon::prelude::*;
use rev8w::Rev8WPrefix;
use right_shift::RightShiftPrefix;
use right_shift_w::RightShiftWPrefix;
use shift_data::ShiftDataPrefix;
use sign_extension::SignExtensionPrefix;
use sign_extension_right_operand::SignExtensionRightOperandPrefix;
use sign_extension_upper_half::SignExtensionUpperHalfPrefix;
use sign_extension_w::SignExtensionWPrefix;
use srlw_sext::SrlwSextPrefix;
use std::{fmt::Display, ops::Index};
use strum::EnumCount;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use align_addr::AlignAddrPrefix;
use and::AndPrefix;
use andn::AndnPrefix;
use div_by_zero::DivByZeroPrefix;
use eq::EqPrefix;
use left_is_zero::LeftOperandIsZeroPrefix;
use left_msb::LeftMsbPrefix;
use left_msb_right_operand::LeftMsbRightOperandPrefix;
use left_msb_right_operand_is_zero::LeftMsbRightOperandIsZeroPrefix;
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
use window_sign::WindowSignPrefix;
use window_sign_pow2::WindowSignPow2Prefix;
use word_msb::WordMsbPrefix;
use xor::XorPrefix;
use xor_rot::XorRotPrefix;
use xor_rotw::XorRotWPrefix;

pub mod align_addr;
pub mod and;
pub mod andn;
pub mod div_by_zero;
pub mod eq;
pub mod left_is_zero;
pub mod left_msb;
pub mod left_msb_right_operand;
pub mod left_msb_right_operand_is_zero;
pub mod left_shift;
pub mod left_shift_helper;
pub mod left_shift_w;
pub mod left_shift_w_helper;
pub mod lower_half_word;
pub mod lower_word;
pub mod lsb;
pub mod lt;
pub mod offset_scale;
pub mod or;
pub mod overflow_bits_zero;
pub mod pow2;
pub mod pow2_offset;
pub mod pow2_w;
pub mod rev8w;
pub mod right_is_zero;
pub mod right_msb;
pub mod right_operand;
pub mod right_operand_w;
pub mod right_shift;
pub mod right_shift_w;
pub mod shift_data;
pub mod sign_extension;
pub mod sign_extension_right_operand;
pub mod sign_extension_upper_half;
pub mod sign_extension_w;
pub mod srlw_sext;
pub mod two_lsb;
pub mod upper_word;
pub mod window_sign;
pub mod window_sign_pow2;
pub mod word_msb;
pub mod xor;
pub mod xor_rot;
pub mod xor_rotw;

pub trait SparseDensePrefix<F: JoltField>: 'static + Sync {
    /// Evaluates the MLE for this prefix:
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
    RightOperand,
    LeftMsbRightOperand,
    LeftMsbRightOperandIsZero,
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
    Pow2OffsetW,
    WindowSign,
    WindowSignPow2,
    XorRotW22,
    XorRotW19,
    XorRotW6,
    WordMsb,
    SignExtensionW,
    SrlwSext,
    Pow2OffsetB,
    Pow2OffsetH,
    ShiftDataB,
    ShiftDataH,
    ShiftDataW,
    OffsetScaleB,
    OffsetScaleH,
    OffsetScaleW,
    AlignAddr,
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
    /// Evaluates the MLE for this prefix:
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
            Prefixes::XorRotW22 => XorRotWPrefix::<XLEN, 22>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::XorRotW19 => XorRotWPrefix::<XLEN, 19>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::XorRotW6 => XorRotWPrefix::<XLEN, 6>::prefix_mle(checkpoints, r_x, c, b, j),
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
            Prefixes::RightOperand => {
                RightOperandPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::LeftMsbRightOperand => {
                LeftMsbRightOperandPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::LeftMsbRightOperandIsZero => {
                LeftMsbRightOperandIsZeroPrefix::prefix_mle(checkpoints, r_x, c, b, j)
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
            Prefixes::Pow2OffsetW => {
                Pow2OffsetPrefix::<XLEN, 2>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::WindowSign => WindowSignPrefix::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::WindowSignPow2 => WindowSignPow2Prefix::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::WordMsb => WordMsbPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::SignExtensionW => {
                SignExtensionWPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::SrlwSext => SrlwSextPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
            Prefixes::Pow2OffsetB => {
                Pow2OffsetPrefix::<XLEN, 0>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::Pow2OffsetH => {
                Pow2OffsetPrefix::<XLEN, 1>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::ShiftDataB => {
                ShiftDataPrefix::<XLEN, 1>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::ShiftDataH => {
                ShiftDataPrefix::<XLEN, 2>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::ShiftDataW => {
                ShiftDataPrefix::<XLEN, 4>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::OffsetScaleB => {
                OffsetScalePrefix::<XLEN, 1>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::OffsetScaleH => {
                OffsetScalePrefix::<XLEN, 2>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::OffsetScaleW => {
                OffsetScalePrefix::<XLEN, 4>::prefix_mle(checkpoints, r_x, c, b, j)
            }
            Prefixes::AlignAddr => AlignAddrPrefix::<XLEN>::prefix_mle(checkpoints, r_x, c, b, j),
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
            Prefixes::XorRotW22 => XorRotWPrefix::<XLEN, 22>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::XorRotW19 => XorRotWPrefix::<XLEN, 19>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::XorRotW6 => XorRotWPrefix::<XLEN, 6>::update_prefix_checkpoint(
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
            Prefixes::RightOperand => RightOperandPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::LeftMsbRightOperand => {
                LeftMsbRightOperandPrefix::<XLEN>::update_prefix_checkpoint(
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                    suffix_len,
                )
            }
            Prefixes::LeftMsbRightOperandIsZero => {
                LeftMsbRightOperandIsZeroPrefix::update_prefix_checkpoint(
                    checkpoints,
                    r_x,
                    r_y,
                    j,
                    suffix_len,
                )
            }
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
            Prefixes::Pow2OffsetW => Pow2OffsetPrefix::<XLEN, 2>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::WindowSign => {
                WindowSignPrefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::WindowSignPow2 => {
                WindowSignPow2Prefix::update_prefix_checkpoint(checkpoints, r_x, r_y, j, suffix_len)
            }
            Prefixes::WordMsb => WordMsbPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::SignExtensionW => SignExtensionWPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::SrlwSext => SrlwSextPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::Pow2OffsetB => Pow2OffsetPrefix::<XLEN, 0>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::Pow2OffsetH => Pow2OffsetPrefix::<XLEN, 1>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::ShiftDataB => ShiftDataPrefix::<XLEN, 1>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::ShiftDataH => ShiftDataPrefix::<XLEN, 2>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::ShiftDataW => ShiftDataPrefix::<XLEN, 4>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::OffsetScaleB => OffsetScalePrefix::<XLEN, 1>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::OffsetScaleH => OffsetScalePrefix::<XLEN, 2>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::OffsetScaleW => OffsetScalePrefix::<XLEN, 4>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
            Prefixes::AlignAddr => AlignAddrPrefix::<XLEN>::update_prefix_checkpoint(
                checkpoints,
                r_x,
                r_y,
                j,
                suffix_len,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::{One, Zero};
    use common::constants::XLEN;
    use rand::prelude::*;

    const LOG_K: usize = 2 * XLEN;

    /// `1 + (2^exp - 1)*r`: the bound per-bit factor of the Pow2Offset and
    /// OffsetScale families.
    fn bit_factor(exp: usize, r: Fr) -> Fr {
        Fr::one() + (Fr::from_u128(1u128 << exp) - Fr::one()) * r
    }

    /// Pins the post-final-round checkpoint state that the production prover
    /// (`read_raf_checking`) actually consumes: binds all LOG_K address
    /// rounds through `Prefixes::update_checkpoints` with the production
    /// suffix_len schedule, then checks every checkpoint of the
    /// byte-addressable prefix families against its closed form at the
    /// challenge point.
    ///
    /// WARNING: the per-table prefix_suffix tests do not cover this state on
    /// their own; a checkpoint updater can diverge from the phased evaluation
    /// path while those tests stay green (this exact gap produced a
    /// production bug in the parallel #1755 implementation).
    fn final_checkpoints_match_closed_forms(log_m: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let r: Vec<Fr> = (0..LOG_K).map(|_| Fr::from_u64(rng.next_u64())).collect();

        let mut checkpoints: Vec<PrefixCheckpoint<Fr>> = vec![None.into(); Prefixes::COUNT];
        for j in (1..LOG_K).step_by(2) {
            // WARNING: keep in sync with the production schedule at
            // read_raf_checking's update site; if that formula changes, this
            // test silently keeps validating the old schedule.
            let suffix_len = LOG_K - (j / log_m + 1) * log_m;
            Prefixes::update_checkpoints::<XLEN, Fr, Fr>(
                &mut checkpoints,
                r[j - 1],
                r[j],
                j,
                suffix_len,
            );
        }
        let value = |p: Prefixes| checkpoints[p as usize].unwrap().0;

        // Raw index bit k binds at round LOG_K - 1 - k; interleaved x_k sits
        // at index bit 2k+1 and y_i at index bit 2i.
        let raw = |k: usize| r[LOG_K - 1 - k];
        let rx = |k: usize| r[LOG_K - 1 - (2 * k + 1)];
        let ry = |i: usize| r[LOG_K - 1 - 2 * i];

        // Pow2Offset: product of per-bit factors over the raw offset bits.
        let pow2_offset =
            |low_bit: usize| -> Fr { (low_bit..3).map(|k| bit_factor(8 << k, raw(k))).product() };
        assert_eq!(value(Prefixes::Pow2OffsetB), pow2_offset(0));
        assert_eq!(value(Prefixes::Pow2OffsetH), pow2_offset(1));
        assert_eq!(value(Prefixes::Pow2OffsetW), pow2_offset(2));

        // AlignAddr: sum of 2^k over raw bits 3..XLEN (carry excluded).
        let align_addr: Fr = (3..XLEN).map(|k| Fr::from_u128(1u128 << k) * raw(k)).sum();
        assert_eq!(value(Prefixes::AlignAddr), align_addr);

        // OffsetScale: product over the interleaved offset y bits; ShiftData
        // is the joint L(x)*P(y) accumulator.
        let offset_scale = |eighths: usize| -> Fr {
            (0..3)
                .filter(|i| (8 - eighths) >> i & 1 == 1)
                .map(|i| bit_factor(8 << i, ry(i)))
                .product()
        };
        let lane = |lane_bits: usize| -> Fr {
            (0..lane_bits)
                .map(|k| Fr::from_u128(1u128 << k) * rx(k))
                .sum()
        };
        assert_eq!(value(Prefixes::OffsetScaleB), offset_scale(1));
        assert_eq!(value(Prefixes::OffsetScaleH), offset_scale(2));
        assert_eq!(value(Prefixes::OffsetScaleW), offset_scale(4));
        assert_eq!(value(Prefixes::ShiftDataB), lane(8) * offset_scale(1));
        assert_eq!(value(Prefixes::ShiftDataH), lane(16) * offset_scale(2));
        assert_eq!(value(Prefixes::ShiftDataW), lane(32) * offset_scale(4));

        // RightShift (generalized pext), WindowSign, and WindowSignPow2, per
        // the PextSigned MLE recurrences (MSB-first).
        let mut pext = Fr::zero();
        let mut sigma = Fr::zero();
        let mut sig2pc = Fr::zero();
        let mut none = Fr::one();
        for k in (0..XLEN).rev() {
            let xy = rx(k) * ry(k);
            pext = pext * (Fr::one() + ry(k)) + xy;
            sig2pc = sig2pc * (Fr::one() + ry(k)) + none * (xy + xy);
            sigma += none * xy;
            none *= Fr::one() - ry(k);
        }
        assert_eq!(value(Prefixes::RightShift), pext);
        assert_eq!(value(Prefixes::WindowSign), sigma);
        assert_eq!(value(Prefixes::WindowSignPow2), sig2pc);
    }

    // Seeds are mnemonics for the byte-addressable stack PRs (#1761, #1768).
    #[test]
    fn final_checkpoints_log_m_16() {
        final_checkpoints_match_closed_forms(16, 0x1761);
    }

    #[test]
    fn final_checkpoints_log_m_8() {
        final_checkpoints_match_closed_forms(8, 0x1768);
    }
}
