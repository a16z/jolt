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
pub const NUM_PREFIXES: usize = 46;

const _: () = assert!(Prefixes::XorRotW16 as usize + 1 == NUM_PREFIXES);

/// All prefix variants in discriminant order.
pub const ALL_PREFIXES: [Prefixes; NUM_PREFIXES] = [
    Prefixes::LowerWord,
    Prefixes::LowerHalfWord,
    Prefixes::UpperWord,
    Prefixes::Eq,
    Prefixes::And,
    Prefixes::Andn,
    Prefixes::Or,
    Prefixes::Xor,
    Prefixes::LessThan,
    Prefixes::LeftOperandIsZero,
    Prefixes::RightOperandIsZero,
    Prefixes::LeftOperandMsb,
    Prefixes::RightOperandMsb,
    Prefixes::DivByZero,
    Prefixes::PositiveRemainderEqualsDivisor,
    Prefixes::PositiveRemainderLessThanDivisor,
    Prefixes::NegativeDivisorZeroRemainder,
    Prefixes::NegativeDivisorEqualsRemainder,
    Prefixes::NegativeDivisorGreaterThanRemainder,
    Prefixes::Lsb,
    Prefixes::Pow2,
    Prefixes::Pow2W,
    Prefixes::Rev8W,
    Prefixes::RightShift,
    Prefixes::SignExtension,
    Prefixes::LeftShift,
    Prefixes::LeftShiftHelper,
    Prefixes::TwoLsb,
    Prefixes::SignExtensionUpperHalf,
    Prefixes::ChangeDivisor,
    Prefixes::ChangeDivisorW,
    Prefixes::RightOperand,
    Prefixes::RightOperandW,
    Prefixes::SignExtensionRightOperand,
    Prefixes::RightShiftW,
    Prefixes::LeftShiftWHelper,
    Prefixes::LeftShiftW,
    Prefixes::OverflowBitsZero,
    Prefixes::XorRot16,
    Prefixes::XorRot24,
    Prefixes::XorRot32,
    Prefixes::XorRot63,
    Prefixes::XorRotW7,
    Prefixes::XorRotW8,
    Prefixes::XorRotW12,
    Prefixes::XorRotW16,
];

/// Dispatches a `SparseDensePrefix` method call to the concrete type for each `Prefixes` variant.
macro_rules! dispatch_prefix {
    ($self:expr, $method:ident, $($args:expr),* $(,)?) => {
        match $self {
            Prefixes::LowerWord => lower_word::LowerWordPrefix::<XLEN>::$method($($args),*),
            Prefixes::LowerHalfWord => lower_half_word::LowerHalfWordPrefix::<XLEN>::$method($($args),*),
            Prefixes::UpperWord => upper_word::UpperWordPrefix::<XLEN>::$method($($args),*),
            Prefixes::Eq => eq::EqPrefix::$method($($args),*),
            Prefixes::And => and::AndPrefix::<XLEN>::$method($($args),*),
            Prefixes::Andn => andn::AndnPrefix::<XLEN>::$method($($args),*),
            Prefixes::Or => or::OrPrefix::<XLEN>::$method($($args),*),
            Prefixes::Xor => xor::XorPrefix::<XLEN>::$method($($args),*),
            Prefixes::LessThan => lt::LessThanPrefix::$method($($args),*),
            Prefixes::LeftOperandIsZero => left_is_zero::LeftOperandIsZeroPrefix::$method($($args),*),
            Prefixes::RightOperandIsZero => right_is_zero::RightOperandIsZeroPrefix::$method($($args),*),
            Prefixes::LeftOperandMsb => left_msb::LeftMsbPrefix::$method($($args),*),
            Prefixes::RightOperandMsb => right_msb::RightMsbPrefix::$method($($args),*),
            Prefixes::DivByZero => div_by_zero::DivByZeroPrefix::$method($($args),*),
            Prefixes::PositiveRemainderEqualsDivisor => positive_remainder_equals_divisor::PositiveRemainderEqualsDivisorPrefix::$method($($args),*),
            Prefixes::PositiveRemainderLessThanDivisor => positive_remainder_less_than_divisor::PositiveRemainderLessThanDivisorPrefix::$method($($args),*),
            Prefixes::NegativeDivisorZeroRemainder => negative_divisor_zero_remainder::NegativeDivisorZeroRemainderPrefix::$method($($args),*),
            Prefixes::NegativeDivisorEqualsRemainder => negative_divisor_equals_remainder::NegativeDivisorEqualsRemainderPrefix::$method($($args),*),
            Prefixes::NegativeDivisorGreaterThanRemainder => negative_divisor_greater_than_remainder::NegativeDivisorGreaterThanRemainderPrefix::$method($($args),*),
            Prefixes::Lsb => lsb::LsbPrefix::<XLEN>::$method($($args),*),
            Prefixes::Pow2 => pow2::Pow2Prefix::<XLEN>::$method($($args),*),
            Prefixes::Pow2W => pow2_w::Pow2WPrefix::<XLEN>::$method($($args),*),
            Prefixes::Rev8W => rev8w::Rev8WPrefix::$method($($args),*),
            Prefixes::RightShift => right_shift::RightShiftPrefix::$method($($args),*),
            Prefixes::SignExtension => sign_extension::SignExtensionPrefix::<XLEN>::$method($($args),*),
            Prefixes::LeftShift => left_shift::LeftShiftPrefix::<XLEN>::$method($($args),*),
            Prefixes::LeftShiftHelper => left_shift_helper::LeftShiftHelperPrefix::$method($($args),*),
            Prefixes::TwoLsb => two_lsb::TwoLsbPrefix::<XLEN>::$method($($args),*),
            Prefixes::SignExtensionUpperHalf => sign_extension_upper_half::SignExtensionUpperHalfPrefix::<XLEN>::$method($($args),*),
            Prefixes::ChangeDivisor => change_divisor::ChangeDivisorPrefix::<XLEN>::$method($($args),*),
            Prefixes::ChangeDivisorW => change_divisor_w::ChangeDivisorWPrefix::<XLEN>::$method($($args),*),
            Prefixes::RightOperand => right_operand::RightOperandPrefix::<XLEN>::$method($($args),*),
            Prefixes::RightOperandW => right_operand_w::RightOperandWPrefix::<XLEN>::$method($($args),*),
            Prefixes::SignExtensionRightOperand => sign_extension_right_operand::SignExtensionRightOperandPrefix::<XLEN>::$method($($args),*),
            Prefixes::RightShiftW => right_shift_w::RightShiftWPrefix::<XLEN>::$method($($args),*),
            Prefixes::LeftShiftWHelper => left_shift_w_helper::LeftShiftWHelperPrefix::<XLEN>::$method($($args),*),
            Prefixes::LeftShiftW => left_shift_w::LeftShiftWPrefix::<XLEN>::$method($($args),*),
            Prefixes::OverflowBitsZero => overflow_bits_zero::OverflowBitsZeroPrefix::<XLEN>::$method($($args),*),
            Prefixes::XorRot16 => xor_rot::XorRotPrefix::<XLEN, 16>::$method($($args),*),
            Prefixes::XorRot24 => xor_rot::XorRotPrefix::<XLEN, 24>::$method($($args),*),
            Prefixes::XorRot32 => xor_rot::XorRotPrefix::<XLEN, 32>::$method($($args),*),
            Prefixes::XorRot63 => xor_rot::XorRotPrefix::<XLEN, 63>::$method($($args),*),
            Prefixes::XorRotW7 => xor_rotw::XorRotWPrefix::<XLEN, 7>::$method($($args),*),
            Prefixes::XorRotW8 => xor_rotw::XorRotWPrefix::<XLEN, 8>::$method($($args),*),
            Prefixes::XorRotW12 => xor_rotw::XorRotWPrefix::<XLEN, 12>::$method($($args),*),
            Prefixes::XorRotW16 => xor_rotw::XorRotWPrefix::<XLEN, 16>::$method($($args),*),
        }
    };
}

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
        PrefixEval(dispatch_prefix!(self, prefix_mle, checkpoints, r_x, c, b, j))
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
        dispatch_prefix!(self, update_prefix_checkpoint, checkpoints, r_x, r_y, j, suffix_len)
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
            let prefix = ALL_PREFIXES[index];
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
