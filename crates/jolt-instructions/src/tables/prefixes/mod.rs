//! Prefix polynomial evaluations for the sparse-dense decomposition.
//!
//! Each prefix captures the "contribution" of high-order bound variables
//! to a lookup table's MLE during sumcheck. Prefixes are field-valued
//! (unlike suffixes which are `u64`), and maintain checkpoints that are
//! updated every two sumcheck rounds.

pub mod and;
pub mod andn;
pub mod eq;
pub mod left_is_zero;
pub mod lower_half_word;
pub mod lower_word;
pub mod lt;
pub mod or;
pub mod upper_word;
pub mod xor;

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
