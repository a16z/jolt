//! Σ-protocol parameters (Whiteout §6.2), derived from the segment table.
//!
//! Both roles derive the Gaussian widths, response digit counts and boxes from
//! the public segment table in integer arithmetic, so they agree on the
//! response vector's layout. The attempt cap `T_att` is the prover's alone and
//! comes from the sampling layer's realized repetition rate.
//!
//! The widths follow the continuous optimum of §6.2, `ω_b = N_b / ΣN` over
//! Gaussian coefficients, so `σ_b = αΓγ_b·√(ΣN/N_b)` with `γ_b²` the
//! segment's honest `ℓ₂²` bound. The derivation never trusts its own output:
//! [`SigmaParams::new`] checks `Σ_b ‖c s_b‖²/σ_b² ≤ α⁻²` exactly, with
//! `σ_b² = k_b²/(2 ln 2)` and `ln 2` replaced by an upper bound, for whatever
//! widths it receives.

use jolt_crypto::challenge::{
    OperatorNormRejection, D64_SELECTIVE_L2_CHALLENGE_CONFIG, MIN_FOLD_CHALLENGE_ENTROPY_BITS,
};
use jolt_crypto::lattice::CHALLENGE_RING_DIM;
#[cfg(feature = "whiteout-prover")]
use jolt_crypto::sample::{BlockParams, GaussianWidth, JointRejection};

use super::segments::{SegmentMask, SegmentTable};
use super::WhiteoutError;

/// Rejection parameter α of §6.2.
pub const ALPHA: u32 = 16;
/// `ε = 2^{−137}` per attempt (§6.2).
pub const EPSILON_BITS: u32 = 137;
/// Response digits are balanced radix 16.
pub const RESPONSE_LOG_BASIS: u32 = 4;
/// The acceptance box keeps `p_𝓑 ≥ 1 − 1/1000`.
const BOX_FAILURE_INVERSE: u64 = 1000;
/// All `T_att` attempts are rejected with probability at most `2^{−128}`.
#[cfg(feature = "whiteout-prover")]
const ALL_REJECTED_BITS: u32 = 128;
/// `⌈2^64 · ln 2⌉`.
const LN2_CEIL_2_64: u128 = 12_786_308_645_202_655_660;
/// Largest digit count whose box stays below the rejection rule's `2^62`.
const MAX_RESPONSE_DIGITS: usize = 15;
/// The sampler's largest width multiplier.
const MAX_WIDTH: u64 = 1 << 31;
/// Bounds of `JointRejection::new`'s exact arithmetic, mirrored so that both
/// roles reject a table the prover could not run.
const MAX_REJECTION_BOUND: u128 = 1 << 62;
const MAX_REJECTION_NUMERATOR: u128 = 1 << 126;

/// One Gaussian-masked segment's realized parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GaussianBlock {
    /// Index into the segment table.
    pub segment: usize,
    /// Width multiplier `k_b`: `σ_b = k_b / √(2 ln 2)`.
    pub width: u64,
    /// Radix-16 digits of each response coefficient.
    pub digits: usize,
    /// Acceptance box `|z| ≤ box_bound`, the positive end of the digits'
    /// balanced range, so `𝓑 ⊆ 𝓑_cert`.
    pub box_bound: u64,
    /// `|v| ≤ κ₁ · B` for every honest shift `v = c·s`.
    pub shift_bound: u64,
    /// Coefficients in the block.
    pub len: usize,
}

/// The Σ-protocol's realized parameters for one segment table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SigmaParams {
    blocks: Vec<GaussianBlock>,
    /// Radix-16 response digits per segment, Gaussian or public.
    response_digits: Vec<usize>,
}

impl SigmaParams {
    /// Derives the widths in closed form and validates them with [`Self::new`].
    pub fn derive(table: &SegmentTable) -> Result<Self, WhiteoutError> {
        let (gamma, _) = challenge_norms();
        let gaussian = table
            .segments()
            .iter()
            .filter(|segment| segment.mask == SegmentMask::Gaussian);
        let total = gaussian
            .clone()
            .map(|segment| segment.num_coefficients() as u128)
            .sum::<u128>();
        // `C = ⌈2 ln 2 · Γ² α² · 2^32⌉`, so `k_b² ≥ C γ_b² ΣN / (N_b 2^32)`.
        let scale = (2 * LN2_CEIL_2_64 * u128::from(gamma).pow(2) * u128::from(ALPHA).pow(2))
            .div_ceil(1 << 32);
        let widths = gaussian
            .map(|segment| {
                let required = scale
                    .checked_mul(segment.honest_l2_sq())
                    .and_then(|value| value.checked_mul(total))
                    .ok_or(WhiteoutError::Overflow)?
                    .div_ceil((segment.num_coefficients() as u128) << 32);
                // The `+ 1` leaves slack far above the check's rounding.
                u64::try_from(ceil_sqrt(required) + 1).map_err(|_| WhiteoutError::Overflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(table, &widths)
    }

    /// Validates `widths`, one per Gaussian segment in table order, and
    /// derives the digits, boxes and shift bounds.
    ///
    /// Rejects an invalid challenge family; widths for which some `s` of the
    /// table's shapes has `‖c s‖_σ > 1/α` for a challenge `c` (Thm
    /// `wo:thm:joint-rejection`); and parameters outside the rejection rule's
    /// exact arithmetic.
    ///
    /// Each block's digit count `D_b` is the fewest with
    /// `⌊L_b²/k_b²⌋ ≥ 1 + ⌈log₂(2000 N)⌉`, `N` the Gaussian coefficient count.
    /// Then `2·exp(−L_b²/(2σ_b²)) = 2^{1−L_b²/k_b²} ≤ 1/(2000 N)` per
    /// coefficient, so `p_𝓑 ≥ 1 − 1/2000` (§2.5's box bound) holds without a
    /// separate check.
    pub fn new(table: &SegmentTable, widths: &[u64]) -> Result<Self, WhiteoutError> {
        OperatorNormRejection::D64_SELECTIVE_L2
            .validate(CHALLENGE_RING_DIM, &D64_SELECTIVE_L2_CHALLENGE_CONFIG)
            .map_err(WhiteoutError::ChallengeFamily)?;
        D64_SELECTIVE_L2_CHALLENGE_CONFIG
            .validate_min_entropy_for_ring_dim(CHALLENGE_RING_DIM, MIN_FOLD_CHALLENGE_ENTROPY_BITS)
            .map_err(WhiteoutError::ChallengeFamily)?;
        let (gamma, kappa) = challenge_norms();
        let gaussian = table
            .segments()
            .iter()
            .filter(|segment| segment.mask == SegmentMask::Gaussian)
            .collect::<Vec<_>>();
        if gaussian.is_empty() || gaussian.len() != widths.len() {
            return Err(WhiteoutError::WidthCount {
                expected: gaussian.len(),
                actual: widths.len(),
            });
        }
        if let Some(&width) = widths.iter().find(|&&k| !(1..=MAX_WIDTH).contains(&k)) {
            return Err(WhiteoutError::WidthOutOfRange { width });
        }

        // Σ_b 2 ln2 Γ² α² γ_b² / k_b² ≤ 1, each term rounded up at 2^{−64}.
        let numerator = 2 * LN2_CEIL_2_64 * u128::from(gamma).pow(2) * u128::from(ALPHA).pow(2);
        let mut sum = 0u128;
        for (segment, &k) in gaussian.iter().zip(widths) {
            let term = numerator
                .checked_mul(segment.honest_l2_sq())
                .ok_or(WhiteoutError::Overflow)?
                .div_ceil(u128::from(k).pow(2));
            sum = sum.checked_add(term).ok_or(WhiteoutError::Overflow)?;
        }
        if sum > 1 << 64 {
            return Err(WhiteoutError::WidthsTooNarrow);
        }

        let total = gaussian
            .iter()
            .map(|segment| segment.num_coefficients() as u128)
            .sum::<u128>();
        let required_exponent = 1 + ceil_log2(2 * u128::from(BOX_FAILURE_INVERSE) * total);
        let mut blocks = Vec::with_capacity(gaussian.len());
        let mut response_digits = Vec::with_capacity(table.segments().len());
        let mut widths = widths.iter();
        for (index, segment) in table.segments().iter().enumerate() {
            let shift_bound = u64::from(kappa) * segment.digit_bound();
            let Some(&width) = (segment.mask == SegmentMask::Gaussian)
                .then(|| widths.next())
                .flatten()
            else {
                response_digits.push(digits_covering(u128::from(shift_bound))?);
                continue;
            };
            let k_sq = u128::from(width).pow(2);
            // Always found: `k ≤ 2^31` needs at most nine digits.
            let digits = (1..=MAX_RESPONSE_DIGITS)
                .find(|&digits| box_end(digits).pow(2) / k_sq >= required_exponent)
                .ok_or(WhiteoutError::Overflow)?;
            response_digits.push(digits);
            blocks.push(GaussianBlock {
                segment: index,
                width,
                digits,
                box_bound: u64::try_from(box_end(digits)).map_err(|_| WhiteoutError::Overflow)?,
                shift_bound,
                len: segment.num_coefficients(),
            });
        }
        check_rejection_arithmetic(&blocks)?;
        Ok(Self {
            blocks,
            response_digits,
        })
    }

    pub fn blocks(&self) -> &[GaussianBlock] {
        &self.blocks
    }

    /// Radix-16 response digits per segment, in table order.
    pub fn response_digits(&self) -> &[usize] {
        &self.response_digits
    }

    /// The joint rejection rule over the Gaussian blocks, in block order.
    #[cfg(feature = "whiteout-prover")]
    pub fn rejection(&self) -> Result<JointRejection, WhiteoutError> {
        let blocks = self
            .blocks
            .iter()
            .map(|block| {
                Ok(BlockParams {
                    width: GaussianWidth::from_multiplier(block.width)?,
                    box_bound: block.box_bound,
                    shift_bound: block.shift_bound,
                    len: block.len,
                })
            })
            .collect::<Result<Vec<_>, WhiteoutError>>()?;
        Ok(JointRejection::new(blocks, ALPHA, EPSILON_BITS)?)
    }

    /// `T_att`, the prover's attempt cap: all attempts are rejected with
    /// probability at most `2^{−128}` (Thm `wo:thm:completeness`), computed
    /// from the rejection rule's realized `M` and `p_𝓑 ≥ 0.999`.
    #[cfg(feature = "whiteout-prover")]
    pub fn max_attempts(&self) -> Result<usize, WhiteoutError> {
        Ok(self
            .rejection()?
            .max_attempts(BOX_FAILURE_INVERSE, ALL_REJECTED_BITS)?)
    }
}

/// Mirrors the parameter bounds of `JointRejection::new`: every box and shift
/// bound at most `2^62`, the product `Q` of the distinct `k_b²` below
/// `2^126`, and the worst-case exponent numerator
/// `Σ_b (2 L_b V_b + V_b²) · len_b · Q/k_b²` below `2^126`.
fn check_rejection_arithmetic(blocks: &[GaussianBlock]) -> Result<(), WhiteoutError> {
    let mut distinct = blocks
        .iter()
        .map(|block| u128::from(block.width).pow(2))
        .collect::<Vec<_>>();
    distinct.sort_unstable();
    distinct.dedup();
    let denominator = distinct
        .iter()
        .try_fold(1u128, |product, &k_sq| product.checked_mul(k_sq))
        .filter(|&product| product < MAX_REJECTION_NUMERATOR)
        .ok_or(WhiteoutError::RejectionArithmetic)?;
    let worst = blocks.iter().try_fold(0u128, |acc, block| {
        let (box_bound, shift) = (u128::from(block.box_bound), u128::from(block.shift_bound));
        if box_bound > MAX_REJECTION_BOUND || shift > MAX_REJECTION_BOUND {
            return None;
        }
        let per_coordinate = box_bound
            .checked_mul(shift)?
            .checked_mul(2)?
            .checked_add(shift.checked_mul(shift)?)?;
        acc.checked_add(
            per_coordinate
                .checked_mul(u128::try_from(block.len).ok()?)?
                .checked_mul(denominator / u128::from(block.width).pow(2))?,
        )
    });
    match worst {
        Some(worst) if worst < MAX_REJECTION_NUMERATOR => Ok(()),
        _ => Err(WhiteoutError::RejectionArithmetic),
    }
}

/// `(Γ, κ₁)`: the operator-norm bound and `‖c‖₁` of the challenge family.
fn challenge_norms() -> (u32, u32) {
    (
        OperatorNormRejection::D64_SELECTIVE_L2.threshold,
        D64_SELECTIVE_L2_CHALLENGE_CONFIG.l1_norm() as u32,
    )
}

/// The positive end of the balanced radix-16 range with `digits` digits,
/// `7·(16^digits − 1)/15`.
fn box_end(digits: usize) -> u128 {
    let radix = 1u128 << RESPONSE_LOG_BASIS;
    (radix / 2 - 1) * (radix.pow(digits as u32) - 1) / (radix - 1)
}

/// The fewest digits whose balanced range covers `[−bound, bound]`.
fn digits_covering(bound: u128) -> Result<usize, WhiteoutError> {
    (1..=MAX_RESPONSE_DIGITS)
        .find(|&digits| box_end(digits) >= bound)
        .ok_or(WhiteoutError::Overflow)
}

/// `⌈√x⌉`.
fn ceil_sqrt(x: u128) -> u128 {
    let root = x.isqrt();
    root + u128::from(root * root < x)
}

/// `⌈log₂ x⌉` for `x ≥ 1`.
fn ceil_log2(x: u128) -> u128 {
    u128::from(u128::BITS - (x - 1).leading_zeros())
}
