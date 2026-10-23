//! Σ-protocol parameters (Whiteout §6.2), derived from the segment table.
//!
//! Both roles derive the Gaussian width, response digit counts and boxes from
//! the public segment table in integer arithmetic, so they agree on the
//! response vector's layout. The attempt cap `T_att` is the prover's alone and
//! comes from the sampling layer's realized repetition rate.
//!
//! Rejection almost never happens (`specs/whiteout-sigma.md` §5.1): `α` is
//! large enough that one attempt is rejected with probability at most about
//! `2^{−13.2}`. Every Gaussian coordinate has one width
//! `σ = αΓ·√(Σ_b γ_b²)`, with `γ_b²` segment `b`'s honest `ℓ₂²` bound, so the
//! rejection rule's denominator is the single `k²`. The derivation never
//! trusts its own output: [`SigmaParams::new`] checks
//! `Σ_b ‖c s_b‖²/σ² ≤ α⁻²` exactly, with `σ² = k²/(2 ln 2)` and `ln 2`
//! replaced by an upper bound, for whatever width it receives.

use jolt_crypto::challenge::{
    OperatorNormRejection, D64_SELECTIVE_L2_CHALLENGE_CONFIG, MIN_FOLD_CHALLENGE_ENTROPY_BITS,
};
use jolt_crypto::lattice::{CHALLENGE_RING_DIM, MAX_CENTERED_BOUND};
#[cfg(feature = "whiteout-prover")]
use jolt_crypto::sample::{BlockParams, GaussianWidth, JointRejection};

use super::segments::{balanced_range, SegmentMask, SegmentTable};
use super::WhiteoutError;

/// Rejection parameter α: the largest power of two whose box fits ten
/// response digits at the headline, so `M − 1 ≈ 1.05·10^{−4}`.
pub const ALPHA: u32 = 1 << 17;
/// `ε = 2^{−137}` per attempt (§6.2).
pub const EPSILON_BITS: u32 = 137;
/// Response digits are balanced radix 16.
pub const RESPONSE_LOG_BASIS: u32 = 4;
/// The response digit count the commitments that `s` opens are priced for
/// (`whiteout-sigma.md` §5.1's regime): [`SigmaParams`] rejects a table whose
/// responses need more, so every certified box is inside the one
/// [`binding_collision_radius`] was computed for.
pub const BINDING_RESPONSE_DIGITS: usize = 10;
/// The acceptance box keeps `p_𝓑 ≥ 1 − 2^{−20}`, far below the rejection
/// rate set by `M`.
const BOX_FAILURE_INVERSE: u64 = 1 << 20;
/// All `T_att` attempts are rejected with probability at most `2^{−128}`.
#[cfg(feature = "whiteout-prover")]
const ALL_REJECTED_BITS: u32 = 128;
/// `⌈2^64 · ln 2⌉`.
const LN2_CEIL_2_64: u128 = 12_786_308_645_202_655_660;
/// Largest response digit count. Its whole balanced range stays within the
/// rejection rule's `2^62` box bound and the Ajtai map's input bound,
/// checked at compile time below.
const MAX_RESPONSE_DIGITS: usize = 15;
/// `8·(16^15 − 1)/15`, the negative end of fifteen balanced radix-16 digits.
const MAX_RESPONSE_MAGNITUDE: u128 = 8 * ((1 << (4 * MAX_RESPONSE_DIGITS)) - 1) / 15;
const _: () = assert!(MAX_RESPONSE_MAGNITUDE <= MAX_REJECTION_BOUND);
const _: () = assert!(MAX_RESPONSE_MAGNITUDE <= MAX_CENTERED_BOUND as u128);

/// The sampler's largest width multiplier, mirrored so that the verifier,
/// which does not build the sampler, rejects a width the prover could not
/// sample.
const MAX_WIDTH: u64 = 1 << 58;
/// Bounds of `JointRejection::new`'s exact arithmetic, mirrored so that both
/// roles reject a table the prover could not run.
const MAX_REJECTION_BOUND: u128 = 1 << 62;
const MAX_REJECTION_NUMERATOR: u128 = 1 << 126;

/// One Gaussian-masked segment's realized parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GaussianBlock {
    /// Index into the segment table.
    pub segment: usize,
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
    alpha: u32,
    /// Width multiplier `k` of every Gaussian coordinate:
    /// `σ = k / √(2 ln 2)`.
    width: u64,
    blocks: Vec<GaussianBlock>,
    /// Radix-16 response digits per segment, Gaussian or public.
    response_digits: Vec<usize>,
    /// A hard bound on every response coefficient with these digits.
    response_bound: u64,
}

impl SigmaParams {
    /// Derives the width in closed form and validates it with [`Self::new`].
    pub fn derive(table: &SegmentTable) -> Result<Self, WhiteoutError> {
        Self::derive_with_alpha(table, ALPHA)
    }

    /// [`Self::derive`] at another rejection parameter α.
    pub(crate) fn derive_with_alpha(
        table: &SegmentTable,
        alpha: u32,
    ) -> Result<Self, WhiteoutError> {
        // `k² ≥ 2 ln 2 · Γ² α² · Σ_b γ_b²`, the bound `new` checks.
        let required = width_scale(alpha)?
            .checked_mul(gaussian_l2_sq(table)?)
            .ok_or(WhiteoutError::Overflow)?
            .div_ceil(1 << 32);
        // The `+ 1` leaves slack far above the check's rounding.
        let width = u64::try_from(ceil_sqrt(required) + 1).map_err(|_| WhiteoutError::Overflow)?;
        Self::new_with_alpha(table, width, alpha)
    }

    /// Validates the common `width` multiplier and derives the digits, boxes
    /// and shift bounds.
    ///
    /// Rejects an invalid challenge family; a width for which some `s` of the
    /// table's shapes has `‖c s‖_σ > 1/α` for a challenge `c` (Thm
    /// `wo:thm:joint-rejection`); and parameters outside the rejection rule's
    /// exact arithmetic.
    ///
    /// Each block's digit count `D_b` is the fewest with
    /// `⌊L_b²/k²⌋ ≥ 1 + ⌈log₂(2 · 2^{20} · N)⌉`, `N` the Gaussian coefficient
    /// count. Then `2·exp(−L_b²/(2σ²)) = 2^{1−L_b²/k²} ≤ 2^{−21}/N` per
    /// coefficient, so `p_𝓑 ≥ 1 − 2^{−21}` holds without a separate check.
    pub fn new(table: &SegmentTable, width: u64) -> Result<Self, WhiteoutError> {
        Self::new_with_alpha(table, width, ALPHA)
    }

    fn new_with_alpha(table: &SegmentTable, width: u64, alpha: u32) -> Result<Self, WhiteoutError> {
        OperatorNormRejection::D64_SELECTIVE_L2
            .validate(CHALLENGE_RING_DIM, &D64_SELECTIVE_L2_CHALLENGE_CONFIG)
            .map_err(WhiteoutError::ChallengeFamily)?;
        D64_SELECTIVE_L2_CHALLENGE_CONFIG
            .validate_min_entropy_for_ring_dim(CHALLENGE_RING_DIM, MIN_FOLD_CHALLENGE_ENTROPY_BITS)
            .map_err(WhiteoutError::ChallengeFamily)?;
        let (_, kappa) = challenge_norms();
        let gaussian = table
            .segments()
            .filter(|segment| segment.mask == SegmentMask::Gaussian)
            .collect::<Vec<_>>();
        if gaussian.is_empty() {
            return Err(WhiteoutError::NoGaussianSegment);
        }
        if !(1..=MAX_WIDTH).contains(&width) {
            return Err(WhiteoutError::WidthOutOfRange { width });
        }

        // 2 ln2 Γ² α² Σ_b γ_b² / k² ≤ 1, the constant rounded up at 2^{−32}.
        let k_sq = u128::from(width).pow(2);
        let lhs = width_scale(alpha)?
            .checked_mul(gaussian_l2_sq(table)?)
            .ok_or(WhiteoutError::Overflow)?;
        if lhs.div_ceil(1 << 32) > k_sq {
            return Err(WhiteoutError::WidthTooNarrow);
        }

        let total = gaussian
            .iter()
            .map(|segment| segment.num_coefficients() as u128)
            .sum::<u128>();
        let required_exponent = 1 + ceil_log2(2 * u128::from(BOX_FAILURE_INVERSE) * total);
        let mut blocks = Vec::with_capacity(gaussian.len());
        let mut response_digits = Vec::with_capacity(table.segments().len());
        for (index, segment) in table.segments().enumerate() {
            let shift_bound = u64::from(kappa) * segment.digit_bound();
            if segment.mask != SegmentMask::Gaussian {
                response_digits.push(digits_covering(u128::from(shift_bound))?);
                continue;
            }
            let digits = (1..=MAX_RESPONSE_DIGITS)
                .find(|&digits| box_end(digits).pow(2) / k_sq >= required_exponent)
                .ok_or(WhiteoutError::Overflow)?;
            response_digits.push(digits);
            blocks.push(GaussianBlock {
                segment: index,
                digits,
                box_bound: u64::try_from(box_end(digits)).map_err(|_| WhiteoutError::Overflow)?,
                shift_bound,
                len: segment.num_coefficients(),
            });
        }
        if let Some(&digits) = response_digits
            .iter()
            .find(|&&digits| digits > BINDING_RESPONSE_DIGITS)
        {
            return Err(WhiteoutError::BindingDigitsExceeded { digits });
        }
        check_rejection_arithmetic(&blocks)?;
        // At most MAX_RESPONSE_MAGNITUDE, within the Ajtai map's input bound.
        let response_bound = response_digits
            .iter()
            .map(|&digits| {
                balanced_range(RESPONSE_LOG_BASIS, digits).map(|(min, _)| min.unsigned_abs())
            })
            .try_fold(0u128, |bound, end| Some(bound.max(end?)))
            .and_then(|bound| u64::try_from(bound).ok())
            .ok_or(WhiteoutError::Overflow)?;
        Ok(Self {
            alpha,
            width,
            blocks,
            response_digits,
            response_bound,
        })
    }

    /// The common width multiplier `k`: `σ = k / √(2 ln 2)`.
    pub fn width(&self) -> u64 {
        self.width
    }

    pub fn blocks(&self) -> &[GaussianBlock] {
        &self.blocks
    }

    /// Radix-16 response digits per segment, in table order.
    pub fn response_digits(&self) -> &[usize] {
        &self.response_digits
    }

    /// A hard bound on every response coefficient, below `2^60`.
    pub(crate) fn response_bound(&self) -> u64 {
        self.response_bound
    }

    /// The joint rejection rule over the Gaussian blocks, in block order.
    #[cfg(feature = "whiteout-prover")]
    pub fn rejection(&self) -> Result<JointRejection, WhiteoutError> {
        let width = GaussianWidth::from_multiplier(self.width)?;
        let blocks = self
            .blocks
            .iter()
            .map(|block| BlockParams {
                width,
                box_bound: block.box_bound,
                shift_bound: block.shift_bound,
                len: block.len,
            })
            .collect();
        Ok(JointRejection::new(blocks, self.alpha, EPSILON_BITS)?)
    }

    /// `T_att`, the prover's attempt cap: all attempts are rejected with
    /// probability at most `2^{−128}` (Thm `wo:thm:completeness`), computed
    /// from the rejection rule's realized `M` and `p_𝓑 ≥ 1 − 2^{−20}`.
    #[cfg(feature = "whiteout-prover")]
    pub fn max_attempts(&self) -> Result<usize, WhiteoutError> {
        Ok(self
            .rejection()?
            .max_attempts(BOX_FAILURE_INVERSE, ALL_REJECTED_BITS)?)
    }
}

/// The coefficient-ℓ∞ radius `4Γ_Δ B` at which a commitment matrix that `s`
/// opens must be Module-SIS hard (Thm `wo:thm:knowledge`; paper §6
/// "Commitment ranks"): `Γ_Δ = 2κ₁` bounds `‖(c − c′)v‖_∞ / ‖v‖_∞` over the
/// challenge family, and `B` is the magnitude of the
/// [`BINDING_RESPONSE_DIGITS`]-digit certified box, the largest response
/// coefficient on any segment. The one owner of the radius: Akita's residual
/// commitment (`A_a`, `[B_a | B′_a]`) and Jolt's pad commitment `A_p`
/// (`jolt_akita`'s `sigma_matrix_rank`, Akita A40) read it through Akita's
/// Σ-binding table.
pub fn binding_collision_radius() -> u128 {
    let (_, kappa) = challenge_norms();
    let magnitude = balanced_range(RESPONSE_LOG_BASIS, BINDING_RESPONSE_DIGITS)
        .map_or(0, |(min, _)| min.unsigned_abs());
    8 * u128::from(kappa) * magnitude
}

/// `⌈2^{32} · 2 ln 2 · Γ² α²⌉`, with `ln 2` rounded up: the width condition
/// is `⌈width_scale · Σ_b γ_b² / 2^{32}⌉ ≤ k²`.
fn width_scale(alpha: u32) -> Result<u128, WhiteoutError> {
    let (gamma, _) = challenge_norms();
    (2 * LN2_CEIL_2_64 * u128::from(gamma).pow(2))
        .checked_mul(u128::from(alpha).pow(2))
        .map(|scale| scale.div_ceil(1 << 32))
        .ok_or(WhiteoutError::Overflow)
}

/// `Σ_b γ_b²` over the Gaussian segments.
fn gaussian_l2_sq(table: &SegmentTable) -> Result<u128, WhiteoutError> {
    table
        .segments()
        .filter(|segment| segment.mask == SegmentMask::Gaussian)
        .try_fold(0u128, |sum, segment| {
            sum.checked_add(segment.honest_l2_sq())
        })
        .ok_or(WhiteoutError::Overflow)
}

/// The verifier's mirror of `JointRejection::new`'s one remaining runtime
/// bound, so that both roles reject a table the prover could not run: the
/// worst-case exponent numerator `Σ_b (2 L_b V_b + V_b²) · len_b` below
/// `2^126`, and shift bounds at most `2^62`. With one width the denominator
/// is `k² < 2^{117}`, and boxes are below `2^62` by the compile-time checks
/// above. At the production α the width check fails first on any table the
/// numerator bound would reject; at α = 16 it can bind.
fn check_rejection_arithmetic(blocks: &[GaussianBlock]) -> Result<(), WhiteoutError> {
    let worst = blocks.iter().try_fold(0u128, |acc, block| {
        let (box_bound, shift) = (u128::from(block.box_bound), u128::from(block.shift_bound));
        if shift > MAX_REJECTION_BOUND {
            return None;
        }
        let per_coordinate = box_bound
            .checked_mul(shift)?
            .checked_mul(2)?
            .checked_add(shift.checked_mul(shift)?)?;
        acc.checked_add(per_coordinate.checked_mul(u128::try_from(block.len).ok()?)?)
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

/// The positive end of the balanced radix-16 range with `digits` digits.
fn box_end(digits: usize) -> u128 {
    balanced_range(RESPONSE_LOG_BASIS, digits).map_or(0, |(_, max)| max.unsigned_abs())
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
