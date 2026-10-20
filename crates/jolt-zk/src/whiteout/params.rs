//! Σ-protocol parameters (Whiteout §6.2), derived from the segment table.
//!
//! Both roles derive the Gaussian widths, response digit counts and boxes from
//! the public segment table in integer arithmetic, so they agree bit for bit:
//! the digit counts fix the layout of the response vector `e`.
//!
//! The widths follow the continuous optimum of §6.2, `ω_b = N_b / ΣN` over
//! Gaussian coefficients, so `σ_b = αΓγ_b·√(ΣN/N_b)` with `γ_b²` the
//! segment's honest `ℓ₂²` bound. The derivation never trusts its own output:
//! [`SigmaParams::new`] checks `Σ_b ‖c s_b‖²/σ_b² ≤ α⁻²` exactly, with
//! `σ_b² = k_b²/(2 ln 2)` and `ln 2` replaced by an upper bound, for whatever
//! widths it receives.

use jolt_crypto::challenge::{OperatorNormRejection, D64_SELECTIVE_L2_CHALLENGE_CONFIG};
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
const BOX_FAILURE_INVERSE: u128 = 1000;
/// All `T_att` attempts are rejected with probability at most `2^{−128}`.
const ALL_REJECTED_BITS: f64 = 128.0;
/// `⌈2^64 · ln 2⌉`.
const LN2_CEIL_2_64: u128 = 12_786_308_645_202_655_660;
/// Largest digit count whose box stays below the rejection rule's `2^62`.
const MAX_RESPONSE_DIGITS: usize = 15;
/// The sampler's largest width multiplier.
const MAX_WIDTH: u64 = 1 << 31;

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
    max_attempts: usize,
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
        let scale = 2 * LN2_CEIL_2_64 * u128::from(gamma).pow(2) * u128::from(ALPHA).pow(2);
        let scale = scale.div_ceil(1 << 32);
        let widths = gaussian
            .map(|segment| {
                let required = scale
                    .checked_mul(segment.honest_l2_sq())
                    .and_then(|value| value.checked_mul(total))
                    .ok_or(WhiteoutError::Parameters("width derivation overflows"))?
                    .div_ceil((segment.num_coefficients() as u128) << 32);
                // The `+ 1` leaves slack far above the check's rounding.
                u64::try_from(ceil_sqrt(required) + 1)
                    .map_err(|_| WhiteoutError::Parameters("width exceeds u64"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(table, &widths)
    }

    /// Validates `widths`, one per Gaussian segment in table order, and
    /// derives the digits, boxes, shift bounds and attempt cap.
    ///
    /// Rejects widths for which some honest `s` of the table's shapes has
    /// `‖c s‖_σ > 1/α` for a challenge `c` (Thm `wo:thm:joint-rejection`), or
    /// for which no digit count keeps `p_𝓑 ≥ 0.999`.
    pub fn new(table: &SegmentTable, widths: &[u64]) -> Result<Self, WhiteoutError> {
        let (gamma, kappa) = challenge_norms();
        let gaussian = table
            .segments()
            .iter()
            .enumerate()
            .filter(|(_, segment)| segment.mask == SegmentMask::Gaussian)
            .collect::<Vec<_>>();
        if gaussian.is_empty() || gaussian.len() != widths.len() {
            return Err(WhiteoutError::Parameters(
                "one width per Gaussian segment, and at least one",
            ));
        }
        if widths.iter().any(|&k| !(1..=MAX_WIDTH).contains(&k)) {
            return Err(WhiteoutError::Parameters(
                "width multiplier outside [1, 2^31]",
            ));
        }

        // Σ_b 2 ln2 Γ² α² γ_b² / k_b² ≤ 1, each term rounded up at 2^{−64}.
        let numerator = 2 * LN2_CEIL_2_64 * u128::from(gamma).pow(2) * u128::from(ALPHA).pow(2);
        let mut sum = 0u128;
        for ((_, segment), &k) in gaussian.iter().zip(widths) {
            let term = numerator
                .checked_mul(segment.honest_l2_sq())
                .ok_or(WhiteoutError::Parameters("width check overflows"))?
                .div_ceil(u128::from(k).pow(2));
            sum = sum
                .checked_add(term)
                .ok_or(WhiteoutError::Parameters("width check overflows"))?;
        }
        if sum > 1 << 64 {
            return Err(WhiteoutError::Parameters(
                "widths admit ‖cs‖_σ > 1/α for an honest s",
            ));
        }

        // Boxes: L_b²/k_b² ≥ 1 + ⌈log₂(2000 N)⌉, then Σ N_b 2^{1−⌊L_b²/k_b²⌋} ≤ 1/1000.
        let total = gaussian
            .iter()
            .map(|(_, segment)| segment.num_coefficients() as u128)
            .sum::<u128>();
        let required_exponent = 1 + ceil_log2(2 * BOX_FAILURE_INVERSE * total);
        let mut blocks = Vec::with_capacity(gaussian.len());
        let mut response_digits = Vec::with_capacity(table.segments().len());
        let mut box_mass = 0u128;
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
            let digits = (1..=MAX_RESPONSE_DIGITS)
                .find(|&digits| box_end(digits).pow(2) / k_sq >= required_exponent)
                .ok_or(WhiteoutError::Parameters("no digit count covers the box"))?;
            let box_bound = box_end(digits);
            let exponent = box_bound.pow(2) / k_sq;
            box_mass = box_mass
                .checked_add(tail_mass(segment.num_coefficients() as u128, exponent)?)
                .ok_or(WhiteoutError::Parameters("box check overflows"))?;
            response_digits.push(digits);
            blocks.push(GaussianBlock {
                segment: index,
                width,
                digits,
                box_bound: u64::try_from(box_bound)
                    .map_err(|_| WhiteoutError::Parameters("box exceeds u64"))?,
                shift_bound,
                len: segment.num_coefficients(),
            });
        }
        if box_mass > (1u128 << 64) / BOX_FAILURE_INVERSE {
            return Err(WhiteoutError::Parameters("acceptance box has p_𝓑 < 0.999"));
        }

        Ok(Self {
            blocks,
            response_digits,
            max_attempts: max_attempts(),
        })
    }

    pub fn blocks(&self) -> &[GaussianBlock] {
        &self.blocks
    }

    /// Radix-16 response digits per segment, in table order.
    pub fn response_digits(&self) -> &[usize] {
        &self.response_digits
    }

    /// `T_att`: the attempt cap.
    pub fn max_attempts(&self) -> usize {
        self.max_attempts
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
        .ok_or(WhiteoutError::Parameters(
            "no digit count covers the public response",
        ))
}

/// `⌈len · 2^{65 − exponent}⌉`: the Gaussian tail mass `len · 2^{1 − exponent}`
/// outside the box, in units of `2^{−64}`.
fn tail_mass(len: u128, exponent: u128) -> Result<u128, WhiteoutError> {
    if exponent <= 65 {
        len.checked_shl(65 - exponent as u32)
            .filter(|mass| mass >> (65 - exponent) == len)
            .ok_or(WhiteoutError::Parameters("box check overflows"))
    } else if exponent - 65 >= 128 {
        Ok(u128::from(len > 0))
    } else {
        Ok(len.div_ceil(1 << (exponent - 65)))
    }
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

/// `T_att`, the fewest attempts with `(1 − (p_𝓑 − ε)/M)^T ≤ 2^{−128}` for
/// `M = exp(t/α + 1/(2α²))`, `t = √(2 ε_bits ln 2)` (Thm
/// `wo:thm:completeness`). It bounds only the prover's retries, so it is
/// computed in `f64`, with the acceptance probability shaded down by `2^{−40}`
/// against rounding.
fn max_attempts() -> usize {
    let alpha = f64::from(ALPHA);
    let t = (2.0 * f64::from(EPSILON_BITS) * std::f64::consts::LN_2).sqrt();
    let repetition = (t / alpha + 1.0 / (2.0 * alpha * alpha)).exp();
    let p_box = 1.0 - 1.0 / BOX_FAILURE_INVERSE as f64;
    let acceptance =
        (p_box - (-f64::from(EPSILON_BITS)).exp2()) / repetition * (1.0 - 2f64.powi(-40));
    (ALL_REJECTED_BITS / -(1.0 - acceptance).log2()).ceil() as usize
}
