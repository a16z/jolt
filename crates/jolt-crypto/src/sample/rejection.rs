//! The joint rejection rule (Whiteout Def. `wo:def:joint-rejection`).
//!
//! Given `z = y + v` with `y ← π_0`, accept with probability
//! `min{1, π_0(z)/(M·π_v(z))}` if `z ∈ 𝓑`, else reject. With realized widths
//! `σ_b² = k_b²/(2 ln 2)` the ratio is `2^X/M` for the exact rational
//! `X = Σ_b (V_b − 2S_b)/k_b²`, `S_b = Σ z_i v_i`, `V_b = Σ v_i²`, which this
//! module evaluates in constant time (`specs/whiteout-gaussian.md` §3).

use rand_core::{CryptoRng, RngCore};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use super::fixed::{
    ct_floor_div, ct_lt_i128, ct_lt_i64, inverse_repetition_rate, pow2_neg_table, Fix192,
};
use super::{GaussianWidth, SampleError};

/// Largest box or shift bound, so negation and clamping stay in `i64`.
const MAX_BOUND: u64 = 1 << 62;
/// Bound on `|P|` and `Q`, so restoring division never overflows `u128`.
const MAX_NUMERATOR: u128 = 1 << 126;
/// Bound on `α` and `ε_bits`, so computing `M` cannot overflow.
const MAX_REJECTION_PARAMETER: u32 = 1 << 16;

/// One block of Gaussian-masked coordinates. Public coordinates of `s` are
/// never part of a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockParams {
    pub width: GaussianWidth,
    /// The acceptance box `|z_i| ≤ box_bound`.
    pub box_bound: u64,
    /// The release-checked honest bound `|v_i| ≤ shift_bound`.
    pub shift_bound: u64,
    pub len: usize,
}

/// The public parameters of one joint rejection rule.
#[derive(Clone, Debug)]
pub struct JointRejection {
    blocks: Vec<BlockParams>,
    /// `Q/k_b²` per block.
    scales: Vec<u128>,
    /// `Q`, the product of the distinct `k_b²`.
    denominator: u128,
    /// `2^{−2^i/Q}` for every bit `i` of `Q`.
    exp_table: Vec<Fix192>,
    inverse_m: Fix192,
    epsilon_bits: u32,
    total_len: usize,
}

/// The secret-dependent pieces of one acceptance decision.
pub(crate) struct Acceptance {
    pub(crate) in_box: Choice,
    pub(crate) shift_ok: Choice,
    /// `2^X/M ≥ 1`.
    pub(crate) saturated: Choice,
    /// `2^X/M` when not saturated.
    pub(crate) threshold: Fix192,
}

impl JointRejection {
    /// `M = exp(t/α + 1/(2α²))` with `t = √(2·epsilon_bits·ln 2)`, so that
    /// `ε = e^{−t²/2} = 2^{−epsilon_bits}`. `M^{−1}` is floored, which runs
    /// the rule with some `M' ≥ M`.
    ///
    /// Requires `M ≤ 4`: the evaluation clamps the exponent so that the
    /// ratio saturates only for `M^{−1} ≥ 1/4` (Whiteout's `M ≈ 2.37`).
    pub fn new(
        blocks: Vec<BlockParams>,
        alpha: u32,
        epsilon_bits: u32,
    ) -> Result<Self, SampleError> {
        if blocks.is_empty() {
            return Err(SampleError::NoBlocks);
        }
        let in_range = |p: u32| (1..=MAX_REJECTION_PARAMETER).contains(&p);
        if !(in_range(alpha) && in_range(epsilon_bits)) {
            return Err(SampleError::BadRejectionParameters);
        }
        let inverse_m = inverse_repetition_rate(alpha, epsilon_bits);
        if inverse_m.0[2] < 1 << 62 {
            return Err(SampleError::BadRejectionParameters);
        }
        if blocks
            .iter()
            .any(|b| b.box_bound > MAX_BOUND || b.shift_bound > MAX_BOUND)
        {
            return Err(SampleError::ArithmeticBound);
        }
        let mut distinct: Vec<u128> = blocks.iter().map(|b| b.width.k_squared()).collect();
        distinct.sort_unstable();
        distinct.dedup();
        let denominator = distinct
            .iter()
            .try_fold(1u128, |q, k2| q.checked_mul(*k2))
            .filter(|q| *q < MAX_NUMERATOR)
            .ok_or(SampleError::ArithmeticBound)?;
        let scales: Vec<u128> = blocks
            .iter()
            .map(|b| denominator / b.width.k_squared())
            .collect();

        // Worst case of |V_b − 2S_b|·Q/k_b² with z clamped to the box.
        let worst = blocks
            .iter()
            .zip(&scales)
            .try_fold(0u128, |acc, (b, scale)| {
                let (l, v) = (u128::from(b.box_bound), u128::from(b.shift_bound));
                let per_coordinate = l
                    .checked_mul(v)?
                    .checked_mul(2)?
                    .checked_add(v.checked_mul(v)?)?;
                acc.checked_add(
                    per_coordinate
                        .checked_mul(u128::try_from(b.len).ok()?)?
                        .checked_mul(*scale)?,
                )
            });
        if worst.is_none_or(|w| w >= MAX_NUMERATOR) {
            return Err(SampleError::ArithmeticBound);
        }
        let total_len = blocks
            .iter()
            .try_fold(0usize, |acc, b| acc.checked_add(b.len))
            .ok_or(SampleError::ArithmeticBound)?;

        Ok(Self {
            exp_table: pow2_neg_table(denominator, 128 - denominator.leading_zeros()),
            inverse_m,
            epsilon_bits,
            blocks,
            scales,
            denominator,
            total_len,
        })
    }

    /// `T_att`: the fewest attempts after which every attempt is rejected
    /// with probability at most `2^{−failure_bits}` (Thm `wo:thm:completeness`).
    ///
    /// Each attempt accepts with probability at least `(p_𝓑 − ε)/M'`
    /// (Thm `wo:thm:joint-rejection`), with `M'` this rule's realized
    /// repetition rate (`M^{−1}` floored) and `p_𝓑 ≥ 1 − 1/box_failure_inverse`
    /// the caller's bound on the box. Every rounding makes the per-attempt
    /// failure probability larger, so the returned count is conservative.
    pub fn max_attempts(
        &self,
        box_failure_inverse: u64,
        failure_bits: u32,
    ) -> Result<usize, SampleError> {
        if box_failure_inverse < 2 {
            return Err(SampleError::BadRejectionParameters);
        }
        // Fractions below one at scale 2^128, rounded against acceptance.
        let box_mass = u128::MAX / u128::from(box_failure_inverse) + 1;
        let epsilon = 1u128 << 128u32.saturating_sub(self.epsilon_bits).min(127);
        let p_box = u128::MAX - box_mass - epsilon;
        let [_, middle, top] = self.inverse_m.0;
        let inverse_m = (u128::from(top) << 64) | u128::from(middle);
        let (acceptance, _) = mul_wide(p_box, inverse_m);
        if acceptance == 0 {
            return Err(SampleError::BadRejectionParameters);
        }
        // `failure = 2^128 − acceptance`, normalized as `m · 2^{−128−e}`.
        let failure = u128::MAX - acceptance + 1;
        let (step, step_exponent) = normalize(failure);
        let (mut mantissa, mut exponent) = (step, step_exponent);
        for attempts in 1..=MAX_ATTEMPTS {
            // `m · 2^{−128−e} ≤ 2^{−bits}` iff `m ≤ 2^{128+e−bits}`.
            let done = match (128 + exponent).checked_sub(failure_bits) {
                Some(shift) if shift >= 128 => true,
                Some(127) => mantissa == 1 << 127,
                _ => false,
            };
            if done {
                return Ok(attempts);
            }
            let (high, low) = mul_wide(mantissa, step);
            let (product, shift) = normalize(high + u128::from(low != 0));
            mantissa = product;
            exponent += step_exponent + shift;
        }
        Err(SampleError::BadRejectionParameters)
    }

    /// Applies the rule to `z = y + v`, where `v = cs` is the exact
    /// centred-integer negacyclic product. Both are laid out block-major.
    /// Every coordinate is processed and the accept bit is the only value
    /// derived from them that leaves.
    ///
    /// [`SampleError::ShiftOutOfRange`] can only fire on a malformed
    /// witness. The caller must treat it as fatal and abort the proof; it
    /// must never retry or record it in the transcript.
    pub fn accept<R: RngCore + CryptoRng>(
        &self,
        z: &[i64],
        v: &[i64],
        rng: &mut R,
    ) -> Result<Choice, SampleError> {
        for got in [z.len(), v.len()] {
            if got != self.total_len {
                return Err(SampleError::LengthMismatch {
                    expected: self.total_len,
                    got,
                });
            }
        }
        let acceptance = self.acceptance(z, v);
        let below = Fix192::random(rng).ct_lt(&acceptance.threshold);
        let accept = acceptance.in_box & (acceptance.saturated | below);
        if bool::from(acceptance.shift_ok) {
            Ok(accept)
        } else {
            Err(SampleError::ShiftOutOfRange)
        }
    }

    // Kept out of line so the reviewed constant-time code is the code that
    // runs, whatever the caller's inlining.
    #[inline(never)]
    pub(crate) fn acceptance(&self, z: &[i64], v: &[i64]) -> Acceptance {
        let mut in_box = Choice::from(1);
        let mut shift_ok = Choice::from(1);
        let mut numerator = 0i128;
        let mut pairs = z.iter().zip(v);
        for (block, scale) in self.blocks.iter().zip(&self.scales) {
            let (l, bound) = (block.box_bound as i64, block.shift_bound as i64);
            let (mut s, mut squares) = (0i128, 0i128);
            for (&zi, &vi) in pairs.by_ref().take(block.len) {
                let above = ct_lt_i64(l, zi);
                let below = ct_lt_i64(zi, -l);
                in_box &= !(above | below);
                shift_ok &= !(ct_lt_i64(bound, vi) | ct_lt_i64(vi, -bound));
                // An out-of-box z is rejected; clamping keeps the sums in
                // range for it. Wrapping only matters when v is out of range,
                // which is reported as an error.
                let clamped = i64::conditional_select(&zi, &l, above);
                let clamped = i64::conditional_select(&clamped, &-l, below);
                s = s.wrapping_add(i128::from(clamped).wrapping_mul(i128::from(vi)));
                squares = squares.wrapping_add(i128::from(vi).wrapping_mul(i128::from(vi)));
            }
            let scaled = squares
                .wrapping_sub(s.wrapping_mul(2))
                .wrapping_mul(*scale as i128);
            numerator = numerator.wrapping_add(scaled);
        }

        // 2^X/M = 2^{n+1}·2^{−(Q−r)/Q}·M^{−1}, with 2^{−(Q−r)/Q} ∈ [1/2, 1).
        let (n, r) = ct_floor_div(numerator, self.denominator);
        let fraction = Fix192::ct_product(&self.exp_table, self.denominator - r);
        let g = fraction.mul(&self.inverse_m);
        let n = i128::conditional_select(&n, &-256, ct_lt_i128(n, -256));
        let n = i128::conditional_select(&n, &2, ct_lt_i128(2, n));
        let m = n + 1;
        let positive = ct_lt_i128(0, m);
        let right = g.ct_shr(i128::conditional_select(&m.wrapping_neg(), &0, positive) as u8);
        let (mut left, mut overflow) = (right, Choice::from(0));
        for shift in 1..=3u32 {
            let (shifted, lost) = g.shl_overflowing(shift);
            let selected = m.ct_eq(&i128::from(shift));
            left = Fix192::conditional_select(&left, &shifted, selected);
            overflow = Choice::conditional_select(&overflow, &lost, selected);
        }
        Acceptance {
            in_box,
            shift_ok,
            saturated: positive & overflow,
            threshold: Fix192::conditional_select(&right, &left, positive),
        }
    }
}

/// Largest attempt cap [`JointRejection::max_attempts`] returns.
const MAX_ATTEMPTS: usize = 1 << 16;

/// The 256-bit product `a·b` as `(high, low)` 128-bit halves.
fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    const LOW: u128 = u64::MAX as u128;
    let (a_high, a_low) = (a >> 64, a & LOW);
    let (b_high, b_low) = (b >> 64, b & LOW);
    let low = a_low * b_low;
    let (middle_a, middle_b) = (a_high * b_low, a_low * b_high);
    let cross = (low >> 64) + (middle_a & LOW) + (middle_b & LOW);
    let high = a_high * b_high + (middle_a >> 64) + (middle_b >> 64) + (cross >> 64);
    (high, (cross << 64) | (low & LOW))
}

/// `x = m · 2^{−shift}` with the top bit of `m` set; `x` must be nonzero.
fn normalize(x: u128) -> (u128, u32) {
    let shift = x.leading_zeros();
    (x << shift, shift)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::print_stdout, reason = "tests report their p-values")]
    #![expect(clippy::unwrap_used, reason = "tests unwrap valid parameters")]
    #![expect(clippy::indexing_slicing, reason = "tests index histograms")]

    use super::*;
    use crate::sample::stats::{chi_square_p, normal_two_sided_p};
    use crate::sample::GaussianSampler;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use std::f64::consts::LN_2;

    const ALPHA: u32 = 16;
    /// The smallest integer `α` with `M < 4` at `ε = 2^{−137}` (`M ≈ 3.99`).
    const ALPHA_NEAR_BOUND: u32 = 10;
    const EPSILON_BITS: u32 = 137;

    fn width(k: u64) -> GaussianWidth {
        GaussianWidth::from_multiplier(k).unwrap()
    }

    fn small_blocks() -> Vec<BlockParams> {
        vec![
            BlockParams {
                width: width(3),
                box_bound: 8,
                shift_bound: 6,
                len: 2,
            },
            BlockParams {
                width: width(5),
                box_bound: 12,
                shift_bound: 6,
                len: 1,
            },
        ]
    }

    /// `M` from the definition in f64.
    fn repetition_rate(alpha: u32) -> f64 {
        let t = (2.0 * f64::from(EPSILON_BITS) * LN_2).sqrt();
        (t / f64::from(alpha) + 1.0 / (2.0 * f64::from(alpha * alpha))).exp()
    }

    /// `ln(π_0(z)/π_v(z)) = Σ_i (v_i² − 2 z_i v_i)/(2σ_{b(i)}²)`.
    fn log_ratio(blocks: &[BlockParams], z: &[i64], v: &[i64]) -> f64 {
        let sigmas = blocks
            .iter()
            .flat_map(|b| std::iter::repeat_n(b.width.sigma(), b.len));
        sigmas
            .zip(z.iter().zip(v))
            .map(|(s, (&zi, &vi))| ((vi * vi) as f64 - 2.0 * (zi * vi) as f64) / (2.0 * s * s))
            .sum()
    }

    fn to_f64(x: &Fix192) -> f64 {
        let [l0, l1, l2] = x.0;
        (l2 as f64 + (l1 as f64 + l0 as f64 / 2f64.powi(64)) / 2f64.powi(64)) / 2f64.powi(64)
    }

    /// mpmath `⌊2^{192}/M⌋` for `α = 16`, `ε = 2^{−137}`.
    #[test]
    fn inverse_repetition_rate_matches_mpmath() {
        assert_eq!(
            inverse_repetition_rate(ALPHA, EPSILON_BITS),
            Fix192([
                0x93f8_0483_22c0_8987,
                0xdfa4_1693_407e_badd,
                0x6bf9_8911_f97d_0900
            ])
        );
    }

    /// mpmath `⌊2^{192}·2^{−2^i/Q}⌋` for `Q = (16,080,468 · 1,005,030)²`, the
    /// headline denominator.
    #[test]
    fn production_denominator_table_matches_mpmath() {
        let blocks = [16_080_468, 1_005_030]
            .map(|k| BlockParams {
                width: width(k),
                box_bound: 1 << 27,
                shift_bound: 1 << 13,
                len: 1 << 22,
            })
            .to_vec();
        let rule = JointRejection::new(blocks, ALPHA, EPSILON_BITS).unwrap();
        assert_eq!(rule.denominator, 261_189_322_840_516_292_736_321_600);
        assert_eq!(rule.exp_table.len(), 88);
        for (i, limbs) in [
            (
                0,
                [
                    0x5b74_9883_8d57_350f,
                    0xffff_ff2d_be4f_d74e,
                    0xffff_ffff_ffff_ffff,
                ],
            ),
            (
                50,
                [
                    0xc7a2_54a2_8203_3dce,
                    0x5d3e_d351_10fb_ba13,
                    0xffff_ffff_fcb6_f93f,
                ],
            ),
            (
                87,
                [
                    0xf4b1_1930_8de2_ad9c,
                    0x6db1_31de_2383_4c04,
                    0xa9c8_6641_d3f2_81ce,
                ],
            ),
        ] {
            assert_eq!(rule.exp_table[i], Fix192(limbs), "i = {i}");
        }
    }

    #[test]
    fn acceptance_probability_matches_definition() {
        for alpha in [ALPHA, ALPHA_NEAR_BOUND] {
            check_acceptance_probability(alpha);
        }
    }

    fn check_acceptance_probability(alpha: u32) {
        let blocks = small_blocks();
        let rule = JointRejection::new(blocks.clone(), alpha, EPSILON_BITS).unwrap();
        let m = repetition_rate(alpha);
        let mut rng = ChaCha20Rng::seed_from_u64(3);
        let (mut interior, mut saturated) = (0, 0);
        for _ in 0..20_000 {
            let mut draw = |range: i64| (rng.next_u64() % (2 * range as u64 + 1)) as i64 - range;
            let z = [draw(10), draw(10), draw(14)];
            let v = [draw(6), draw(6), draw(6)];
            let a = rule.acceptance(&z, &v);
            let inside = z[..2].iter().all(|x| x.abs() <= 8) && z[2].abs() <= 12;
            assert_eq!(bool::from(a.in_box), inside);
            assert!(bool::from(a.shift_ok));
            if !inside {
                continue;
            }
            let exact = log_ratio(&blocks, &z, &v).exp() / m;
            if exact >= 1.0 {
                saturated += 1;
                assert!(bool::from(a.saturated), "z = {z:?}, v = {v:?}");
            } else {
                interior += 1;
                assert!(!bool::from(a.saturated));
                let got = to_f64(&a.threshold);
                assert!(
                    (got - exact).abs() <= 1e-12 * exact,
                    "z = {z:?}, v = {v:?}: {got} vs {exact}"
                );
            }
        }
        assert!(interior > 1000 && saturated > 1000);
    }

    /// The exact discrete Gaussian pmf of width `σ` at `z`.
    fn pmf(sigma: f64, z: i64) -> f64 {
        let rho = |x: i64| (-((x * x) as f64) / (2.0 * sigma * sigma)).exp();
        let total: f64 = (-200..=200).map(rho).sum();
        rho(z) / total
    }

    #[test]
    fn rejection_matches_exact_rate_and_law() {
        check_rate_and_law(ALPHA, 4);
    }

    /// `M ≈ 3.99`, just inside the `M ≤ 4` bound of the saturation clamp.
    #[test]
    fn rejection_matches_exact_rate_and_law_near_m_bound() {
        check_rate_and_law(ALPHA_NEAR_BOUND, 8);
    }

    fn check_rate_and_law(alpha: u32, seed: u64) {
        let blocks = small_blocks();
        let rule = JointRejection::new(blocks.clone(), alpha, EPSILON_BITS).unwrap();
        let m = repetition_rate(alpha);
        let v = [3i64, -5, 6];
        let samplers: Vec<GaussianSampler> = blocks
            .iter()
            .map(|b| GaussianSampler::new(b.width))
            .collect();

        // Accepted mass at z ∈ 𝓑 is min{π_v(z), π_0(z)/M}.
        let (l0, l1) = (8i64, 12i64);
        let index = |z: &[i64; 3]| {
            (((z[0] + l0) * (2 * l0 + 1) + z[1] + l0) * (2 * l1 + 1) + z[2] + l1) as usize
        };
        let cells = ((2 * l0 + 1) * (2 * l0 + 1) * (2 * l1 + 1)) as usize;
        let mut accepted_mass = vec![0.0; cells];
        let sigma = [
            blocks[0].width.sigma(),
            blocks[0].width.sigma(),
            blocks[1].width.sigma(),
        ];
        for a in -l0..=l0 {
            for b in -l0..=l0 {
                for c in -l1..=l1 {
                    let z = [a, b, c];
                    let pi0: f64 = (0..3).map(|i| pmf(sigma[i], z[i])).product();
                    let piv: f64 = (0..3).map(|i| pmf(sigma[i], z[i] - v[i])).product();
                    accepted_mass[index(&z)] = piv.min(pi0 / m);
                }
            }
        }
        let exact_rate: f64 = accepted_mass.iter().sum();

        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let trials = 1_000_000u64;
        let mut accepted = 0u64;
        let mut counts = vec![0u64; cells];
        for _ in 0..trials {
            let y = [
                samplers[0].sample(&mut rng),
                samplers[0].sample(&mut rng),
                samplers[1].sample(&mut rng),
            ];
            let z = [y[0] + v[0], y[1] + v[1], y[2] + v[2]];
            if bool::from(rule.accept(&z, &v, &mut rng).unwrap()) {
                accepted += 1;
                counts[index(&z)] += 1;
            }
        }
        let rate = accepted as f64 / trials as f64;
        let score = (rate - exact_rate) / (exact_rate * (1.0 - exact_rate) / trials as f64).sqrt();
        let p_rate = normal_two_sided_p(score);
        let expected: Vec<f64> = accepted_mass
            .iter()
            .map(|w| w / exact_rate * accepted as f64)
            .collect();
        let p_law = chi_square_p(&counts, &expected);
        println!("alpha = {alpha}: acceptance {rate:.5} vs exact {exact_rate:.5}: p = {p_rate:.4}; accepted law chi-square p = {p_law:.4}");
        assert!(p_rate > 1e-4 && p_law > 1e-4);
    }

    #[test]
    fn rejects_bad_shapes_and_shifts() {
        let rule = JointRejection::new(small_blocks(), ALPHA, EPSILON_BITS).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(5);
        assert_eq!(
            rule.accept(&[0, 0], &[0, 0, 0], &mut rng).err(),
            Some(SampleError::LengthMismatch {
                expected: 3,
                got: 2
            })
        );
        assert_eq!(
            rule.accept(&[0, 0, 0], &[0, 7, 0], &mut rng).err(),
            Some(SampleError::ShiftOutOfRange)
        );
        assert_eq!(
            JointRejection::new(vec![], ALPHA, EPSILON_BITS).err(),
            Some(SampleError::NoBlocks)
        );
        let huge = vec![BlockParams {
            width: width(1 << 31),
            box_bound: 1 << 62,
            shift_bound: 1 << 62,
            len: 1 << 10,
        }];
        assert_eq!(
            JointRejection::new(huge, ALPHA, EPSILON_BITS).err(),
            Some(SampleError::ArithmeticBound)
        );
        let block = |k: u64, bound: u64| BlockParams {
            width: width(k),
            box_bound: bound,
            shift_bound: 1,
            len: 1,
        };
        assert_eq!(
            JointRejection::new(vec![block(3, MAX_BOUND + 1)], ALPHA, EPSILON_BITS).err(),
            Some(SampleError::ArithmeticBound)
        );
        // Q = (2^31)²·(2^31 − 1)²·3² ≥ 2^126.
        let wide = vec![block(1 << 31, 1), block((1 << 31) - 1, 1), block(3, 1)];
        assert_eq!(
            JointRejection::new(wide, ALPHA, EPSILON_BITS).err(),
            Some(SampleError::ArithmeticBound)
        );
    }

    #[test]
    fn rejects_bad_rejection_parameters() {
        // α = 8 gives M ≈ 5.64 > 4, outside the saturation clamp's range.
        for (alpha, epsilon_bits) in [
            (0, EPSILON_BITS),
            (ALPHA, 0),
            (8, EPSILON_BITS),
            (MAX_REJECTION_PARAMETER + 1, EPSILON_BITS),
            (ALPHA, MAX_REJECTION_PARAMETER + 1),
        ] {
            assert_eq!(
                JointRejection::new(small_blocks(), alpha, epsilon_bits).err(),
                Some(SampleError::BadRejectionParameters),
                "alpha = {alpha}, epsilon_bits = {epsilon_bits}"
            );
        }
        assert!(JointRejection::new(small_blocks(), ALPHA_NEAR_BOUND, EPSILON_BITS).is_ok());
    }

    /// `T_att` against the completeness bound in `f64` with the exact `M`:
    /// 163 at the paper's parameters (§6.2), and the same rule at the
    /// near-bound `α`.
    #[test]
    fn max_attempts_meets_the_completeness_bound() {
        for (alpha, expected) in [(ALPHA, 163), (ALPHA_NEAR_BOUND, 0)] {
            let rule = JointRejection::new(small_blocks(), alpha, EPSILON_BITS).unwrap();
            let attempts = rule.max_attempts(1000, 128).unwrap();
            let t = (2.0 * f64::from(EPSILON_BITS) * LN_2).sqrt();
            let m = (t / f64::from(alpha) + 0.5 / f64::from(alpha).powi(2)).exp();
            let failure = 1.0 - (0.999 - (-f64::from(EPSILON_BITS)).exp2()) / m;
            let bits = |n: usize| -(n as f64) * failure.log2();
            assert!(
                bits(attempts) >= 128.0 && bits(attempts - 1) < 128.0,
                "α = {alpha}"
            );
            if expected > 0 {
                assert_eq!(attempts, expected);
            }
        }
        let rule = JointRejection::new(small_blocks(), ALPHA, EPSILON_BITS).unwrap();
        assert!(rule.max_attempts(1, 128).is_err());
    }

    #[test]
    fn wide_product_matches_its_halves() {
        assert_eq!(mul_wide(u128::MAX, u128::MAX), (u128::MAX - 1, 1));
        assert_eq!(mul_wide(1 << 100, 1 << 100), (1 << 72, 0));
        assert_eq!(mul_wide(3, u128::MAX), (2, u128::MAX - 2));
    }
}
