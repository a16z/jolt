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
    pub fn new(
        blocks: Vec<BlockParams>,
        alpha: u32,
        epsilon_bits: u32,
    ) -> Result<Self, SampleError> {
        if blocks.is_empty() {
            return Err(SampleError::NoBlocks);
        }
        if alpha == 0 || epsilon_bits == 0 {
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
            inverse_m: inverse_repetition_rate(alpha, epsilon_bits),
            blocks,
            scales,
            denominator,
            total_len,
        })
    }

    /// Applies the rule to `z = y + v`, where `v = cs` is the exact
    /// centred-integer negacyclic product. Both are laid out block-major.
    /// Every coordinate is processed and the accept bit is the only value
    /// derived from them that leaves.
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

    const ALPHA: u32 = 16;
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
    fn repetition_rate() -> f64 {
        let t = (2.0 * f64::from(EPSILON_BITS) * std::f64::consts::LN_2).sqrt();
        (t / f64::from(ALPHA) + 1.0 / (2.0 * f64::from(ALPHA * ALPHA))).exp()
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
        let blocks = small_blocks();
        let rule = JointRejection::new(blocks.clone(), ALPHA, EPSILON_BITS).unwrap();
        let m = repetition_rate();
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
        let blocks = small_blocks();
        let rule = JointRejection::new(blocks.clone(), ALPHA, EPSILON_BITS).unwrap();
        let m = repetition_rate();
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

        let mut rng = ChaCha20Rng::seed_from_u64(4);
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
        println!("acceptance {rate:.5} vs exact {exact_rate:.5}: p = {p_rate:.4}; accepted law chi-square p = {p_law:.4}");
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
    }
}
