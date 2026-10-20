//! The centred discrete Gaussian `D_σ(z) ∝ exp(−z²/(2σ²))` on Z, for
//! `σ = k·σ_2` with `σ_2 = 1/√(2 ln 2)`.
//!
//! BLISS's binary-Gaussian method (Ducas–Durmus–Lepoint–Lyubashevsky 2013),
//! made isochronous as in Howe–Prest–Ricosset–Rossi 2020. One trial draws
//! `x ∝ 2^{−x²}` on `[0, 13]` by a full CDT scan, `y` uniform on `[0, k)`,
//! sets `z = kx + y`, and accepts with probability `2^{−(y² + 2kxy)/k²}`; a
//! sign bit then maps `D⁺_σ` onto Z, rejecting `−0`. A trial runs a fixed
//! instruction sequence; the loop around it repeats on its accept bit, and a
//! rejection sampler's accept/reject sequence is independent of the value it
//! finally returns.

use rand_core::{CryptoRng, RngCore};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

use super::fixed::{binary_gaussian_cdt, pow2_neg_table, Fix192};
use super::SampleError;

/// `σ_2 = 1/√(2 ln 2)`, the width of the binary Gaussian `∝ 2^{−x²}`.
const BINARY_GAUSSIAN_SIGMA: f64 = 0.849_321_800_288_019_1;

const MAX_MULTIPLIER: u64 = 1 << 31;

/// Thresholds `j ∈ [0, 12]`; `Pr[X ≥ 14] ≈ 2^{−196.6}` is below resolution.
const CDT_LEN: u32 = 13;

/// A Gaussian width `σ = k·σ_2` with integer `k ∈ [1, 2^{31}]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GaussianWidth {
    k: u64,
}

impl GaussianWidth {
    pub fn from_multiplier(k: u64) -> Result<Self, SampleError> {
        if (1..=MAX_MULTIPLIER).contains(&k) {
            Ok(Self { k })
        } else {
            Err(SampleError::WidthOutOfRange)
        }
    }

    pub fn sigma(&self) -> f64 {
        self.k as f64 * BINARY_GAUSSIAN_SIGMA
    }

    /// The hard tail cut: every sample `z = ±(k·x + y)` with `x ≤ 13` and
    /// `y < k` has `|z| < 14k`, so `|z| ≤ 14k − 1`. A bound for exact
    /// arithmetic on samples, never a statistical one.
    pub fn tail_bound(&self) -> u64 {
        (u64::from(CDT_LEN) + 1) * self.k - 1
    }

    /// `k²`, the denominator of the trial's exponent.
    pub(crate) fn k_squared(&self) -> u128 {
        u128::from(self.k) * u128::from(self.k)
    }
}

/// A constant-time sampler for one width. Construction computes its
/// constants from the public width; sampling touches secret data only
/// through constant-time arithmetic.
#[derive(Clone, Debug)]
pub struct GaussianSampler {
    width: GaussianWidth,
    cdt: Vec<Fix192>,
    /// `2^{−2^i/k²}` for every bit `i` of `y² + 2kxy < 27k²`.
    exp_table: Vec<Fix192>,
}

impl GaussianSampler {
    pub fn new(width: GaussianWidth) -> Self {
        let k_squared = width.k_squared();
        let exponent_bits = 128 - (27 * k_squared - 1).leading_zeros();
        Self {
            width,
            cdt: binary_gaussian_cdt(CDT_LEN),
            exp_table: pow2_neg_table(k_squared, exponent_bits),
        }
    }

    pub fn width(&self) -> GaussianWidth {
        self.width
    }

    pub fn sample<R: RngCore + CryptoRng>(&self, rng: &mut R) -> i64 {
        loop {
            let (z, accept) = self.trial(rng);
            if bool::from(accept) {
                return z;
            }
        }
    }

    pub fn fill<R: RngCore + CryptoRng>(&self, rng: &mut R, out: &mut [i64]) {
        for slot in out {
            *slot = self.sample(rng);
        }
    }

    // Kept out of line so the reviewed constant-time code is the code that
    // runs, whatever the caller's inlining.
    #[inline(never)]
    fn trial<R: RngCore>(&self, rng: &mut R) -> (i64, Choice) {
        let k = self.width.k;
        let u = Fix192::random(rng);
        let x: u64 = self
            .cdt
            .iter()
            .map(|threshold| u64::from((!u.ct_lt(threshold)).unwrap_u8()))
            .sum();
        let y = uniform_below(rng, k);
        let z = k * x + y;
        let exponent = u128::from(y) * u128::from(y) + 2 * u128::from(k * x) * u128::from(y);
        let keep = Fix192::random(rng).ct_lt(&Fix192::ct_product(&self.exp_table, exponent));
        let negate = Choice::from((rng.next_u32() & 1) as u8);
        let accept = keep & !(z.ct_eq(&0) & negate);
        let z = z as i64;
        (
            i64::conditional_select(&z, &z.wrapping_neg(), negate),
            accept,
        )
    }
}

/// Uniform on `[0, bound)` for a public `bound ≥ 1`, by masked rejection.
/// The number of draws is independent of the returned value.
fn uniform_below<R: RngCore>(rng: &mut R, bound: u64) -> u64 {
    let mask = u64::MAX
        .checked_shr((bound - 1).leading_zeros())
        .unwrap_or(0);
    loop {
        let candidate = rng.next_u64() & mask;
        if candidate < bound {
            return candidate;
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::print_stdout, reason = "tests report their p-values")]
    #![expect(clippy::unwrap_used, reason = "tests unwrap valid widths")]
    #![expect(clippy::indexing_slicing, reason = "tests index histograms")]

    use super::super::fixed::Fix192;
    use super::*;
    use crate::sample::stats::{chi_square_p, ks_p, normal_cdf};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// mpmath (400-bit) `⌊2^{192}·2^{−2^i/k²}⌋`, little-endian limbs.
    const EXP_VECTORS: [(u64, usize, [u64; 3]); 6] = [
        (
            16_080_468,
            0,
            [
                0xec5f_af25_d97b_3072,
                0x1aae_016c_7eca_fe3d,
                0xffff_ffff_ffff_3ed8,
            ],
        ),
        (
            16_080_468,
            20,
            [
                0xb12c_1d6d_66f9_5e39,
                0xf55a_5c48_597d_f2a9,
                0xffff_fff3_ed81_ab28,
            ],
        ),
        (
            16_080_468,
            52,
            [
                0xaf92_8c80_2f66_5871,
                0x01c8_3ac9_c3d2_8171,
                0x0000_5fe6_0f7b_783e,
            ],
        ),
        (
            1_005_030,
            44,
            [
                0x3129_7937_cdaf_da66,
                0xdf83_f61a_b111_5c3f,
                0x0000_5fe6_80b8_ad73,
            ],
        ),
        (
            3,
            0,
            [
                0xc45c_c465_d068_67fa,
                0x7a6e_01a4_677f_45e0,
                0xed06_2009_77a2_65f0,
            ],
        ),
        (
            3,
            3,
            [
                0x458f_6452_3602_ad20,
                0xc196_0e85_ec87_2371,
                0x8a3f_65c4_2266_8a2e,
            ],
        ),
    ];

    /// mpmath `⌊2^{192}·Pr[X ≤ j]⌋` for `Pr[X = x] ∝ 2^{−x²}`.
    const CDT_VECTORS: [(usize, [u64; 3]); 4] = [
        (
            0,
            [
                0x15ff_f7fa_3d27_1d0e,
                0x85db_b564_c791_ff8b,
                0xa3a2_446c_95b5_b009,
            ],
        ),
        (
            1,
            [
                0xa0ff_f3f7_5bba_ab95,
                0x48c9_9017_2b5a_ff50,
                0xf573_66a2_e090_880e,
            ],
        ),
        (
            5,
            [
                0xe075_cefd_f4a1_eb3b,
                0x70cb_c3e6_cd77_623e,
                0xffff_ffff_f5c5_89e7,
            ],
        ),
        (
            12,
            [
                0xffff_ffff_ffae_2edd,
                0xffff_ffff_ffff_ffff,
                0xffff_ffff_ffff_ffff,
            ],
        ),
    ];

    #[test]
    fn constants_match_mpmath() {
        for (k, i, limbs) in EXP_VECTORS {
            let sampler = GaussianSampler::new(GaussianWidth::from_multiplier(k).unwrap());
            assert_eq!(sampler.exp_table[i], Fix192(limbs), "k = {k}, i = {i}");
        }
        let cdt = binary_gaussian_cdt(CDT_LEN);
        for (j, limbs) in CDT_VECTORS {
            assert_eq!(cdt[j], Fix192(limbs), "j = {j}");
        }
    }

    /// The table has exactly one factor per bit of the largest exponent
    /// `27k² − 1`.
    #[test]
    fn exponent_tables_cover_every_trial() {
        for k in [1, 3, 1_005_030, 16_080_468, MAX_MULTIPLIER] {
            let sampler = GaussianSampler::new(GaussianWidth::from_multiplier(k).unwrap());
            let len = sampler.exp_table.len() as u32;
            let max_exponent = 27 * u128::from(k) * u128::from(k) - 1;
            assert!(
                1u128 << (len - 1) <= max_exponent && max_exponent < 1u128 << len,
                "k = {k}"
            );
        }
    }

    #[test]
    fn rejects_out_of_range_multipliers() {
        for k in [0, MAX_MULTIPLIER + 1] {
            assert_eq!(
                GaussianWidth::from_multiplier(k),
                Err(SampleError::WidthOutOfRange)
            );
        }
    }

    /// Exact `D_σ` pmf on `[−bound, bound]` from the definition.
    fn exact_pmf(sigma: f64, bound: i64) -> Vec<f64> {
        let weights: Vec<f64> = (-bound..=bound)
            .map(|z| (-((z * z) as f64) / (2.0 * sigma * sigma)).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        weights.into_iter().map(|w| w / total).collect()
    }

    #[test]
    fn small_widths_match_exact_pmf() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        for k in [1, 2, 3, 10] {
            let sampler = GaussianSampler::new(GaussianWidth::from_multiplier(k).unwrap());
            let sigma = sampler.width().sigma();
            let bound = (6.0 * sigma).ceil() as i64;
            let pmf = exact_pmf(sigma, bound);
            let n = 1_000_000;
            let mut counts = vec![0u64; pmf.len()];
            let mut outside = 0u64;
            for _ in 0..n {
                let z = sampler.sample(&mut rng);
                match usize::try_from(z + bound)
                    .ok()
                    .filter(|&i| i < counts.len())
                {
                    Some(i) => counts[i] += 1,
                    None => outside += 1,
                }
            }
            let expected: Vec<f64> = pmf.iter().map(|p| p * n as f64).collect();
            let p = chi_square_p(&counts, &expected);
            println!(
                "k = {k}: chi-square over {} cells, p = {p:.4}, outside 6σ = {outside}",
                pmf.len()
            );
            assert!(p > 1e-4, "k = {k}: p = {p}");
            assert!(outside < 10, "k = {k}: {outside} samples beyond 6σ");
        }
    }

    #[test]
    fn production_widths_match_normal_law() {
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        for k in [16_080_468u64, 1_005_030] {
            let sampler = GaussianSampler::new(GaussianWidth::from_multiplier(k).unwrap());
            let sigma = sampler.width().sigma();
            let n = 2_000_000usize;
            let mut samples = vec![0i64; n];
            sampler.fill(&mut rng, &mut samples);

            // For σ ≫ η_ε(Z), the mass of the integers in [a, b] equals the
            // normal mass of [a − 1/2, b + 1/2] to within f64 precision.
            let cdf = |z: f64| normal_cdf((z + 0.5) / sigma);
            let edges: Vec<i64> = (-20..=20)
                .map(|j| (j as f64 * sigma / 4.0).round() as i64)
                .collect();
            let mut counts = vec![0u64; edges.len() + 1];
            for &z in &samples {
                counts[edges.partition_point(|&e| e <= z)] += 1;
            }
            let mut expected = Vec::with_capacity(counts.len());
            let mut previous = 0.0;
            for &e in &edges {
                let c = cdf(e as f64 - 1.0);
                expected.push((c - previous) * n as f64);
                previous = c;
            }
            expected.push((1.0 - previous) * n as f64);
            let p_chi = chi_square_p(&counts, &expected);

            samples.sort_unstable();
            let d = samples
                .iter()
                .enumerate()
                .map(|(i, &z)| {
                    let c = cdf(z as f64);
                    (c - i as f64 / n as f64)
                        .abs()
                        .max(((i + 1) as f64 / n as f64 - c).abs())
                })
                .fold(0.0, f64::max);
            let p_ks = ks_p(d, n);

            let mean = samples.iter().map(|&z| z as f64).sum::<f64>() / n as f64;
            let m2 = samples.iter().map(|&z| (z as f64).powi(2)).sum::<f64>() / n as f64;
            let m4 = samples.iter().map(|&z| (z as f64).powi(4)).sum::<f64>() / n as f64;
            let sigma2 = sigma * sigma;
            // Standard errors under the normal law: σ/√n, σ²√(2/n), and
            // √(96/n) for the kurtosis m4/σ⁴.
            let z_mean = mean / (sigma / (n as f64).sqrt());
            let z_var = (m2 - sigma2) / (sigma2 * (2.0 / n as f64).sqrt());
            let z_kurt = (m4 / (sigma2 * sigma2) - 3.0) / (96.0 / n as f64).sqrt();
            println!(
                "k = {k}: chi-square p = {p_chi:.4}, KS D = {d:.2e} p = {p_ks:.4}, \
                 mean z = {z_mean:.2}, variance z = {z_var:.2}, kurtosis z = {z_kurt:.2}"
            );
            assert!(p_chi > 1e-4 && p_ks > 1e-4, "k = {k}");
            assert!(
                z_mean.abs() < 5.0 && z_var.abs() < 5.0 && z_kurt.abs() < 5.0,
                "k = {k}"
            );
        }
    }

    /// Every sample lies within the hard tail cut, which the samples come
    /// close to at small widths.
    #[test]
    fn samples_stay_within_the_tail_bound() {
        let mut rng = ChaCha20Rng::seed_from_u64(77);
        for k in [1u64, 2, 3, 1000] {
            let width = GaussianWidth::from_multiplier(k).unwrap();
            let sampler = GaussianSampler::new(width);
            let bound = width.tail_bound();
            let largest = (0..200_000)
                .map(|_| sampler.sample(&mut rng).unsigned_abs())
                .max()
                .unwrap();
            assert!(largest <= bound, "k = {k}: |z| = {largest} > {bound}");
        }
        assert_eq!(GaussianWidth::from_multiplier(1).unwrap().tail_bound(), 13);
    }
}
