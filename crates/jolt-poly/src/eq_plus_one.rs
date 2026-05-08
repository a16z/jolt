//! Equality-plus-one polynomial for shift sumcheck.
//!
//! The MLE `eq+1(x, y)` evaluates to 1 when `y = x + 1` (as integers in
//! `[0, 2^l − 2]`) and 0 otherwise. There is no wrap-around: when `x` is all
//! ones (the maximum), the polynomial outputs 0 for every `y`.
//!
//! This is used in the Spartan shift sumcheck to relate polynomial evaluations
//! at consecutive cycles.
//!
//! Both `x` and `y` are in **big-endian** bit ordering (`point[0]` = MSB).

use jolt_field::Field;

use crate::thread::unsafe_allocate_zero_vec;
use crate::EqPolynomial;

/// MLE evaluating to 1 iff `y = x + 1` (no wrap at `2^l − 1`).
///
/// Stores a fixed point `x` in big-endian order. Call [`evaluate`](Self::evaluate)
/// to compute `eq+1(x, y)` at any `y`.
pub struct EqPlusOnePolynomial<F: Field> {
    /// Fixed point (big-endian: `point[0]` = MSB).
    point: Vec<F>,
}

impl<F: Field> EqPlusOnePolynomial<F> {
    pub fn new(point: Vec<F>) -> Self {
        Self { point }
    }

    /// Evaluates `eq+1(x, y)` at a single point `y` (big-endian).
    ///
    /// The identity decomposes by suffix length `k` of consecutive 1-bits in `x`:
    /// - The bottom `k` bits of `x` are 1 and the corresponding bits of `y` are 0.
    /// - Bit `k` flips: `x[k] = 0`, `y[k] = 1`.
    /// - All higher bits match.
    pub fn evaluate(&self, y: &[F]) -> F {
        let l = self.point.len();
        let x = &self.point;
        assert_eq!(y.len(), l, "point length mismatch");
        let one = F::one();

        (0..l)
            .map(|k| {
                let lower_bits: F = (0..k)
                    .map(|i| x[l - 1 - i] * (one - y[l - 1 - i]))
                    .product();
                let flip = (one - x[l - 1 - k]) * y[l - 1 - k];
                let higher_bits: F = ((k + 1)..l)
                    .map(|i| {
                        x[l - 1 - i] * y[l - 1 - i] + (one - x[l - 1 - i]) * (one - y[l - 1 - i])
                    })
                    .product();
                lower_bits * flip * higher_bits
            })
            .sum()
    }

    /// Computes full evaluation tables `(eq_evals, eq_plus_one_evals)` over the
    /// Boolean hypercube, where:
    ///
    /// - `eq_evals[j] = eq(r, j)`
    /// - `eq_plus_one_evals[j] = eq+1(r, j)`
    ///
    /// Both tables are indexed in big-endian order: `j = 0` corresponds to
    /// the all-zeros vertex.
    ///
    /// The `eq` table is built incrementally prefix-by-prefix. At each step
    /// the `eq+1` contribution for the new bit position is derived from the
    /// partial `eq` table and a product of the remaining `r` coordinates.
    pub fn evals(r: &[F], scaling_factor: Option<F>) -> (Vec<F>, Vec<F>) {
        let ell = r.len();
        let size = 1usize << ell;
        let mut eq_evals: Vec<F> = unsafe_allocate_zero_vec(size);
        eq_evals[0] = scaling_factor.unwrap_or(F::one());
        let mut eq_plus_one_evals: Vec<F> = unsafe_allocate_zero_vec(size);

        // Build tables incrementally. After processing bit i, the eq table
        // encodes a prefix of length i+1, stored at strided positions.
        //
        // At each step:
        // 1. Derive eq+1 contributions from the current eq prefix.
        // 2. Extend the eq table by one more variable r[i].
        for i in 0..ell {
            let step = 1usize << (ell - i);
            let half_step = step / 2;

            // r_lower_product = (1 - r[i]) · Π_{j > i} r[j]
            let mut r_lower_product = F::one();
            for &x in r.iter().skip(i + 1) {
                r_lower_product *= x;
            }
            r_lower_product *= F::one() - r[i];

            // Fill eq+1 entries for bit position i.
            let mut idx = half_step;
            while idx < size {
                eq_plus_one_evals[idx] = eq_evals[idx - half_step] * r_lower_product;
                idx += step;
            }

            // Extend eq table by variable r[i].
            // The eq table after i steps has 2^i nonzero entries at stride 2^(ell-i).
            // After extension, it has 2^(i+1) entries at stride 2^(ell-i-1).
            // Selected indices: 0, eq_step, 2·eq_step, ... where eq_step = 2^(ell-i-1).
            // Pairs: (k, k+eq_step) → eq[k+eq_step] = eq[k]·r[i]; eq[k] -= eq[k+eq_step].
            let eq_step = 1usize << (ell - i - 1);
            let mut k = 0;
            while k < size {
                let val = eq_evals[k] * r[i];
                eq_evals[k + eq_step] = val;
                eq_evals[k] -= val;
                k += eq_step * 2;
            }
        }

        (eq_evals, eq_plus_one_evals)
    }
}

/// Prefix-suffix decomposition of `eq+1` for sumcheck optimization.
///
/// Decomposes `eq+1((r_hi, r_lo), (y_hi, y_lo))` into two rank-1 terms:
///
/// ```text
///   prefix_0(y_lo) · suffix_0(y_hi) + prefix_1(y_lo) · suffix_1(y_hi)
/// ```
///
/// where `r = (r_hi, r_lo)` is split at the midpoint. This enables the first
/// half of the shift sumcheck to operate on √N-sized buffers rather than N.
///
/// See <https://eprint.iacr.org/2025/611.pdf> (Appendix A).
pub struct EqPlusOnePrefixSuffix<F: Field> {
    /// Evals of `eq+1(r_lo, j)` for `j ∈ {0,1}^{n/2}`.
    pub prefix_0: Vec<F>,
    /// Evals of `eq(r_hi, j)` for `j ∈ {0,1}^{n/2}`.
    pub suffix_0: Vec<F>,
    /// `is_max(r_lo) · is_min(j)` — nonzero only at `j = 0`.
    pub prefix_1: Vec<F>,
    /// Evals of `eq+1(r_hi, j)` for `j ∈ {0,1}^{n/2}`.
    pub suffix_1: Vec<F>,
}

impl<F: Field> EqPlusOnePrefixSuffix<F> {
    /// Creates the decomposition from a big-endian point `r`.
    ///
    /// Splits at `r.len() / 2`: the first half is `r_hi`, the second is `r_lo`.
    pub fn new(r: &[F]) -> Self {
        let mid = r.len() / 2;
        let (r_hi, r_lo) = r.split_at(mid);

        // is_max(r_lo) = eq((1,...,1), r_lo) = Π r_lo[i]
        let ones: Vec<F> = vec![F::one(); r_lo.len()];
        let is_max_eval = EqPolynomial::<F>::mle(&ones, r_lo);

        let mut prefix_1 = crate::thread::unsafe_allocate_zero_vec(1 << r_lo.len());
        prefix_1[0] = is_max_eval;

        let (suffix_0, suffix_1) = EqPlusOnePolynomial::evals(r_hi, None);

        Self {
            prefix_0: EqPlusOnePolynomial::evals(r_lo, None).1,
            suffix_0,
            prefix_1,
            suffix_1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn index_to_bits(idx: usize, n: usize) -> Vec<Fr> {
        (0..n)
            .map(|i| {
                if (idx >> (n - 1 - i)) & 1 == 1 {
                    Fr::one()
                } else {
                    Fr::zero()
                }
            })
            .collect()
    }

    #[test]
    fn successor_at_boolean_points() {
        // For l=3, eq+1(x, y) = 1 iff y = x + 1 (no wrap at 7).
        let l = 3;
        for x_int in 0..(1 << l) {
            let x_bits = index_to_bits(x_int, l);
            let eq_plus_one = EqPlusOnePolynomial::new(x_bits);

            for y_int in 0..(1 << l) {
                let y_bits = index_to_bits(y_int, l);
                let val = eq_plus_one.evaluate(&y_bits);
                if x_int < (1 << l) - 1 && y_int == x_int + 1 {
                    assert_eq!(val, Fr::one(), "eq+1({x_int}, {y_int}) should be 1");
                } else {
                    assert!(val.is_zero(), "eq+1({x_int}, {y_int}) should be 0");
                }
            }
        }
    }

    #[test]
    fn no_wraparound_at_max() {
        // eq+1(all_ones, 0) = 0 (no wrap-around).
        let l = 4;
        let x = vec![Fr::one(); l];
        let y = vec![Fr::zero(); l];
        let eq_plus_one = EqPlusOnePolynomial::new(x);
        assert!(eq_plus_one.evaluate(&y).is_zero());
    }

    #[test]
    fn evals_table_matches_pointwise() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let l = 4;
        let r: Vec<Fr> = (0..l).map(|_| Fr::random(&mut rng)).collect();

        let (eq_evals, eq_plus_one_evals) = EqPlusOnePolynomial::evals(&r, None);

        let eq_poly = EqPolynomial::new(r.clone());
        let eq_plus_one_poly = EqPlusOnePolynomial::new(r);

        for idx in 0..(1 << l) {
            let bits = index_to_bits(idx, l);
            assert_eq!(
                eq_evals[idx],
                eq_poly.evaluate(&bits),
                "eq mismatch at {idx}"
            );
            assert_eq!(
                eq_plus_one_evals[idx],
                eq_plus_one_poly.evaluate(&bits),
                "eq+1 mismatch at {idx}"
            );
        }
    }

    #[test]
    fn evals_with_scaling() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let l = 3;
        let r: Vec<Fr> = (0..l).map(|_| Fr::random(&mut rng)).collect();
        let scale = Fr::from_u64(5);

        let (eq_unscaled, _) = EqPlusOnePolynomial::evals(&r, None);
        let (eq_scaled, _) = EqPlusOnePolynomial::evals(&r, Some(scale));

        for (u, s) in eq_unscaled.iter().zip(eq_scaled.iter()) {
            assert_eq!(*u * scale, *s);
        }
    }

    #[test]
    fn prefix_suffix_matches_direct() {
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let l = 4;
        let r: Vec<Fr> = (0..l).map(|_| Fr::random(&mut rng)).collect();

        let eq_plus_one_direct = EqPlusOnePolynomial::new(r.clone());

        let ps = EqPlusOnePrefixSuffix::new(&r);

        // Verify at a random evaluation point y = (y_hi, y_lo).
        let y: Vec<Fr> = (0..l).map(|_| Fr::random(&mut rng)).collect();
        let (y_hi, y_lo) = y.split_at(l / 2);

        let p0_eval = crate::Polynomial::new(ps.prefix_0).evaluate(y_lo);
        let s0_eval = crate::Polynomial::new(ps.suffix_0).evaluate(y_hi);
        let p1_eval = crate::Polynomial::new(ps.prefix_1).evaluate(y_lo);
        let s1_eval = crate::Polynomial::new(ps.suffix_1).evaluate(y_hi);

        let via_decomp = p0_eval * s0_eval + p1_eval * s1_eval;
        let via_direct = eq_plus_one_direct.evaluate(&y);
        assert_eq!(via_decomp, via_direct);
    }

    #[test]
    fn prefix_suffix_multiple_random_points() {
        let mut rng = ChaCha20Rng::seed_from_u64(456);
        for l in [4, 6, 8] {
            let r: Vec<Fr> = (0..l).map(|_| Fr::random(&mut rng)).collect();
            let direct = EqPlusOnePolynomial::new(r.clone());
            let ps = EqPlusOnePrefixSuffix::new(&r);

            for _ in 0..5 {
                let y: Vec<Fr> = (0..l).map(|_| Fr::random(&mut rng)).collect();
                let (y_hi, y_lo) = y.split_at(l / 2);

                let p0 = crate::Polynomial::new(ps.prefix_0.clone()).evaluate(y_lo);
                let s0 = crate::Polynomial::new(ps.suffix_0.clone()).evaluate(y_hi);
                let p1 = crate::Polynomial::new(ps.prefix_1.clone()).evaluate(y_lo);
                let s1 = crate::Polynomial::new(ps.suffix_1.clone()).evaluate(y_hi);

                assert_eq!(
                    p0 * s0 + p1 * s1,
                    direct.evaluate(&y),
                    "decomposition mismatch for l={l}"
                );
            }
        }
    }

    #[test]
    fn eq_plus_one_sum_over_hypercube() {
        // For random r, sum_y eq+1(r, y) should equal 1 - eq(r, max).
        // Because eq+1 maps x → x+1 for x in [0, 2^l-2], so it covers
        // all y in [1, 2^l-1], missing y=0 and hitting y=(2^l-1) only if
        // x=(2^l-2). The sum should be 1 - Π r_i (the missing all-ones term).
        let mut rng = ChaCha20Rng::seed_from_u64(789);
        let l = 5;
        let r: Vec<Fr> = (0..l).map(|_| Fr::random(&mut rng)).collect();

        let (_, eq_plus_one_evals) = EqPlusOnePolynomial::evals(&r, None);
        let sum: Fr = eq_plus_one_evals.iter().copied().sum();

        let r_product: Fr = r.iter().copied().product();
        let expected = Fr::one() - r_product;
        assert_eq!(sum, expected);
    }
}
