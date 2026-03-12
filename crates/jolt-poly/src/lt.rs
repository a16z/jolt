//! Less-than polynomial for value accumulation sumchecks.
//!
//! The MLE `LT(x, y)` evaluates to 1 on Boolean inputs when `x < y` as
//! integers and 0 otherwise. Its multilinear extension is:
//!
//! $$\text{LT}(x, y) = \sum_{i} (1 - x_i) \cdot y_i \cdot \text{eq}(x_{i+1:}, y_{i+1:})$$
//!
//! where the sum runs from MSB to LSB (big-endian bit ordering).
//!
//! Used in the register/RAM value evaluation sumcheck to accumulate writes
//! that occurred before a given cycle point.
//!
//! # Split optimization
//!
//! Rather than materializing the full `2^n` table and binding it each round
//! (O(n·2^n) total work, O(2^n) memory), `LtPolynomial` splits the point
//! `r` at the midpoint into `(r_hi, r_lo)` and stores three √N-sized tables:
//!
//! ```text
//! LT(j, r) = LT(j_hi, r_hi) + eq(j_hi, r_hi) · LT(j_lo, r_lo)
//! ```
//!
//! where `j = (j_hi, j_lo)`. Binding proceeds HighToLow: first all hi vars
//! (shrinking `lt_hi` and `eq_hi`), then all lo vars (shrinking `lt_lo`).
//! Total memory stays at 3 · √N throughout.

use jolt_field::Field;

use crate::EqPolynomial;

/// Split less-than polynomial for efficient sumcheck binding.
///
/// Stores three sub-tables of size ≤ √N each, reconstructing full-table
/// values on demand via `LT[j] = lt_hi[j_hi] + eq_hi[j_hi] · lt_lo[j_lo]`.
///
/// Supports HighToLow binding only (MSB first).
pub struct LtPolynomial<F: Field> {
    lt_lo: Vec<F>,
    lt_hi: Vec<F>,
    eq_hi: Vec<F>,
    n_lo_vars: usize,
    n_hi_vars: usize,
}

impl<F: Field> LtPolynomial<F> {
    /// Creates a split LT polynomial for the fixed point `r` (big-endian).
    ///
    /// Splits at `r.len() / 2`: the first half is `r_hi`, the second is `r_lo`.
    /// For odd-length `r`, `r_hi` gets the extra variable.
    pub fn new(r: &[F]) -> Self {
        let mid = r.len() / 2;
        let (r_hi, r_lo) = r.split_at(r.len() - mid);

        Self {
            lt_lo: lt_evals(r_lo),
            lt_hi: lt_evals(r_hi),
            eq_hi: EqPolynomial::new(r_hi.to_vec()).evaluations(),
            n_lo_vars: r_lo.len(),
            n_hi_vars: r_hi.len(),
        }
    }

    /// Total number of remaining variables.
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.n_hi_vars + self.n_lo_vars
    }

    /// Effective table size `2^num_vars`.
    #[inline]
    pub fn len(&self) -> usize {
        self.lt_hi.len() * self.lt_lo.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lt_hi.is_empty()
    }

    /// Reconstructs `LT[idx]` from the split tables.
    #[inline]
    fn get(&self, idx: usize) -> F {
        let lo_size = self.lt_lo.len();
        let i_hi = idx / lo_size;
        let i_lo = idx % lo_size;
        self.lt_hi[i_hi] + self.eq_hi[i_hi] * self.lt_lo[i_lo]
    }

    /// Returns `(LT[j], LT[j + half])` for HighToLow sumcheck pairing.
    ///
    /// In the hi-binding phase, the pairing splits across hi-table halves.
    /// In the lo-binding phase (all hi vars bound), it splits across lo-table halves.
    #[inline]
    pub fn sumcheck_eval_pair(&self, j: usize) -> (F, F) {
        let half = self.len() / 2;
        (self.get(j), self.get(j + half))
    }

    /// Binds the MSB (HighToLow), halving the effective table size.
    pub fn bind(&mut self, challenge: F) {
        if self.n_hi_vars > 0 {
            bind_in_place(&mut self.lt_hi, challenge);
            bind_in_place(&mut self.eq_hi, challenge);
            self.n_hi_vars -= 1;
        } else {
            assert!(self.n_lo_vars > 0, "no variables left to bind");
            bind_in_place(&mut self.lt_lo, challenge);
            self.n_lo_vars -= 1;
        }
    }

    /// Materializes the full `2^n` evaluation table `[LT(0, r), ..., LT(2^n - 1, r)]`.
    ///
    /// Big-endian index order: `j = 0` corresponds to the all-zeros vertex.
    pub fn evaluations(r: &[F]) -> Vec<F> {
        lt_evals(r)
    }

    /// Evaluates `LT(x, r)` at a single point without materializing the full table.
    ///
    /// Computes `Σ_i (1 - x_i) · r_i · eq(x[0..i], r[0..i])` iteratively
    /// from MSB to LSB, accumulating the prefix eq product.
    ///
    /// Both `x` and `r` are big-endian. Time: O(n). Space: O(1).
    pub fn evaluate(x: &[F], r: &[F]) -> F {
        assert_eq!(x.len(), r.len(), "LT point dimension mismatch");
        let mut lt = F::zero();
        let mut eq_prefix = F::one();
        for (&xi, &ri) in x.iter().zip(r.iter()) {
            lt += (F::one() - xi) * ri * eq_prefix;
            eq_prefix *= xi * ri + (F::one() - xi) * (F::one() - ri);
        }
        lt
    }
}

/// Materializes `[LT(0, r), LT(1, r), ..., LT(2^n - 1, r)]` in big-endian order.
///
/// Uses an in-place doubling construction. For each bit position `i` (LSB to MSB):
/// - Left half `x`: `x' = x + r_i - x·r_i` (accumulates `(1-x_i)·r_i·eq_suffix`)
/// - Right half `y`: `y' = x·r_i` (propagates eq term through x_i=1)
///
/// Time: O(n·2^n). Space: O(2^n).
fn lt_evals<F: Field>(r: &[F]) -> Vec<F> {
    let n = r.len();
    let mut evals = vec![F::zero(); 1usize << n];
    for (i, &ri) in r.iter().rev().enumerate() {
        let (left, right) = evals.split_at_mut(1 << i);
        left.iter_mut().zip(right.iter_mut()).for_each(|(x, y)| {
            *y = *x * ri;
            *x += ri - *y;
        });
    }
    evals
}

/// In-place HighToLow bind: `v[j] = v[j] + challenge · (v[j+half] - v[j])`.
#[inline]
fn bind_in_place<F: Field>(v: &mut Vec<F>, challenge: F) {
    let half = v.len() / 2;
    for j in 0..half {
        let lo = v[j];
        let hi = v[j + half];
        v[j] = lo + challenge * (hi - lo);
    }
    v.truncate(half);
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
    fn boolean_correctness() {
        // LT(x, r) = 1 iff x < r on Boolean inputs.
        for n in 1..=5 {
            for r_int in 0..(1u64 << n) {
                let r_bits = index_to_bits(r_int as usize, n);
                let table = LtPolynomial::evaluations(&r_bits);

                for x_int in 0..(1u64 << n) {
                    let expected = if x_int < r_int { Fr::one() } else { Fr::zero() };
                    assert_eq!(
                        table[x_int as usize], expected,
                        "LT({x_int}, {r_int}) wrong for n={n}"
                    );
                }
            }
        }
    }

    #[test]
    fn evaluations_matches_inline() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        for n in 2..=6 {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let table = LtPolynomial::evaluations(&r);

            for (idx, &entry) in table.iter().enumerate() {
                let x = index_to_bits(idx, n);
                let inline = LtPolynomial::evaluate(&x, &r);
                assert_eq!(entry, inline, "mismatch at idx={idx}, n={n}");
            }
        }
    }

    #[test]
    fn split_matches_full_table() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        for n in 2..=8 {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let full_table = LtPolynomial::evaluations(&r);
            let split = LtPolynomial::new(&r);

            assert_eq!(split.len(), full_table.len());
            for (j, &expected) in full_table.iter().enumerate() {
                assert_eq!(split.get(j), expected, "split mismatch at j={j}, n={n}");
            }
        }
    }

    #[test]
    fn sumcheck_eval_pair_matches_full_table() {
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        for n in 2..=7 {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let full_table = LtPolynomial::evaluations(&r);
            let split = LtPolynomial::new(&r);

            let half = full_table.len() / 2;
            for (j, (&expected_lo, &expected_hi)) in full_table[..half]
                .iter()
                .zip(full_table[half..].iter())
                .enumerate()
            {
                let (lo, hi) = split.sumcheck_eval_pair(j);
                assert_eq!(lo, expected_lo, "lo mismatch at j={j}, n={n}");
                assert_eq!(hi, expected_hi, "hi mismatch at j={j}, n={n}");
            }
        }
    }

    #[test]
    fn sequential_bind_converges() {
        // Bind all variables → single scalar = evaluate(challenges, r).
        let mut rng = ChaCha20Rng::seed_from_u64(200);
        for n in 2..=8 {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let challenges: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

            let mut split = LtPolynomial::new(&r);
            for &c in &challenges {
                split.bind(c);
            }

            assert_eq!(split.num_vars(), 0);
            assert_eq!(split.len(), 1);
            let final_val = split.get(0);

            let expected = LtPolynomial::evaluate(&challenges, &r);
            assert_eq!(final_val, expected, "bind convergence failed for n={n}");
        }
    }

    #[test]
    fn bind_matches_full_table_bind() {
        // Verify that binding the split matches binding the full table.
        let mut rng = ChaCha20Rng::seed_from_u64(300);
        for n in 3..=7 {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let challenge = Fr::random(&mut rng);

            let mut split = LtPolynomial::new(&r);
            split.bind(challenge);

            let mut full = LtPolynomial::evaluations(&r);
            bind_in_place(&mut full, challenge);

            assert_eq!(split.len(), full.len());
            for (j, &expected) in full.iter().enumerate() {
                assert_eq!(split.get(j), expected, "post-bind mismatch at j={j}, n={n}");
            }
        }
    }

    #[test]
    fn multi_round_bind_matches_full_table() {
        // Bind several rounds and verify each round matches the full table.
        let mut rng = ChaCha20Rng::seed_from_u64(400);
        let n = 6;
        let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let challenges: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let mut split = LtPolynomial::new(&r);
        let mut full = LtPolynomial::evaluations(&r);

        for (round, &c) in challenges.iter().enumerate() {
            split.bind(c);
            bind_in_place(&mut full, c);

            assert_eq!(split.len(), full.len(), "size mismatch after round {round}");
            for (j, &expected) in full.iter().enumerate() {
                assert_eq!(
                    split.get(j),
                    expected,
                    "mismatch at j={j} after round {round}"
                );
            }
        }
    }

    #[test]
    fn inline_evaluate_matches_table() {
        let mut rng = ChaCha20Rng::seed_from_u64(500);
        for n in 2..=6 {
            let x: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

            let via_table = {
                let table = LtPolynomial::evaluations(&r);
                let eq_evals = EqPolynomial::new(x.clone()).evaluations();
                table
                    .iter()
                    .zip(eq_evals.iter())
                    .map(|(&t, &e)| t * e)
                    .sum::<Fr>()
            };

            let via_inline = LtPolynomial::evaluate(&x, &r);
            assert_eq!(via_table, via_inline, "inline vs table mismatch for n={n}");
        }
    }

    #[test]
    fn sum_over_hypercube() {
        // Σ_x LT(x, r) = r interpreted as an integer in [0, 2^n).
        // Actually: Σ_x LT(x, r) for Boolean x gives the number of x < r,
        // but for random r this is a multilinear extension.
        // Simple check: for Boolean r, the sum should equal r_int.
        for n in 1..=5 {
            for r_int in 0..(1u64 << n) {
                let r_bits = index_to_bits(r_int as usize, n);
                let table = LtPolynomial::evaluations(&r_bits);
                let sum: Fr = table.iter().copied().sum();
                assert_eq!(
                    sum,
                    Fr::from_u64(r_int),
                    "hypercube sum wrong for r={r_int}, n={n}"
                );
            }
        }
    }

    #[test]
    fn odd_num_vars() {
        // Verify split works correctly when n is odd (hi gets extra var).
        let mut rng = ChaCha20Rng::seed_from_u64(600);
        for n in [3, 5, 7] {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let full_table = LtPolynomial::evaluations(&r);
            let split = LtPolynomial::new(&r);

            let mid = n / 2;
            assert_eq!(split.n_hi_vars, n - mid);
            assert_eq!(split.n_lo_vars, mid);

            for (j, &expected) in full_table.iter().enumerate() {
                assert_eq!(split.get(j), expected, "odd split mismatch at j={j}, n={n}");
            }
        }
    }
}
