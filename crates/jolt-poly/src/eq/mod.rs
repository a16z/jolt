use crate::MultilinearEvaluation;
use jolt_field::Field;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub mod alloc;
mod build;
mod cached;
mod scalar;
use jolt_field::JoltField;
pub use scalar::{
    boolean_bits_msb, boolean_index_msb, boolean_point_msb, eq_index, eq_index_msb, try_eq_mle,
};

/// Equality polynomial $\widetilde{eq}(x, r) = \prod_{i=1}^{n}(r_i x_i + (1-r_i)(1-x_i))$.
///
/// Given a fixed point $r \in \mathbb{F}^n$, the equality polynomial evaluates to 1
/// when $x = r$ on the Boolean hypercube and 0 at all other Boolean points. Its
/// multilinear extension interpolates these values over all of $\mathbb{F}^n$.
///
/// This polynomial is fundamental to sumcheck-based protocols where it selects
/// a single evaluation from a multilinear polynomial:
/// $$f(r) = \sum_{x \in \{0,1\}^n} f(x) \cdot \widetilde{eq}(x, r)$$
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: DeserializeOwned"))]
pub struct EqPolynomial<F: Field> {
    point: Vec<F>,
}

#[cfg(feature = "parallel")]
const PAR_THRESHOLD: usize = 1024;

impl<F: Field> EqPolynomial<F> {
    pub fn new(point: Vec<F>) -> Self {
        Self { point }
    }

    pub fn num_vars(&self) -> usize {
        self.point.len()
    }

    /// Materializes all $2^n$ evaluations of $\widetilde{eq}(\cdot, r)$ over the Boolean hypercube.
    ///
    /// Uses a bottom-up doubling construction: starting from `[1]`, each coordinate
    /// $r_i$ doubles the table by producing entries $e \cdot (1 - r_i)$ and $e \cdot r_i$.
    ///
    /// Time: $O(2^n)$. Space: $O(2^n)$.
    #[tracing::instrument(skip_all, name = "EqPolynomial::evaluations")]
    pub fn evaluations(&self) -> Vec<F> {
        let n = self.point.len();
        let size = 1usize << n;
        let mut table = Vec::with_capacity(size);
        table.push(F::one());

        for &r_i in &self.point {
            let one_minus_r_i = F::one() - r_i;
            let prev_len = table.len();

            table.resize(prev_len * 2, F::zero());

            // Process in reverse to avoid overwriting entries we still need.
            // After this loop, table[2*j] = old[j] * (1 - r_i) and
            // table[2*j+1] = old[j] * r_i.
            #[cfg(feature = "parallel")]
            {
                if prev_len >= PAR_THRESHOLD {
                    use rayon::prelude::*;
                    let prev: Vec<F> = table[..prev_len].to_vec();
                    let dest = &mut table[..prev_len * 2];
                    dest.par_chunks_mut(2)
                        .zip(prev.par_iter())
                        .for_each(|(pair, &base)| {
                            pair[0] = base * one_minus_r_i;
                            pair[1] = base * r_i;
                        });
                } else {
                    for j in (0..prev_len).rev() {
                        let base = table[j];
                        table[2 * j] = base * one_minus_r_i;
                        table[2 * j + 1] = base * r_i;
                    }
                }
            }

            #[cfg(not(feature = "parallel"))]
            {
                for j in (0..prev_len).rev() {
                    let base = table[j];
                    table[2 * j] = base * one_minus_r_i;
                    table[2 * j + 1] = base * r_i;
                }
            }
        }

        table
    }

    /// Evaluates $\widetilde{eq}(p, r)$ at a single point without materializing the full table.
    ///
    /// Computes the product $\prod_{i} (r_i \cdot p_i + (1 - r_i)(1 - p_i))$ directly.
    ///
    /// Time: $O(n)$. Space: $O(1)$.
    #[inline]
    pub fn evaluate(&self, point: &[F]) -> F {
        assert_eq!(
            self.point.len(),
            point.len(),
            "eq polynomial dimension mismatch"
        );
        self.point
            .iter()
            .zip(point.iter())
            .fold(F::one(), |acc, (&r_i, &p_i)| {
                acc * (r_i * p_i + (F::one() - r_i) * (F::one() - p_i))
            })
    }
}

impl<F: JoltField> MultilinearEvaluation<F> for EqPolynomial<F> {
    fn num_vars(&self) -> usize {
        self.point.len()
    }

    fn len(&self) -> usize {
        1 << self.point.len()
    }

    fn evaluate(&self, point: &[F]) -> F {
        EqPolynomial::evaluate(self, point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_field::{Field, Ring};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn suffix_sums_match_dense_boolean_tables() {
        let mut rng = ChaCha20Rng::seed_from_u64(849);
        for bits in 0..9 {
            for kind in 0..3 {
                let point: Vec<Fr> = (0..bits)
                    .map(|_| match kind {
                        0 => Fr::zero(),
                        1 => Fr::one(),
                        _ => Fr::random(&mut rng),
                    })
                    .collect();
                let table: Vec<Fr> = EqPolynomial::evals(&point, None);
                for index in 0..table.len() {
                    assert_eq!(
                        EqPolynomial::suffix_sum(&point, index),
                        table[index..].iter().copied().sum::<Fr>()
                    );
                }
                assert_eq!(EqPolynomial::suffix_sum(&point, table.len()), Fr::zero());
            }
        }
    }

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
    fn sum_over_hypercube_is_one() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 4;
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let eq = EqPolynomial::new(point);
        let table = eq.evaluations();
        let sum: Fr = table.iter().copied().sum();
        assert_eq!(sum, Fr::one());
    }

    #[test]
    fn evaluations_matches_evaluate_pointwise() {
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let n = 5;
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let eq = EqPolynomial::new(point);
        let table = eq.evaluations();

        for (idx, &entry) in table.iter().enumerate() {
            let bits = index_to_bits(idx, n);
            assert_eq!(entry, eq.evaluate(&bits));
        }
    }

    #[test]
    fn parallel_evaluations_sum_is_one() {
        // num_vars=11 -> 2048 entries, above PAR_THRESHOLD=1024
        // Verifies the parallel path produces a valid eq table whose entries sum to 1.
        let mut rng = ChaCha20Rng::seed_from_u64(300);
        let n = 11;
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let eq = EqPolynomial::new(point);
        let table = eq.evaluations();

        assert_eq!(table.len(), 1 << n);
        let sum: Fr = table.iter().copied().sum();
        assert_eq!(sum, Fr::one());
    }

    #[test]
    fn evaluate_cross_verification_random_point() {
        let mut rng = ChaCha20Rng::seed_from_u64(302);
        let n = 6;
        let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let eq = EqPolynomial::new(r);
        let table = eq.evaluations();

        let p: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let direct = eq.evaluate(&p);

        let eq_p = EqPolynomial::new(p);
        let table_p = eq_p.evaluations();
        let via_tables: Fr = table.iter().zip(table_p.iter()).map(|(&a, &b)| a * b).sum();
        assert_eq!(direct, via_tables);
    }

    #[test]
    fn eq_at_boolean_point_is_one() {
        for n in 1..=5 {
            for idx in 0..(1 << n) {
                let bits = index_to_bits(idx, n);
                let eq = EqPolynomial::new(bits.clone());
                assert_eq!(
                    eq.evaluate(&bits),
                    Fr::one(),
                    "eq(b, b) != 1 for n={n}, idx={idx}"
                );
            }
        }
    }

    #[test]
    fn eq_at_distinct_boolean_points_is_zero() {
        let n = 3;
        for i in 0..(1 << n) {
            for j in 0..(1 << n) {
                if i == j {
                    continue;
                }
                let bi = index_to_bits(i, n);
                let bj = index_to_bits(j, n);
                let eq = EqPolynomial::new(bi);
                assert!(
                    eq.evaluate(&bj).is_zero(),
                    "eq(b_i, b_j) != 0 for i={i}, j={j}"
                );
            }
        }
    }

    #[test]
    fn evals_serial_matches_instance() {
        let mut rng = ChaCha20Rng::seed_from_u64(400);
        for n in 1..=8 {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let instance = EqPolynomial::new(r.clone()).evaluations();
            let via_static = EqPolynomial::<Fr>::evals_serial(&r, None);
            assert_eq!(instance, via_static, "mismatch for n={n}");
        }
    }

    #[test]
    fn evals_parallel_matches_serial() {
        let mut rng = ChaCha20Rng::seed_from_u64(401);
        for n in [5, 10, 12] {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let serial = EqPolynomial::<Fr>::evals_serial(&r, None);
            let parallel = EqPolynomial::<Fr>::evals_parallel(&r, None);
            assert_eq!(serial, parallel, "serial vs parallel mismatch for n={n}");
        }
    }

    #[test]
    fn evals_cached_msb_prefix_consistency() {
        let mut rng = ChaCha20Rng::seed_from_u64(402);
        for n in 2..=10 {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let cached = EqPolynomial::<Fr>::evals_cached_msb(&r, None);
            assert_eq!(cached.len(), n + 1);
            assert_eq!(cached[0], vec![Fr::one()]);
            for i in 0..=n {
                assert_eq!(cached[i].len(), 1 << i);
                let direct = EqPolynomial::<Fr>::evals_serial(&r[..i], None);
                assert_eq!(cached[i], direct, "cached[{i}] mismatch for n={n}");
            }
        }
    }

    #[test]
    fn mle_static_matches_instance_evaluate() {
        let mut rng = ChaCha20Rng::seed_from_u64(404);
        let n = 5;
        let x: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let y: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let via_instance = EqPolynomial::new(x.clone()).evaluate(&y);
        let via_static = EqPolynomial::<Fr>::mle(&x, &y);
        assert_eq!(via_instance, via_static);
    }

    #[test]
    fn parallel_evaluations_pointwise_correctness() {
        // Verifies that the parallel path in evaluations() produces the correct
        // entry at every index — catches layout mismatches (blocked vs interleaved).
        let mut rng = ChaCha20Rng::seed_from_u64(500);
        let n = 12; // 4096 entries, well above PAR_THRESHOLD=1024
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let eq = EqPolynomial::new(point);
        let table = eq.evaluations();
        assert_eq!(table.len(), 1 << n);

        for (idx, &entry) in table.iter().enumerate() {
            let bits = index_to_bits(idx, n);
            let expected = eq.evaluate(&bits);
            assert_eq!(entry, expected, "mismatch at index {idx}");
        }
    }

    #[test]
    fn evals_with_scaling() {
        let mut rng = ChaCha20Rng::seed_from_u64(406);
        let n = 4;
        let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let scale = Fr::from_u64(7);

        let unscaled = EqPolynomial::<Fr>::evals_serial(&r, None);
        let scaled = EqPolynomial::<Fr>::evals_serial(&r, Some(scale));

        for (u, s) in unscaled.iter().zip(scaled.iter()) {
            assert_eq!(*u * scale, *s);
        }
    }
}

#[cfg(test)]
mod endianness_tests;

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests assert valid nonzero tags and allocations"
)]
mod allocation_tests;

#[cfg(test)]
mod failure_policy;
#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "policy contract tests assert errors and successful allocations"
)]
mod policy_tests;
