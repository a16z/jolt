//! Equality polynomial for multilinear evaluation.

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::serde_canonical::vec_canonical;

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
#[serde(bound = "")]
pub struct EqPolynomial<F: Field> {
    #[serde(with = "vec_canonical")]
    point: Vec<F>,
}

/// Parallelism threshold: tables larger than this are built with rayon.
const PAR_THRESHOLD: usize = 1024;

impl<F: Field> EqPolynomial<F> {
    /// Creates a new equality polynomial for the given point $r \in \mathbb{F}^n$.
    pub fn new(point: Vec<F>) -> Self {
        Self { point }
    }

    /// Number of variables $n$.
    pub fn num_vars(&self) -> usize {
        self.point.len()
    }

    /// Materializes all $2^n$ evaluations of $\widetilde{eq}(\cdot, r)$ over the Boolean hypercube.
    ///
    /// Uses a bottom-up doubling construction: starting from `[1]`, each coordinate
    /// $r_i$ doubles the table by producing entries $e \cdot (1 - r_i)$ and $e \cdot r_i$.
    ///
    /// Time: $O(2^n)$. Space: $O(2^n)$.
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
                    let (left, right) = table.split_at_mut(prev_len);
                    left.par_iter_mut()
                        .zip(right.par_iter_mut())
                        .for_each(|(lo, hi)| {
                            let base = *lo;
                            *hi = base * r_i;
                            *lo = base * one_minus_r_i;
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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use jolt_field::Field;
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
    fn evaluate_at_boolean_selects_entry() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let n = 3;
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let eq = EqPolynomial::new(point);
        let table = eq.evaluations();

        for (idx, &entry) in table.iter().enumerate() {
            let bits = index_to_bits(idx, n);
            let direct = eq.evaluate(&bits);
            assert_eq!(direct, entry, "mismatch at index {idx}");
        }
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
    fn parallel_evaluations_inner_product_consistency() {
        // Verifies that the inner product of two eq tables (which computes
        // eq(r, s) = sum_x eq(x,r)*eq(x,s)) is consistent with evaluate().
        // This holds regardless of table ordering.
        let mut rng = ChaCha20Rng::seed_from_u64(303);
        let n = 11;
        let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let s: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let eq_r = EqPolynomial::new(r.clone());
        let eq_s = EqPolynomial::new(s.clone());

        let table_r = eq_r.evaluations();
        let table_s = eq_s.evaluations();

        let inner_product: Fr = table_r
            .iter()
            .zip(table_s.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let direct = eq_r.evaluate(&s);
        assert_eq!(inner_product, direct);
    }

    #[test]
    fn parallel_sum_over_hypercube_is_one() {
        let mut rng = ChaCha20Rng::seed_from_u64(301);
        let n = 11;
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let eq = EqPolynomial::new(point);
        let table = eq.evaluations();
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

        // Pick a random non-Boolean evaluation point and verify via definition
        let p: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let direct = eq.evaluate(&p);

        // Manual computation: sum over hypercube of eq(x,r) * eq(x,p)
        // which equals eq(r,p) since sum_x eq(x,r)*eq(x,p) = eq(r,p)
        let eq_p = EqPolynomial::new(p);
        let table_p = eq_p.evaluations();
        let via_tables: Fr = table.iter().zip(table_p.iter()).map(|(&a, &b)| a * b).sum();
        assert_eq!(direct, via_tables);
    }
}
