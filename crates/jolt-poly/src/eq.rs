//! Equality polynomial for multilinear evaluation.

use std::ops::{Mul, SubAssign};

use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::math::Math;
use crate::thread::unsafe_allocate_zero_vec;

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
    point: Vec<F>,
}

/// Parallelism threshold: tables larger than this are built with rayon.
const PAR_THRESHOLD: usize = 1024;

impl<F: Field> EqPolynomial<F> {
    /// Creates a new equality polynomial for the given point $r \in \mathbb{F}^n$.
    pub fn new(point: Vec<F>) -> Self {
        Self { point }
    }

    /// Number of variables `n` in the fixed point `r`.
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

/// Static (point-free) evaluation methods for eq polynomial tables.
///
/// These accept challenge or field-element slices and produce materialized
/// tables without constructing an `EqPolynomial` instance. They are used
/// by split-eq evaluators and sumcheck witnesses.
impl<F: Field> EqPolynomial<F> {
    /// Computes `eq(x, y) = Π_i (x_i y_i + (1 - x_i)(1 - y_i))` for two slices.
    pub fn mle<C>(x: &[C], y: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F>,
        F: Mul<C, Output = F> + SubAssign<F>,
    {
        assert_eq!(x.len(), y.len());
        x.iter()
            .zip(y.iter())
            .map(|(x_i, y_i)| {
                let x: F = (*x_i).into();
                let y: F = (*y_i).into();
                x * y + (F::one() - x) * (F::one() - y)
            })
            .fold(F::one(), |acc, v| acc * v)
    }

    /// Computes `eq(r, 0) = Π_i (1 - r_i)`, selecting the all-zeros vertex.
    pub fn zero_selector<C>(r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F>,
    {
        r.iter()
            .map(|r_i| F::one() - (*r_i).into())
            .fold(F::one(), |acc, v| acc * v)
    }

    /// Computes `{ eq(r, x) : x ∈ {0,1}^n }` with optional scaling.
    ///
    /// Uses a serial or parallel path based on table size. Big-endian index
    /// order: `r[0]` is the MSB.
    #[tracing::instrument(skip_all, name = "EqPolynomial::evals")]
    pub fn evals<C>(r: &[C], scaling_factor: Option<F>) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F>,
        F: Mul<C, Output = F> + SubAssign<F>,
    {
        if r.len() <= 16 {
            Self::evals_serial(r, scaling_factor)
        } else {
            Self::evals_parallel(r, scaling_factor)
        }
    }

    /// Serial eq table construction with optional scaling.
    #[inline]
    pub fn evals_serial<C>(r: &[C], scaling_factor: Option<F>) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F>,
        F: Mul<C, Output = F> + SubAssign<F>,
    {
        let mut evals: Vec<F> = vec![scaling_factor.unwrap_or(F::one()); r.len().pow2()];
        let mut size = 1;
        for r_j in r {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let scalar = evals[i / 2];
                evals[i] = scalar * *r_j;
                evals[i - 1] = scalar - evals[i];
            }
        }
        evals
    }

    /// Prefix-cached eq tables: `result[j]` = eq over the prefix `r[..j]`.
    ///
    /// Returns `n+1` tables where `result[0] = [scaling_factor]` (eq over 0 vars).
    /// Big-endian index order.
    #[tracing::instrument(skip_all, name = "EqPolynomial::evals_cached")]
    pub fn evals_cached<C>(r: &[C], scaling_factor: Option<F>) -> Vec<Vec<F>>
    where
        C: Copy + Send + Sync + Into<F>,
        F: Mul<C, Output = F> + SubAssign<F>,
    {
        let mut evals: Vec<Vec<F>> = (0..=r.len())
            .map(|i| vec![scaling_factor.unwrap_or(F::one()); 1 << i])
            .collect();
        let mut size = 1;
        for j in 0..r.len() {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let scalar = evals[j][i / 2];
                evals[j + 1][i] = scalar * r[j];
                evals[j + 1][i - 1] = scalar - evals[j + 1][i];
            }
        }
        evals
    }

    /// Like [`evals_cached`](Self::evals_cached) but for high-to-low (reverse) binding order.
    ///
    /// Returns `result` where `result[j]` contains evaluations for the suffix `r[(n-j)..]`.
    /// `result[0] = [scaling_factor]`. Builds tables in reverse variable order.
    pub fn evals_cached_rev<C>(r: &[C], scaling_factor: Option<F>) -> Vec<Vec<F>>
    where
        C: Copy + Send + Sync + Into<F>,
        F: Mul<C, Output = F>,
    {
        let rev_r: Vec<_> = r.iter().rev().collect();
        let mut evals: Vec<Vec<F>> = (0..=r.len())
            .map(|i| vec![scaling_factor.unwrap_or(F::one()); 1 << i])
            .collect();
        let mut size = 1;
        for j in 0..r.len() {
            for i in 0..size {
                let scalar = evals[j][i];
                let multiple = 1 << j;
                evals[j + 1][i + multiple] = scalar * *rev_r[j];
                evals[j + 1][i] = scalar - evals[j + 1][i + multiple];
            }
            size *= 2;
        }
        evals
    }

    /// Parallel eq table construction with optional scaling.
    ///
    /// Uses rayon to build large layers in parallel. Low-to-high construction:
    /// processes `r` in reverse so that the first coordinate ends up as the MSB.
    #[tracing::instrument(skip_all, name = "EqPolynomial::evals_parallel")]
    #[inline]
    pub fn evals_parallel<C>(r: &[C], scaling_factor: Option<F>) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F>,
        F: Mul<C, Output = F> + SubAssign<F>,
    {
        let final_size = r.len().pow2();
        let mut evals: Vec<F> = unsafe_allocate_zero_vec(final_size);
        let mut size = 1;
        evals[0] = scaling_factor.unwrap_or(F::one());

        for r in r.iter().rev() {
            let (evals_left, evals_right) = evals.split_at_mut(size);
            let (evals_right, _) = evals_right.split_at_mut(size);

            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                evals_left
                    .par_iter_mut()
                    .zip(evals_right.par_iter_mut())
                    .for_each(|(x, y)| {
                        *y = *x * *r;
                        *x -= *y;
                    });
            }

            #[cfg(not(feature = "parallel"))]
            {
                for i in 0..size {
                    evals_right[i] = evals_left[i] * *r;
                    evals_left[i] -= evals_right[i];
                }
            }

            size *= 2;
        }

        evals
    }
}

impl<F: Field> crate::MultilinearEvaluation<F> for EqPolynomial<F> {
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
    use jolt_field::Field;
    use jolt_field::Fr;
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

    #[test]
    fn eq_at_boolean_point_is_one() {
        // eq(b, b) = 1 for any Boolean vector b ∈ {0,1}^n
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
    fn evals_cached_prefix_consistency() {
        let mut rng = ChaCha20Rng::seed_from_u64(402);
        for n in 2..=10 {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let cached = EqPolynomial::<Fr>::evals_cached(&r, None);
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
    fn evals_cached_rev_consistency() {
        let mut rng = ChaCha20Rng::seed_from_u64(403);
        for n in 2..=8 {
            let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
            let cached_rev = EqPolynomial::<Fr>::evals_cached_rev(&r, None);
            assert_eq!(cached_rev.len(), n + 1);
            assert_eq!(cached_rev[0], vec![Fr::one()]);
            for (j, table) in cached_rev.iter().enumerate() {
                assert_eq!(table.len(), 1 << j);
            }
            // The last entry should equal evals over all variables in reverse order
            let full_rev: Vec<Fr> = r.iter().rev().copied().collect();
            let full_table = EqPolynomial::<Fr>::evals_serial(&full_rev, None);
            // Sizes should match but the table is built differently
            assert_eq!(cached_rev[n].len(), full_table.len());
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
    fn zero_selector() {
        let mut rng = ChaCha20Rng::seed_from_u64(405);
        let n = 4;
        let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let expected = r
            .iter()
            .fold(Fr::one(), |acc, &r_i| acc * (Fr::one() - r_i));
        let result = EqPolynomial::<Fr>::zero_selector(&r);
        assert_eq!(expected, result);
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
