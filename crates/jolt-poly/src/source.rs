//! Abstract multilinear polynomial trait and compositions.
//!
//! [`MultilinearPoly`] is the core abstraction over multilinear polynomials
//! in evaluation form. Implementations range from dense evaluation tables
//! ([`Polynomial<F>`](crate::Polynomial)) to structured sparse representations
//! ([`OneHotPolynomial`](crate::OneHotPolynomial)) to lazy compositions ([`RlcSource`]). The trait
//! decouples polynomial *access* from *storage*, enabling streaming opening
//! proofs where the full $2^n$ table never resides in memory simultaneously.
//!
//! [`RlcSource`] composes multiple polynomials via random linear combination
//! without materializing the combined table. Its [`fold_rows`] distributes
//! across constituents, avoiding allocation of the combined table.
//!
//! [`MultilinearPoly`]: trait.MultilinearPoly.html
//! [`RlcSource`]: struct.RlcSource.html
//! [`OneHotPolynomial`]: crate::OneHotPolynomial
//! [`fold_rows`]: trait.MultilinearPoly.html#method.fold_rows

use jolt_field::Field;

use crate::Polynomial;

/// A multilinear polynomial $f : \{0,1\}^n \to \mathbb{F}$ in evaluation form.
///
/// The evaluation table can be viewed as a $(2^\nu \times 2^\sigma)$ matrix
/// where $\nu + \sigma = n$. Implementations range from dense evaluation
/// tables ([`Polynomial<F>`](crate::Polynomial)) to structured sparse forms
/// ([`OneHotPolynomial`](crate::OneHotPolynomial)) to lazy compositions ([`RlcSource`]).
///
/// Core operations:
/// - [`num_vars`](Self::num_vars) / [`evaluate`](Self::evaluate): metadata and point evaluation
/// - [`for_each_row`](Self::for_each_row): row-wise iteration (streaming commit, row-based MSM)
/// - [`fold_rows`](Self::fold_rows): matrix-vector product $v \cdot M$ (opening protocols)
/// - [`is_sparse`](Self::is_sparse) / [`for_each_nonzero`](Self::for_each_nonzero): sparsity
///   hints for PCS commit optimization (e.g., batch addition instead of MSM)
pub trait MultilinearPoly<F: Field>: Send + Sync {
    /// Number of variables $n$. The polynomial has $2^n$ evaluations.
    fn num_vars(&self) -> usize;

    /// Evaluates $f(r)$ at an arbitrary point $r \in \mathbb{F}^n$.
    fn evaluate(&self, point: &[F]) -> F;

    /// Iterates over the evaluation table in row-major order.
    ///
    /// The table is interpreted as a $(2^\nu \times 2^\sigma)$ matrix where
    /// $\sigma$ is the number of column variables and $\nu = n - \sigma$.
    /// The closure receives `(row_index, row_data)` pairs in order.
    ///
    /// For in-memory polynomials, rows are borrowed slices (zero-copy).
    /// For lazy sources, each row may be computed on-the-fly.
    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F]));

    /// Folds a left vector against the $(2^\nu \times 2^\sigma)$ matrix form.
    ///
    /// Computes:
    /// $$\text{result}\[c\] = \sum_{r=0}^{2^\nu - 1} \text{left}\[r\] \cdot M\[r\]\[c\]$$
    ///
    /// where $M\[r\]\[c\] = f(\text{bits}(r \cdot 2^\sigma + c))$ and
    /// $\nu = n - \sigma$.
    ///
    /// The default implementation iterates rows via [`for_each_row`](Self::for_each_row).
    /// Implementations with distributable structure (e.g., [`RlcSource`]) or
    /// sparse representations (e.g., one-hot polynomials) should override
    /// for better performance.
    ///
    /// # Panics
    ///
    /// Panics if `left.len() != 2^(num_vars - sigma)`.
    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        let num_cols = 1usize << sigma;
        let mut result = vec![F::zero(); num_cols];
        self.for_each_row(sigma, &mut |row_idx, row| {
            let l = left[row_idx];
            for (r, &val) in result.iter_mut().zip(row.iter()) {
                *r += l * val;
            }
        });
        result
    }

    /// Whether this polynomial has sparse structure that allows more efficient
    /// commitment (e.g., batch affine addition instead of full MSM).
    ///
    /// When true, PCS backends should use [`for_each_nonzero`](Self::for_each_nonzero)
    /// to access only the nonzero entries.
    fn is_sparse(&self) -> bool {
        false
    }

    /// Iterates over nonzero entries as `(flat_index, value)` pairs.
    ///
    /// For dense polynomials, the default scans the full table. Structured
    /// sparse types (e.g., [`OneHotPolynomial`](crate::OneHotPolynomial)) yield only O(T) entries.
    fn for_each_nonzero(&self, f: &mut dyn FnMut(usize, F)) {
        let n = self.num_vars();
        let total = 1usize << n;
        self.for_each_row(n, &mut |_, row| {
            for (i, &val) in row.iter().take(total).enumerate() {
                if !val.is_zero() {
                    f(i, val);
                }
            }
        });
    }
}

impl<F: Field> MultilinearPoly<F> for Polynomial<F> {
    #[inline]
    fn num_vars(&self) -> usize {
        Polynomial::num_vars(self)
    }

    fn evaluate(&self, point: &[F]) -> F {
        Polynomial::evaluate(self, point)
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F])) {
        let num_cols = 1usize << sigma;
        for (i, row) in self.evaluations().chunks(num_cols).enumerate() {
            f(i, row);
        }
    }

    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        let num_cols = 1usize << sigma;
        let evals = self.evaluations();
        debug_assert_eq!(
            left.len(),
            evals.len() / num_cols,
            "left vector length must equal number of rows"
        );

        let mut result = vec![F::zero(); num_cols];
        for (row_idx, row) in evals.chunks(num_cols).enumerate() {
            let l = left[row_idx];
            for (r, &val) in result.iter_mut().zip(row.iter()) {
                *r += l * val;
            }
        }
        result
    }
}

impl<F: Field> MultilinearPoly<F> for [F] {
    #[inline]
    fn num_vars(&self) -> usize {
        if self.is_empty() {
            return 0;
        }
        assert!(
            self.len().is_power_of_two(),
            "slice length must be a power of two, got {}",
            self.len()
        );
        self.len().trailing_zeros() as usize
    }

    fn evaluate(&self, point: &[F]) -> F {
        let eq_evals = crate::EqPolynomial::new(point.to_vec()).evaluations();
        self.iter().zip(eq_evals.iter()).map(|(&f, &e)| f * e).sum()
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F])) {
        let num_cols = 1usize << sigma;
        for (i, row) in self.chunks(num_cols).enumerate() {
            f(i, row);
        }
    }

    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        let num_cols = 1usize << sigma;
        let mut result = vec![F::zero(); num_cols];
        for (row_idx, row) in self.chunks(num_cols).enumerate() {
            let l = left[row_idx];
            for (r, &val) in result.iter_mut().zip(row.iter()) {
                *r += l * val;
            }
        }
        result
    }
}

impl<F: Field> MultilinearPoly<F> for Vec<F> {
    #[inline]
    fn num_vars(&self) -> usize {
        self.as_slice().num_vars()
    }

    fn evaluate(&self, point: &[F]) -> F {
        self.as_slice().evaluate(point)
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F])) {
        self.as_slice().for_each_row(sigma, f);
    }

    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        self.as_slice().fold_rows(left, sigma)
    }
}

/// Lazy RLC composition of multilinear polynomials.
///
/// Represents $f(x) = \sum_{i=0}^{k-1} s_i \cdot f_i(x)$ without
/// materializing the combined evaluation table. Operations distribute
/// over the constituents:
///
/// - [`evaluate`](MultilinearPoly::evaluate): $\sum_i s_i \cdot f_i(r)$
/// - [`fold_rows`](MultilinearPoly::fold_rows): $\sum_i s_i \cdot (v \cdot M_i)$ —
///   each polynomial computes its own fold, results are combined with scalars.
///   No evaluation table is ever materialized.
pub struct RlcSource<F: Field, S: MultilinearPoly<F>> {
    sources: Vec<S>,
    scalars: Vec<F>,
    num_vars: usize,
}

impl<F: Field, S: MultilinearPoly<F>> RlcSource<F, S> {
    /// Creates a lazy RLC composition.
    ///
    /// # Panics
    ///
    /// Panics if `sources` and `scalars` have different lengths, or if
    /// sources have inconsistent `num_vars`.
    pub fn new(sources: Vec<S>, scalars: Vec<F>) -> Self {
        assert_eq!(sources.len(), scalars.len());
        let num_vars = sources.first().map_or(0, |s| s.num_vars());
        debug_assert!(
            sources.iter().all(|s| s.num_vars() == num_vars),
            "all sources must have the same num_vars"
        );
        Self {
            sources,
            scalars,
            num_vars,
        }
    }

    pub fn sources(&self) -> &[S] {
        &self.sources
    }

    pub fn scalars(&self) -> &[F] {
        &self.scalars
    }
}

impl<F: Field, S: MultilinearPoly<F>> MultilinearPoly<F> for RlcSource<F, S> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, point: &[F]) -> F {
        self.sources
            .iter()
            .zip(&self.scalars)
            .map(|(source, &scalar)| scalar * source.evaluate(point))
            .fold(F::zero(), |acc, x| acc + x)
    }

    /// Iterates over combined rows by collecting each source's rows and combining.
    ///
    /// Memory: O(k × 2^σ) where k = number of sources.
    /// For streaming-critical paths, prefer [`fold_rows`](Self::fold_rows) which
    /// distributes without materializing any rows.
    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F])) {
        if self.sources.is_empty() {
            return;
        }

        let num_cols = 1usize << sigma;
        let nu = self.num_vars.saturating_sub(sigma);
        let num_rows = 1usize << nu;

        // Collect all rows from all sources.
        // Each inner vec has num_rows entries, each of length num_cols.
        let all_rows: Vec<Vec<Vec<F>>> = self
            .sources
            .iter()
            .map(|source| {
                let mut rows = Vec::with_capacity(num_rows);
                source.for_each_row(sigma, &mut |_idx, row| {
                    rows.push(row.to_vec());
                });
                rows
            })
            .collect();

        let mut combined = vec![F::zero(); num_cols];
        for row_idx in 0..num_rows {
            combined.fill(F::zero());
            for (source_rows, &scalar) in all_rows.iter().zip(&self.scalars) {
                for (dst, &val) in combined.iter_mut().zip(source_rows[row_idx].iter()) {
                    *dst += scalar * val;
                }
            }
            f(row_idx, &combined);
        }
    }

    /// Distributes fold_rows across constituent sources.
    ///
    /// Computes $\sum_i s_i \cdot (v \cdot M_i)$ by having each source
    /// independently compute its own fold. No evaluation table is
    /// ever materialized — this is the key streaming win.
    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        let num_cols = 1usize << sigma;
        let mut result = vec![F::zero(); num_cols];
        for (source, &scalar) in self.sources.iter().zip(&self.scalars) {
            let contribution = source.fold_rows(left, sigma);
            for (r, &c) in result.iter_mut().zip(contribution.iter()) {
                *r += scalar * c;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use num_traits::Zero;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn polynomial_for_each_row_matches_chunks() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let poly = Polynomial::<Fr>::random(4, &mut rng);
        let sigma = 2;
        let num_cols = 1usize << sigma;

        let mut rows = Vec::new();
        poly.for_each_row(sigma, &mut |_idx, row| {
            rows.push(row.to_vec());
        });

        assert_eq!(rows.len(), poly.len() / num_cols);
        for (i, row) in rows.iter().enumerate() {
            let start = i * num_cols;
            assert_eq!(row.as_slice(), &poly.evaluations()[start..start + num_cols]);
        }
    }

    #[test]
    fn polynomial_fold_rows_matches_manual_vmp() {
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        let num_vars = 4;
        let sigma = 2;
        let nu = num_vars - sigma;
        let num_cols = 1usize << sigma;
        let num_rows = 1usize << nu;

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let left: Vec<Fr> = (0..num_rows).map(|_| Fr::random(&mut rng)).collect();

        let result = poly.fold_rows(&left, sigma);

        // Manual VMP
        let mut expected = vec![Fr::zero(); num_cols];
        for (row, &l) in left.iter().enumerate() {
            for (col, dest) in expected.iter_mut().enumerate() {
                *dest += l * poly.evaluations()[row * num_cols + col];
            }
        }

        assert_eq!(result, expected);
    }

    #[test]
    fn rlc_source_evaluate_matches_manual() {
        let mut rng = ChaCha20Rng::seed_from_u64(10);
        let num_vars = 3;

        let p1 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let p2 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let s1 = Fr::random(&mut rng);
        let s2 = Fr::random(&mut rng);

        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        let rlc = RlcSource::new(vec![p1.clone(), p2.clone()], vec![s1, s2]);
        let result = rlc.evaluate(&point);
        let expected = s1 * p1.evaluate(&point) + s2 * p2.evaluate(&point);

        assert_eq!(result, expected);
    }

    #[test]
    fn rlc_source_fold_rows_matches_materialized() {
        let mut rng = ChaCha20Rng::seed_from_u64(20);
        let num_vars = 4;
        let sigma = 2;
        let nu = num_vars - sigma;
        let num_rows = 1usize << nu;

        let p1 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let p2 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let s1 = Fr::random(&mut rng);
        let s2 = Fr::random(&mut rng);
        let left: Vec<Fr> = (0..num_rows).map(|_| Fr::random(&mut rng)).collect();

        // Lazy fold
        let rlc = RlcSource::new(vec![p1.clone(), p2.clone()], vec![s1, s2]);
        let lazy_result = rlc.fold_rows(&left, sigma);

        // Materialized fold
        let combined_evals: Vec<Fr> = p1
            .evaluations()
            .iter()
            .zip(p2.evaluations().iter())
            .map(|(&a, &b)| s1 * a + s2 * b)
            .collect();
        let combined = Polynomial::new(combined_evals);
        let materialized_result = combined.fold_rows(&left, sigma);

        assert_eq!(lazy_result, materialized_result);
    }

    #[test]
    fn rlc_source_for_each_row_matches_materialized() {
        let mut rng = ChaCha20Rng::seed_from_u64(30);
        let num_vars = 3;
        let sigma = 1;

        let p1 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let p2 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let s1 = Fr::random(&mut rng);
        let s2 = Fr::random(&mut rng);

        let rlc = RlcSource::new(vec![p1.clone(), p2.clone()], vec![s1, s2]);

        let mut lazy_rows = Vec::new();
        rlc.for_each_row(sigma, &mut |_idx, row| {
            lazy_rows.push(row.to_vec());
        });

        let combined_evals: Vec<Fr> = p1
            .evaluations()
            .iter()
            .zip(p2.evaluations().iter())
            .map(|(&a, &b)| s1 * a + s2 * b)
            .collect();
        let combined = Polynomial::new(combined_evals);
        let mut materialized_rows = Vec::new();
        combined.for_each_row(sigma, &mut |_idx, row| {
            materialized_rows.push(row.to_vec());
        });

        assert_eq!(lazy_rows, materialized_rows);
    }

    #[test]
    fn rlc_source_fold_equals_evaluate_at_point() {
        use crate::eq::EqPolynomial;

        let mut rng = ChaCha20Rng::seed_from_u64(40);
        let num_vars = 4;
        let sigma = 2;
        let nu = num_vars - sigma;

        let p1 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let p2 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let p3 = Polynomial::<Fr>::random(num_vars, &mut rng);
        let s1 = Fr::random(&mut rng);
        let s2 = Fr::random(&mut rng);
        let s3 = Fr::random(&mut rng);

        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        // Split point into row-point (first nu vars) and col-point (last sigma vars)
        let row_point = &point[..nu];
        let col_point = &point[nu..];

        let rlc = RlcSource::new(vec![p1.clone(), p2.clone(), p3.clone()], vec![s1, s2, s3]);

        // fold_rows with eq(row_point) as left vector, then dot with eq(col_point)
        let eq_rows = EqPolynomial::new(row_point.to_vec()).evaluations();
        let folded = rlc.fold_rows(&eq_rows, sigma);
        let eq_cols = EqPolynomial::new(col_point.to_vec()).evaluations();
        let via_fold: Fr = folded
            .iter()
            .zip(eq_cols.iter())
            .map(|(&a, &b)| a * b)
            .sum();

        // Direct evaluation
        let via_eval = rlc.evaluate(&point);

        assert_eq!(via_fold, via_eval);
    }

    #[test]
    fn default_fold_rows_matches_override() {
        let mut rng = ChaCha20Rng::seed_from_u64(50);
        let num_vars = 4;
        let sigma = 2;
        let nu = num_vars - sigma;
        let num_rows = 1usize << nu;

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let left: Vec<Fr> = (0..num_rows).map(|_| Fr::random(&mut rng)).collect();

        // Use the default impl (via for_each_row)
        let default_result = default_fold_rows(&poly, &left, sigma);

        // Use the overridden impl
        let override_result = poly.fold_rows(&left, sigma);

        assert_eq!(default_result, override_result);
    }

    /// Calls the default `fold_rows` implementation (via `for_each_row`).
    fn default_fold_rows<F: Field>(
        source: &impl MultilinearPoly<F>,
        left: &[F],
        sigma: usize,
    ) -> Vec<F> {
        let num_cols = 1usize << sigma;
        let mut result = vec![F::zero(); num_cols];
        source.for_each_row(sigma, &mut |row_idx, row| {
            let l = left[row_idx];
            for (r, &val) in result.iter_mut().zip(row.iter()) {
                *r += l * val;
            }
        });
        result
    }

    #[test]
    fn empty_rlc_source() {
        let rlc: RlcSource<Fr, Polynomial<Fr>> = RlcSource::new(vec![], vec![]);
        assert_eq!(rlc.num_vars(), 0);
    }

    #[test]
    fn single_source_rlc_is_scaled_original() {
        let mut rng = ChaCha20Rng::seed_from_u64(60);
        let num_vars = 3;
        let sigma = 1;
        let nu = num_vars - sigma;
        let num_rows = 1usize << nu;

        let poly = Polynomial::<Fr>::random(num_vars, &mut rng);
        let scalar = Fr::random(&mut rng);
        let left: Vec<Fr> = (0..num_rows).map(|_| Fr::random(&mut rng)).collect();

        let rlc = RlcSource::new(vec![poly.clone()], vec![scalar]);
        let rlc_result = rlc.fold_rows(&left, sigma);

        // Manually scale the polynomial fold
        let direct_result = poly.fold_rows(&left, sigma);
        let scaled: Vec<Fr> = direct_result.iter().map(|&v| scalar * v).collect();

        assert_eq!(rlc_result, scaled);
    }
}
