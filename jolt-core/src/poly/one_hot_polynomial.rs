//! This is an implementation of one-hot multilinear polynomials as
//! necessary for Dory. In particular, this implementation is _not_ used
//! in the Twist/Shout PIOP implementations in Jolt.

use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use allocative::Allocative;
use ark_bn254::G1Affine;
use ark_ec::CurveGroup;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

#[cfg(test)]
use crate::poly::dense_mlpoly::DensePolynomial;

/// Represents a one-hot multilinear polynomial (ra/wa) used
/// in Twist/Shout. Perhaps somewhat unintuitively, the implementation
/// in this file is currently only used to compute the Dory
/// commitment.
#[derive(Clone, Debug, Allocative)]
pub struct OneHotPolynomial<F: JoltField> {
    /// The size of the "address" space for this polynomial.
    pub K: usize,
    /// The indices of the nonzero coefficients for each j \in {0, 1}^T.
    /// In other words, the raf/waf corresponding to this
    /// ra/wa polynomial.
    /// If empty, this polynomial is 0 for all j.
    pub nonzero_indices: Arc<Vec<Option<u8>>>,
    /// PhantomData to hold the field type parameter.
    _marker: PhantomData<F>,
}

impl<F: JoltField> PartialEq for OneHotPolynomial<F> {
    fn eq(&self, other: &Self) -> bool {
        self.K == other.K && self.nonzero_indices == other.nonzero_indices
    }
}

impl<F: JoltField> Default for OneHotPolynomial<F> {
    fn default() -> Self {
        Self {
            K: 1,
            nonzero_indices: Arc::new(vec![]),
            _marker: PhantomData,
        }
    }
}

impl<F: JoltField> OneHotPolynomial<F> {
    /// The number of rows in the coefficient matrix used to
    /// commit to this polynomial using Dory
    pub fn num_rows(&self) -> usize {
        let T = self.nonzero_indices.len() as u128;
        let row_length = DoryGlobals::get_num_columns() as u128;
        (T * self.K as u128 / row_length) as usize
    }

    pub fn get_num_vars(&self) -> usize {
        self.K.log_2() + self.nonzero_indices.len().log_2()
    }

    #[cfg(test)]
    fn to_dense_poly(&self) -> DensePolynomial<F> {
        let T = DoryGlobals::get_T();
        let mut dense_coeffs: Vec<F> = vec![F::zero(); self.K * T];
        for (t, k) in self.nonzero_indices.iter().enumerate() {
            if let Some(k) = k {
                let log_K = self.K.log_2();
                // NOTE: log_K variables are reversed for testing purposes of low to high
                let k = (k & !((1 << log_K) - 1))
                    | ((k & ((1 << log_K) - 1)).reverse_bits() >> (u8::BITS as usize - log_K));
                dense_coeffs[k as usize * T + t] = F::one();
            }
        }
        DensePolynomial::new(dense_coeffs)
    }

    pub fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        assert_eq!(r.len(), self.get_num_vars());
        let (r_left, r_right) = r.split_at(self.num_rows().log_2());
        let eq_left = EqPolynomial::<F>::evals(r_left);
        let eq_right = EqPolynomial::<F>::evals(r_right);
        let mut left_product = unsafe_allocate_zero_vec(eq_right.len());
        self.vector_matrix_product(&eq_left, F::one(), &mut left_product);
        left_product
            .into_par_iter()
            .zip_eq(eq_right.par_iter())
            .map(|(l, r)| l * r)
            .sum()
    }

    pub fn from_indices(nonzero_indices: Vec<Option<u8>>, K: usize) -> Self {
        debug_assert_eq!(DoryGlobals::get_T(), nonzero_indices.len());
        assert!(K <= 1usize << u8::BITS, "K must be <= 256 for indices");

        Self {
            K,
            nonzero_indices: Arc::new(nonzero_indices),
            _marker: PhantomData,
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomial::commit_rows")]
    pub fn commit_rows<G: CurveGroup<ScalarField = F> + VariableBaseMSM>(
        &self,
        bases: &[G::Affine],
    ) -> Vec<G> {
        let num_rows = self.num_rows();
        tracing::debug!("Committing to one-hot polynomial with {num_rows} rows");
        let row_len = DoryGlobals::get_num_columns();
        let T = DoryGlobals::get_T();

        let rows_per_k = T / row_len;
        if rows_per_k >= rayon::current_num_threads() {
            // This is the typical case (T >> K)

            let chunk_commitments: Vec<Vec<_>> = self
                .nonzero_indices
                .par_chunks(row_len)
                .map(|chunk| {
                    // Collect indices for each k
                    let mut indices_per_k: Vec<Vec<usize>> = vec![Vec::new(); self.K];

                    for (col_index, k) in chunk.iter().enumerate() {
                        if let Some(k) = k {
                            indices_per_k[*k as usize].push(col_index);
                        }
                    }

                    // Safety: This function is only called with G1Affine
                    let g1_bases =
                        unsafe { std::mem::transmute::<&[G::Affine], &[G1Affine]>(bases) };

                    // Vectorized batch addition for all k values at once
                    let results =
                        jolt_optimizations::batch_g1_additions_multi(g1_bases, &indices_per_k);

                    // Convert results to row_commitments
                    let mut row_commitments = vec![G::zero(); self.K];
                    for (k, result) in results.into_iter().enumerate() {
                        if !indices_per_k[k].is_empty() {
                            // Convert G1Affine to G1Projective, then cast to G
                            let projective = ark_bn254::G1Projective::from(result);
                            // Safety: We know G is G1Projective in practice when called from dory
                            row_commitments[k] = unsafe { std::mem::transmute_copy(&projective) };
                        }
                    }

                    row_commitments
                })
                .collect();
            let mut result = vec![G::zero(); num_rows];
            for (chunk_index, commitments) in chunk_commitments.iter().enumerate() {
                result
                    .par_iter_mut()
                    .skip(chunk_index)
                    .step_by(rows_per_k)
                    .zip(commitments.into_par_iter())
                    .for_each(|(dest, src)| *dest = *src);
            }

            result
        } else {
            let num_chunks = rayon::current_num_threads().next_power_of_two();
            let chunk_size = std::cmp::max(1, num_rows / num_chunks);
            // row_len is always a power of two (from DoryGlobals::calculate_dimensions)
            let log_row_len = row_len.trailing_zeros();
            let row_len_mask = (row_len - 1) as u64;

            // Iterate over chunks of contiguous rows in parallel
            let mut result: Vec<G> = vec![G::zero(); num_rows];

            // First, collect indices for each row
            let mut row_indices: Vec<Vec<usize>> = vec![Vec::new(); num_rows];

            for (t, k) in self.nonzero_indices.iter().enumerate() {
                if let Some(k) = k {
                    let global_index = *k as u64 * T as u64 + t as u64;
                    let row_index = (global_index >> log_row_len) as usize;
                    let col_index = (global_index & row_len_mask) as usize;
                    row_indices[row_index].push(col_index);
                }
            }

            // Process rows in parallel chunks
            // Safety: This function is only called with G1Affine
            let g1_bases = unsafe { std::mem::transmute::<&[G::Affine], &[G1Affine]>(bases) };

            result
                .par_chunks_mut(chunk_size)
                .zip(row_indices.par_chunks(chunk_size))
                .for_each(|(result_chunk, indices_chunk)| {
                    let results =
                        jolt_optimizations::batch_g1_additions_multi(g1_bases, indices_chunk);

                    for (row_result, (indices, result)) in result_chunk
                        .iter_mut()
                        .zip(indices_chunk.iter().zip(results.into_iter()))
                    {
                        if !indices.is_empty() {
                            // Convert G1Affine to G1Projective, then cast to G
                            let projective = ark_bn254::G1Projective::from(result);
                            // Safety: We know G is G1Projective in practice when called from dory
                            *row_result = unsafe { std::mem::transmute_copy(&projective) };
                        }
                    }
                });
            result
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomial::commit_one_hot_batch")]
    pub fn commit_one_hot_batch<U, G: CurveGroup<ScalarField = F> + VariableBaseMSM>(
        one_hot_polys: &[U],
        bases: &[G::Affine],
    ) -> Vec<Vec<G>>
    where
        U: std::borrow::Borrow<OneHotPolynomial<F>> + Sync,
    {
        let row_len = DoryGlobals::get_num_columns();
        let T = DoryGlobals::get_T();
        let rows_per_k = T / row_len;

        // Phase 1: Collect all chunks from all polynomials
        #[derive(Clone)]
        struct ChunkWork {
            poly_idx: usize,
            chunk_idx: usize,
            chunk_start: usize,
            chunk_len: usize,
            K: usize,
        }

        let all_chunks: Vec<ChunkWork> = one_hot_polys
            .iter()
            .enumerate()
            .flat_map(|(poly_idx, poly)| {
                let poly = poly.borrow();
                let num_chunks = poly.nonzero_indices.len().div_ceil(row_len);
                (0..num_chunks).map(move |chunk_idx| {
                    let chunk_start = chunk_idx * row_len;
                    let chunk_len =
                        std::cmp::min(row_len, poly.nonzero_indices.len() - chunk_start);
                    ChunkWork {
                        poly_idx,
                        chunk_idx,
                        chunk_start,
                        chunk_len,
                        K: poly.K,
                    }
                })
            })
            .collect();

        // Phase 2: Process all chunks in parallel (flat parallelism)
        let chunk_results: Vec<_> = all_chunks
            .par_iter()
            .map(|work| {
                let poly = one_hot_polys[work.poly_idx].borrow();
                let chunk =
                    &poly.nonzero_indices[work.chunk_start..work.chunk_start + work.chunk_len];

                // Collect indices for each k
                let mut indices_per_k: Vec<Vec<usize>> = vec![Vec::new(); work.K];
                for (col_index, k) in chunk.iter().enumerate() {
                    if let Some(k) = k {
                        indices_per_k[*k as usize].push(col_index);
                    }
                }

                // Safety: This function is only called with G1Affine
                let g1_bases = unsafe { std::mem::transmute::<&[G::Affine], &[G1Affine]>(bases) };

                // Vectorized batch addition for all k values at once
                let results =
                    jolt_optimizations::batch_g1_additions_multi(g1_bases, &indices_per_k);

                // Convert results to row_commitments
                let mut row_commitments = vec![G::zero(); work.K];
                for (k, result) in results.into_iter().enumerate() {
                    if !indices_per_k[k].is_empty() {
                        // Convert G1Affine to G1Projective, then cast to G
                        let projective = ark_bn254::G1Projective::from(result);
                        // Safety: We know G is G1Projective in practice when called from dory
                        row_commitments[k] = unsafe { std::mem::transmute_copy(&projective) };
                    }
                }

                (work.poly_idx, work.chunk_idx, row_commitments)
            })
            .collect();

        // Phase 3: Reassemble results by polynomial
        let mut poly_results: Vec<Vec<G>> = one_hot_polys
            .iter()
            .map(|poly| vec![G::zero(); poly.borrow().num_rows()])
            .collect();

        // Group results by polynomial
        let mut results_by_poly: Vec<Vec<_>> = vec![Vec::new(); one_hot_polys.len()];
        for (poly_idx, chunk_idx, commitments) in chunk_results {
            results_by_poly[poly_idx].push((chunk_idx, commitments));
        }

        // Scatter into final results (can be done in parallel per polynomial)
        poly_results
            .par_iter_mut()
            .enumerate()
            .for_each(|(poly_idx, result)| {
                let poly = &one_hot_polys[poly_idx];
                let num_rows = poly.borrow().num_rows();

                for (chunk_idx, commitments) in &results_by_poly[poly_idx] {
                    // Scatter this chunk's results into the output
                    for (k, commitment) in commitments.iter().enumerate() {
                        let row_idx = k * rows_per_k + chunk_idx;
                        if row_idx < num_rows {
                            result[row_idx] = *commitment;
                        }
                    }
                }
            });

        poly_results
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomial::vector_matrix_product")]
    pub fn vector_matrix_product(&self, left_vec: &[F], coeff: F, result: &mut [F]) {
        let T = DoryGlobals::get_T();
        let num_columns = DoryGlobals::get_num_columns();
        debug_assert_eq!(result.len(), num_columns);
        let row_len = num_columns;

        if T >= row_len {
            // This is the typical case (T >= K)
            let rows_per_k = T / row_len;
            result
                .par_iter_mut()
                .enumerate()
                .for_each(|(col_index, dest)| {
                    let mut col_dot_product = F::zero();
                    for (row_offset, t) in (col_index..T).step_by(row_len).enumerate() {
                        if let Some(k) = self.nonzero_indices[t] {
                            let row_index = k as usize * rows_per_k + row_offset;
                            col_dot_product += left_vec[row_index];
                        }
                    }
                    *dest += coeff * col_dot_product;
                });
        } else {
            let num_chunks = rayon::current_num_threads().next_power_of_two();
            let chunk_size = std::cmp::max(1, num_columns / num_chunks);
            // row_len and chunk_size are powers of two (from DoryGlobals and next_power_of_two)
            let log_row_len = row_len.trailing_zeros();
            let row_len_mask = (row_len - 1) as u128;
            let chunk_size_mask = chunk_size - 1;

            result
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_index, chunk)| {
                    let min_col_index = chunk_index * chunk_size;
                    let max_col_index = min_col_index + chunk_size;
                    for (t, k) in self.nonzero_indices.iter().enumerate() {
                        if let Some(k) = k {
                            let global_index = *k as u128 * T as u128 + t as u128;
                            let col_index = (global_index & row_len_mask) as usize;
                            // If this coefficient falls in the chunk of rows corresponding
                            // to `chunk_index`, compute its contribution to the result.
                            if col_index >= min_col_index && col_index < max_col_index {
                                let row_index = (global_index >> log_row_len) as usize;
                                chunk[col_index & chunk_size_mask] += coeff * left_vec[row_index];
                            }
                        }
                    }
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_core::RngCore;
    use serial_test::serial;

    fn evaluate_test<const LOG_K: usize, const LOG_T: usize>() {
        let K: usize = 1 << LOG_K;
        let T: usize = 1 << LOG_T;
        let _guard = DoryGlobals::initialize(K, T);

        let mut rng = test_rng();

        let nonzero_indices: Vec<_> = (0..T)
            .map(|_| Some((rng.next_u64() % K as u64) as u8))
            .collect();
        let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, K);
        let dense_poly = one_hot_poly.to_dense_poly();

        let mut r: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(LOG_K + LOG_T)
                .collect();
        let r_one_hot = r.clone();
        // We reverse because `one_hot_poly.to_dense_poly()` has K variables reversed
        r[..LOG_K].reverse();

        assert_eq!(one_hot_poly.evaluate(&r_one_hot), dense_poly.evaluate(&r));
    }

    #[test]
    #[serial]
    fn evaluate_K_less_than_T() {
        evaluate_test::<5, 6>();
    }

    #[test]
    #[serial]
    fn evaluate_K_equals_T() {
        evaluate_test::<6, 6>();
    }

    #[test]
    #[serial]
    fn evaluate_K_greater_than_T() {
        evaluate_test::<6, 5>();
    }
}
