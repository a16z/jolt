//! This is an implementation of one-hot multilinear polynomials as
//! necessary for Dory. In particular, this implementation is _not_ used
//! in the Twist/Shout PIOP implementations in Jolt.

use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::{DoryGlobals, DoryLayout};
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
    /// commit to this polynomial using Dory.
    /// For AddressMajor layout, this is always the dimension (square matrix).
    /// For CycleMajor layout, this is K * T / num_columns.
    pub fn num_rows(&self) -> usize {
        match DoryGlobals::get_layout() {
            DoryLayout::AddressMajor => DoryGlobals::get_dimension(),
            DoryLayout::CycleMajor => {
                let T = self.nonzero_indices.len() as u128;
                let row_length = DoryGlobals::get_num_columns() as u128;
                (T * self.K as u128 / row_length) as usize
            }
        }
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
        match DoryGlobals::get_layout() {
            DoryLayout::AddressMajor => {
                // AddressMajor: split into address and cycle components
                let T = DoryGlobals::get_T();
                let (r_address, r_cycle) = r.split_at(r.len() - T.log_2());
                let eq_r_address = EqPolynomial::<F>::evals(r_address);
                let eq_r_cycle = EqPolynomial::<F>::evals(r_cycle);
                self.nonzero_indices
                    .par_iter()
                    .zip(eq_r_cycle.par_iter())
                    .map(|(k, eq_cycle)| {
                        if let Some(k) = k {
                            eq_r_address[*k as usize] * eq_cycle
                        } else {
                            F::zero()
                        }
                    })
                    .sum()
            }
            DoryLayout::CycleMajor => {
                // CycleMajor: use vector_matrix_product approach
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
        }
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
        match DoryGlobals::get_layout() {
            DoryLayout::AddressMajor => self.commit_rows_address_major(bases),
            DoryLayout::CycleMajor => self.commit_rows_cycle_major(bases),
        }
    }

    /// Commit rows for AddressMajor layout.
    /// In this layout, each row corresponds to `cycles_per_row` consecutive cycles.
    /// Each cycle has at most one nonzero entry (at the accessed address).
    fn commit_rows_address_major<G: CurveGroup<ScalarField = F> + VariableBaseMSM>(
        &self,
        bases: &[G::Affine],
    ) -> Vec<G> {
        let num_rows = DoryGlobals::get_dimension();
        let row_len = DoryGlobals::get_dimension();
        let T = DoryGlobals::get_T();

        tracing::debug!("Committing to one-hot polynomial (AddressMajor) with {num_rows} rows");

        if T < num_rows {
            // Edge case where T < dimension; each cycle spans multiple rows
            let rows_per_cycle = num_rows / T;

            let mut row_commitments = vec![G::zero(); num_rows];

            for (cycle, k) in self.nonzero_indices.iter().enumerate() {
                if let Some(k) = k {
                    // In AddressMajor with T < dimension:
                    // The coefficient at (address=k, cycle=t) goes to:
                    // row = cycle * rows_per_cycle + k / row_len
                    // col = k % row_len
                    let row_index = cycle * rows_per_cycle + (*k as usize) / row_len;
                    let col_index = (*k as usize) % row_len;
                    if row_index < num_rows && col_index < bases.len() {
                        // Each row commitment is just a single basis element
                        let projective = G::from(bases[col_index]);
                        row_commitments[row_index] = projective;
                    }
                }
            }

            return row_commitments;
        }

        // Normal case: T >= dimension
        let cycles_per_row = T / num_rows;

        // For AddressMajor, each row contains `cycles_per_row` consecutive cycles.
        // Within each row, we need to collect all the nonzero indices and their column positions.
        // Column position for cycle t within the row = (t % cycles_per_row) * K + address

        // Safety: This function is only called with G1Affine
        let g1_bases = unsafe { std::mem::transmute::<&[G::Affine], &[G1Affine]>(bases) };

        // Collect indices for each row
        let mut row_indices: Vec<Vec<usize>> = vec![Vec::new(); num_rows];

        for (cycle, k) in self.nonzero_indices.iter().enumerate() {
            if let Some(k) = k {
                let row_index = cycle / cycles_per_row;
                let col_within_row = (cycle % cycles_per_row) * self.K + (*k as usize);
                if row_index < num_rows && col_within_row < row_len {
                    row_indices[row_index].push(col_within_row);
                }
            }
        }

        // Process rows using batch additions
        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = std::cmp::max(1, num_rows / num_chunks);

        let mut result: Vec<G> = vec![G::zero(); num_rows];

        result
            .par_chunks_mut(chunk_size)
            .zip(row_indices.par_chunks(chunk_size))
            .for_each(|(result_chunk, indices_chunk)| {
                let results = jolt_optimizations::batch_g1_additions_multi(g1_bases, indices_chunk);

                for (row_result, (indices, batch_result)) in result_chunk
                    .iter_mut()
                    .zip(indices_chunk.iter().zip(results.into_iter()))
                {
                    if !indices.is_empty() {
                        let projective = ark_bn254::G1Projective::from(batch_result);
                        *row_result = unsafe { std::mem::transmute_copy(&projective) };
                    }
                }
            });

        result
    }

    /// Commit rows for CycleMajor layout (original implementation).
    fn commit_rows_cycle_major<G: CurveGroup<ScalarField = F> + VariableBaseMSM>(
        &self,
        bases: &[G::Affine],
    ) -> Vec<G> {
        let num_rows = self.num_rows();
        tracing::debug!("Committing to one-hot polynomial (CycleMajor) with {num_rows} rows");
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
        match DoryGlobals::get_layout() {
            DoryLayout::AddressMajor => {
                self.vector_matrix_product_address_major(left_vec, coeff, result)
            }
            DoryLayout::CycleMajor => {
                self.vector_matrix_product_cycle_major(left_vec, coeff, result)
            }
        }
    }

    /// Vector-matrix product for AddressMajor layout.
    fn vector_matrix_product_address_major(&self, left_vec: &[F], coeff: F, result: &mut [F]) {
        let T = DoryGlobals::get_T();
        let num_columns = DoryGlobals::get_dimension();
        let num_rows = DoryGlobals::get_dimension();
        debug_assert_eq!(result.len(), num_columns);

        if T < num_rows {
            // Edge case where T < dimension
            let rows_per_cycle = num_rows / T;

            for (cycle, k) in self.nonzero_indices.iter().enumerate() {
                if let Some(k) = k {
                    let row_index = cycle * rows_per_cycle + (*k as usize) / num_columns;
                    let col_index = (*k as usize) % num_columns;
                    if row_index < left_vec.len() && col_index < result.len() {
                        result[col_index] += coeff * left_vec[row_index];
                    }
                }
            }
            return;
        }

        // Normal case: T >= dimension
        let cycles_per_row = T / num_rows;

        // For AddressMajor layout:
        // - Each row contains `cycles_per_row` consecutive cycles
        // - Within each row, column = (cycle % cycles_per_row) * K + address
        let K = self.K;
        let nonzero_indices = &self.nonzero_indices;
        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(col_index, dest)| {
                let mut col_dot_product = F::zero();

                // For this column, determine which cycles and addresses map here
                // col_index = (cycle_within_row) * K + address
                // where cycle_within_row = cycle % cycles_per_row
                let address_in_col = col_index % K;
                let cycle_offset_in_row = col_index / K;

                if cycle_offset_in_row < cycles_per_row {
                    // Iterate over all rows
                    for row in 0..num_rows {
                        let cycle = row * cycles_per_row + cycle_offset_in_row;
                        if cycle < T {
                            if let Some(addr) = nonzero_indices[cycle] {
                                if addr as usize == address_in_col {
                                    col_dot_product += left_vec[row];
                                }
                            }
                        }
                    }
                }

                *dest += coeff * col_dot_product;
            });
    }

    /// Vector-matrix product for CycleMajor layout (original implementation).
    fn vector_matrix_product_cycle_major(&self, left_vec: &[F], coeff: F, result: &mut [F]) {
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
