//! This is an implementation of one-hot multilinear polynomials as
//! necessary for Dory and the opening proof reduction sumcheck in
//! `opening_proof.rs`. In particular, this implementation is _not_ used
//! in the Twist/Shout PIOP implementations in Jolt.

use super::multilinear_polynomial::BindingOrder;
use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialBinding};
use crate::poly::ra_poly::RaPolynomial;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::utils::expanding_table::ExpandingTable;
use crate::utils::math::Math;
use crate::utils::thread::drop_in_background_thread;
use crate::utils::thread::unsafe_allocate_zero_vec;
use allocative::Allocative;
use ark_bn254::G1Affine;
use ark_ec::CurveGroup;
use fixedbitset::FixedBitSet;
use num_traits::Zero;
use rayon::prelude::*;
use std::mem;
use std::sync::{Arc, RwLock};

#[cfg(test)]
use crate::poly::dense_mlpoly::DensePolynomial;

/// Represents a one-hot multilinear polynomial (ra/wa) used
/// in Twist/Shout. Perhaps somewhat unintuitively, the implementation
/// in this file is currently only used to compute the Dory
/// commitment and in the opening proof reduction sumcheck.
#[derive(Clone, Debug, Allocative)]
pub struct OneHotPolynomial<F: JoltField> {
    /// The size of the "address" space for this polynomial.
    pub K: usize,
    /// The indices of the nonzero coefficients for each j \in {0, 1}^T.
    /// In other words, the raf/waf corresponding to this
    /// ra/wa polynomial.
    /// If empty, this polynomial is 0 for all j.
    pub nonzero_indices: Arc<Vec<Option<u8>>>,
    /// The number of variables that have been bound over the
    /// course of sumcheck so far.
    num_variables_bound: usize,
    /// The array described in Section 6.3 of the Twist/Shout paper.
    G: Vec<F>,
    /// The array described in Section 6.3 of the Twist/Shout paper.
    H: Arc<RwLock<RaPolynomial<u8, F>>>,
}

impl<F: JoltField> PartialEq for OneHotPolynomial<F> {
    fn eq(&self, other: &Self) -> bool {
        self.K == other.K
            && self.nonzero_indices == other.nonzero_indices
            && self.num_variables_bound == other.num_variables_bound
            && self.G == other.G
            && *self.H.read().unwrap() == *other.H.read().unwrap()
    }
}

/// State related to the address variable (i.e. k) terms appearing in the opening
/// proof reduction sumcheck.
#[derive(Clone, Debug, Allocative)]
pub struct EqAddressState<F: JoltField> {
    /// B stores eq(r, k), see Equation (53)
    pub B: MultilinearPolynomial<F>,
    /// F will maintain an array that, at the end of sumcheck round m, has size 2^m
    /// and stores all 2^m values eq((k_1, ..., k_m), (r_1, ..., r_m))
    pub F: ExpandingTable<F>,
    /// The number of variables that have been bound during sumcheck so far
    pub num_variables_bound: usize,
}

/// State related to the cycle variable (i.e. j) terms appearing in the opening
/// proof reduction sumcheck.
#[derive(Clone, Debug, Allocative)]
pub struct EqCycleState<F: JoltField> {
    /// D stores eq(r', j), see Equation (54) but with Gruen X Dao-Thaler optimizations
    pub D: GruenSplitEqPolynomial<F>,
    /// The number of variables that have been bound during sumcheck so far
    pub num_variables_bound: usize,
}

impl<F: JoltField> EqAddressState<F> {
    #[tracing::instrument(skip_all, name = "EqAddressState::new")]
    pub fn new(r_address: &[F::Challenge]) -> Self {
        let K = 1 << r_address.len();
        // F will maintain an array that, at the end of sumcheck round m, has size 2^m
        // and stores all 2^m values eq((k_1, ..., k_m), (r_1, ..., r_m))
        let mut F = ExpandingTable::new(K, BindingOrder::HighToLow);
        F.reset(F::one());

        Self {
            B: MultilinearPolynomial::from(EqPolynomial::<F>::evals(r_address)),
            F,
            num_variables_bound: 0,
        }
    }
}

impl<F: JoltField> EqCycleState<F> {
    #[tracing::instrument(skip_all, name = "EqCycleState::new")]
    pub fn new(r_cycle: &[F::Challenge]) -> Self {
        let D = GruenSplitEqPolynomial::new(r_cycle, BindingOrder::LowToHigh);
        Self {
            D,
            num_variables_bound: 0,
        }
    }
}

/// The opening proof reduction sumcheck is a batched sumcheck where
/// each sumcheck instance in the batch corresponds to one opening.
/// The sumcheck instance for a one-hot polynomial opening has the form
///   \sum eq(k, r_address) * eq(j, r_cycle) * ra(k, j)
/// so we use a simplified version of the prover algorithm for the
/// Booleanity sumcheck described in Section 6.3 of the Twist/Shout paper.
#[derive(Clone, Allocative)]
pub struct OneHotPolynomialProverOpening<F: JoltField> {
    pub log_T: usize,
    pub polynomial: OneHotPolynomial<F>,
    pub eq_address_state: Arc<RwLock<EqAddressState<F>>>,
    pub eq_cycle_state: Arc<RwLock<EqCycleState<F>>>,
}

impl<F: JoltField> OneHotPolynomialProverOpening<F> {
    #[tracing::instrument(skip_all, name = "OneHotPolynomialProverOpening::new")]
    pub fn new(
        eq_address_state: Arc<RwLock<EqAddressState<F>>>,
        eq_cycle_state: Arc<RwLock<EqCycleState<F>>>,
    ) -> Self {
        Self {
            log_T: 0,
            polynomial: OneHotPolynomial::default(),
            eq_address_state,
            eq_cycle_state,
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomialProverOpening::initialize")]
    pub fn initialize(&mut self, mut polynomial: OneHotPolynomial<F>) {
        let nonzero_indices = &polynomial.nonzero_indices;
        let T = nonzero_indices.len();

        let eq = self.eq_cycle_state.read().unwrap();
        // Compute G without materializing the dense eq(r_cycle, j) table:
        // G[k] = sum_{j: nonzero_indices[j] = k} eq(r_cycle, j).
        // (described in Section 6.3 of the Twist/Shout paper)
        // Custom parallelization: parallel over E_out, sequential over E_in and the last bit.
        // This avoids nested Rayon layers while still computing d_j on-the-fly from the split representation.
        let E_in = eq.D.E_in_current();
        let E_out = eq.D.E_out_current();
        let w_current = eq.D.get_current_w();
        let factor_0 = F::one() - w_current;
        let factor_1: F = w_current.into();

        // Precompute merged inner weights once (in unreduced form): [low*(1-w), low*w] for all x_in
        let in_len = E_in.len();
        let x_in_bits = in_len.log_2();
        let merged_in_unreduced: Vec<F::Unreduced<9>> = {
            let mut merged: Vec<F::Unreduced<9>> = unsafe_allocate_zero_vec(2 * in_len);
            merged
                .par_chunks_exact_mut(2)
                .zip(E_in.par_iter())
                .for_each(|(chunk, &low)| {
                    chunk[0] = low.mul_unreduced::<9>(factor_0);
                    chunk[1] = low.mul_unreduced::<9>(factor_1);
                });
            merged
        };

        let G = E_out
            .par_iter()
            .enumerate()
            .fold(
                || unsafe_allocate_zero_vec(polynomial.K),
                |mut partial, (x_out, &e_out)| {
                    let mut local_unreduced: Vec<F::Unreduced<9>> =
                        unsafe_allocate_zero_vec(polynomial.K);
                    let mut touched_flags = FixedBitSet::with_capacity(polynomial.K);
                    let x_out_base = x_out << (x_in_bits + 1);

                    for x_in in 0..in_len {
                        let j0 = x_out_base + (x_in << 1);
                        let j1 = j0 + 1;
                        let off = 2 * x_in;
                        let add0_unr = merged_in_unreduced[off];
                        let add1_unr = merged_in_unreduced[off + 1];

                        if let Some(k0) = nonzero_indices[j0] {
                            let idx = k0 as usize;
                            if !touched_flags.contains(idx) {
                                touched_flags.insert(idx);
                            }
                            local_unreduced[idx] += add0_unr;
                        }
                        if let Some(k1) = nonzero_indices[j1] {
                            let idx = k1 as usize;
                            if !touched_flags.contains(idx) {
                                touched_flags.insert(idx);
                            }
                            local_unreduced[idx] += add1_unr;
                        }
                    }

                    for idx in touched_flags.ones() {
                        let reduced = F::from_montgomery_reduce::<9>(local_unreduced[idx]);
                        partial[idx] += e_out * reduced;
                    }

                    partial
                },
            )
            .reduce(
                || unsafe_allocate_zero_vec(polynomial.K),
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            );

        polynomial.G = G;
        self.polynomial = polynomial;
        self.log_T = T.log_2();
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomialProverOpening::compute_message")]
    pub fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let shared_eq_address = self.eq_address_state.read().unwrap();
        let shared_eq_cycle = self.eq_cycle_state.read().unwrap();
        let polynomial = &self.polynomial;

        if round < polynomial.K.log_2() {
            let num_unbound_address_variables = polynomial.K.log_2() - round;
            let B = &shared_eq_address.B;
            let F = &shared_eq_address.F;
            let G = &polynomial.G;

            let unreduced_univariate_poly_evals = (0..B.len() / 2)
                .into_par_iter()
                .map(|k_prime| {
                    let B_evals = B.sumcheck_evals_array::<2>(k_prime, BindingOrder::HighToLow);
                    let inner_sum = G
                        .par_iter()
                        .enumerate()
                        .skip(k_prime)
                        .step_by(B.len() / 2)
                        .map(|(k, &G_k)| {
                            let k_m = (k >> (num_unbound_address_variables - 1)) & 1;
                            let F_k = F[k >> num_unbound_address_variables];
                            let G_times_F = G_k * F_k;

                            let eval_c0 = if k_m == 0 { G_times_F } else { F::zero() };
                            let eval_c2 = if k_m == 0 {
                                -G_times_F
                            } else {
                                G_times_F + G_times_F
                            };

                            [eval_c0, eval_c2]
                        })
                        .fold_with([F::zero(); 2], |running, new| {
                            [running[0] + new[0], running[1] + new[1]]
                        })
                        .reduce(
                            || [F::zero(); 2],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    [
                        B_evals[0].mul_unreduced::<9>(inner_sum[0]),
                        B_evals[1].mul_unreduced::<9>(inner_sum[1]),
                    ]
                })
                .reduce(
                    || [F::Unreduced::<9>::zero(); 2],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                );

            let univariate_poly_evals = unreduced_univariate_poly_evals
                .into_iter()
                .map(|evals| F::from_montgomery_reduce(evals))
                .collect::<Vec<_>>();

            UniPoly::from_evals_and_hint(previous_claim, &univariate_poly_evals)
        } else {
            // T-variable rounds
            let B = &shared_eq_address.B;
            let d_gruen = &shared_eq_cycle.D;
            let eq_r_address_claim = B.final_sumcheck_claim();
            let H = &polynomial.H.read().unwrap();

            let [gruen_eval_0] =
                d_gruen.par_fold_out_in_unreduced::<9, 1>(&|g| [H.get_bound_coeff(2 * g)]);

            let gruen_univariate_evals =
                d_gruen.gruen_poly_deg_2(gruen_eval_0, previous_claim / eq_r_address_claim);

            gruen_univariate_evals * eq_r_address_claim
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomialProverOpening::bind")]
    pub fn bind(&mut self, r: F::Challenge, round: usize) {
        let mut shared_eq_address = self.eq_address_state.write().unwrap();
        let mut shared_eq_cycle = self.eq_cycle_state.write().unwrap();
        let polynomial = &mut self.polynomial;
        let num_variables_bound =
            shared_eq_address.num_variables_bound + shared_eq_cycle.num_variables_bound;

        // Bind shared state if not already bound
        if num_variables_bound <= round {
            if round < polynomial.K.log_2() {
                shared_eq_address
                    .B
                    .bind_parallel(r, BindingOrder::HighToLow);

                shared_eq_address.F.update(r);
                shared_eq_address.num_variables_bound += 1;
            } else {
                shared_eq_cycle.D.bind(r);
                shared_eq_cycle.num_variables_bound += 1;
            }
        }

        // For the first two log T rounds we want to use F still
        if round == polynomial.K.log_2() - 1 {
            let nonzero_indices = &polynomial.nonzero_indices;

            let mut lock = polynomial.H.write().unwrap();
            if matches!(*lock, RaPolynomial::None) {
                *lock =
                    RaPolynomial::new(nonzero_indices.clone(), shared_eq_address.F.clone_values());
            }

            let g = mem::take(&mut polynomial.G);
            drop_in_background_thread(g);
        } else if round >= polynomial.K.log_2() {
            // Bind H for subsequent T rounds
            let mut H = polynomial.H.write().unwrap();
            if H.len().log_2() == self.log_T + polynomial.K.log_2() - round {
                H.bind_parallel(r, BindingOrder::LowToHigh);
            }
        }
    }

    pub fn final_sumcheck_claim(&self) -> F {
        self.polynomial.H.read().unwrap().final_sumcheck_claim()
    }
}

impl<F: JoltField> Default for OneHotPolynomial<F> {
    fn default() -> Self {
        Self {
            K: 1,
            nonzero_indices: Arc::new(vec![]),
            num_variables_bound: 0,
            G: vec![],
            H: Arc::new(RwLock::new(RaPolynomial::None)),
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
                dense_coeffs[*k as usize * T + t] = F::one();
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
        assert!(K <= 1 << 8, "K must be <= 256 for index to fit into u8");

        Self {
            K,
            nonzero_indices: Arc::new(nonzero_indices),
            ..Default::default()
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

            // Iterate over chunks of contiguous rows in parallel
            let mut result: Vec<G> = vec![G::zero(); num_rows];

            // First, collect indices for each row
            let mut row_indices: Vec<Vec<usize>> = vec![Vec::new(); num_rows];

            for (t, k) in self.nonzero_indices.iter().enumerate() {
                if let Some(k) = k {
                    let global_index = *k as u64 * T as u64 + t as u64;
                    let row_index = (global_index / row_len as u64) as usize;
                    let col_index = (global_index % row_len as u64) as usize;
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

            result
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_index, chunk)| {
                    let min_col_index = chunk_index * chunk_size;
                    let max_col_index = min_col_index + chunk_size;
                    for (t, k) in self.nonzero_indices.iter().enumerate() {
                        if let Some(k) = k {
                            let global_index = *k as u128 * T as u128 + t as u128;
                            let col_index = (global_index % row_len as u128) as usize;
                            // If this coefficient falls in the chunk of rows corresponding
                            // to `chunk_index`, compute its contribution to the result.
                            if col_index >= min_col_index && col_index < max_col_index {
                                let row_index = (global_index / row_len as u128) as usize;
                                chunk[col_index % chunk_size] += coeff * left_vec[row_index];
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
    use crate::poly::unipoly::UniPoly;
    use ark_bn254::Fr;
    use ark_std::{test_rng, Zero};
    use rand_core::RngCore;
    use serial_test::serial;

    fn dense_polynomial_equivalence<const LOG_K: usize, const LOG_T: usize>() {
        let K: usize = 1 << LOG_K;
        let T: usize = 1 << LOG_T;
        let _guard = DoryGlobals::initialize(K, T);

        let mut rng = test_rng();

        let nonzero_indices: Vec<_> =
            std::iter::repeat_with(|| Some(rng.next_u64() as u8 % K as u8))
                .take(T)
                .collect();
        let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, K);
        let mut dense_poly = one_hot_poly.to_dense_poly();

        let r_address = std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
            .take(LOG_K)
            .collect::<Vec<_>>();
        let r_cycle = std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
            .take(LOG_T)
            .collect::<Vec<_>>();

        let eq_address_state = EqAddressState::new(&r_address);
        let eq_cycle_state = EqCycleState::new(&r_cycle);

        let mut one_hot_opening = OneHotPolynomialProverOpening::new(
            Arc::new(RwLock::new(eq_address_state)),
            Arc::new(RwLock::new(eq_cycle_state)),
        );
        one_hot_opening.initialize(one_hot_poly.clone());

        let r_concat = [r_address.as_slice(), r_cycle.as_slice()].concat();
        let mut eq = DensePolynomial::new(EqPolynomial::<Fr>::evals(&r_concat));

        // Compute the initial input claim
        let input_claim: Fr = (0..dense_poly.len()).map(|i| dense_poly[i] * eq[i]).sum();
        let mut previous_claim = input_claim;

        for round in 0..LOG_K + LOG_T {
            let one_hot_message = one_hot_opening.compute_message(round, previous_claim);
            // Evals at [0, 2].
            let mut expected_message = vec![Fr::zero(), Fr::zero()];
            let mle_half = dense_poly.len() / 2;

            // We bind first log_K vars HighToLow and then log_T vars LowToHigh
            if round < LOG_K {
                expected_message[0] = (0..mle_half).map(|i| dense_poly[i] * eq[i]).sum();
                expected_message[1] = (0..mle_half)
                    .map(|i| {
                        let poly_bound_point =
                            dense_poly[i + mle_half] + dense_poly[i + mle_half] - dense_poly[i];
                        let eq_bound_point = eq[i + mle_half] + eq[i + mle_half] - eq[i];
                        poly_bound_point * eq_bound_point
                    })
                    .sum();
            } else {
                expected_message[0] = (0..mle_half).map(|i| dense_poly[2 * i] * eq[2 * i]).sum();
                expected_message[1] = (0..mle_half)
                    .map(|i| {
                        let poly_bound_point =
                            dense_poly[2 * i + 1] + dense_poly[2 * i + 1] - dense_poly[2 * i];
                        let eq_bound_point = eq[2 * i + 1] + eq[2 * i + 1] - eq[2 * i];
                        poly_bound_point * eq_bound_point
                    })
                    .sum();
            }
            assert_eq!(
                [
                    one_hot_message.eval_at_zero(),
                    one_hot_message.evaluate::<Fr>(&Fr::from(2))
                ],
                *expected_message,
                "round {round} prover message mismatch"
            );

            let r = <Fr as JoltField>::Challenge::random(&mut rng);

            // Update previous_claim by evaluating the univariate polynomial at r
            let eval_at_1 = previous_claim - expected_message[0];
            let univariate_evals = vec![expected_message[0], eval_at_1, expected_message[1]];
            let univariate_poly = UniPoly::from_evals(&univariate_evals);
            previous_claim = univariate_poly.evaluate(&r);

            one_hot_opening.bind(r, round);
            if round < LOG_K {
                dense_poly.bind_parallel(r, BindingOrder::HighToLow);
                eq.bind_parallel(r, BindingOrder::HighToLow);
            } else {
                dense_poly.bind_parallel(r, BindingOrder::LowToHigh);
                eq.bind_parallel(r, BindingOrder::LowToHigh);
            }
        }
        assert_eq!(
            one_hot_opening.final_sumcheck_claim(),
            dense_poly[0],
            "final sumcheck claim"
        );
    }

    #[test]
    #[serial]
    fn sumcheck_K_less_than_T() {
        dense_polynomial_equivalence::<5, 6>();
    }

    #[test]
    #[serial]
    fn sumcheck_K_equals_T() {
        dense_polynomial_equivalence::<6, 6>();
    }

    #[test]
    #[serial]
    fn sumcheck_K_greater_than_T() {
        dense_polynomial_equivalence::<6, 5>();
    }

    fn evaluate_test<const LOG_K: usize, const LOG_T: usize>() {
        let K: usize = 1 << LOG_K;
        let T: usize = 1 << LOG_T;
        let _guard = DoryGlobals::initialize(K, T);

        let mut rng = test_rng();

        let nonzero_indices: Vec<_> =
            std::iter::repeat_with(|| Some(rng.next_u64() as u8 % K as u8))
                .take(T)
                .collect();
        let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, K);
        let dense_poly = one_hot_poly.to_dense_poly();

        let r: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(LOG_K + LOG_T)
                .collect();

        assert_eq!(one_hot_poly.evaluate(&r), dense_poly.evaluate(&r));
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
