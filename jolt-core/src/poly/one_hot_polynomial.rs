//! This is an implementation of one-hot multilinear polynomials as
//! necessary for Dory and the opening proof reduction sumcheck in
//! `opening_proof.rs`. In particular, this implementation is _not_ used
//! in the Twist/Shout PIOP implementations in Jolt.

use std::sync::{Arc, Mutex};

use super::multilinear_polynomial::BindingOrder;
use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::{DoryGlobals, JoltGroupWrapper};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::subprotocols::sparse_dense_shout::ExpandingTable;
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use ark_ec::CurveGroup;
use rayon::prelude::*;

/// Represents a one-hot multilinear polynomial (ra/wa) used
/// in Twist/Shout. Perhaps somewhat unintuitively, the implementation
/// in this file is currently only used to compute the Dory
/// commitment and in the opening proof reduction sumcheck.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct OneHotPolynomial<F: JoltField> {
    /// The size of the "address" space for this polynomial.
    pub K: usize,
    /// The indices of the nonzero coefficients for each j \in {0, 1}^T.
    /// In other words, the raf/waf corresponding to this
    /// ra/wa polynomial.
    /// If None, this polynomial is 0 for this j.
    pub nonzero_indices: Vec<Option<usize>>,
    /// The number of variables that have been bound over the
    /// course of sumcheck so far.
    num_variables_bound: usize,
    /// The array described in Section 6.3 of the Twist/Shout paper.
    G: Option<Vec<F>>,
    /// The array described in Section 6.3 of the Twist/Shout paper.
    H: Option<DensePolynomial<F>>,
}

/// State related to the EQ(k, j) term appearing in the opening
/// proof reduction sumcheck.
///
/// The opening proof reduction sumcheck is a batched sumcheck where
/// each sumcheck instance in the batch corresponds to one opening.
/// The sumcheck instance for a one-hot polynomial opening has the form
///   \sum eq(k, r_address) * eq(j, r_cycle) * ra(k, j)
/// so we use a simplified version of the prover algorithm for the
/// Booleanity sumcheck described in Section 6.3 of the Twist/Shout paper.
#[derive(Clone, Debug)]
pub struct OneHotSumcheckState<F: JoltField> {
    /// B stores eq(r, k) using Gruen optimization, see Equation (53)
    pub B: GruenSplitEqPolynomial<F>,
    /// D stores eq(r', j) using Gruen optimization, see Equation (54)
    pub D: GruenSplitEqPolynomial<F>,
    /// TODO(markosg04) we can eliminate this by potentially doing tensor product with Gruen D
    pub D_eq: MultilinearPolynomial<F>,
    /// F will maintain an array that, at the end of sumcheck round m, has size 2^m
    /// and stores all 2^m values eq((k_1, ..., k_m), (r_1, ..., r_m))
    pub F: ExpandingTable<F>,
    /// The number of variables that have been bound during sumcheck so far
    pub num_variables_bound: usize,
}

impl<F: JoltField> OneHotSumcheckState<F> {
    #[tracing::instrument(skip_all, name = "OneHotSumcheckState::new")]
    pub fn new(r_address: &[F], r_cycle: &[F]) -> Self {
        let K = 1 << r_address.len();
        // F will maintain an array that, at the end of sumcheck round m, has size 2^m
        // and stores all 2^m values eq((k_1, ..., k_m), (r_1, ..., r_m))
        // See Equation (55)
        let mut F = ExpandingTable::new(K);
        F.reset(F::one());
        Self {
            B: GruenSplitEqPolynomial::new_rev(r_address), // Equation (53)
            D: GruenSplitEqPolynomial::new_rev(r_cycle),   // Equation (54)
            D_eq: MultilinearPolynomial::from(EqPolynomial::evals(r_cycle)), // Original D
            F,
            num_variables_bound: 0,
        }
    }
}

#[derive(Clone)]
pub struct OneHotPolynomialProverOpening<F: JoltField> {
    pub polynomial: Option<OneHotPolynomial<F>>,
    pub eq_state: Arc<Mutex<OneHotSumcheckState<F>>>,
    pub previous_claim: F,
}

impl<F: JoltField> OneHotPolynomialProverOpening<F> {
    #[tracing::instrument(skip_all, name = "OneHotPolynomialProverOpening::new")]
    pub fn new(eq_state: Arc<Mutex<OneHotSumcheckState<F>>>) -> Self {
        Self {
            polynomial: None,
            eq_state,
            previous_claim: F::zero(),
        }
    }

    pub fn initialize(&mut self, mut polynomial: OneHotPolynomial<F>) {
        let T = polynomial.nonzero_indices.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);
        let D_eq = &self.eq_state.lock().unwrap().D_eq;

        // Compute G as described in Section 6.3
        let G = polynomial
            .nonzero_indices
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, chunk)| {
                let mut result = unsafe_allocate_zero_vec(polynomial.K);
                let mut j = chunk_index * chunk_size;
                for k in chunk {
                    if let Some(k) = k {
                        result[*k] += D_eq.get_bound_coeff(j);
                    }
                    j += 1;
                }
                result
            })
            .reduce(
                || unsafe_allocate_zero_vec(polynomial.K),
                |mut running, new| {
                    running
                        .par_iter_mut()
                        .zip(new.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running
                },
            );

        polynomial.G = Some(G);
        self.polynomial = Some(polynomial);
    }

    #[tracing::instrument(
        skip_all,
        name = "OneHotPolynomialProverOpening::compute_prover_message"
    )]
    pub fn compute_prover_message(&mut self, round: usize) -> Vec<F> {
        let shared_eq = self.eq_state.lock().unwrap();
        let polynomial = self.polynomial.as_ref().unwrap();

        if round < polynomial.K.log_2() {
            let num_unbound_address_variables = polynomial.K.log_2() - round;
            let B = &shared_eq.B;
            let F = &shared_eq.F;
            let G = polynomial.G.as_ref().unwrap();

            // Compute constant coefficient of the quadratic polynomial
            let eval_at_0: F = (0..B.E_out_current_len() / 2)
                .into_par_iter()
                .map(|k_prime| {
                    let inner_sum = G
                        .par_iter()
                        .enumerate()
                        .skip(k_prime)
                        .step_by(B.E_out_current_len() / 2)
                        .map(|(k, &G_k)| {
                            // k_m is the bit corresponding to the variable we'll be binding next
                            let k_m = (k >> (num_unbound_address_variables - 1)) & 1;
                            // We then index into F using the high order bits of k
                            let F_k = F[k >> num_unbound_address_variables];

                            // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                            let G_times_F = G_k * F_k;

                            // Compute constant coefficient: eq(k_m, 0) * G_times_F
                            match k_m {
                                0 => G_times_F, // eq(0, 0) = 1
                                1 => F::zero(), // eq(1, 0) = 0
                                _ => unreachable!(),
                            }
                        })
                        .sum::<F>();

                    inner_sum
                })
                .sum::<F>();

            // Use Gruen's to get evaluations at {0, 2}
            let quadratic_evals = B.gruen_evals_deg_2(eval_at_0, self.previous_claim);
            quadratic_evals.to_vec()
        } else {
            let H = polynomial.H.as_ref().unwrap();
            let B = &shared_eq.B;
            let D = &shared_eq.D;
            let n = H.len() / 2;

            // Compute constant coefficient from H
            let eval_at_0: F = (0..n)
                .into_par_iter()
                .map(|j| {
                    let H_evals = H.sumcheck_evals(j, 2, BindingOrder::HighToLow);
                    // H is linear: H(X) = H[j] + (H[j+n] - H[j]) * X
                    // We only need the constant term H(0) = H[j]
                    H_evals[0]
                })
                .sum::<F>();

            // Use sumcheck_quadratic_evals_from_linear on D with H's constant coefficient
            let quadratic_evals = D.gruen_evals_deg_2(eval_at_0, self.previous_claim);

            let eq_r_address_claim = B.current_scalar;
            vec![
                eq_r_address_claim * quadratic_evals[0],
                eq_r_address_claim * quadratic_evals[1],
            ]
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomialProverOpening::bind")]
    pub fn bind(&mut self, r: F, round: usize) {
        let mut shared_eq = self.eq_state.lock().unwrap();
        let polynomial = self.polynomial.as_mut().unwrap();
        let num_variables_bound = shared_eq.num_variables_bound;
        if round < polynomial.K.log_2() {
            if num_variables_bound <= round {
                shared_eq.B.bind(r);
                // Update F for this round (see Equation 55)
                shared_eq.F.update(r);

                shared_eq.num_variables_bound += 1;
            }

            if round == polynomial.K.log_2() - 1 {
                let F = &shared_eq.F;
                // Transition point; initialize H
                polynomial.H = Some(DensePolynomial::new(
                    polynomial
                        .nonzero_indices
                        .par_iter()
                        .map(|&k| k.map_or(F::zero(), |k| F[k]))
                        .collect::<Vec<_>>(),
                ));
            }
        } else {
            // Last log(T) rounds of sumcheck

            if num_variables_bound <= round {
                shared_eq.D.bind(r);
                shared_eq.num_variables_bound += 1;
            }

            polynomial
                .H
                .as_mut()
                .unwrap()
                .bind_parallel(r, BindingOrder::HighToLow)
        }
    }

    pub fn final_sumcheck_claim(&self) -> F {
        self.polynomial.as_ref().unwrap().H.as_ref().unwrap().Z[0]
    }

    pub fn set_previous_claim(&mut self, claim: F) {
        self.previous_claim = claim;
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
                dense_coeffs[k * T + t] = F::one();
            }
        }
        DensePolynomial::new(dense_coeffs)
    }

    pub fn evaluate(&self, r: &[F]) -> F {
        assert_eq!(r.len(), self.get_num_vars());
        let (r_left, r_right) = r.split_at(self.num_rows().log_2());
        let eq_left = EqPolynomial::evals(r_left);
        let eq_right = EqPolynomial::evals(r_right);
        self.vector_matrix_product(&eq_left)
            .into_par_iter()
            .zip_eq(eq_right.par_iter())
            .map(|(l, r)| l * r)
            .sum()
    }

    pub fn from_indices(nonzero_indices: Vec<Option<usize>>, K: usize) -> Self {
        debug_assert_eq!(DoryGlobals::get_T(), nonzero_indices.len());

        Self {
            K,
            nonzero_indices,
            num_variables_bound: 0,
            G: None,
            H: None,
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomial::commit_rows")]
    pub fn commit_rows<G: CurveGroup<ScalarField = F> + VariableBaseMSM>(
        &self,
        bases: &[G::Affine],
    ) -> Vec<JoltGroupWrapper<G>> {
        let num_rows = self.num_rows();
        println!("# rows = {num_rows}");
        let row_len = DoryGlobals::get_num_columns();
        let T = DoryGlobals::get_T();

        let rows_per_k = T / row_len;
        if rows_per_k >= rayon::current_num_threads() {
            // This is the typical case (T >> K)

            let chunk_commitments: Vec<Vec<_>> = self
                .nonzero_indices
                .par_chunks(row_len)
                .map(|chunk| {
                    let mut row_commitments = vec![JoltGroupWrapper(G::zero()); self.K];
                    for (col_index, k) in chunk.iter().enumerate() {
                        // All the nonzero coefficients are 1, so we simply add
                        // the associated base to the result.
                        if let Some(k) = k {
                            row_commitments[*k].0 += bases[col_index];
                        }
                    }
                    row_commitments
                })
                .collect();
            let mut result = vec![JoltGroupWrapper(G::zero()); num_rows];
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
            let mut result: Vec<JoltGroupWrapper<G>> = vec![JoltGroupWrapper(G::zero()); num_rows];
            result
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_index, chunk)| {
                    let min_row_index = chunk_index * chunk_size;
                    let max_row_index = min_row_index + chunk_size;

                    for (t, k) in self.nonzero_indices.iter().enumerate() {
                        if let Some(k) = k {
                            let global_index = *k as u64 * T as u64 + t as u64;
                            let row_index = (global_index / row_len as u64) as usize;

                            // If this coefficient falls in the chunk of rows corresponding
                            // to `chunk_index`, add its contribution to the result
                            if row_index >= min_row_index && row_index < max_row_index {
                                let col_index = global_index % row_len as u64;
                                // All the nonzero coefficients are 1, so we simply add
                                // the associated base to the result.
                                chunk[row_index % chunk_size].0 += bases[col_index as usize];
                            }
                        }
                    }
                });
            result
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomial::vector_matrix_product")]
    pub fn vector_matrix_product(&self, left_vec: &[F]) -> Vec<F> {
        let T = DoryGlobals::get_T();
        let num_columns = DoryGlobals::get_num_columns();
        let row_len = num_columns;
        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = std::cmp::max(1, num_columns / num_chunks);

        // TODO(moodlezoup): Optimize this
        let mut result = unsafe_allocate_zero_vec(num_columns);
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
                            chunk[col_index % chunk_size] += left_vec[row_index];
                        }
                    }
                }
            });
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::{test_rng, Zero};
    use rand_core::RngCore;
    use serial_test::serial;

    fn dense_polynomial_equivalence<const LOG_K: usize, const LOG_T: usize>() {
        let K: usize = 1 << LOG_K;
        let T: usize = 1 << LOG_T;
        let _guard = DoryGlobals::initialize(K, T);

        let mut rng = test_rng();

        let nonzero_indices: Vec<_> = std::iter::repeat_with(|| Some(rng.next_u64() as usize % K))
            .take(T)
            .collect();
        let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, K);
        let mut dense_poly = one_hot_poly.to_dense_poly();

        let r_address: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(LOG_K)
            .collect();
        let r_cycle: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(LOG_T)
            .collect();

        let one_hot_sumcheck_state = OneHotSumcheckState::new(&r_address, &r_cycle);
        let mut one_hot_opening =
            OneHotPolynomialProverOpening::new(Arc::new(Mutex::new(one_hot_sumcheck_state)));
        one_hot_opening.initialize(one_hot_poly);

        let r_concat = [r_address.as_slice(), r_cycle.as_slice()].concat();
        let mut eq = DensePolynomial::new(EqPolynomial::evals(&r_concat));

        for round in 0..LOG_K + LOG_T {
            let one_hot_message = one_hot_opening.compute_prover_message(round);
            let mut expected_message = vec![Fr::zero(), Fr::zero()];
            let mle_half = dense_poly.len() / 2;
            expected_message[0] = (0..mle_half).map(|i| dense_poly[i] * eq[i]).sum();
            expected_message[1] = (0..mle_half)
                .map(|i| {
                    let poly_bound_point =
                        dense_poly[i + mle_half] + dense_poly[i + mle_half] - dense_poly[i];
                    let eq_bound_point = eq[i + mle_half] + eq[i + mle_half] - eq[i];
                    poly_bound_point * eq_bound_point
                })
                .sum();
            assert_eq!(
                one_hot_message, expected_message,
                "round {round} prover message mismatch"
            );

            let r = Fr::random(&mut rng);
            one_hot_opening.bind(r, round);
            dense_poly.bind_parallel(r, BindingOrder::HighToLow);
            eq.bind_parallel(r, BindingOrder::HighToLow);
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

        let nonzero_indices: Vec<_> = std::iter::repeat_with(|| Some(rng.next_u64() as usize % K))
            .take(T)
            .collect();
        let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, K);
        let dense_poly = one_hot_poly.to_dense_poly();

        let r: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
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
