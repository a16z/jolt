use std::cell::RefCell;
use std::rc::Rc;

use super::multilinear_polynomial::BindingOrder;
use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::JoltGroupWrapper;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{
    MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use crate::poly::rlc_polynomial::{get_T, get_num_columns};
use crate::subprotocols::sparse_dense_shout::ExpandingTable;
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use ark_ec::CurveGroup;
use rayon::prelude::*;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct OneHotPolynomial<F: JoltField> {
    pub K: usize,
    pub nonzero_indices: Vec<usize>,
    num_variables_bound: usize,
    G: Option<Vec<F>>,
    H: Option<DensePolynomial<F>>,
}

#[derive(Clone, Debug)]
pub struct OneHotSumcheckState<F: JoltField> {
    pub B: MultilinearPolynomial<F>, // Equation (53)
    pub D: MultilinearPolynomial<F>, // Equation (54)
    pub F: ExpandingTable<F>,        // Equation (55)
    num_variables_bound: usize,
}

impl<F: JoltField> OneHotSumcheckState<F> {
    pub fn new(r_address: &[F], r_cycle: &[F]) -> Self {
        let K = 1 << r_address.len();
        // F will maintain an array that, at the end of sumcheck round m, has size 2^m
        // and stores all 2^m values eq((k_1, ..., k_m), (r_1, ..., r_m))
        // See Equation (55)
        let mut F = ExpandingTable::new(K);
        F.reset(F::one());
        Self {
            B: MultilinearPolynomial::from(EqPolynomial::evals(&r_address)), // Equation (53)
            D: MultilinearPolynomial::from(EqPolynomial::evals(&r_cycle)),   // Equation (54)
            F,
            num_variables_bound: 0,
        }
    }
}

pub struct OneHotPolynomialProverOpening<F: JoltField> {
    pub polynomial: OneHotPolynomial<F>,
    pub eq_state: Rc<RefCell<OneHotSumcheckState<F>>>,
}

impl<F: JoltField> OneHotPolynomialProverOpening<F> {
    pub fn new(
        mut polynomial: OneHotPolynomial<F>,
        eq_state: Rc<RefCell<OneHotSumcheckState<F>>>,
    ) -> Self {
        let T = polynomial.nonzero_indices.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);

        let eq_rc = eq_state.clone();
        let D = &eq_rc.borrow().D;

        let G = polynomial
            .nonzero_indices
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, chunk)| {
                let mut result = unsafe_allocate_zero_vec(polynomial.K);
                let mut j = chunk_index * chunk_size;
                for k in chunk {
                    result[*k] += D.get_bound_coeff(j);
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

        Self {
            polynomial,
            eq_state,
        }
    }

    pub fn compute_prover_message(&self, round: usize) -> Vec<F> {
        let shared_eq = self.eq_state.borrow();

        if round < self.polynomial.K.log_2() {
            let num_unbound_address_variables = self.polynomial.K.log_2() - round;
            let B = &shared_eq.B;
            let F = &shared_eq.F;
            let G = self.polynomial.G.as_ref().unwrap();

            let univariate_poly_evals: [F; 2] = (0..B.len() / 2)
                .into_par_iter()
                .map(|k_prime| {
                    let B_evals = B.sumcheck_evals(k_prime, 2, BindingOrder::HighToLow);
                    let inner_sum = G
                        .par_iter()
                        .enumerate()
                        .skip(k_prime)
                        .step_by(B.len() / 2)
                        .map(|(k, &G_k)| {
                            // k_m is the bit corresponding to the variable we'll be binding next
                            let k_m = (k >> (num_unbound_address_variables - 1)) & 1;
                            // We then index into F using the high order bits of k
                            let F_k = F[k >> num_unbound_address_variables];

                            // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                            let G_times_F = G_k * F_k;

                            // For c \in {0, 2} compute:
                            //    G[k] * F[k_1, ...., k_{m-1}, c]
                            //    = G[k] * F[k_1, ...., k_{m-1}] * eq(k_m, c)
                            //    = G_times_F * eq(k_m, c)
                            let eval_c0 = match k_m {
                                0 => G_times_F, // eq(0, 0) = 0 * 0 + (1 - 0) * (1 - 0) = 1,
                                1 => F::zero(), // eq(1, 0) = 1 * 0 + (1 - 1) * (1 - 0) = 0
                                _ => unreachable!(),
                            };

                            let eval_c2 = match k_m {
                                0 => -G_times_F,            // eq(0, 2) = 0 * 2 + (1 - 0) * (1 - 2) = -1,
                                1 => G_times_F + G_times_F, // eq(1, 2) = 1 * 2 + (1 - 1) * (1 - 2) = 2
                                _ => unreachable!(),
                            };
                            [eval_c0, eval_c2]
                        })
                        .reduce(
                            || [F::zero(); 2],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    [B_evals[0] * inner_sum[0], B_evals[1] * inner_sum[1]]
                })
                .reduce(
                    || [F::zero(); 2],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                );

            univariate_poly_evals.to_vec()
        } else {
            let H = self.polynomial.H.as_ref().unwrap();
            let B = &shared_eq.B;
            let D = &shared_eq.D;
            let n = H.len() / 2;

            let univariate_poly_evals: [F; 2] = (0..n)
                .into_par_iter()
                .map(|j| {
                    let H_evals = H.sumcheck_evals(j, 2, BindingOrder::HighToLow);
                    let D_evals = D.sumcheck_evals(j, 2, BindingOrder::HighToLow);
                    [H_evals[0] * D_evals[0], H_evals[1] * D_evals[1]]
                })
                .reduce(
                    || [F::zero(); 2],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                );

            let eq_r_address_claim = B.final_sumcheck_claim();
            vec![
                eq_r_address_claim * univariate_poly_evals[0],
                eq_r_address_claim * univariate_poly_evals[1],
            ]
        }
    }

    pub fn bind(&mut self, r: F, round: usize) {
        let mut shared_eq = self.eq_state.borrow_mut();
        let num_variables_bound = shared_eq.num_variables_bound;
        if round < self.polynomial.K.log_2() {
            if num_variables_bound <= round {
                shared_eq.B.bind_parallel(r, BindingOrder::HighToLow);
                // Update F for this round (see Equation 55)
                shared_eq.F.update(r);

                shared_eq.num_variables_bound += 1;
            }

            if round == self.polynomial.K.log_2() - 1 {
                let F = &shared_eq.F;
                // Transition point; initialize H
                self.polynomial.H = Some(DensePolynomial::new(
                    self.polynomial
                        .nonzero_indices
                        .par_iter()
                        .map(|&k| F[k])
                        .collect::<Vec<_>>(),
                ));
            }
        } else {
            // Last log(T) rounds of sumcheck

            if num_variables_bound <= round {
                shared_eq.D.bind_parallel(r, BindingOrder::HighToLow);
                shared_eq.num_variables_bound += 1;
            }

            self.polynomial
                .H
                .as_mut()
                .unwrap()
                .bind_parallel(r, BindingOrder::HighToLow)
        }
    }

    pub fn final_sumcheck_claim(&self) -> F {
        self.polynomial.H.as_ref().unwrap().Z[0]
    }
}

impl<F: JoltField> OneHotPolynomial<F> {
    pub fn num_rows(&self) -> usize {
        let T = self.nonzero_indices.len() as u128;
        let row_length = get_num_columns() as u128;
        (T * self.K as u128 / row_length) as usize
    }

    #[cfg(test)]
    fn to_dense_poly(&self) -> DensePolynomial<F> {
        let T = get_T();
        let mut dense_coeffs: Vec<F> = vec![F::zero(); self.K * T];
        for (t, k) in self.nonzero_indices.iter().enumerate() {
            dense_coeffs[k * T + t] = F::one();
        }
        DensePolynomial::new(dense_coeffs)
    }

    pub fn from_indices(nonzero_indices: Vec<usize>, K: usize) -> Self {
        debug_assert_eq!(get_T(), nonzero_indices.len());

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
        let row_len = get_num_columns();
        let T = get_T();

        let num_chunks = 4 * rayon::current_num_threads().next_power_of_two();
        let chunk_size = std::cmp::max(1, num_rows / num_chunks);
        let num_chunks = num_rows / chunk_size;

        // TODO(moodlezoup): Optimize this
        (0..num_chunks)
            .into_par_iter()
            .flat_map(|chunk_index| {
                let min_row_index = chunk_index * chunk_size;
                let max_row_index = min_row_index + chunk_size;

                let mut result: Vec<JoltGroupWrapper<G>> =
                    vec![JoltGroupWrapper(G::zero()); chunk_size];

                for (t, k) in self.nonzero_indices.iter().enumerate() {
                    let global_index = *k as u128 * T as u128 + t as u128;
                    let row_index = (global_index / row_len as u128) as usize;

                    if row_index >= min_row_index && row_index < max_row_index {
                        let col_index = global_index % row_len as u128;
                        result[row_index % chunk_size].0 += bases[col_index as usize];
                    }
                }

                result
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomial::vector_matrix_product")]
    pub fn vector_matrix_product(&self, left_vec: &[F]) -> Vec<F> {
        let T = get_T();
        let num_columns = get_num_columns();
        let row_len = num_columns;
        let num_chunks = 4 * rayon::current_num_threads().next_power_of_two();
        let chunk_size = std::cmp::max(1, num_columns / num_chunks);
        let num_chunks = num_columns / chunk_size;

        // TODO(moodlezoup): Optimize this
        let product: Vec<_> = (0..num_chunks)
            .into_par_iter()
            .flat_map(|chunk_index| {
                let min_col_index = chunk_index * chunk_size;
                let max_col_index = min_col_index + chunk_size;
                let mut result: Vec<F> = unsafe_allocate_zero_vec(chunk_size);
                for (t, k) in self.nonzero_indices.iter().enumerate() {
                    let global_index = *k as u128 * T as u128 + t as u128;
                    let col_index = (global_index % row_len as u128) as usize;
                    if col_index >= min_col_index && col_index < max_col_index {
                        let row_index = (global_index / row_len as u128) as usize;
                        result[col_index % chunk_size] += left_vec[row_index];
                    }
                }

                result
            })
            .collect();

        product
    }
}

#[cfg(test)]
mod tests {
    use crate::poly::rlc_polynomial::RLCPolynomial;

    use super::*;
    use ark_bn254::Fr;
    use ark_std::{test_rng, Zero};
    use rand_core::RngCore;

    fn dense_polynomial_equivalence<const LOG_K: usize, const LOG_T: usize>() {
        let K: usize = 1 << LOG_K;
        let T: usize = 1 << LOG_T;
        RLCPolynomial::<Fr>::initialize(K, T);

        let mut rng = test_rng();

        let nonzero_indices: Vec<_> = std::iter::repeat_with(|| rng.next_u64() as usize % K)
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
        let mut one_hot_opening = OneHotPolynomialProverOpening::new(
            one_hot_poly,
            Rc::new(RefCell::new(one_hot_sumcheck_state)),
        );

        let r_concat = [r_address.as_slice(), r_cycle.as_slice()].concat();
        let mut eq = DensePolynomial::new(EqPolynomial::evals(&r_concat));

        for round in 0..LOG_K + LOG_T {
            let one_hot_message = one_hot_opening.compute_prover_message(round);
            let mut expected_message = vec![Fr::zero(), Fr::zero()];
            let mle_half = dense_poly.len() / 2;
            (0..mle_half).for_each(|i| {
                let eval_0 = dense_poly[i] * eq[i];
                let poly_bound_point =
                    dense_poly[i + mle_half] + dense_poly[i + mle_half] - dense_poly[i];
                let eq_bound_point = eq[i + mle_half] + eq[i + mle_half] - eq[i];
                let eval_2 = poly_bound_point * eq_bound_point;
            });
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

    // #[test]
    // fn K_less_than_T() {
    //     dense_polynomial_equivalence::<5, 6>();
    // }

    // #[test]
    // fn K_equals_T() {
    //     dense_polynomial_equivalence::<6, 6>();
    // }

    #[test]
    fn K_greater_than_T() {
        dense_polynomial_equivalence::<6, 5>();
    }
}
