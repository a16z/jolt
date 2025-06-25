use std::cell::RefCell;
use std::rc::Rc;

use super::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::JoltGroupWrapper;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use crate::poly::rlc_polynomial::{get_T, get_num_columns};
use crate::poly::split_eq_poly::SplitEqPolynomial;
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use ark_ec::CurveGroup;
use rayon::prelude::*;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct OneHotPolynomial<F: JoltField> {
    pub K: usize,
    pub nonzero_indices: Vec<usize>,
    H: Option<DensePolynomial<F>>,
    num_variables_bound: usize,
}

pub struct OneHotEqState<F: JoltField> {
    pub B: MultilinearPolynomial<F>,
    pub D: MultilinearPolynomial<F>,
    pub F: Vec<F>,
    num_variables_bound: usize,
}

pub struct OneHotPolynomialProverOpening<F: JoltField> {
    pub polynomial: OneHotPolynomial<F>,
    pub eq_state: Rc<RefCell<OneHotEqState<F>>>,
}

impl<F: JoltField> OneHotPolynomialProverOpening<F> {
    fn compute_prover_message(&self, _: usize) -> Vec<F> {
        let shared_eq = self.eq_state.borrow();
        todo!()
    }

    fn bind(&mut self, r_j: F, round: usize) {
        let mut shared_eq = self.eq_state.borrow_mut();
        if shared_eq.num_variables_bound <= round {
            shared_eq
                .eq_poly
                .bind_parallel(r_j, BindingOrder::HighToLow);
            shared_eq.num_variables_bound += 1;
        }

        self.polynomial.bind_parallel(r_j, BindingOrder::HighToLow);
    }

    fn final_sumcheck_claim(&self) -> F {
        self.polynomial.final_sumcheck_claim()
    }
}

impl<F: JoltField> OneHotEqState<F> {
    pub fn new(r_address: &[F], r_cycle: &[F]) -> Self {
        let K = 1 << r_address.len();
        // F will maintain an array that, at the end of sumcheck round m, has size 2^m
        // and stores all 2^m values eq((k_1, ..., k_m), (r_1, ..., r_m))
        // See Equation (55)
        let mut F = unsafe_allocate_zero_vec(K);
        F[0] = F::one();
        Self {
            B: MultilinearPolynomial::from(EqPolynomial::evals(&r_address)), // Equation (53)
            D: MultilinearPolynomial::from(EqPolynomial::evals(&r_cycle)),   // Equation (54)
            F,
            num_variables_bound: 0,
        }
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
        assert!(!self.is_bound());
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
            F: unsafe_allocate_zero_vec(K),
            H: None,
            num_variables_bound: 0,
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

    #[tracing::instrument(skip_all, name = "OneHotPolynomial::compute_sumcheck_prover_message")]
    pub fn compute_sumcheck_prover_message(&self, eq_poly: &SplitEqPolynomial<F>) -> Vec<F> {
        if self.num_variables_bound < self.K.log_2() {
            todo!()
        } else {
            let H = self.H.as_ref().unwrap();
            debug_assert_eq!(H.len(), eq_poly.E1_len);
            let n = H.len() / 2;

            let univariate_poly_evals: [F; 2] = (0..n)
                .into_par_iter()
                .map(|j| {
                    let H_evals = H.sumcheck_evals(j, 2, BindingOrder::HighToLow);
                    [
                        H_evals[0] * eq_poly.E1[j],
                        H_evals[1] * (eq_poly.E1[j + n] + eq_poly.E1[j + n] - eq_poly.E1[j]),
                    ]
                })
                .reduce(
                    || [F::zero(); 2],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                );
            univariate_poly_evals.to_vec()
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for OneHotPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.num_variables_bound > 0
    }

    fn bind(&mut self, _: F, _: BindingOrder) {
        unimplemented!("Always use bind_parallel")
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomial::bind_parallel")]
    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        assert_eq!(order, BindingOrder::HighToLow);

        if self.num_variables_bound < self.K.log_2() {
            // println!(
            //     "TODO(moodlezoup): does this need to be changed for high-to-low binding order?"
            // );
            // // Update F for this round (see Equation 55)
            // let (F_left, F_right) = self.F.split_at_mut(1 << self.num_variables_bound);
            // F_left
            //     .par_iter_mut()
            //     .zip(F_right.par_iter_mut())
            //     .for_each(|(x, y)| {
            //         *y = *x * r;
            //         *x -= *y;
            //     });

            if self.num_variables_bound == self.K.log_2() - 1 {
                // Transition point; initialize H
                self.H = Some(DensePolynomial::new(
                    self.nonzero_indices
                        .par_iter()
                        .map(|&k| self.F[k])
                        .collect::<Vec<_>>(),
                ));
            }
        } else {
            // Last log(T) rounds of sumcheck
            self.H
                .as_mut()
                .unwrap()
                .bind_parallel(r, BindingOrder::HighToLow)
        }

        self.num_variables_bound += 1;
    }

    fn final_sumcheck_claim(&self) -> F {
        self.H.as_ref().unwrap().Z[0]
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
        let mut one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, K);
        let mut dense_poly = one_hot_poly.to_dense_poly();

        let r_address: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(LOG_K)
            .collect();
        let r_cycle: Vec<Fr> = std::iter::repeat_with(|| Fr::random(&mut rng))
            .take(LOG_T)
            .collect();
        let mut split_eq = SplitEqPolynomial::new_with_split(&r_cycle, &r_address);
        let mut eq = split_eq.merge(BindingOrder::HighToLow);

        for round in 0..LOG_K + LOG_T {
            let one_hot_message = one_hot_poly.compute_sumcheck_prover_message(&split_eq);
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
            one_hot_poly.bind_parallel(r, BindingOrder::HighToLow);
            split_eq.bind(r, BindingOrder::HighToLow);
            dense_poly.bind_parallel(r, BindingOrder::HighToLow);
            eq.bind_parallel(r, BindingOrder::HighToLow);
        }
        assert_eq!(
            one_hot_poly.final_sumcheck_claim(),
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
