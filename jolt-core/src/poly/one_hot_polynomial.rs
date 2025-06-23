use super::multilinear_polynomial::{BindingOrder, PolynomialBinding};
use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::commitment::dory::JoltGroupWrapper;
#[cfg(test)]
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::sparse_matrix_polynomial::{get_T, get_num_columns};
use crate::poly::split_eq_poly::SplitEqPolynomial;
use crate::utils::math::Math;
use ark_ec::CurveGroup;
use rayon::prelude::*;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct OneHotPolynomial<F: JoltField> {
    pub K: usize,
    pub nonzero_indices: Vec<(usize, usize)>,
    pub bound_coeffs: Vec<(usize, usize, F)>,
    binding_scratch_space: Vec<(usize, usize, F)>,
    num_variables_bound: usize,
}

impl<F: JoltField> OneHotPolynomial<F> {
    pub fn num_rows(&self) -> usize {
        let T = self.nonzero_indices.len() as u128;
        let row_length = get_num_columns() as u128;
        (T * self.K as u128 / row_length) as usize
    }

    pub fn commit_rows<G>(&self, g1_generators: &[JoltGroupWrapper<G>]) -> Vec<JoltGroupWrapper<G>>
    where
        G: CurveGroup<ScalarField = F> + VariableBaseMSM,
    {
        todo!();
        vec![]
    }

    #[cfg(test)]
    fn to_dense_poly(&self) -> DensePolynomial<F> {
        use crate::utils::thread::unsafe_allocate_zero_vec;
        let T = get_T();
        let num_cycle_variables = T.log_2();

        if !self.is_bound() {
            let mut dense_coeffs: Vec<F> = unsafe_allocate_zero_vec(self.K * T);
            for (t, k) in self.nonzero_indices.iter() {
                dense_coeffs[k * T + t] = F::one();
            }
            DensePolynomial::new(dense_coeffs)
        } else if self.num_variables_bound < num_cycle_variables {
            let T_bound = T >> self.num_variables_bound;
            let mut dense_coeffs: Vec<F> = unsafe_allocate_zero_vec(self.K * T_bound);
            for (t, k, coeff) in self.bound_coeffs.iter() {
                dense_coeffs[k * T_bound + t] += *coeff;
            }
            DensePolynomial::new(dense_coeffs)
        } else {
            let num_address_variables_bound = self.num_variables_bound - num_cycle_variables;
            let K_bound = self.K >> num_address_variables_bound;
            let mut dense_coeffs: Vec<F> = unsafe_allocate_zero_vec(K_bound);
            for (_, k, coeff) in self.bound_coeffs.iter() {
                dense_coeffs[*k] += *coeff;
            }
            DensePolynomial::new(dense_coeffs)
        }
    }

    pub fn from_indices(indices: Vec<usize>, K: usize) -> Self {
        debug_assert_eq!(get_T(), indices.len());

        Self {
            K,
            // Annoying that we have to do this, but we can't chain
            // enumerate() with par_chunk_by(), which we want to do for
            // the first `compute_prover_message` and `bind`
            nonzero_indices: indices.into_par_iter().enumerate().collect(),
            bound_coeffs: vec![],
            binding_scratch_space: vec![],
            num_variables_bound: 0,
        }
    }

    fn evaluate(&self, r: &[F]) -> F {
        let T = get_T();
        debug_assert_eq!(self.nonzero_indices.len(), T);
        assert_eq!(self.K.log_2() + T.log_2(), r.len());

        let (r_left, r_right) = r.split_at(self.K.log_2());
        let (eq_left, eq_right) = rayon::join(
            || EqPolynomial::evals(r_left),
            || EqPolynomial::evals(r_right),
        );

        self.nonzero_indices
            .par_iter()
            .map(|(t, k)| eq_left[*k] * eq_right[*t])
            .sum()
    }

    pub fn compute_sumcheck_prover_message(&self, eq_poly: &SplitEqPolynomial<F>) -> Vec<F> {
        // SplitEqPolynomial only supports binding from low to high, where
        // cycle variables are bound before address variables.

        let num_cycle_variables = get_T().log_2();

        if self.num_variables_bound == 0 {
            let eval_0: F = self
                .nonzero_indices
                .par_iter()
                .step_by(2)
                .map(|(t, k)| {
                    let eq_address = eq_poly.E2[*k];
                    let eq_cycle = eq_poly.E1[*t];
                    eq_address * eq_cycle
                })
                .sum();
            let eval_2: F = self
                .nonzero_indices
                .par_chunk_by(|(t1, k1), (t2, k2)| (t1 >> 1 == t2 >> 1) && k1 == k2)
                .map(|chunk| match chunk {
                    [(t, k)] => {
                        let eq_address = eq_poly.E2[*k];
                        if t % 2 == 0 {
                            let eq_cycle = eq_poly.E1[*t + 1] + eq_poly.E1[*t + 1] - eq_poly.E1[*t];
                            // poly[t + 1] = 0, poly[t] = 1
                            // => 2 * poly[t + 1] - poly[t] = -1
                            -eq_address * eq_cycle
                        } else {
                            let eq_cycle = eq_poly.E1[*t] + eq_poly.E1[*t] - eq_poly.E1[*t - 1];
                            let eq_eval = eq_address * eq_cycle;
                            // poly[t + 1] = 1, poly[t] = 0
                            // => 2 * poly[t + 1] - poly[t] = 2
                            eq_eval + eq_eval
                        }
                    }
                    [(t1, k1), (t2, k2)] => {
                        debug_assert_eq!(t1 % 2, 0);
                        debug_assert_eq!(*t2, t1 + 1);
                        let eq_address = eq_poly.E2[*k2] + eq_poly.E2[*k2] - eq_poly.E2[*k1];
                        let eq_cycle = eq_poly.E1[*t2] + eq_poly.E1[*t2] - eq_poly.E1[*t1];
                        // poly[t + 1] = 1, poly[t] = 1
                        // => 2 * poly[t + 1] - poly[t] = 1
                        eq_address * eq_cycle
                    }
                    _ => panic!("Unexpected chunk with length > 2: {:?}", chunk),
                })
                .sum();
            vec![eval_0, eval_2]
        } else if self.num_variables_bound < num_cycle_variables {
            let eval_0: F = self
                .bound_coeffs
                .par_iter()
                .filter_map(|(t, k, coeff)| {
                    if t % 2 == 0 {
                        let eq_address = eq_poly.E2[*k];
                        let eq_cycle = eq_poly.E1[*t];
                        Some(eq_address * eq_cycle * coeff)
                    } else {
                        None
                    }
                })
                .sum();

            let eval_2: F = self
                .bound_coeffs
                .par_chunk_by(|(t1, k1, _), (t2, k2, _)| (t1 >> 1 == t2 >> 1) && k1 == k2)
                .map(|chunk| match chunk {
                    [(t, k, coeff)] => {
                        let eq_address = eq_poly.E2[*k];
                        if t % 2 == 0 {
                            let eq_cycle = eq_poly.E1[*t + 1] + eq_poly.E1[*t + 1] - eq_poly.E1[*t];
                            // poly[t + 1] = 0, poly[t] = coeff
                            // => 2 * poly[t + 1] - poly[t] = -coeff
                            -eq_address * eq_cycle * coeff
                        } else {
                            let eq_cycle = eq_poly.E1[*t] + eq_poly.E1[*t] - eq_poly.E1[*t - 1];
                            let eq_times_coeff = eq_address * eq_cycle * coeff;
                            // poly[t + 1] = 1, poly[t] = 0
                            // => 2 * poly[t + 1] - poly[t] = 2 * coeff
                            eq_times_coeff + eq_times_coeff
                        }
                    }
                    [(t1, k1, coeff1), (t2, k2, coeff2)] => {
                        debug_assert_eq!(t1 % 2, 0);
                        debug_assert_eq!(*t2, t1 + 1);
                        let eq_address = eq_poly.E2[*k2] + eq_poly.E2[*k2] - eq_poly.E2[*k1];
                        let eq_cycle = eq_poly.E1[*t2] + eq_poly.E1[*t2] - eq_poly.E1[*t1];
                        let poly_eval = *coeff2 + coeff2 - coeff1;
                        eq_address * eq_cycle * poly_eval
                    }
                    _ => panic!("Unexpected chunk with length > 2: {:?}", chunk),
                })
                .sum();
            vec![eval_0, eval_2]
        } else {
            let eval_0: F = self
                .bound_coeffs
                .par_iter()
                .filter_map(|(_, k, coeff)| {
                    if k % 2 == 0 {
                        Some(eq_poly.E2[*k] * coeff)
                    } else {
                        None
                    }
                })
                .sum();

            let eval_2: F = self
                .bound_coeffs
                .par_chunk_by(|(_, k1, _), (_, k2, _)| k1 >> 1 == k2 >> 1)
                .map(|chunk| match chunk {
                    [(t, k, coeff)] => {
                        debug_assert_eq!(*t, 0);
                        if k % 2 == 0 {
                            let eq_address =
                                eq_poly.E2[*k + 1] + eq_poly.E2[*k + 1] - eq_poly.E2[*k];
                            // poly[k + 1] = 0, poly[k] = coeff
                            // => 2 * poly[k + 1] - poly[k] = -coeff
                            -eq_address * coeff
                        } else {
                            let eq_address = eq_poly.E2[*k] + eq_poly.E2[*k] - eq_poly.E2[*k - 1];
                            let eq_times_coeff = eq_address * coeff;
                            // poly[k + 1] = 1, poly[k] = 0
                            // => 2 * poly[k + 1] - poly[k] = 2 * coeff
                            eq_times_coeff + eq_times_coeff
                        }
                    }
                    [(t1, k1, coeff1), (t2, k2, coeff2)] => {
                        debug_assert_eq!(*t1, 0);
                        debug_assert_eq!(*t2, 0);
                        debug_assert_eq!(*k1 + 1, *k2);
                        let eq_address = eq_poly.E2[*k2] + eq_poly.E2[*k2] - eq_poly.E2[*k1];
                        let poly_eval = *coeff2 + coeff2 - coeff1;
                        eq_address * poly_eval
                    }
                    _ => panic!("Unexpected chunk with length > 2: {:?}", chunk),
                })
                .sum();
            vec![eval_0, eval_2]
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for OneHotPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.num_variables_bound > 0
    }

    fn bind(&mut self, r: F, order: BindingOrder) {
        assert_eq!(order, BindingOrder::LowToHigh);
        todo!()
    }

    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        assert_eq!(order, BindingOrder::LowToHigh);
        let num_cycle_variables = get_T().log_2();
        if self.num_variables_bound == 0 {
            // Bind cycle variable
            self.bound_coeffs = self
                .nonzero_indices
                .par_chunk_by(|(t1, k1), (t2, k2)| (t1 >> 1 == t2 >> 1) && k1 == k2)
                .map(|chunk| match chunk {
                    [(t, k)] => {
                        let bound_coeff = if t % 2 == 0 { F::one() - r } else { r };
                        (t / 2, *k, bound_coeff)
                    }
                    [(t1, k1), (t2, k2)] => {
                        debug_assert_eq!(*t2, t1 + 1);
                        debug_assert_eq!(k1, k2);
                        (t1 / 2, *k1, F::one())
                    }
                    _ => panic!("Unexpected chunk with length > 2: {:?}", chunk),
                })
                .collect();
        } else if self.num_variables_bound < num_cycle_variables {
            // Bind cycle variable
            self.binding_scratch_space = self
                .bound_coeffs
                .par_chunk_by(|(t1, k1, _), (t2, k2, _)| (t1 >> 1 == t2 >> 1) && k1 == k2)
                .map(|chunk| match chunk {
                    [(t, k, coeff)] => {
                        let bound_coeff = if *t % 2 == 0 {
                            *coeff * (F::one() - r)
                        } else {
                            *coeff * r
                        };
                        (*t / 2, *k, bound_coeff)
                    }
                    [(t1, k1, coeff1), (t2, k2, coeff2)] => {
                        debug_assert_eq!(*t1 % 2, 0);
                        debug_assert_eq!(*t2, *t1 + 1);
                        debug_assert_eq!(k1, k2);
                        let bound_coeff = *coeff1 + (*coeff2 - coeff1) * r;
                        (*t1 / 2, *k1, bound_coeff)
                    }
                    _ => panic!("Unexpected chunk with length > 2: {:?}", chunk),
                })
                .collect();
            std::mem::swap(&mut self.bound_coeffs, &mut self.binding_scratch_space);
        } else {
            // Bind address variable
            self.binding_scratch_space = self
                .bound_coeffs
                .par_chunk_by(|(_, k1, _), (_, k2, _)| k1 >> 1 == k2 >> 1)
                .map(|chunk| match chunk {
                    [(0, k, coeff)] => {
                        let bound_coeff = if *k % 2 == 0 {
                            *coeff * (F::one() - r)
                        } else {
                            *coeff * r
                        };
                        (0, *k / 2, bound_coeff)
                    }
                    [(0, k1, coeff1), (0, k2, coeff2)] => {
                        debug_assert_eq!(*k1 + 1, *k2);
                        let bound_coeff = *coeff1 + (*coeff2 - coeff1) * r;
                        (0, *k1 / 2, bound_coeff)
                    }
                    _ => panic!("Unexpected chunk: {:?}", chunk),
                })
                .collect();

            std::mem::swap(&mut self.bound_coeffs, &mut self.binding_scratch_space);
        }
        self.num_variables_bound += 1;
        debug_assert!(self.num_variables_bound <= self.K.log_2() + num_cycle_variables);

        if self.num_variables_bound == num_cycle_variables {
            println!("Sorting bound_coeffs...");
            // TODO(moodlezoup): avoid sorting
            self.bound_coeffs.sort_unstable_by_key(|(_, k, _)| *k);
            self.binding_scratch_space = self
                .bound_coeffs
                .par_chunk_by(|(_, k1, _), (_, k2, _)| k1 == k2)
                .map(|chunk| {
                    let k = chunk[0].1;
                    let mut result = (0, k, F::zero());
                    for (t, _, coeff) in chunk.iter() {
                        debug_assert_eq!(*t, 0);
                        result.2 += *coeff;
                    }
                    result
                })
                .collect();
            std::mem::swap(&mut self.bound_coeffs, &mut self.binding_scratch_space);
        }
    }

    fn final_sumcheck_claim(&self) -> F {
        assert_eq!(self.bound_coeffs.len(), 1);
        self.bound_coeffs[0].2
    }
}

#[cfg(test)]
mod tests {
    use crate::poly::sparse_matrix_polynomial::SparseMatrixPolynomial;

    use super::*;
    use ark_bn254::Fr;
    use ark_std::{test_rng, Zero};
    use rand_core::RngCore;

    fn dense_polynomial_equivalence<const LOG_K: usize, const LOG_T: usize>() {
        let K: usize = 1 << LOG_K;
        let T: usize = 1 << LOG_T;
        SparseMatrixPolynomial::<Fr>::initialize(K, T);

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
        let mut eq = split_eq.merge();

        for round in 0..LOG_K + LOG_T {
            let one_hot_message = one_hot_poly.compute_sumcheck_prover_message(&split_eq);
            let mut expected_message = vec![Fr::zero(), Fr::zero()];
            let mle_half = dense_poly.len() / 2;
            expected_message[0] = (0..mle_half).map(|i| dense_poly[2 * i] * eq[2 * i]).sum();
            expected_message[1] = (0..mle_half)
                .map(|i| {
                    let poly_bound_point =
                        dense_poly[2 * i + 1] + dense_poly[2 * i + 1] - dense_poly[2 * i];
                    let eq_bound_point = eq[2 * i + 1] + eq[2 * i + 1] - eq[2 * i];
                    poly_bound_point * eq_bound_point
                })
                .sum();
            assert_eq!(
                one_hot_message, expected_message,
                "round {round} prover message mismatch"
            );

            let r = Fr::random(&mut rng);
            one_hot_poly.bind_parallel(r, BindingOrder::LowToHigh);
            split_eq.bind(r);
            dense_poly.bind_parallel(r, BindingOrder::LowToHigh);
            eq.bind_parallel(r, BindingOrder::LowToHigh);
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
