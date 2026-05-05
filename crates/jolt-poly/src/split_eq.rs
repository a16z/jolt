//! Split equality polynomial used by sumcheck provers.
//!
//! This implements the Dao-Thaler/Gruen factorization used by Jolt's larger
//! sumchecks. It stores prefix tables for two halves of the remaining equality
//! polynomial and tracks already-bound variables in a scalar.

use jolt_field::Field;

use crate::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GruenSplitEqPolynomial<F: Field> {
    current_index: usize,
    current_scalar: F,
    w: Vec<F>,
    e_in_vec: Vec<Vec<F>>,
    e_out_vec: Vec<Vec<F>>,
    binding_order: BindingOrder,
}

impl<F: Field> GruenSplitEqPolynomial<F> {
    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::new_with_scaling")]
    pub fn new_with_scaling(
        w: &[F],
        binding_order: BindingOrder,
        scaling_factor: Option<F>,
    ) -> Self {
        assert!(!w.is_empty(), "split eq requires at least one variable");
        match binding_order {
            BindingOrder::LowToHigh => {
                let split = w.len() / 2;
                let w_prime = &w[..w.len() - 1];
                let (w_out, w_in) = w_prime.split_at(split);
                let (e_out_vec, e_in_vec) = join_or_serial(
                    || EqPolynomial::evals_cached(w_out, None),
                    || EqPolynomial::evals_cached(w_in, None),
                );
                Self {
                    current_index: w.len(),
                    current_scalar: scaling_factor.unwrap_or_else(F::one),
                    w: w.to_vec(),
                    e_in_vec,
                    e_out_vec,
                    binding_order,
                }
            }
            BindingOrder::HighToLow => {
                let w_prime = &w[1..];
                let split = w.len() / 2;
                let (w_in, w_out) = w_prime.split_at(split);
                let (e_in_vec, e_out_vec) = join_or_serial(
                    || EqPolynomial::evals_cached_rev(w_in, None),
                    || EqPolynomial::evals_cached_rev(w_out, None),
                );
                Self {
                    current_index: 0,
                    current_scalar: scaling_factor.unwrap_or_else(F::one),
                    w: w.to_vec(),
                    e_in_vec,
                    e_out_vec,
                    binding_order,
                }
            }
        }
    }

    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::new")]
    pub fn new(w: &[F], binding_order: BindingOrder) -> Self {
        Self::new_with_scaling(w, binding_order, None)
    }

    #[inline]
    pub fn num_vars(&self) -> usize {
        self.w.len()
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self.binding_order {
            BindingOrder::LowToHigh => 1 << self.current_index,
            BindingOrder::HighToLow => 1 << (self.w.len() - self.current_index),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        false
    }

    #[inline]
    pub fn num_bound_vars(&self) -> usize {
        match self.binding_order {
            BindingOrder::LowToHigh => self.w.len() - self.current_index,
            BindingOrder::HighToLow => self.current_index,
        }
    }

    #[inline]
    pub fn e_in_current_len(&self) -> usize {
        self.e_in_current().len()
    }

    #[inline]
    pub fn e_out_current_len(&self) -> usize {
        self.e_out_current().len()
    }

    #[inline]
    pub fn e_in_current(&self) -> &[F] {
        &self.e_in_vec[self.e_in_vec.len() - 1]
    }

    #[inline]
    pub fn e_out_current(&self) -> &[F] {
        &self.e_out_vec[self.e_out_vec.len() - 1]
    }

    pub fn e_out_in_for_window(&self, window_size: usize) -> (&[F], &[F]) {
        assert_eq!(
            self.binding_order,
            BindingOrder::LowToHigh,
            "streaming windows are not defined for high-to-low split eq"
        );
        let num_unbound = self.current_index;
        let window_size = window_size.min(num_unbound);
        let head_len = num_unbound.saturating_sub(window_size);
        let split = self.w.len() / 2;
        let head_out_bits = head_len.min(split);
        let head_in_bits = head_len.saturating_sub(head_out_bits);
        (&self.e_out_vec[head_out_bits], &self.e_in_vec[head_in_bits])
    }

    pub fn e_active_for_window(&self, window_size: usize) -> Vec<F> {
        if window_size <= 1 {
            return vec![F::one()];
        }
        assert_eq!(
            self.binding_order,
            BindingOrder::LowToHigh,
            "streaming windows are not defined for high-to-low split eq"
        );
        let num_unbound = self.current_index;
        if window_size > num_unbound {
            return vec![F::one()];
        }
        let remaining_w = &self.w[..num_unbound];
        let window_start = remaining_w.len() - window_size;
        let (_head, w_window) = remaining_w.split_at(window_start);
        let (w_active, _w_current) = w_window.split_at(window_size - 1);
        EqPolynomial::<F>::evals(w_active, None)
    }

    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::bind")]
    pub fn bind(&mut self, r: F) {
        match self.binding_order {
            BindingOrder::LowToHigh => {
                let w = self.w[self.current_index - 1];
                let prod = w * r;
                self.current_scalar *= F::one() - w - r + prod + prod;
                self.current_index -= 1;
                if self.w.len() / 2 < self.current_index && self.e_in_vec.len() > 1 {
                    let _ = self.e_in_vec.pop();
                } else if 0 < self.current_index && self.e_out_vec.len() > 1 {
                    let _ = self.e_out_vec.pop();
                }
            }
            BindingOrder::HighToLow => {
                let w = self.w[self.current_index];
                let prod = w * r;
                self.current_scalar *= F::one() - w - r + prod + prod;
                self.current_index += 1;
                if self.current_index <= self.w.len() / 2 && self.e_in_vec.len() > 1 {
                    let _ = self.e_in_vec.pop();
                } else if self.current_index <= self.w.len() && self.e_out_vec.len() > 1 {
                    let _ = self.e_out_vec.pop();
                }
            }
        }
    }

    pub fn gruen_poly_deg_3(
        &self,
        q_constant: F,
        q_quadratic_coeff: F,
        s_0_plus_s_1: F,
    ) -> UnivariatePoly<F> {
        let eq_eval_1 = self.current_scalar * self.current_w();
        let eq_eval_0 = self.current_scalar - eq_eval_1;
        let eq_slope = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_slope;
        let eq_eval_3 = eq_eval_2 + eq_slope;

        let quadratic_eval_0 = q_constant;
        let cubic_eval_0 = eq_eval_0 * quadratic_eval_0;
        let cubic_eval_1 = s_0_plus_s_1 - cubic_eval_0;
        let quadratic_eval_1 = cubic_eval_1 / eq_eval_1;
        let e_times_2 = q_quadratic_coeff + q_quadratic_coeff;
        let quadratic_eval_2 = quadratic_eval_1 + quadratic_eval_1 - quadratic_eval_0 + e_times_2;
        let quadratic_eval_3 =
            quadratic_eval_2 + quadratic_eval_1 - quadratic_eval_0 + e_times_2 + e_times_2;

        UnivariatePoly::from_evals(&[
            cubic_eval_0,
            cubic_eval_1,
            eq_eval_2 * quadratic_eval_2,
            eq_eval_3 * quadratic_eval_3,
        ])
    }

    pub fn gruen_poly_deg_2(&self, q_0: F, previous_claim: F) -> UnivariatePoly<F> {
        let eq_eval_1 = self.current_scalar * self.current_w();
        let eq_eval_0 = self.current_scalar - eq_eval_1;
        let eq_slope = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_slope;

        let quadratic_eval_0 = eq_eval_0 * q_0;
        let quadratic_eval_1 = previous_claim - quadratic_eval_0;
        let linear_eval_1 = quadratic_eval_1 / eq_eval_1;
        let linear_eval_2 = linear_eval_1 + linear_eval_1 - q_0;

        UnivariatePoly::from_evals(&[
            quadratic_eval_0,
            quadratic_eval_1,
            eq_eval_2 * linear_eval_2,
        ])
    }

    pub fn gruen_poly_from_evals(&self, q_evals: &[F], s_0_plus_s_1: F) -> UnivariatePoly<F> {
        assert!(!q_evals.is_empty(), "q_evals must be non-empty");
        let r_round = self.current_w();
        let l_at_0 = self.current_scalar * EqPolynomial::<F>::mle(&[F::zero()], &[r_round]);
        let l_at_1 = self.current_scalar * EqPolynomial::<F>::mle(&[F::one()], &[r_round]);
        let q_at_0 = (s_0_plus_s_1 - l_at_1 * q_evals[0]) / l_at_0;

        let mut full_q_evals = Vec::with_capacity(q_evals.len() + 1);
        full_q_evals.push(q_at_0);
        full_q_evals.extend_from_slice(q_evals);
        let q = UnivariatePoly::from_evals_toom(&full_q_evals);

        let l_c0 = l_at_0;
        let l_c1 = l_at_1 - l_at_0;
        let q_coeffs = q.into_coefficients();
        let mut s_coeffs = vec![F::zero(); q_coeffs.len() + 1];
        for (index, q_coeff) in q_coeffs.into_iter().enumerate() {
            s_coeffs[index] += q_coeff * l_c0;
            s_coeffs[index + 1] += q_coeff * l_c1;
        }
        UnivariatePoly::new(s_coeffs)
    }

    pub fn merge(&self) -> Polynomial<F> {
        let evals = match self.binding_order {
            BindingOrder::LowToHigh => {
                EqPolynomial::evals(&self.w[..self.current_index], Some(self.current_scalar))
            }
            BindingOrder::HighToLow => {
                EqPolynomial::evals(&self.w[self.current_index..], Some(self.current_scalar))
            }
        };
        Polynomial::new(evals)
    }

    #[inline]
    pub fn current_scalar(&self) -> F {
        self.current_scalar
    }

    #[inline]
    pub fn current_w(&self) -> F {
        match self.binding_order {
            BindingOrder::LowToHigh => self.w[self.current_index - 1],
            BindingOrder::HighToLow => self.w[self.current_index],
        }
    }

    #[inline]
    pub fn group_index(&self, x_out: usize, x_in: usize) -> usize {
        let num_x_in_bits = self.e_in_current_len().trailing_zeros() as usize;
        (x_out << num_x_in_bits) | x_in
    }

    pub fn fold_out_in<
        OuterAcc: Send,
        InnerAcc: Send,
        MakeInner: Fn() -> InnerAcc + Sync + Send,
        InnerStep: Fn(&mut InnerAcc, usize, usize, F) + Sync + Send,
        OuterStep: Fn(usize, F, InnerAcc) -> OuterAcc + Sync + Send,
        Merge: Fn(OuterAcc, OuterAcc) -> OuterAcc + Sync + Send,
    >(
        &self,
        make_inner: MakeInner,
        inner_step: InnerStep,
        outer_step: OuterStep,
        merge: Merge,
    ) -> OuterAcc {
        let e_out = self.e_out_current();
        let e_in = self.e_in_current();

        #[cfg(feature = "parallel")]
        {
            let result = (0..e_out.len())
                .into_par_iter()
                .map(|x_out| {
                    let mut inner_acc = make_inner();
                    for (x_in, &e_in) in e_in.iter().enumerate() {
                        let group = self.group_index(x_out, x_in);
                        inner_step(&mut inner_acc, group, x_in, e_in);
                    }
                    outer_step(x_out, e_out[x_out], inner_acc)
                })
                .reduce_with(merge);
            if let Some(result) = result {
                result
            } else {
                assert!(!e_out.is_empty(), "split eq e_out invariant");
                std::process::abort();
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut iter = (0..e_out.len()).map(|x_out| {
                let mut inner_acc = make_inner();
                for (x_in, &e_in) in e_in.iter().enumerate() {
                    let group = self.group_index(x_out, x_in);
                    inner_step(&mut inner_acc, group, x_in, e_in);
                }
                outer_step(x_out, e_out[x_out], inner_acc)
            });
            let first = iter.next().expect("split eq e_out invariant");
            iter.fold(first, merge)
        }
    }
}

#[cfg(feature = "parallel")]
fn join_or_serial<A: Send, B: Send>(
    left: impl FnOnce() -> A + Send,
    right: impl FnOnce() -> B + Send,
) -> (A, B) {
    rayon::join(left, right)
}

#[cfg(not(feature = "parallel"))]
fn join_or_serial<A, B>(left: impl FnOnce() -> A, right: impl FnOnce() -> B) -> (A, B) {
    (left(), right())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Math;
    use jolt_field::{Field, Fr};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn bind_low_to_high_matches_dense_eq() {
        let mut rng = ChaCha20Rng::seed_from_u64(700);
        let point: Vec<Fr> = (0..10).map(|_| Fr::random(&mut rng)).collect();
        let mut dense = Polynomial::new(EqPolynomial::<Fr>::evals(&point, None));
        let mut split = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
        assert_eq!(dense, split.merge());

        for _ in 0..point.len() {
            let r = Fr::random(&mut rng);
            dense.bind_with_order(r, BindingOrder::LowToHigh);
            split.bind(r);
            assert_eq!(dense, split.merge());
        }
    }

    #[test]
    fn bind_high_to_low_matches_dense_eq() {
        let mut rng = ChaCha20Rng::seed_from_u64(701);
        let point: Vec<Fr> = (0..10).map(|_| Fr::random(&mut rng)).collect();
        let mut dense = Polynomial::new(EqPolynomial::<Fr>::evals(&point, None));
        let mut split = GruenSplitEqPolynomial::new(&point, BindingOrder::HighToLow);
        assert_eq!(dense, split.merge());

        for _ in 0..point.len() {
            let r = Fr::random(&mut rng);
            dense.bind_with_order(r, BindingOrder::HighToLow);
            split.bind(r);
            assert_eq!(dense, split.merge());
        }
    }

    #[test]
    fn window_size_one_factors_current_head() {
        let mut rng = ChaCha20Rng::seed_from_u64(702);
        let point: Vec<Fr> = (0..10).map(|_| Fr::random(&mut rng)).collect();
        let mut split = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);

        for _round in 0..point.len() {
            let num_unbound = split.current_index;
            if num_unbound <= 1 {
                break;
            }
            let (e_out, e_in) = split.e_out_in_for_window(1);
            let head = EqPolynomial::<Fr>::evals(&split.w[..num_unbound - 1], None);
            assert_eq!(e_out.len() * e_in.len(), head.len());

            let x_in_bits = e_in.len().log_2();
            for (x_out, &e_out) in e_out.iter().enumerate() {
                for (x_in, &e_in) in e_in.iter().enumerate() {
                    let index = (x_out << x_in_bits) | x_in;
                    assert_eq!(e_out * e_in, head[index]);
                }
            }

            split.bind(Fr::random(&mut rng));
        }
    }

    #[test]
    fn gruen_degree_two_matches_direct_interpolation() {
        let mut rng = ChaCha20Rng::seed_from_u64(703);
        let point: Vec<Fr> = (0..5).map(|_| Fr::random(&mut rng)).collect();
        let split = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
        let q0 = Fr::from_u64(11);
        let q1 = Fr::from_u64(29);
        let l1 = split.current_scalar() * split.current_w();
        let l0 = split.current_scalar() - l1;
        let previous_claim = l0 * q0 + l1 * q1;

        let poly = split.gruen_poly_deg_2(q0, previous_claim);
        let q2 = q1 + q1 - q0;
        let l2 = l1 + (l1 - l0);
        assert_eq!(poly.evaluate(Fr::zero()), l0 * q0);
        assert_eq!(poly.evaluate(Fr::one()), l1 * q1);
        assert_eq!(poly.evaluate(Fr::from_u64(2)), l2 * q2);
    }
}
