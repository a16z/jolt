//! Split equality tables for sqrt-memory sumcheck kernels.

use jolt_field::Field;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorEqTable<F: Field> {
    e_out: Vec<F>,
    e_in: Vec<F>,
    in_bits: usize,
}

impl<F: Field> TensorEqTable<F> {
    pub fn new(point: &[F]) -> Self {
        let split = point.len() / 2;
        let (out_point, in_point) = point.split_at(split);
        #[cfg(feature = "parallel")]
        let (e_out, e_in) = rayon::join(
            || EqPolynomial::<F>::evals(out_point, None),
            || EqPolynomial::<F>::evals(in_point, None),
        );
        #[cfg(not(feature = "parallel"))]
        let (e_out, e_in) = (
            EqPolynomial::<F>::evals(out_point, None),
            EqPolynomial::<F>::evals(in_point, None),
        );
        Self {
            e_out,
            e_in,
            in_bits: in_point.len(),
        }
    }

    pub fn len(&self) -> usize {
        self.e_out.len() * self.e_in.len()
    }

    pub fn is_empty(&self) -> bool {
        self.e_out.is_empty() || self.e_in.is_empty()
    }

    pub fn e_out(&self) -> &[F] {
        &self.e_out
    }

    pub fn e_in(&self) -> &[F] {
        &self.e_in
    }

    pub fn evaluate_index(&self, index: usize) -> F {
        let x_out = index >> self.in_bits;
        let x_in = index & ((1usize << self.in_bits) - 1);
        self.e_out[x_out] * self.e_in[x_in]
    }

    pub fn evaluate_slices(&self, values: &[&[F]]) -> Vec<F> {
        if values.is_empty() {
            return Vec::new();
        }
        debug_assert!(
            values.iter().all(|values| values.len() == self.len()),
            "TensorEqTable::evaluate_slices length mismatch"
        );

        self.par_fold_out_in(
            || vec![F::zero(); values.len()],
            |inner, row, _x_in, e_in| {
                if e_in.is_zero() {
                    return;
                }
                for (accumulator, values) in inner.iter_mut().zip(values) {
                    *accumulator += e_in * values[row];
                }
            },
            |_x_out, e_out, mut inner| {
                if e_out.is_zero() {
                    inner.fill(F::zero());
                } else {
                    for value in &mut inner {
                        *value *= e_out;
                    }
                }
                inner
            },
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            },
        )
    }

    #[inline(always)]
    pub fn group_index(&self, x_out: usize, x_in: usize) -> usize {
        (x_out << self.in_bits) | x_in
    }

    #[inline]
    pub fn par_fold_out_in<
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
        #[cfg(feature = "parallel")]
        {
            (0..self.e_out.len())
                .into_par_iter()
                .map(|x_out| {
                    let mut inner_acc = make_inner();
                    for (x_in, &e_in) in self.e_in.iter().enumerate() {
                        let row = self.group_index(x_out, x_in);
                        inner_step(&mut inner_acc, row, x_in, e_in);
                    }
                    outer_step(x_out, self.e_out[x_out], inner_acc)
                })
                .reduce_with(merge)
                .unwrap_or_else(|| {
                    let inner_acc = make_inner();
                    outer_step(0, F::zero(), inner_acc)
                })
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut acc = None;
            for (x_out, &e_out) in self.e_out.iter().enumerate() {
                let mut inner_acc = make_inner();
                for (x_in, &e_in) in self.e_in.iter().enumerate() {
                    let row = self.group_index(x_out, x_in);
                    inner_step(&mut inner_acc, row, x_in, e_in);
                }
                let value = outer_step(x_out, e_out, inner_acc);
                acc = Some(match acc {
                    Some(acc) => merge(acc, value),
                    None => value,
                });
            }
            acc.unwrap_or_else(|| {
                let inner_acc = make_inner();
                outer_step(0, F::zero(), inner_acc)
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GruenSplitEqPolynomial<F: Field> {
    current_index: usize,
    current_scalar: F,
    point: Vec<F>,
    e_in_vec: Vec<Vec<F>>,
    e_out_vec: Vec<Vec<F>>,
    binding_order: BindingOrder,
}

impl<F: Field> GruenSplitEqPolynomial<F> {
    pub fn new(point: &[F], binding_order: BindingOrder) -> Self {
        Self::new_with_scaling(point, binding_order, None)
    }

    pub fn new_with_scaling(
        point: &[F],
        binding_order: BindingOrder,
        scaling_factor: Option<F>,
    ) -> Self {
        if point.is_empty() {
            return Self {
                current_index: match binding_order {
                    BindingOrder::LowToHigh => 0,
                    BindingOrder::HighToLow => 0,
                },
                current_scalar: scaling_factor.unwrap_or(F::one()),
                point: Vec::new(),
                e_in_vec: vec![vec![F::one()]],
                e_out_vec: vec![vec![F::one()]],
                binding_order,
            };
        }

        match binding_order {
            BindingOrder::LowToHigh => {
                let split = point.len() / 2;
                let head = &point[..point.len() - 1];
                let (out_point, in_point) = head.split_at(split.min(head.len()));
                #[cfg(feature = "parallel")]
                let (e_out_vec, e_in_vec) = rayon::join(
                    || EqPolynomial::<F>::evals_cached(out_point, None),
                    || EqPolynomial::<F>::evals_cached(in_point, None),
                );
                #[cfg(not(feature = "parallel"))]
                let (e_out_vec, e_in_vec) = (
                    EqPolynomial::<F>::evals_cached(out_point, None),
                    EqPolynomial::<F>::evals_cached(in_point, None),
                );
                Self {
                    current_index: point.len(),
                    current_scalar: scaling_factor.unwrap_or(F::one()),
                    point: point.to_vec(),
                    e_in_vec,
                    e_out_vec,
                    binding_order,
                }
            }
            BindingOrder::HighToLow => {
                let split = point.len() / 2;
                let tail = &point[1..];
                let (in_point, out_point) = tail.split_at(split.min(tail.len()));
                #[cfg(feature = "parallel")]
                let (e_in_vec, e_out_vec) = rayon::join(
                    || EqPolynomial::<F>::evals_cached_rev(in_point, None),
                    || EqPolynomial::<F>::evals_cached_rev(out_point, None),
                );
                #[cfg(not(feature = "parallel"))]
                let (e_in_vec, e_out_vec) = (
                    EqPolynomial::<F>::evals_cached_rev(in_point, None),
                    EqPolynomial::<F>::evals_cached_rev(out_point, None),
                );
                Self {
                    current_index: 0,
                    current_scalar: scaling_factor.unwrap_or(F::one()),
                    point: point.to_vec(),
                    e_in_vec,
                    e_out_vec,
                    binding_order,
                }
            }
        }
    }

    pub fn current_scalar(&self) -> F {
        self.current_scalar
    }

    pub fn current_linear_evals(&self) -> (F, F) {
        let point = match self.binding_order {
            BindingOrder::LowToHigh => self.point[self.current_index - 1],
            BindingOrder::HighToLow => self.point[self.current_index],
        };
        let at_one = self.current_scalar * point;
        (self.current_scalar - at_one, at_one)
    }

    pub fn current_index(&self) -> usize {
        self.current_index
    }

    pub fn e_in_current(&self) -> &[F] {
        &self.e_in_vec[self.e_in_vec.len() - 1]
    }

    pub fn e_out_current(&self) -> &[F] {
        &self.e_out_vec[self.e_out_vec.len() - 1]
    }

    pub fn e_in_current_len(&self) -> usize {
        self.e_in_current().len()
    }

    pub fn e_out_current_len(&self) -> usize {
        self.e_out_current().len()
    }

    pub fn e_out_in_for_window(&self, window_size: usize) -> (&[F], &[F]) {
        assert!(
            matches!(self.binding_order, BindingOrder::LowToHigh),
            "streaming split-eq windows are only defined for low-to-high"
        );

        let window_size = core::cmp::min(window_size, self.current_index);
        let head_len = self.current_index.saturating_sub(window_size);
        let split = self.point.len() / 2;

        let head_out_bits = core::cmp::min(head_len, split);
        let head_in_bits = head_len.saturating_sub(head_out_bits);

        debug_assert_eq!(head_out_bits + head_in_bits, head_len);
        debug_assert!(head_out_bits < self.e_out_vec.len());
        debug_assert!(head_in_bits < self.e_in_vec.len());

        (&self.e_out_vec[head_out_bits], &self.e_in_vec[head_in_bits])
    }

    pub fn e_active_for_window(&self, window_size: usize) -> Vec<F> {
        assert!(
            matches!(self.binding_order, BindingOrder::LowToHigh),
            "streaming split-eq windows are only defined for low-to-high"
        );

        if window_size <= 1 {
            return vec![F::one()];
        }

        let num_unbound = self.current_index;
        if window_size > num_unbound {
            return vec![F::one()];
        }

        let remaining_point = &self.point[..num_unbound];
        let window_start = remaining_point.len() - window_size;
        let (_, window_point) = remaining_point.split_at(window_start);
        let (active_point, _) = window_point.split_at(window_size - 1);
        EqPolynomial::<F>::evals(active_point, None)
    }

    pub fn bind(&mut self, challenge: F) {
        if self.point.is_empty() {
            return;
        }

        match self.binding_order {
            BindingOrder::LowToHigh => {
                let point = self.point[self.current_index - 1];
                let product = point * challenge;
                self.current_scalar *= F::one() - point - challenge + product + product;
                self.current_index -= 1;
                if self.point.len() / 2 < self.current_index && self.e_in_vec.len() > 1 {
                    let _ = self.e_in_vec.pop();
                } else if 0 < self.current_index && self.e_out_vec.len() > 1 {
                    let _ = self.e_out_vec.pop();
                }
            }
            BindingOrder::HighToLow => {
                let point = self.point[self.current_index];
                let product = point * challenge;
                self.current_scalar *= F::one() - point - challenge + product + product;
                self.current_index += 1;
                if self.current_index <= self.point.len() / 2 && self.e_in_vec.len() > 1 {
                    let _ = self.e_in_vec.pop();
                } else if self.current_index <= self.point.len() && self.e_out_vec.len() > 1 {
                    let _ = self.e_out_vec.pop();
                }
            }
        }
    }

    pub fn merge(&self) -> Polynomial<F> {
        let evals = match self.binding_order {
            BindingOrder::LowToHigh => EqPolynomial::<F>::evals(
                &self.point[..self.current_index],
                Some(self.current_scalar),
            ),
            BindingOrder::HighToLow => EqPolynomial::<F>::evals(
                &self.point[self.current_index..],
                Some(self.current_scalar),
            ),
        };
        Polynomial::new(evals)
    }

    /// Computes `s(X) = l(X) * q(X)` where `l` is the current linear eq
    /// factor and `q` is quadratic, represented by its constant and quadratic
    /// coefficients plus the sumcheck hint `s(0) + s(1)`.
    #[expect(clippy::expect_used)]
    pub fn gruen_poly_deg_3(
        &self,
        q_constant: F,
        q_quadratic_coeff: F,
        s_0_plus_s_1: F,
    ) -> UnivariatePoly<F> {
        let eq_eval_1 = self.current_scalar
            * match self.binding_order {
                BindingOrder::LowToHigh => self.point[self.current_index - 1],
                BindingOrder::HighToLow => self.point[self.current_index],
            };
        let eq_eval_0 = self.current_scalar - eq_eval_1;
        let eq_m = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_m;
        let eq_eval_3 = eq_eval_2 + eq_m;

        let quadratic_eval_0 = q_constant;
        let cubic_eval_0 = eq_eval_0 * quadratic_eval_0;
        let cubic_eval_1 = s_0_plus_s_1 - cubic_eval_0;
        let quadratic_eval_1 = cubic_eval_1
            * eq_eval_1
                .inverse()
                .expect("current eq evaluation at one must be invertible");
        let e_times_2 = q_quadratic_coeff + q_quadratic_coeff;
        let quadratic_eval_2 = quadratic_eval_1 + quadratic_eval_1 - quadratic_eval_0 + e_times_2;
        let quadratic_eval_3 =
            quadratic_eval_2 + quadratic_eval_1 - quadratic_eval_0 + e_times_2 + e_times_2;

        UnivariatePoly::interpolate_over_integers(&[
            cubic_eval_0,
            cubic_eval_1,
            eq_eval_2 * quadratic_eval_2,
            eq_eval_3 * quadratic_eval_3,
        ])
    }

    #[expect(clippy::expect_used)]
    pub fn gruen_poly_from_evals(&self, q_evals: &[F], s_0_plus_s_1: F) -> UnivariatePoly<F> {
        let r_round = match self.binding_order {
            BindingOrder::LowToHigh => self.point[self.current_index - 1],
            BindingOrder::HighToLow => self.point[self.current_index],
        };

        let l_at_0 = self.current_scalar * (F::one() - r_round);
        let l_at_1 = self.current_scalar * r_round;
        let q_at_0 = (s_0_plus_s_1 - l_at_1 * q_evals[0])
            * l_at_0
                .inverse()
                .expect("current eq evaluation at zero must be invertible");

        let mut full_q_evals = Vec::with_capacity(q_evals.len() + 1);
        full_q_evals.push(q_at_0);
        full_q_evals.extend_from_slice(q_evals);
        let q_coeffs = UnivariatePoly::from_evals_toom(&full_q_evals).into_coefficients();

        let l_c0 = l_at_0;
        let l_c1 = l_at_1 - l_at_0;
        let mut s_coeffs = vec![F::zero(); q_coeffs.len() + 1];
        for (index, q_coeff) in q_coeffs.into_iter().enumerate() {
            s_coeffs[index] += q_coeff * l_c0;
            s_coeffs[index + 1] += q_coeff * l_c1;
        }

        UnivariatePoly::new(s_coeffs)
    }

    #[inline(always)]
    pub fn group_index(&self, x_out: usize, x_in: usize) -> usize {
        let in_bits = self.e_in_current_len().trailing_zeros() as usize;
        (x_out << in_bits) | x_in
    }

    #[inline]
    pub fn par_fold_out_in<
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
            (0..e_out.len())
                .into_par_iter()
                .map(|x_out| {
                    let mut inner_acc = make_inner();
                    for (x_in, &e_in) in e_in.iter().enumerate() {
                        let row = self.group_index(x_out, x_in);
                        inner_step(&mut inner_acc, row, x_in, e_in);
                    }
                    outer_step(x_out, e_out[x_out], inner_acc)
                })
                .reduce_with(merge)
                .unwrap_or_else(|| {
                    let inner_acc = make_inner();
                    outer_step(0, F::zero(), inner_acc)
                })
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut acc = None;
            for (x_out, &e_out_val) in e_out.iter().enumerate() {
                let mut inner_acc = make_inner();
                for (x_in, &e_in_val) in e_in.iter().enumerate() {
                    let row = self.group_index(x_out, x_in);
                    inner_step(&mut inner_acc, row, x_in, e_in_val);
                }
                let value = outer_step(x_out, e_out_val, inner_acc);
                acc = Some(match acc {
                    Some(acc) => merge(acc, value),
                    None => value,
                });
            }
            acc.unwrap_or_else(|| {
                let inner_acc = make_inner();
                outer_step(0, F::zero(), inner_acc)
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use jolt_field::{FieldCore, Fr, FromPrimitiveInt};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::*;

    fn random_point(len: usize, seed: u64) -> Vec<Fr> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..len).map(|_| Fr::random(&mut rng)).collect()
    }

    #[test]
    fn tensor_eq_table_factors_full_eq_table() {
        for vars in 0..=10 {
            let point = random_point(vars, 100 + vars as u64);
            let tensor = TensorEqTable::<Fr>::new(&point);
            let full = EqPolynomial::<Fr>::evals(&point, None);
            assert_eq!(tensor.len(), full.len());
            for x_out in 0..tensor.e_out().len() {
                for x_in in 0..tensor.e_in().len() {
                    let row = tensor.group_index(x_out, x_in);
                    assert_eq!(tensor.e_out()[x_out] * tensor.e_in()[x_in], full[row]);
                }
            }
        }
    }

    #[test]
    fn tensor_eq_fold_matches_full_table_dot_product() {
        let point = random_point(9, 211);
        let values = random_point(1 << point.len(), 307);
        let tensor = TensorEqTable::<Fr>::new(&point);
        let folded = tensor.par_fold_out_in(
            || Fr::from_u64(0),
            |inner, row, _x_in, e_in| {
                *inner += e_in * values[row];
            },
            |_x_out, e_out, inner| e_out * inner,
            |left, right| left + right,
        );
        let full = EqPolynomial::<Fr>::evals(&point, None)
            .into_iter()
            .zip(values)
            .map(|(eq, value)| eq * value)
            .sum::<Fr>();
        assert_eq!(folded, full);
    }

    #[test]
    fn tensor_eq_evaluates_slices_in_one_fold() {
        let point = random_point(8, 811);
        let values = [
            random_point(1 << point.len(), 907),
            random_point(1 << point.len(), 1009),
            random_point(1 << point.len(), 1103),
        ];
        let slices = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let tensor = TensorEqTable::<Fr>::new(&point);
        let actual = tensor.evaluate_slices(&slices);
        let eq = EqPolynomial::<Fr>::evals(&point, None);
        let expected = values
            .iter()
            .map(|values| {
                eq.iter()
                    .zip(values)
                    .map(|(&eq, &value)| eq * value)
                    .sum::<Fr>()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn gruen_low_to_high_merge_matches_bound_eq() {
        let point = random_point(10, 401);
        let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);
        let mut dense = Polynomial::new(EqPolynomial::<Fr>::evals(&point, None));
        assert_eq!(split.merge(), dense);

        let challenges = random_point(point.len(), 509);
        for challenge in challenges {
            split.bind(challenge);
            dense.bind_with_order(challenge, BindingOrder::LowToHigh);
            assert_eq!(split.merge(), dense);
        }
    }

    #[test]
    fn gruen_high_to_low_merge_matches_bound_eq() {
        let point = random_point(10, 601);
        let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::HighToLow);
        let mut dense = Polynomial::new(EqPolynomial::<Fr>::evals(&point, None));
        assert_eq!(split.merge(), dense);

        let challenges = random_point(point.len(), 709);
        for challenge in challenges {
            split.bind(challenge);
            dense.bind_with_order(challenge, BindingOrder::HighToLow);
            assert_eq!(split.merge(), dense);
        }
    }

    #[test]
    fn gruen_current_linear_factor_matches_merged_sumcheck_pair() {
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            let point = random_point(8, 811);
            let challenges = random_point(4, 919);
            let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, order);
            for (round, challenge) in challenges.into_iter().enumerate() {
                let merged = split.merge();
                let (linear_0, linear_1) = split.current_linear_evals();
                for x_out in 0..split.e_out_current_len() {
                    for x_in in 0..split.e_in_current_len() {
                        let row = split.group_index(x_out, x_in);
                        let dense_row = match order {
                            BindingOrder::LowToHigh => row,
                            BindingOrder::HighToLow => {
                                let out_bits = split.e_out_current_len().trailing_zeros() as usize;
                                (x_in << out_bits) | x_out
                            }
                        };
                        let head = split.e_out_current()[x_out] * split.e_in_current()[x_in];
                        let (dense_0, dense_1) = merged.sumcheck_eval_pair(dense_row, order);
                        assert_eq!(
                            head * linear_0,
                            dense_0,
                            "{order:?} round {round} row {row} eval 0"
                        );
                        assert_eq!(
                            head * linear_1,
                            dense_1,
                            "{order:?} round {round} row {row} eval 1"
                        );
                    }
                }
                split.bind(challenge);
            }
        }
    }
}
