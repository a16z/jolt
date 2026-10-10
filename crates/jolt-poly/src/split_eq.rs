use jolt_field::{Field, JoltField};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};

/// Recovers the missing endpoint of `q` for a round polynomial `s = l*q`.
///
/// `s_known` is the value of `s` at one endpoint, and `l_missing` is the
/// value of the linear factor at the other. When `l_missing` is nonzero,
/// solves `s_known + l_missing*q = s_0_plus_s_1` without calling `q_missing`.
/// Otherwise calls `q_missing` once, returning its value if that equation
/// holds or `Err` containing the actual endpoint sum if it does not.
/// This holds in every characteristic and checks no other property of `s` or `q`.
pub fn gruen_recover_endpoint<F: Field>(
    s_known: F,
    l_missing: F,
    s_0_plus_s_1: F,
    q_missing: impl FnOnce() -> F,
) -> Result<F, F> {
    if let Some(inverse) = l_missing.inverse() {
        Ok((s_0_plus_s_1 - s_known) * inverse)
    } else {
        let endpoint = q_missing();
        let actual = s_known + l_missing * endpoint;
        if actual == s_0_plus_s_1 {
            Ok(endpoint)
        } else {
            Err(actual)
        }
    }
}

/// Multiplies `q` by the linear polynomial with endpoint values `linear_evals`.
///
/// Input and output coefficients are in ascending degree order. The output
/// has `q_coeffs.len() + 1` coefficients, retaining trailing zeros; an empty
/// slice gives one zero coefficient. This holds in every characteristic and
/// does not check a degree bound or evaluate `q` at interpolation nodes.
pub fn gruen_mul_linear<F: Field>(linear_evals: (F, F), q_coeffs: &[F]) -> UnivariatePoly<F> {
    let (l_zero, l_one) = linear_evals;
    let l_slope = l_one - l_zero;
    let mut coefficients = vec![F::zero(); q_coeffs.len() + 1];
    for (index, q_coeff) in q_coeffs.iter().copied().enumerate() {
        coefficients[index] += q_coeff * l_zero;
        coefficients[index + 1] += q_coeff * l_slope;
    }
    UnivariatePoly::new(coefficients)
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorEqTable<F: JoltField> {
    e_out: Vec<F>,
    e_in: Vec<F>,
    in_bits: usize,
}

impl<F: JoltField> TensorEqTable<F> {
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

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GruenSplitEqPolynomial<F: JoltField> {
    current_index: usize,
    current_scalar: F,
    point: Vec<F>,
    e_in_vec: Vec<Vec<F>>,
    e_out_vec: Vec<Vec<F>>,
    binding_order: BindingOrder,
}

impl<F: JoltField> GruenSplitEqPolynomial<F> {
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

    /// Computes the cubic `s = l*q` from `q(0)`, its leading coefficient,
    /// and the sumcheck hint. The missing endpoint is evaluated only when
    /// `l(1)` vanishes. An error carries the actual endpoint sum.
    ///
    /// Integer nodes `0, 1, 2, 3` must be distinct in `F`, requiring field
    /// characteristic greater than three. In characteristic two, use
    /// [`Self::recover_q_one`] and [`Self::round_poly_from_q_coeffs`].
    ///
    /// # Panics
    ///
    /// If interpolation is reached with repeated integer nodes,
    /// [`UnivariatePoly::interpolate`] panics when a node difference has no inverse.
    pub fn gruen_poly_deg_3(
        &self,
        q_constant: F,
        q_quadratic_coeff: F,
        s_0_plus_s_1: F,
        q_at_one: impl FnOnce() -> F,
    ) -> Result<UnivariatePoly<F>, F> {
        let (eq_eval_0, eq_eval_1) = self.current_linear_evals();
        if self.current_scalar.is_zero() {
            return Self::zero_round(4, s_0_plus_s_1);
        }
        let eq_m = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_m;
        let eq_eval_3 = eq_eval_2 + eq_m;
        let cubic_eval_0 = eq_eval_0 * q_constant;
        let cubic_eval_1 = s_0_plus_s_1 - cubic_eval_0;
        let quadratic_eval_1 =
            gruen_recover_endpoint(cubic_eval_0, eq_eval_1, s_0_plus_s_1, q_at_one)?;
        let e_times_2 = q_quadratic_coeff + q_quadratic_coeff;
        let quadratic_eval_2 = quadratic_eval_1 + quadratic_eval_1 - q_constant + e_times_2;
        let quadratic_eval_3 =
            quadratic_eval_2 + quadratic_eval_1 - q_constant + e_times_2 + e_times_2;
        Ok(UnivariatePoly::interpolate_over_integers(&[
            cubic_eval_0,
            cubic_eval_1,
            eq_eval_2 * quadratic_eval_2,
            eq_eval_3 * quadratic_eval_3,
        ]))
    }

    /// Toom samples are `q(1)..q(d-1), q`'s leading coefficient, with `d>=2`.
    /// `q_evals` must contain at least two entries.
    /// The missing `q(0)` is evaluated only when `l(0)` vanishes.
    ///
    /// The finite integer nodes `0, 1, ..., d-1` must be distinct in `F`.
    /// With three or more finite samples, the factorials through `(d-1)!`
    /// must be invertible, requiring field characteristic greater than `d-1`.
    /// Two finite samples use only `0, 1` and work in characteristic two;
    /// for more samples there, use [`Self::recover_q_one`] and
    /// [`Self::round_poly_from_q_coeffs`].
    ///
    /// # Panics
    ///
    /// If Toom interpolation is reached without these conditions,
    /// [`UnivariatePoly::from_evals`] panics when a required factorial has no inverse.
    pub fn gruen_poly_from_evals(
        &self,
        q_evals: &[F],
        s_0_plus_s_1: F,
        q_at_zero: impl FnOnce() -> F,
    ) -> Result<UnivariatePoly<F>, F> {
        if self.current_scalar.is_zero() {
            return Self::zero_round(q_evals.len() + 2, s_0_plus_s_1);
        }
        let q_zero = self.recover_q_zero(q_evals[0], s_0_plus_s_1, q_at_zero)?;
        let mut full_q_evals = Vec::with_capacity(q_evals.len() + 1);
        full_q_evals.push(q_zero);
        full_q_evals.extend_from_slice(q_evals);
        let q_coeffs = UnivariatePoly::from_evals_toom(&full_q_evals).into_coefficients();
        Ok(self.round_poly_from_q_coeffs(&q_coeffs))
    }

    /// Degree-two message for a linear inner factor, with lazy `q(0)` recovery.
    pub fn gruen_poly_deg_2(
        &self,
        q_one: F,
        s_0_plus_s_1: F,
        q_at_zero: impl FnOnce() -> F,
    ) -> Result<UnivariatePoly<F>, F> {
        if self.current_scalar.is_zero() {
            return Self::zero_round(3, s_0_plus_s_1);
        }
        let q_zero = self.recover_q_zero(q_one, s_0_plus_s_1, q_at_zero)?;
        Ok(self.round_poly_from_q_coeffs(&[q_zero, q_one - q_zero]))
    }

    fn recover_q_zero(&self, q_one: F, hint: F, q_at_zero: impl FnOnce() -> F) -> Result<F, F> {
        let (l_zero, l_one) = self.current_linear_evals();
        gruen_recover_endpoint(l_one * q_one, l_zero, hint, q_at_zero)
    }

    /// Recovers `q(1)` for the current round `s = l*q` from `q(0)` and its claim.
    ///
    /// Uses [`gruen_recover_endpoint`] with the current linear factor. Calls
    /// `q_at_one` once only when `l(1)` is zero, including a zero current scalar.
    /// An error contains the actual endpoint sum; a zero scalar therefore
    /// accepts a zero claim and returns `Err(0)` for a nonzero claim.
    /// A variable must remain to bind, as for [`Self::current_linear_evals`];
    /// this precondition and the claimed degree of `q` are not checked.
    pub fn recover_q_one(
        &self,
        q_zero: F,
        s_0_plus_s_1: F,
        q_at_one: impl FnOnce() -> F,
    ) -> Result<F, F> {
        let (l_zero, l_one) = self.current_linear_evals();
        gruen_recover_endpoint(l_zero * q_zero, l_one, s_0_plus_s_1, q_at_one)
    }

    /// Builds the current round polynomial `l*q` from ascending coefficients of `q`.
    ///
    /// Uses [`gruen_mul_linear`] with the current linear factor, retaining
    /// `q_coeffs.len() + 1` coefficients, including trailing zeros. A zero
    /// current scalar yields that many zero coefficients, and an empty slice
    /// yields one zero coefficient. A variable must remain to bind, as for
    /// [`Self::current_linear_evals`]; this precondition and any degree bound
    /// on `q` are not checked.
    pub fn round_poly_from_q_coeffs(&self, q_coeffs: &[F]) -> UnivariatePoly<F> {
        gruen_mul_linear(self.current_linear_evals(), q_coeffs)
    }

    fn zero_round(coefficients: usize, hint: F) -> Result<UnivariatePoly<F>, F> {
        if hint.is_zero() {
            Ok(UnivariatePoly::new(vec![F::zero(); coefficients]))
        } else {
            Err(F::zero())
        }
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
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test module asserts successful reconstruction and forbidden lazy endpoint evaluation"
)]
mod tests {
    #[cfg(feature = "binary")]
    use jolt_field::F128;
    use jolt_field::{Field, Fr, Prime128OffsetA7F7, Ring};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    use std::cell::Cell;

    use super::*;

    fn random_point(len: usize, seed: u64) -> Vec<Fr> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..len).map(|_| Fr::random(&mut rng)).collect()
    }

    fn gruen_functions_match_products<F: JoltField>() {
        let mut rng = ChaCha20Rng::seed_from_u64(5101);
        for degree in 0..=5 {
            let linear = (F::random(&mut rng), F::random(&mut rng));
            assert!(!linear.0.is_zero() && !linear.1.is_zero());
            let mut q_coeffs: Vec<F> = (0..=degree).map(|_| F::random(&mut rng)).collect();
            for zero_top in [false, true] {
                if zero_top {
                    q_coeffs[degree] = F::zero();
                }
                let q = UnivariatePoly::new(q_coeffs.clone());
                let product = gruen_mul_linear(linear, &q_coeffs);
                assert_eq!(product.coefficients().len(), q_coeffs.len() + 1);
                let mut points = vec![F::zero(), F::one()];
                while points.len() < product.coefficients().len() {
                    let point = F::random(&mut rng);
                    if !points.contains(&point) {
                        points.push(point);
                    }
                }
                for x in points {
                    assert_eq!(
                        product.evaluate(x),
                        ((F::one() - x) * linear.0 + x * linear.1) * q.evaluate(x)
                    );
                }
                let endpoint_q = (q.evaluate(F::zero()), q.evaluate(F::one()));
                let endpoint_s = (linear.0 * endpoint_q.0, linear.1 * endpoint_q.1);
                let claim = endpoint_s.0 + endpoint_s.1;
                assert_eq!(
                    gruen_recover_endpoint(endpoint_s.0, linear.1, claim, || panic!(
                        "nonzero missing factor must not evaluate the endpoint"
                    )),
                    Ok(endpoint_q.1)
                );
                assert_eq!(
                    gruen_recover_endpoint(endpoint_s.1, linear.0, claim, || panic!(
                        "nonzero missing factor must not evaluate the endpoint"
                    )),
                    Ok(endpoint_q.0)
                );
            }
        }
        assert_eq!(
            gruen_mul_linear((F::one(), F::one()), &[]).coefficients(),
            &[F::zero()]
        );
        for known in [F::zero(), F::random(&mut rng)] {
            let endpoint = F::random(&mut rng);
            for claim in [known, known + F::one()] {
                let calls = Cell::new(0);
                let recovered = gruen_recover_endpoint(known, F::zero(), claim, || {
                    calls.set(calls.get() + 1);
                    endpoint
                });
                assert_eq!(calls.get(), 1);
                assert_eq!(
                    recovered,
                    if claim == known {
                        Ok(endpoint)
                    } else {
                        Err(known)
                    }
                );
            }
        }
    }

    #[test]
    fn characteristic_free_gruen_functions_bn254() {
        gruen_functions_match_products::<Fr>();
    }

    #[test]
    fn characteristic_free_gruen_functions_akita_field() {
        gruen_functions_match_products::<Prime128OffsetA7F7>();
    }

    #[cfg(feature = "binary")]
    #[test]
    fn characteristic_free_gruen_functions_binary() {
        gruen_functions_match_products::<F128>();
    }

    fn direct_boolean_weight<F: Field>(point: &[F], index: usize) -> F {
        point
            .iter()
            .enumerate()
            .map(|(coordinate, &value)| {
                if index & (1 << (point.len() - coordinate - 1)) == 0 {
                    F::one() - value
                } else {
                    value
                }
            })
            .product()
    }

    fn direct_table_value<F: Field>(table: &[F], point: &[F]) -> F {
        table
            .iter()
            .enumerate()
            .map(|(index, &value)| direct_boolean_weight(point, index) * value)
            .sum()
    }

    struct GruenRoundTables<F> {
        a: Vec<F>,
        b: Vec<F>,
        w: Vec<F>,
        order: BindingOrder,
    }

    impl<F: JoltField> GruenRoundTables<F> {
        fn assignment(&self, bound: &[F], x: F, index: usize) -> Vec<F> {
            let remaining = self.w.len() - bound.len() - 1;
            let mut point = vec![F::zero(); self.w.len()];
            match self.order {
                BindingOrder::LowToHigh => {
                    for (round, &challenge) in bound.iter().enumerate() {
                        point[self.w.len() - round - 1] = challenge;
                    }
                    point[remaining] = x;
                    for (coordinate, value) in point.iter_mut().take(remaining).enumerate() {
                        *value = if index & (1 << (remaining - coordinate - 1)) == 0 {
                            F::zero()
                        } else {
                            F::one()
                        };
                    }
                }
                BindingOrder::HighToLow => {
                    point[..bound.len()].copy_from_slice(bound);
                    point[bound.len()] = x;
                    for (coordinate, value) in point.iter_mut().skip(bound.len() + 1).enumerate() {
                        *value = if index & (1 << (remaining - coordinate - 1)) == 0 {
                            F::zero()
                        } else {
                            F::one()
                        };
                    }
                }
            }
            point
        }

        fn q_endpoints_and_leading(&self, bound: &[F]) -> (F, F, F) {
            let remaining_w = match self.order {
                BindingOrder::LowToHigh => &self.w[..self.w.len() - bound.len() - 1],
                BindingOrder::HighToLow => &self.w[bound.len() + 1..],
            };
            let mut q_zero = F::zero();
            let mut q_one = F::zero();
            let mut q_quadratic = F::zero();
            for index in 0..1 << remaining_w.len() {
                let weight = direct_boolean_weight(remaining_w, index);
                let zero = self.assignment(bound, F::zero(), index);
                let one = self.assignment(bound, F::one(), index);
                let a_zero = direct_table_value(&self.a, &zero);
                let a_one = direct_table_value(&self.a, &one);
                let b_zero = direct_table_value(&self.b, &zero);
                let b_one = direct_table_value(&self.b, &one);
                q_zero += weight * a_zero * b_zero;
                q_one += weight * a_one * b_one;
                q_quadratic += weight * (a_one - a_zero) * (b_one - b_zero);
            }
            (q_zero, q_one, q_quadratic)
        }

        fn round_value(&self, bound: &[F], x: F) -> F {
            (0..1 << (self.w.len() - bound.len() - 1))
                .map(|index| {
                    let point = self.assignment(bound, x, index);
                    let eq: F = self
                        .w
                        .iter()
                        .zip(&point)
                        .map(|(&w, &coordinate)| {
                            (F::one() - w) * (F::one() - coordinate) + w * coordinate
                        })
                        .product();
                    eq * direct_table_value(&self.a, &point) * direct_table_value(&self.b, &point)
                })
                .sum()
        }
    }

    fn gruen_methods_match_direct_sums<F: JoltField>(check_cubic: bool) {
        let mut rng = ChaCha20Rng::seed_from_u64(5209);
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            let a: Vec<F> = (0..16).map(|_| F::random(&mut rng)).collect();
            let b: Vec<F> = (0..16).map(|_| F::random(&mut rng)).collect();
            let original_w: Vec<F> = (0..4).map(|_| F::random(&mut rng)).collect();
            let challenges: Vec<F> = (0..4).map(|_| F::random(&mut rng)).collect();
            let extra_points = [F::random(&mut rng), F::random(&mut rng)];
            for exceptional in [None, Some(F::zero()), Some(F::one())] {
                let mut w = original_w.clone();
                if let Some(coordinate) = exceptional {
                    w[1] = coordinate;
                }
                let fixture = GruenRoundTables {
                    a: a.clone(),
                    b: b.clone(),
                    w,
                    order,
                };
                let mut claim: F = fixture
                    .a
                    .iter()
                    .zip(&fixture.b)
                    .enumerate()
                    .map(|(index, (&a, &b))| direct_boolean_weight(&fixture.w, index) * a * b)
                    .sum();
                let mut split = GruenSplitEqPolynomial::new(&fixture.w, order);
                for (round, &challenge) in challenges.iter().enumerate() {
                    let bound = &challenges[..round];
                    let (q_zero, direct_q_one, q_quadratic) =
                        fixture.q_endpoints_and_leading(bound);
                    let calls = Cell::new(0);
                    let q_one = split
                        .recover_q_one(q_zero, claim, || {
                            calls.set(calls.get() + 1);
                            direct_q_one
                        })
                        .unwrap();
                    assert_eq!(q_one, direct_q_one);
                    assert_eq!(
                        calls.get(),
                        usize::from(split.current_linear_evals().1.is_zero())
                    );
                    let coefficients = [q_zero, q_one - q_zero - q_quadratic, q_quadratic];
                    let polynomial = split.round_poly_from_q_coeffs(&coefficients);
                    assert_eq!(polynomial.coefficients().len(), 4);
                    for x in [F::zero(), F::one(), extra_points[0], extra_points[1]] {
                        assert_eq!(polynomial.evaluate(x), fixture.round_value(bound, x));
                    }
                    if check_cubic {
                        let cubic = split
                            .gruen_poly_deg_3(q_zero, q_quadratic, claim, || direct_q_one)
                            .unwrap();
                        for x in [F::zero(), F::one(), extra_points[0], extra_points[1]] {
                            assert_eq!(cubic.evaluate(x), fixture.round_value(bound, x));
                        }
                    }
                    claim = polynomial.evaluate(challenge);
                    split.bind(challenge);
                }

                let zero =
                    GruenSplitEqPolynomial::new_with_scaling(&fixture.w, order, Some(F::zero()));
                let endpoint = F::random(&mut rng);
                for claim in [F::zero(), F::one()] {
                    let calls = Cell::new(0);
                    let result = zero.recover_q_one(F::one(), claim, || {
                        calls.set(calls.get() + 1);
                        endpoint
                    });
                    assert_eq!(calls.get(), 1);
                    assert_eq!(
                        result,
                        if claim.is_zero() {
                            Ok(endpoint)
                        } else {
                            Err(F::zero())
                        }
                    );
                }
                for coefficients in [&[][..], &[F::one(), endpoint, F::zero()][..]] {
                    assert_eq!(
                        zero.round_poly_from_q_coeffs(coefficients).coefficients(),
                        vec![F::zero(); coefficients.len() + 1]
                    );
                }
            }
        }
    }

    #[test]
    fn characteristic_free_gruen_methods_bn254() {
        gruen_methods_match_direct_sums::<Fr>(true);
    }

    #[test]
    fn characteristic_free_gruen_methods_akita_field() {
        gruen_methods_match_direct_sums::<Prime128OffsetA7F7>(false);
    }

    #[cfg(feature = "binary")]
    #[test]
    fn characteristic_free_gruen_methods_binary() {
        gruen_methods_match_direct_sums::<F128>(false);
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

    fn eq_factor(w: Fr, c: Fr) -> Fr {
        (Fr::one() - w) * (Fr::one() - c) + w * c
    }

    /// The current round's eq factor `l(X) = scalar * ((1-w)(1-X) + wX)`,
    /// built by hand from the point coordinate and an independently tracked
    /// scalar, never from the struct's internals.
    fn hand_built_linear(scalar: Fr, w: Fr) -> UnivariatePoly<Fr> {
        UnivariatePoly::new(vec![scalar * (Fr::one() - w), scalar * (w + w - Fr::one())])
    }

    fn current_round_variable(
        split: &GruenSplitEqPolynomial<Fr>,
        point: &[Fr],
        order: BindingOrder,
    ) -> Fr {
        match order {
            BindingOrder::LowToHigh => point[split.current_index() - 1],
            BindingOrder::HighToLow => point[split.current_index()],
        }
    }

    #[test]
    fn gruen_poly_deg_3_equals_hand_built_linear_times_quadratic() {
        for (order, seed) in [
            (BindingOrder::LowToHigh, 1301u64),
            (BindingOrder::HighToLow, 1409),
        ] {
            let point = random_point(6, seed);
            let challenges = random_point(3, seed + 1);
            let q = UnivariatePoly::new(random_point(3, seed + 2));

            let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, order);
            let mut hand_scalar = Fr::one();
            for (round, challenge) in challenges.into_iter().enumerate() {
                let w = current_round_variable(&split, &point, order);
                let l = hand_built_linear(hand_scalar, w);
                let hint = l.evaluate(Fr::zero()) * q.evaluate(Fr::zero())
                    + l.evaluate(Fr::one()) * q.evaluate(Fr::one());
                let s = split
                    .gruen_poly_deg_3(q.coefficients()[0], q.coefficients()[2], hint, || {
                        q.evaluate(Fr::one())
                    })
                    .unwrap();
                assert_eq!(s.coefficients().len(), 4, "{order:?} round {round}");
                // s and l*q both have degree <= 3, so agreement on 8 points
                // plus a random one forces polynomial equality
                for x in (0..8u64).map(Fr::from_u64).chain([Fr::random(
                    &mut ChaCha20Rng::seed_from_u64(seed + 3 + round as u64),
                )]) {
                    assert_eq!(
                        s.evaluate(x),
                        l.evaluate(x) * q.evaluate(x),
                        "{order:?} round {round}"
                    );
                }
                hand_scalar *= eq_factor(w, challenge);
                split.bind(challenge);
            }
        }
    }

    #[test]
    fn gruen_poly_from_evals_equals_hand_built_linear_times_q() {
        for (order, seed) in [
            (BindingOrder::LowToHigh, 2003u64),
            (BindingOrder::HighToLow, 2087),
        ] {
            // gruen_poly_from_evals requires degree >= 2: q_evals[0] must be q(1)
            for degree in [2usize, 3] {
                let point = random_point(5, seed + degree as u64);
                let challenges = random_point(2, seed + 10 + degree as u64);
                let q = UnivariatePoly::new(random_point(degree + 1, seed + 20 + degree as u64));

                let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, order);
                let mut hand_scalar = Fr::one();
                for &challenge in &challenges {
                    let w = current_round_variable(&split, &point, order);
                    hand_scalar *= eq_factor(w, challenge);
                    split.bind(challenge);
                }

                let w = current_round_variable(&split, &point, order);
                let l = hand_built_linear(hand_scalar, w);
                // Toom layout: [q(1), ..., q(degree-1), leading coefficient]
                let mut q_evals: Vec<Fr> = (1..degree as u64)
                    .map(|x| q.evaluate(Fr::from_u64(x)))
                    .collect();
                q_evals.push(q.coefficients()[degree]);
                let hint = l.evaluate(Fr::zero()) * q.evaluate(Fr::zero())
                    + l.evaluate(Fr::one()) * q.evaluate(Fr::one());

                let s = split
                    .gruen_poly_from_evals(&q_evals, hint, || q.evaluate(Fr::zero()))
                    .unwrap();
                assert_eq!(s.coefficients().len(), degree + 2, "{order:?} deg {degree}");
                for x in (0..2 * degree as u64 + 3).map(Fr::from_u64) {
                    assert_eq!(
                        s.evaluate(x),
                        l.evaluate(x) * q.evaluate(x),
                        "{order:?} deg {degree}"
                    );
                }
            }
        }
    }

    fn exceptional_rounds_match_products<F: JoltField>() {
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            for coordinate in [F::zero(), F::one(), F::from_u64(7)] {
                for zero_prefix in [false, true] {
                    let mut split = GruenSplitEqPolynomial::new_with_scaling(
                        &[coordinate],
                        order,
                        Some(if zero_prefix {
                            F::zero()
                        } else {
                            F::from_u64(11)
                        }),
                    );
                    for killed_prefix in [false, true] {
                        if killed_prefix {
                            let point = match order {
                                BindingOrder::LowToHigh => vec![coordinate, F::zero()],
                                BindingOrder::HighToLow => vec![F::zero(), coordinate],
                            };
                            split = GruenSplitEqPolynomial::new_with_scaling(
                                &point,
                                order,
                                Some(F::from_u64(11)),
                            );
                            split.bind(F::one());
                        }
                        let scalar = if zero_prefix || killed_prefix {
                            F::zero()
                        } else {
                            F::from_u64(11)
                        };
                        let l_zero = scalar * (F::one() - coordinate);
                        let l_one = scalar * coordinate;
                        for degree in [1usize, 2, 3] {
                            let q = UnivariatePoly::new(
                                (0..=degree)
                                    .map(|i| F::from_u64(2 + 3 * i as u64))
                                    .collect(),
                            );
                            let hint =
                                l_zero * q.evaluate(F::zero()) + l_one * q.evaluate(F::one());
                            let called = Cell::new(false);
                            let polynomial = if degree == 1 {
                                split
                                    .gruen_poly_deg_2(q.evaluate(F::one()), hint, || {
                                        called.set(true);
                                        q.evaluate(F::zero())
                                    })
                                    .unwrap()
                            } else {
                                let mut samples: Vec<F> = (1..degree)
                                    .map(|i| q.evaluate(F::from_u64(i as u64)))
                                    .collect();
                                samples.push(q.coefficients()[degree]);
                                split
                                    .gruen_poly_from_evals(&samples, hint, || {
                                        called.set(true);
                                        q.evaluate(F::zero())
                                    })
                                    .unwrap()
                            };
                            assert_eq!(called.get(), !scalar.is_zero() && l_zero.is_zero());
                            if !scalar.is_zero() && l_zero.is_zero() {
                                let bad_hint = hint + F::one();
                                if degree == 1 {
                                    assert_eq!(
                                        split.gruen_poly_deg_2(
                                            q.evaluate(F::one()),
                                            bad_hint,
                                            || q.evaluate(F::zero())
                                        ),
                                        Err(hint)
                                    );
                                } else {
                                    let mut samples: Vec<F> = (1..degree)
                                        .map(|i| q.evaluate(F::from_u64(i as u64)))
                                        .collect();
                                    samples.push(q.coefficients()[degree]);
                                    assert_eq!(
                                        split.gruen_poly_from_evals(&samples, bad_hint, || q
                                            .evaluate(F::zero())),
                                        Err(hint)
                                    );
                                }
                            }
                            assert_eq!(polynomial.coefficients().len(), degree + 2);
                            for x in (0..8).map(F::from_u64) {
                                let linear = l_zero + (l_one - l_zero) * x;
                                assert_eq!(polynomial.evaluate(x), linear * q.evaluate(x));
                            }
                            if degree == 2 {
                                called.set(false);
                                let cubic = split
                                    .gruen_poly_deg_3(
                                        q.coefficients()[0],
                                        q.coefficients()[2],
                                        hint,
                                        || {
                                            called.set(true);
                                            q.evaluate(F::one())
                                        },
                                    )
                                    .unwrap();
                                assert_eq!(called.get(), !scalar.is_zero() && l_one.is_zero());
                                assert_eq!(cubic.coefficients().len(), 4);
                                assert_eq!(cubic.coefficients(), polynomial.coefficients());
                                if !scalar.is_zero() && l_one.is_zero() {
                                    assert_eq!(
                                        split.gruen_poly_deg_3(
                                            q.coefficients()[0],
                                            q.coefficients()[2],
                                            hint + F::one(),
                                            || q.evaluate(F::one())
                                        ),
                                        Err(hint)
                                    );
                                }
                            }
                        }
                        if scalar.is_zero() {
                            assert_eq!(
                                split.gruen_poly_deg_3(F::one(), F::one(), F::one(), || panic!(
                                    "zero factor must not scan"
                                )),
                                Err(F::zero())
                            );
                            assert_eq!(
                                split.gruen_poly_from_evals(
                                    &[F::one(), F::one()],
                                    F::one(),
                                    || panic!("zero factor must not scan")
                                ),
                                Err(F::zero())
                            );
                            assert_eq!(
                                split.gruen_poly_deg_2(F::one(), F::one(), || panic!(
                                    "zero factor must not scan"
                                )),
                                Err(F::zero())
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exceptional_gruen_rounds_match_true_products_bn254() {
        exceptional_rounds_match_products::<Fr>();
    }

    #[test]
    fn exceptional_gruen_rounds_match_true_products_akita_field() {
        exceptional_rounds_match_products::<Prime128OffsetA7F7>();
    }

    #[test]
    fn e_out_in_for_window_factors_naive_head_eq_table() {
        let point = random_point(9, 1601);
        let challenges = random_point(9, 1607);
        let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);

        for &challenge in &challenges {
            let current = split.current_index();
            for window in 1..=current {
                let (e_out, e_in) = split.e_out_in_for_window(window);
                let head_len = current - window;
                let head = EqPolynomial::<Fr>::evals(&point[..head_len], None);
                let in_bits = e_in.len().trailing_zeros() as usize;
                assert_eq!(
                    e_out.len() * e_in.len(),
                    head.len(),
                    "window {window} at index {current}"
                );
                for x_out in 0..e_out.len() {
                    for x_in in 0..e_in.len() {
                        assert_eq!(
                            e_out[x_out] * e_in[x_in],
                            head[(x_out << in_bits) | x_in],
                            "window {window} at index {current}, out {x_out}, in {x_in}"
                        );
                    }
                }
            }
            let (e_out, e_in) = split.e_out_in_for_window(current + 3);
            assert_eq!((e_out, e_in), (&[Fr::one()][..], &[Fr::one()][..]));
            split.bind(challenge);
        }
    }

    #[test]
    fn e_active_for_window_reconstructs_naive_eq_table_with_linear_factor() {
        let point = random_point(8, 1701);
        let challenges = random_point(3, 1709);
        let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, BindingOrder::LowToHigh);
        let mut hand_scalar = Fr::one();

        for stage in 0..=challenges.len() {
            if stage > 0 {
                let w = point[split.current_index() - 1];
                hand_scalar *= eq_factor(w, challenges[stage - 1]);
                split.bind(challenges[stage - 1]);
            }
            let current = split.current_index();
            let w_current = point[current - 1];
            let (lin_0, lin_1) = split.current_linear_evals();
            assert_eq!(
                (lin_0, lin_1),
                (
                    hand_scalar * (Fr::one() - w_current),
                    hand_scalar * w_current
                ),
                "stage {stage}"
            );

            assert_eq!(split.e_active_for_window(0), vec![Fr::one()]);
            assert_eq!(split.e_active_for_window(1), vec![Fr::one()]);
            assert_eq!(split.e_active_for_window(current + 1), vec![Fr::one()]);

            let full = EqPolynomial::<Fr>::evals(&point[..current], None);
            for window in 2..=current {
                let (e_out, e_in) = split.e_out_in_for_window(window);
                let active = split.e_active_for_window(window);
                assert_eq!(active.len(), 1 << (window - 1), "stage {stage}");
                let in_bits = e_in.len().trailing_zeros() as usize;
                for head_index in 0..e_out.len() * e_in.len() {
                    let head =
                        e_out[head_index >> in_bits] * e_in[head_index & ((1usize << in_bits) - 1)];
                    for (active_index, &active_value) in active.iter().enumerate() {
                        for last_bit in 0..2usize {
                            let index =
                                (((head_index << (window - 1)) | active_index) << 1) | last_bit;
                            let last = if last_bit == 0 {
                                Fr::one() - w_current
                            } else {
                                w_current
                            };
                            assert_eq!(
                                head * active_value * last,
                                full[index],
                                "stage {stage} window {window} index {index}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn new_with_scaling_scales_bound_and_merged_results_by_factor() {
        let scaling = random_point(1, 3001)[0];
        for (order, seed) in [
            (BindingOrder::LowToHigh, 3103u64),
            (BindingOrder::HighToLow, 3203),
        ] {
            let point = random_point(7, seed);
            let challenges = random_point(7, seed + 1);
            let mut scaled =
                GruenSplitEqPolynomial::<Fr>::new_with_scaling(&point, order, Some(scaling));
            let mut plain = GruenSplitEqPolynomial::<Fr>::new(&point, order);

            assert_eq!(scaled.current_scalar(), scaling);
            assert_eq!(scaled.merge(), plain.merge() * scaling);
            for &challenge in &challenges {
                let (scaled_0, scaled_1) = scaled.current_linear_evals();
                let (plain_0, plain_1) = plain.current_linear_evals();
                assert_eq!(
                    (scaled_0, scaled_1),
                    (plain_0 * scaling, plain_1 * scaling),
                    "{order:?}"
                );
                scaled.bind(challenge);
                plain.bind(challenge);
                assert_eq!(
                    scaled.current_scalar(),
                    plain.current_scalar() * scaling,
                    "{order:?}"
                );
                assert_eq!(scaled.merge(), plain.merge() * scaling, "{order:?}");
            }
        }

        let empty = GruenSplitEqPolynomial::<Fr>::new_with_scaling(
            &[],
            BindingOrder::LowToHigh,
            Some(scaling),
        );
        assert_eq!(empty.current_scalar(), scaling);
        assert_eq!(empty.merge().evaluations(), &[scaling]);
    }

    #[test]
    fn gruen_par_fold_out_in_matches_sequential_reference_fold() {
        for (order, seed) in [
            (BindingOrder::LowToHigh, 4001u64),
            (BindingOrder::HighToLow, 4103),
        ] {
            let point = random_point(7, seed);
            let challenges = random_point(4, seed + 1);
            let mut split = GruenSplitEqPolynomial::<Fr>::new(&point, order);

            for stage in 0..=challenges.len() {
                if stage > 0 {
                    split.bind(challenges[stage - 1]);
                }
                let e_out = split.e_out_current().to_vec();
                let e_in = split.e_in_current().to_vec();
                let values = random_point(e_out.len() * e_in.len(), seed + 10 + stage as u64);
                let out_weights = random_point(e_out.len(), seed + 20 + stage as u64);
                let in_weights = random_point(e_in.len(), seed + 30 + stage as u64);

                let folded = split.par_fold_out_in(
                    Fr::zero,
                    |inner, row, x_in, e_in_value| {
                        *inner += e_in_value * in_weights[x_in] * values[row];
                    },
                    |x_out, e_out_value, inner| e_out_value * out_weights[x_out] * inner,
                    |left, right| left + right,
                );

                let mut expected = Fr::zero();
                for (x_out, &e_out_value) in e_out.iter().enumerate() {
                    let mut inner = Fr::zero();
                    for (x_in, &e_in_value) in e_in.iter().enumerate() {
                        inner += e_in_value * in_weights[x_in] * values[x_out * e_in.len() + x_in];
                    }
                    expected += e_out_value * out_weights[x_out] * inner;
                }
                assert_eq!(folded, expected, "{order:?} stage {stage}");
            }
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
