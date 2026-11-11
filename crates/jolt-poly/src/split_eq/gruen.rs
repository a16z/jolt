use crate::{
    BindingOrder, Endianness, EqPolynomial, PointLike, PointRef, Polynomial, UnivariatePoly,
    HIGH_TO_LOW,
};
use jolt_field::JoltField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
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
    pub fn new<P: PointLike<Coord = F> + ?Sized>(point: &P, binding_order: BindingOrder) -> Self {
        Self::new_with_scaling(point, binding_order, None)
    }

    /// Builds tables in significance order and stores one MSB-first point copy.
    pub fn new_with_scaling<P: PointLike<Coord = F> + ?Sized>(
        point: &P,
        binding_order: BindingOrder,
        scaling_factor: Option<F>,
    ) -> Self {
        if P::ENDIANNESS == HIGH_TO_LOW {
            Self::from_coords(
                PointRef::high_to_low(point.coords()),
                binding_order,
                scaling_factor,
            )
        } else {
            Self::from_coords(
                PointRef::low_to_high(point.coords()),
                binding_order,
                scaling_factor,
            )
        }
    }

    fn from_coords<const E: Endianness>(
        point: PointRef<'_, E, F>,
        binding_order: BindingOrder,
        scaling_factor: Option<F>,
    ) -> Self {
        if point.num_vars() == 0 {
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

        let (current_index, e_in_vec, e_out_vec) = match binding_order {
            BindingOrder::LowToHigh => {
                let split = (point.num_vars() / 2).min(point.num_vars() - 1);
                let out_point = point.msb_range(0..split);
                let in_point = point.msb_range(split..point.num_vars() - 1);
                #[cfg(feature = "parallel")]
                let (e_out_vec, e_in_vec) = rayon::join(
                    || EqPolynomial::<F>::evals_cached_msb(&out_point, None),
                    || EqPolynomial::<F>::evals_cached_msb(&in_point, None),
                );
                #[cfg(not(feature = "parallel"))]
                let (e_out_vec, e_in_vec) = (
                    EqPolynomial::<F>::evals_cached_msb(&out_point, None),
                    EqPolynomial::<F>::evals_cached_msb(&in_point, None),
                );
                (point.num_vars(), e_in_vec, e_out_vec)
            }
            BindingOrder::HighToLow => {
                let split = (point.num_vars() / 2).min(point.num_vars() - 1) + 1;
                let in_point = point.msb_range(1..split);
                let out_point = point.msb_range(split..point.num_vars());
                #[cfg(feature = "parallel")]
                let (e_in_vec, e_out_vec) = rayon::join(
                    || EqPolynomial::<F>::evals_cached_lsb(&in_point, None),
                    || EqPolynomial::<F>::evals_cached_lsb(&out_point, None),
                );
                #[cfg(not(feature = "parallel"))]
                let (e_in_vec, e_out_vec) = (
                    EqPolynomial::<F>::evals_cached_lsb(&in_point, None),
                    EqPolynomial::<F>::evals_cached_lsb(&out_point, None),
                );
                (0, e_in_vec, e_out_vec)
            }
        };
        Self {
            current_index,
            current_scalar: scaling_factor.unwrap_or(F::one()),
            point: if E == HIGH_TO_LOW {
                point.coords().to_vec()
            } else {
                (0..point.num_vars()).map(|k| point.msb(k)).collect()
            },
            e_in_vec,
            e_out_vec,
            binding_order,
        }
    }

    /// Bytes [`Self::new`] requests for a nonempty `vars`-variable point bound
    /// [`BindingOrder::LowToHigh`], or `None` past `usize::MAX`: the copied
    /// point and the [`EqPolynomial::evals_cached_msb`] tables over the point
    /// without its last variable, split at `vars / 2` into `2P - 1` out entries
    /// for `P = 2^⌊vars/2⌋` and `H - 1` in entries for `H = 2^⌈vars/2⌉`, with
    /// their `vars + 1` `Vec` headers. The empty point's layout differs.
    /// Excludes allocator slack.
    pub fn low_to_high_bytes(vars: usize) -> Option<usize> {
        let pow2 = |bits: usize| 1usize.checked_shl(u32::try_from(bits).ok()?);
        let out = pow2(vars / 2)?;
        let inner = pow2(vars.div_ceil(2))?;
        vars.checked_add(out.checked_mul(2)?)?
            .checked_add(inner)?
            .checked_sub(2)?
            .checked_mul(size_of::<F>())?
            .checked_add(vars.checked_add(1)?.checked_mul(size_of::<Vec<F>>())?)
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
        let quadratic_eval_1 = if let Some(inverse) = eq_eval_1.inverse() {
            cubic_eval_1 * inverse
        } else {
            let endpoint = q_at_one();
            let actual = cubic_eval_0 + eq_eval_1 * endpoint;
            if actual != s_0_plus_s_1 {
                return Err(actual);
            }
            endpoint
        };
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
        Ok(self.multiply_linear_factor(&q_coeffs))
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
        Ok(self.multiply_linear_factor(&[q_zero, q_one - q_zero]))
    }

    fn recover_q_zero(&self, q_one: F, hint: F, q_at_zero: impl FnOnce() -> F) -> Result<F, F> {
        let (l_zero, l_one) = self.current_linear_evals();
        if let Some(inverse) = l_zero.inverse() {
            Ok((hint - l_one * q_one) * inverse)
        } else {
            let q_zero = q_at_zero();
            let actual = l_zero * q_zero + l_one * q_one;
            if actual == hint {
                Ok(q_zero)
            } else {
                Err(actual)
            }
        }
    }

    fn multiply_linear_factor(&self, q_coeffs: &[F]) -> UnivariatePoly<F> {
        let (l_zero, l_one) = self.current_linear_evals();
        let l_slope = l_one - l_zero;
        let mut coefficients = vec![F::zero(); q_coeffs.len() + 1];
        for (index, q_coeff) in q_coeffs.iter().copied().enumerate() {
            coefficients[index] += q_coeff * l_zero;
            coefficients[index + 1] += q_coeff * l_slope;
        }
        UnivariatePoly::new(coefficients)
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

    /// Sum of the remaining-variable equality weights on `start..domain_len`.
    /// The current variable's linear factor and bound scalar are applied by
    /// the Gruen reconstruction, not included in this weight.
    /// Only defined for low-to-high binding, where group and polynomial order agree.
    pub fn suffix_weight(&self, start: usize) -> F {
        assert!(
            matches!(self.binding_order, BindingOrder::LowToHigh),
            "split-eq suffix weights are only defined for low-to-high"
        );
        EqPolynomial::suffix_sum(&self.point[..self.current_index.saturating_sub(1)], start)
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
        self.par_fold_out_in_prefix(
            self.e_out_current_len() * self.e_in_current_len(),
            make_inner,
            inner_step,
            outer_step,
            merge,
        )
    }

    /// Fold the first `len` remaining-variable rows in group order. `len` counts
    /// polynomial rows only for [`BindingOrder::LowToHigh`]; in that order a caller
    /// can add a constant suffix's contribution using [`Self::suffix_weight`].
    pub fn par_fold_out_in_prefix<
        OuterAcc: Send,
        InnerAcc: Send,
        MakeInner: Fn() -> InnerAcc + Sync + Send,
        InnerStep: Fn(&mut InnerAcc, usize, usize, F) + Sync + Send,
        OuterStep: Fn(usize, F, InnerAcc) -> OuterAcc + Sync + Send,
        Merge: Fn(OuterAcc, OuterAcc) -> OuterAcc + Sync + Send,
    >(
        &self,
        len: usize,
        make_inner: MakeInner,
        inner_step: InnerStep,
        outer_step: OuterStep,
        merge: Merge,
    ) -> OuterAcc {
        let e_out = self.e_out_current();
        let e_in = self.e_in_current();
        assert!(len <= e_out.len() * e_in.len());
        let outer_len = len.div_ceil(e_in.len());
        #[cfg(feature = "parallel")]
        {
            (0..outer_len)
                .into_par_iter()
                .map(|x_out| {
                    let mut inner_acc = make_inner();
                    let inner_len = (len - x_out * e_in.len()).min(e_in.len());
                    for (x_in, &e_in) in e_in[..inner_len].iter().enumerate() {
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
            for (x_out, &e_out_val) in e_out[..outer_len].iter().enumerate() {
                let mut inner_acc = make_inner();
                let inner_len = (len - x_out * e_in.len()).min(e_in.len());
                for (x_in, &e_in_val) in e_in[..inner_len].iter().enumerate() {
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
#[path = "tests.rs"]
mod tests;
