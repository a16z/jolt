use super::{GruenRound, GruenRoundMut};
use crate::split_eq::recovery::{mul_linear, PreparedLinearRecovery};
use crate::{BindingOrder, UnivariatePoly};
use jolt_field::Field;

impl<'state, F: Field> GruenRound<'state, F> {
    pub fn coordinate(&self) -> F {
        self.coordinate
    }

    /// Evaluations `(l(0), l(1))` of the scaled current equality factor.
    pub fn linear_evals(&self) -> (F, F) {
        let at_one = self.state.current_scalar * self.coordinate;
        (self.state.current_scalar - at_one, at_one)
    }

    /// Prepares one inversion for this round; `None` exactly when `l(1) = 0`.
    /// The recovery borrows the state, including when the handle is temporary.
    ///
    /// ```compile_fail,E0502
    /// use jolt_field::Field;
    /// use jolt_poly::GruenSplitEqPolynomial;
    /// fn stale<F: Field>(state: &mut GruenSplitEqPolynomial<F>, challenge: F) {
    ///     let recovery = state.round().and_then(|r| r.prepare_linear_q_recovery());
    ///     if let Some(recovery) = recovery {
    ///         if let Some(handle) = state.round_mut() { handle.bind(challenge); }
    ///         recovery.gruen_poly_deg_3(challenge, challenge, challenge);
    ///     }
    /// }
    /// ```
    pub fn prepare_linear_q_recovery(&self) -> Option<PreparedLinearRecovery<'state, F>> {
        let (at_zero, at_one) = self.linear_evals();
        PreparedLinearRecovery::new(at_zero, at_one)
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
        let (eq_eval_0, eq_eval_1) = self.linear_evals();
        if self.state.current_scalar.is_zero() {
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
    /// The caller must supply at least two entries in `q_evals`.
    /// The missing `q(0)` is evaluated only when `l(0)` vanishes.
    pub fn gruen_poly_from_evals(
        &self,
        q_evals: &[F],
        s_0_plus_s_1: F,
        q_at_zero: impl FnOnce() -> F,
    ) -> Result<UnivariatePoly<F>, F> {
        if self.state.current_scalar.is_zero() {
            return Self::zero_round(q_evals.len() + 2, s_0_plus_s_1);
        }
        let q_zero = self.recover_q_zero(q_evals[0], s_0_plus_s_1, q_at_zero)?;
        let mut full_q_evals = Vec::with_capacity(q_evals.len() + 1);
        full_q_evals.push(q_zero);
        full_q_evals.extend_from_slice(q_evals);
        let q_coeffs = UnivariatePoly::from_evals_toom(&full_q_evals).into_coefficients();
        Ok(self.gruen_mul(&q_coeffs))
    }

    /// Degree-two message for a linear inner factor, with lazy `q(0)` recovery.
    pub fn gruen_poly_deg_2(
        &self,
        q_one: F,
        s_0_plus_s_1: F,
        q_at_zero: impl FnOnce() -> F,
    ) -> Result<UnivariatePoly<F>, F> {
        if self.state.current_scalar.is_zero() {
            return Self::zero_round(3, s_0_plus_s_1);
        }
        let q_zero = self.recover_q_zero(q_one, s_0_plus_s_1, q_at_zero)?;
        Ok(self.gruen_mul(&[q_zero, q_one - q_zero]))
    }

    fn recover_q_zero(&self, q_one: F, hint: F, q_at_zero: impl FnOnce() -> F) -> Result<F, F> {
        let (l_zero, l_one) = self.linear_evals();
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

    /// Multiplies the coefficient-form inner polynomial by the current linear factor.
    pub fn gruen_mul(&self, q_coeffs: &[F]) -> UnivariatePoly<F> {
        let (at_zero, at_one) = self.linear_evals();
        mul_linear(q_coeffs, at_zero, at_one - at_zero)
    }

    fn zero_round(coefficients: usize, hint: F) -> Result<UnivariatePoly<F>, F> {
        if hint.is_zero() {
            Ok(UnivariatePoly::new(vec![F::zero(); coefficients]))
        } else {
            Err(F::zero())
        }
    }
}

impl<F: Field> GruenRoundMut<'_, F> {
    pub fn round(&self) -> GruenRound<'_, F> {
        GruenRound {
            state: self.state,
            coordinate: self.coordinate,
        }
    }

    pub fn bind(self, challenge: F) {
        match self.state.binding_order {
            BindingOrder::LowToHigh => {
                let point = self.coordinate;
                let product = point * challenge;
                self.state.current_scalar *= F::one() - point - challenge + product + product;
                self.state.current_index -= 1;
                if self.state.point.len() / 2 < self.state.current_index
                    && self.state.e_in_vec.len() > 1
                {
                    let _ = self.state.e_in_vec.pop();
                } else if 0 < self.state.current_index && self.state.e_out_vec.len() > 1 {
                    let _ = self.state.e_out_vec.pop();
                }
            }
            BindingOrder::HighToLow => {
                let point = self.coordinate;
                let product = point * challenge;
                self.state.current_scalar *= F::one() - point - challenge + product + product;
                self.state.current_index += 1;
                if self.state.current_index <= self.state.point.len() / 2
                    && self.state.e_in_vec.len() > 1
                {
                    let _ = self.state.e_in_vec.pop();
                } else if self.state.current_index <= self.state.point.len()
                    && self.state.e_out_vec.len() > 1
                {
                    let _ = self.state.e_out_vec.pop();
                }
            }
        }
    }
}
