//! Split equality polynomial evaluator for efficient sumcheck.
//!
//! Implements the Dao-Thaler + Gruen optimization for eq polynomial evaluations.
//! See <https://eprint.iacr.org/2024/1210.pdf>.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use jolt_field::{FieldAccumulator, WithChallenge};
use jolt_poly::{
    math::Math, BindingOrder, EqPolynomial, Polynomial, UnivariatePoly,
};

/// Split equality polynomial evaluator for efficient sumcheck.
///
/// Factors `eq(w, x)` into three parts and precomputes prefix tables:
///
/// ```text
/// eq(w, x) = eq(w_out, x_out) · eq(w_in, x_in) · eq(w_last, x_last)
/// ```
///
/// By caching prefix eq tables for `w_out` and `w_in`, we avoid redundant
/// computation across sumcheck rounds. The split structure enables
/// efficient parallel accumulation over `(x_out, x_in)` pairs via
/// [`par_fold_out_in`](Self::par_fold_out_in).
///
/// # Variable Layout (LowToHigh binding)
///
/// ```text
/// w = [w_0, w_1, ..., w_{m-1}, w_m, ..., w_{n-2}, w_{n-1}]
///      |------w_out-------|    |-----w_in-----|   w_last
///            m vars              (n-1-m) vars     1 var
/// ```
///
/// where `m = n / 2` and `n = w.len()`.
///
/// # Cached Tables
///
/// `E_out_vec[k]` and `E_in_vec[k]` hold eq tables over `k` variables:
///
/// ```text
/// E_out_vec[k] = eq(w_out[..k], ·) over {0,1}^k    (size 2^k)
/// E_in_vec[k]  = eq(w_in[..k], ·)  over {0,1}^k    (size 2^k)
///
/// Index 0 is always [1] (eq over 0 vars).
/// ```
///
/// This uniform indexing eliminates special-casing throughout the code.
/// Tables are popped during [`bind`](Self::bind) as variables are consumed.
#[allow(non_snake_case)]
#[derive(Clone, Debug)]
pub struct SplitEqEvaluator<F: WithChallenge> {
    /// Number of unbound variables remaining (LowToHigh) or
    /// number of bound variables (HighToLow).
    pub(crate) current_index: usize,
    /// Accumulated `eq(w_bound, r_bound)` from already-bound variables.
    pub(crate) current_scalar: F,
    /// The full challenge vector w.
    pub(crate) w: Vec<F::Challenge>,
    /// Prefix eq tables for w_in.
    pub(crate) E_in_vec: Vec<Vec<F>>,
    /// Prefix eq tables for w_out.
    pub(crate) E_out_vec: Vec<Vec<F>>,
    /// Binding order: LowToHigh (LSB first) or HighToLow (MSB first).
    pub(crate) binding_order: BindingOrder,
}

#[allow(non_snake_case)]
impl<F: WithChallenge> SplitEqEvaluator<F> {
    #[tracing::instrument(skip_all, name = "SplitEqEvaluator::new_with_scaling")]
    pub fn new_with_scaling(
        w: &[F::Challenge],
        binding_order: BindingOrder,
        scaling_factor: Option<F>,
    ) -> Self {
        match binding_order {
            BindingOrder::LowToHigh => {
                let m = w.len() / 2;
                let (_w_last, wprime) = w.split_last().unwrap();
                let (w_out, w_in) = wprime.split_at(m);

                #[cfg(feature = "parallel")]
                let (E_out_vec, E_in_vec) = rayon::join(
                    || EqPolynomial::<F>::evals_cached(w_out, None),
                    || EqPolynomial::<F>::evals_cached(w_in, None),
                );
                #[cfg(not(feature = "parallel"))]
                let (E_out_vec, E_in_vec) = (
                    EqPolynomial::<F>::evals_cached(w_out, None),
                    EqPolynomial::<F>::evals_cached(w_in, None),
                );

                Self {
                    current_index: w.len(),
                    current_scalar: scaling_factor.unwrap_or(F::one()),
                    w: w.to_vec(),
                    E_in_vec,
                    E_out_vec,
                    binding_order,
                }
            }
            BindingOrder::HighToLow => {
                let (_, wprime) = w.split_first().unwrap();
                let m = w.len() / 2;
                let (w_in, w_out) = wprime.split_at(m);

                #[cfg(feature = "parallel")]
                let (E_in_vec, E_out_vec) = rayon::join(
                    || EqPolynomial::<F>::evals_cached_rev(w_in, None),
                    || EqPolynomial::<F>::evals_cached_rev(w_out, None),
                );
                #[cfg(not(feature = "parallel"))]
                let (E_in_vec, E_out_vec) = (
                    EqPolynomial::<F>::evals_cached_rev(w_in, None),
                    EqPolynomial::<F>::evals_cached_rev(w_out, None),
                );

                Self {
                    current_index: 0,
                    current_scalar: scaling_factor.unwrap_or(F::one()),
                    w: w.to_vec(),
                    E_in_vec,
                    E_out_vec,
                    binding_order,
                }
            }
        }
    }

    #[tracing::instrument(skip_all, name = "SplitEqEvaluator::new")]
    pub fn new(w: &[F::Challenge], binding_order: BindingOrder) -> Self {
        Self::new_with_scaling(w, binding_order, None)
    }

    pub fn get_num_vars(&self) -> usize {
        self.w.len()
    }

    pub fn len(&self) -> usize {
        match self.binding_order {
            BindingOrder::LowToHigh => 1 << self.current_index,
            BindingOrder::HighToLow => 1 << (self.w.len() - self.current_index),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of variables that have already been bound into `current_scalar`.
    pub fn num_challenges(&self) -> usize {
        match self.binding_order {
            BindingOrder::LowToHigh => self.w.len() - self.current_index,
            BindingOrder::HighToLow => self.current_index,
        }
    }

    pub fn E_in_current_len(&self) -> usize {
        self.E_in_vec.last().expect("E_in_vec is never empty").len()
    }

    pub fn E_out_current_len(&self) -> usize {
        self.E_out_vec
            .last()
            .expect("E_out_vec is never empty")
            .len()
    }

    /// Current E_in table slice. Always at least `[1]` when fully bound.
    pub fn E_in_current(&self) -> &[F] {
        self.E_in_vec.last().expect("E_in_vec is never empty")
    }

    /// Current E_out table slice. Always at least `[1]` when fully bound.
    pub fn E_out_current(&self) -> &[F] {
        self.E_out_vec.last().expect("E_out_vec is never empty")
    }

    /// Returns the `(E_out, E_in)` tables for a streaming window of `window_size`.
    ///
    /// For LowToHigh binding, this factors `eq(w_head, x)` where `w_head` is
    /// the prefix before the current window:
    ///
    /// ```text
    /// log2(|E_out|) + log2(|E_in|) + log2(|E_active|) + 1 = #unbound bits
    /// ```
    ///
    /// where the `+ 1` accounts for the current linear Gruen bit.
    pub fn E_out_in_for_window(&self, window_size: usize) -> (&[F], &[F]) {
        match self.binding_order {
            BindingOrder::LowToHigh => {
                let num_unbound = self.current_index;
                let window_size = core::cmp::min(window_size, num_unbound);
                let head_len = num_unbound.saturating_sub(window_size);

                let n = self.w.len();
                let m = n / 2;
                let head_out_bits = core::cmp::min(head_len, m);
                let head_in_bits = head_len.saturating_sub(head_out_bits);

                debug_assert_eq!(head_out_bits + head_in_bits, head_len);
                debug_assert!(head_out_bits < self.E_out_vec.len());
                debug_assert!(head_in_bits < self.E_in_vec.len());

                (&self.E_out_vec[head_out_bits], &self.E_in_vec[head_in_bits])
            }
            BindingOrder::HighToLow => {
                panic!("E_out_in_for_window not implemented for HighToLow");
            }
        }
    }

    /// Returns the eq table over the "active" window bits (all but the
    /// last variable in the current streaming window).
    pub fn E_active_for_window(&self, window_size: usize) -> Vec<F> {
        if window_size <= 1 {
            return vec![F::one()];
        }

        match self.binding_order {
            BindingOrder::LowToHigh => {
                let num_unbound = self.current_index;
                if window_size > num_unbound {
                    return vec![F::one()];
                }
                let remaining_w = &self.w[..num_unbound];
                let window_start = remaining_w.len() - window_size;
                let (_w_body, w_window) = remaining_w.split_at(window_start);
                let (w_active, _w_curr) = w_window.split_at(window_size - 1);
                EqPolynomial::<F>::evals(w_active, None)
            }
            BindingOrder::HighToLow => {
                panic!("E_active_for_window not implemented for HighToLow");
            }
        }
    }

    /// Binds the current variable to challenge `r`, updating `current_scalar`
    /// by `eq(w_i, r) = 1 - w_i - r + 2·w_i·r` and popping the consumed eq table.
    #[tracing::instrument(skip_all, name = "SplitEqEvaluator::bind")]
    pub fn bind(&mut self, r: F::Challenge) {
        match self.binding_order {
            BindingOrder::LowToHigh => {
                // Convert to field for the eq evaluation (avoids Challenge×Challenge)
                let w_i: F = self.w[self.current_index - 1].into();
                let r_f: F = r.into();
                let prod = w_i * r_f;
                self.current_scalar *= F::one() - w_i - r_f + prod + prod;
                self.current_index -= 1;
                // Pop the consumed table; never pop the [1] at index 0
                if self.w.len() / 2 < self.current_index && self.E_in_vec.len() > 1 {
                    let _ = self.E_in_vec.pop();
                } else if 0 < self.current_index && self.E_out_vec.len() > 1 {
                    let _ = self.E_out_vec.pop();
                }
            }
            BindingOrder::HighToLow => {
                let w_i: F = self.w[self.current_index].into();
                let r_f: F = r.into();
                let prod = w_i * r_f;
                self.current_scalar *= F::one() - w_i - r_f + prod + prod;
                self.current_index += 1;
                if self.current_index <= self.w.len() / 2 && self.E_in_vec.len() > 1 {
                    let _ = self.E_in_vec.pop();
                } else if self.current_index <= self.w.len() && self.E_out_vec.len() > 1 {
                    let _ = self.E_out_vec.pop();
                }
            }
        }
    }

    /// Computes the cubic round polynomial `s(X) = l(X) · q(X)` where `l(X)` is
    /// the linear eq polynomial and `q(X) = c + dX + eX²`.
    ///
    /// # Arguments
    /// - `q_constant`: constant term of q
    /// - `q_quadratic_coeff`: quadratic coefficient of q
    /// - `s_0_plus_s_1`: the previous round claim `s(0) + s(1)`
    pub fn gruen_poly_deg_3(
        &self,
        q_constant: F,
        q_quadratic_coeff: F,
        s_0_plus_s_1: F,
    ) -> UnivariatePoly<F> {
        // l(X) evaluations via the current eq variable
        let eq_eval_1 = self.current_scalar * self.get_current_w();
        let eq_eval_0 = self.current_scalar - eq_eval_1;
        let eq_m = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_m;
        let eq_eval_3 = eq_eval_2 + eq_m;

        // q(X) evaluations from partial info
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

    /// Computes the quadratic round polynomial `s(X) = l(X) · q(X)` where
    /// both `l` and `q` are linear.
    ///
    /// # Arguments
    /// - `q_0`: `q(0)` (constant term)
    /// - `previous_claim`: `s(0) + s(1)`
    pub fn gruen_poly_deg_2(&self, q_0: F, previous_claim: F) -> UnivariatePoly<F> {
        let eq_eval_1 = self.current_scalar * self.get_current_w();
        let eq_eval_0 = self.current_scalar - eq_eval_1;
        let eq_m = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_m;

        let linear_eval_0 = q_0;
        let quadratic_eval_0 = eq_eval_0 * linear_eval_0;
        let quadratic_eval_1 = previous_claim - quadratic_eval_0;
        let linear_eval_1 = quadratic_eval_1 / eq_eval_1;
        let linear_eval_2 = linear_eval_1 + linear_eval_1 - linear_eval_0;

        UnivariatePoly::from_evals(&[
            quadratic_eval_0,
            quadratic_eval_1,
            eq_eval_2 * linear_eval_2,
        ])
    }

    /// Computes the round polynomial for arbitrary degree from partial evaluations.
    ///
    /// `q_evals` contains `[q(1), q(2), ..., q(deg-1), q(∞)]` where `q(∞)` is the
    /// leading coefficient. Recovers `q(0)` from the claim, interpolates via
    /// Toom-Cook, then multiplies by the linear eq polynomial.
    pub fn gruen_poly_from_evals(&self, q_evals: &[F], s_0_plus_s_1: F) -> UnivariatePoly<F> {
        let r_round: F = self.get_current_w().into();
        let l_at_0 = self.current_scalar * (F::one() - r_round);
        let l_at_1 = self.current_scalar * r_round;

        // Recover q(0): s(0) + s(1) = l(0)·q(0) + l(1)·q(1)
        let q_at_0 = (s_0_plus_s_1 - l_at_1 * q_evals[0]) / l_at_0;

        // Interpolate q from [q(0), q(1), ..., q(deg-1), q(∞)]
        let mut full_q_evals = q_evals.to_vec();
        full_q_evals.insert(0, q_at_0);
        let q = UnivariatePoly::from_evals_toom(&full_q_evals);

        // s(X) = l(X) · q(X) where l(X) = l_c0 + l_c1·X
        let l_c0 = l_at_0;
        let l_c1 = l_at_1 - l_at_0;
        let q_coeffs = q.coefficients();
        let mut s_coeffs = vec![F::zero(); q_coeffs.len() + 1];
        for (i, &q_ci) in q_coeffs.iter().enumerate() {
            s_coeffs[i] += q_ci * l_c0;
            s_coeffs[i + 1] += q_ci * l_c1;
        }
        UnivariatePoly::new(s_coeffs)
    }

    /// Materializes the remaining unbound eq polynomial as a dense polynomial.
    pub fn merge(&self) -> Polynomial<F> {
        let evals = match self.binding_order {
            BindingOrder::LowToHigh => EqPolynomial::<F>::evals_parallel(
                &self.w[..self.current_index],
                Some(self.current_scalar),
            ),
            BindingOrder::HighToLow => EqPolynomial::<F>::evals_parallel(
                &self.w[self.current_index..],
                Some(self.current_scalar),
            ),
        };
        Polynomial::new(evals)
    }

    pub fn get_current_scalar(&self) -> F {
        self.current_scalar
    }

    pub fn get_current_w(&self) -> F::Challenge {
        match self.binding_order {
            BindingOrder::LowToHigh => self.w[self.current_index - 1],
            BindingOrder::HighToLow => self.w[self.current_index],
        }
    }

    #[inline(always)]
    pub fn group_index(&self, x_out: usize, x_in: usize) -> usize {
        let num_x_in_bits = self.E_in_current_len().log_2();
        (x_out << num_x_in_bits) | x_in
    }

    /// Parallel fold over split-eq weights.
    ///
    /// Computes:
    /// ```text
    /// Σ_{x_out} E_out[x_out] · fold_{x_in}(E_in[x_in] · custom(x_out, x_in))
    /// ```
    ///
    /// Parallelizes over `x_out` (outer loop); inner loop over `x_in` is sequential.
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
        let e_out = self.E_out_current();
        let e_in = self.E_in_current();
        let out_len = e_out.len();

        let compute = |x_out: usize| -> OuterAcc {
            let mut inner_acc = make_inner();
            for (x_in, &e_in_val) in e_in.iter().enumerate() {
                let g = self.group_index(x_out, x_in);
                inner_step(&mut inner_acc, g, x_in, e_in_val);
            }
            outer_step(x_out, e_out[x_out], inner_acc)
        };

        #[cfg(feature = "parallel")]
        {
            (0..out_len)
                .into_par_iter()
                .map(compute)
                .reduce_with(&merge)
                .expect("par_fold_out_in: empty E_out; invariant violation")
        }

        #[cfg(not(feature = "parallel"))]
        {
            (0..out_len)
                .map(compute)
                .reduce(merge)
                .expect("par_fold_out_in: empty E_out; invariant violation")
        }
    }

    /// Delayed-reduction fold over split-eq weights.
    ///
    /// Inner loop accumulates with [`FieldAccumulator::fmadd`], reduces once
    /// per `x_out`, and returns `[F; NUM_OUT]` after final reduction.
    #[inline]
    pub fn par_fold_out_in_unreduced<const NUM_OUT: usize>(
        &self,
        per_g_values: &(impl Fn(usize) -> [F; NUM_OUT] + Sync + Send),
    ) -> [F; NUM_OUT] {
        self.par_fold_out_in(
            || [F::Accumulator::default(); NUM_OUT],
            |inner, g, _x_in, e_in| {
                let vals = per_g_values(g);
                for k in 0..NUM_OUT {
                    inner[k].fmadd(e_in, vals[k]);
                }
            },
            |_x_out, e_out, inner| {
                let mut outer = [F::Accumulator::default(); NUM_OUT];
                for k in 0..NUM_OUT {
                    let inner_red = inner[k].reduce();
                    outer[k].fmadd(e_out, inner_red);
                }
                outer
            },
            |mut a, b| {
                for k in 0..NUM_OUT {
                    a[k].merge(b[k]);
                }
                a
            },
        )
        .map(|acc| acc.reduce())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr, WithChallenge};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// Binds the highest (MSB) variable in an evaluation table.
    fn bind_top(evals: &mut Vec<Fr>, r: Fr) {
        let n = evals.len() / 2;
        for i in 0..n {
            evals[i] = evals[i] + r * (evals[i + n] - evals[i]);
        }
        evals.truncate(n);
    }

    /// Binds the lowest (LSB) variable in an evaluation table.
    /// Pairs `evals[2i]` (x_0=0) and `evals[2i+1]` (x_0=1).
    fn bind_bottom(evals: &mut Vec<Fr>, r: Fr) {
        let half = evals.len() / 2;
        for i in 0..half {
            evals[i] = evals[2 * i] + r * (evals[2 * i + 1] - evals[2 * i]);
        }
        evals.truncate(half);
    }

    #[test]
    fn bind_low_to_high() {
        const NUM_VARS: usize = 10;
        let mut rng = ChaCha20Rng::seed_from_u64(100);
        let w: Vec<<Fr as WithChallenge>::Challenge> =
            std::iter::repeat_with(|| <Fr as WithChallenge>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let mut regular_evals = EqPolynomial::<Fr>::evals(&w, None);
        let mut split_eq = SplitEqEvaluator::new(&w, BindingOrder::LowToHigh);

        let merged = split_eq.merge();
        assert_eq!(
            &regular_evals,
            merged.evaluations(),
            "initial mismatch"
        );

        for _ in 0..NUM_VARS {
            let r = <Fr as WithChallenge>::Challenge::random(&mut rng);
            let r_f: Fr = r.into();
            bind_bottom(&mut regular_evals, r_f);
            split_eq.bind(r);

            let merged = split_eq.merge();
            assert_eq!(
                &regular_evals,
                merged.evaluations(),
            );
        }
    }

    #[test]
    fn bind_high_to_low() {
        const NUM_VARS: usize = 10;
        let mut rng = ChaCha20Rng::seed_from_u64(101);
        let w: Vec<<Fr as WithChallenge>::Challenge> =
            std::iter::repeat_with(|| <Fr as WithChallenge>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let mut regular_evals = EqPolynomial::<Fr>::evals(&w, None);
        let mut split_eq = SplitEqEvaluator::new(&w, BindingOrder::HighToLow);

        let merged = split_eq.merge();
        assert_eq!(&regular_evals, merged.evaluations());

        for _ in 0..NUM_VARS {
            let r = <Fr as WithChallenge>::Challenge::random(&mut rng);
            let r_f: Fr = r.into();
            bind_top(&mut regular_evals, r_f);
            split_eq.bind(r);

            let merged = split_eq.merge();
            assert_eq!(&regular_evals, merged.evaluations());
        }
    }

    #[test]
    fn window_out_in() {
        const NUM_VARS: usize = 17;
        let mut rng = ChaCha20Rng::seed_from_u64(102);
        let w: Vec<<Fr as WithChallenge>::Challenge> =
            std::iter::repeat_with(|| <Fr as WithChallenge>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let split_eq = SplitEqEvaluator::<Fr>::new(&w, BindingOrder::LowToHigh);
        let (e_prime_out, e_prime_in) = split_eq.E_out_in_for_window(1);
        assert_eq!(split_eq.E_out_current_len(), 1 << 8);
        assert_eq!(e_prime_out.len(), split_eq.E_out_current().len());
        assert_eq!(e_prime_in.len(), split_eq.E_in_current().len());
    }

    #[test]
    fn window_size_one_matches_current() {
        const NUM_VARS: usize = 10;
        let mut rng = ChaCha20Rng::seed_from_u64(103);
        let w: Vec<<Fr as WithChallenge>::Challenge> =
            std::iter::repeat_with(|| <Fr as WithChallenge>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let mut split_eq = SplitEqEvaluator::<Fr>::new(&w, BindingOrder::LowToHigh);

        for _round in 0..NUM_VARS {
            let num_unbound = split_eq.current_index;
            if num_unbound <= 1 {
                break;
            }

            let (e_out_window, e_in_window) = split_eq.E_out_in_for_window(1);
            let w_head = &split_eq.w[..num_unbound - 1];
            let head_evals = EqPolynomial::<Fr>::evals(w_head, None);

            let num_x_out = e_out_window.len();
            let num_x_in = e_in_window.len();
            assert_eq!(num_x_out * num_x_in, head_evals.len());

            let x_in_bits = num_x_in.log_2();
            for (x_out, &e_out_val) in e_out_window.iter().enumerate() {
                for (x_in, &e_in_val) in e_in_window.iter().enumerate() {
                    let idx = (x_out << x_in_bits) | x_in;
                    assert_eq!(
                        e_out_val * e_in_val,
                        head_evals[idx],
                        "factorisation mismatch at round={_round}, x_out={x_out}, x_in={x_in}",
                    );
                }
            }

            let r = <Fr as WithChallenge>::Challenge::random(&mut rng);
            split_eq.bind(r);
        }
    }

    #[test]
    fn window_bit_accounting_invariants() {
        const NUM_VARS: usize = 8;
        let mut rng = ChaCha20Rng::seed_from_u64(104);
        let w: Vec<<Fr as WithChallenge>::Challenge> =
            std::iter::repeat_with(|| <Fr as WithChallenge>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let mut split_eq = SplitEqEvaluator::<Fr>::new(&w, BindingOrder::LowToHigh);

        for _round in 0..NUM_VARS {
            let num_unbound = split_eq.len().log_2();
            if num_unbound == 0 {
                break;
            }

            for window_size in 1..=num_unbound {
                let (e_out, e_in) = split_eq.E_out_in_for_window(window_size);
                let e_active = split_eq.E_active_for_window(window_size);

                let bits_out = e_out.len().log_2();
                let bits_in = e_in.len().log_2();
                let bits_active = e_active.len().log_2();

                assert_eq!(
                    bits_out + bits_in + bits_active + 1,
                    num_unbound,
                    "bit accounting failed for window_size={window_size}",
                );
            }

            let r = <Fr as WithChallenge>::Challenge::random(&mut rng);
            split_eq.bind(r);
        }
    }

    #[test]
    fn e_vec_invariant_preserved_through_binding() {
        const NUM_VARS: usize = 10;
        let mut rng = ChaCha20Rng::seed_from_u64(105);
        let w: Vec<<Fr as WithChallenge>::Challenge> =
            std::iter::repeat_with(|| <Fr as WithChallenge>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let mut split_eq = SplitEqEvaluator::<Fr>::new(&w, BindingOrder::LowToHigh);

        // E_*_vec[0] = [1] at construction
        assert_eq!(split_eq.E_out_vec[0], vec![Fr::one()]);
        assert_eq!(split_eq.E_in_vec[0], vec![Fr::one()]);

        for _ in 0..NUM_VARS {
            let r = <Fr as WithChallenge>::Challenge::random(&mut rng);
            split_eq.bind(r);
            // Invariant: E_*_vec never empty, index 0 always [1]
            assert!(!split_eq.E_out_vec.is_empty());
            assert!(!split_eq.E_in_vec.is_empty());
            assert_eq!(split_eq.E_out_vec[0], vec![Fr::one()]);
            assert_eq!(split_eq.E_in_vec[0], vec![Fr::one()]);
        }
    }

    #[test]
    fn par_fold_out_in_unreduced_matches_direct() {
        const NUM_VARS: usize = 8;
        let mut rng = ChaCha20Rng::seed_from_u64(106);
        let w: Vec<<Fr as WithChallenge>::Challenge> =
            std::iter::repeat_with(|| <Fr as WithChallenge>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let split_eq = SplitEqEvaluator::<Fr>::new(&w, BindingOrder::LowToHigh);

        // par_fold_out_in operates over the (x_out, x_in) factored tables,
        // NOT the full merged table. Compute expected values directly from
        // the same E_out/E_in tables.
        let e_out = split_eq.E_out_current();
        let e_in = split_eq.E_in_current();
        let in_bits = e_in.len().log_2();

        let mut expected_sum = Fr::zero();
        let mut expected_sum_sq = Fr::zero();
        for (x_out, &e_out_val) in e_out.iter().enumerate() {
            for (x_in, &e_in_val) in e_in.iter().enumerate() {
                let g = (x_out << in_bits) | x_in;
                let g_f = Fr::from_u64(g as u64);
                let weight = e_out_val * e_in_val;
                expected_sum += weight * g_f;
                expected_sum_sq += weight * g_f * g_f;
            }
        }

        let [got_sum, got_sum_sq] = split_eq.par_fold_out_in_unreduced(&|g| {
            let g_f = Fr::from_u64(g as u64);
            [g_f, g_f * g_f]
        });

        assert_eq!(got_sum, expected_sum);
        assert_eq!(got_sum_sq, expected_sum_sq);
    }

    #[test]
    fn gruen_poly_deg_2_consistency() {
        // l(x) = eq_eval_0 + (eq_eval_1 - eq_eval_0)·x
        // q(x) = q_0 + q_1·x
        // s(x) = l(x)·q(x)
        // Verify s(0) + s(1) = previous_claim
        let mut rng = ChaCha20Rng::seed_from_u64(107);
        let w: Vec<<Fr as WithChallenge>::Challenge> =
            std::iter::repeat_with(|| <Fr as WithChallenge>::Challenge::random(&mut rng))
                .take(5)
                .collect();

        let split_eq = SplitEqEvaluator::<Fr>::new(&w, BindingOrder::LowToHigh);

        let q_0 = Fr::from_u64(7);
        let q_1 = Fr::from_u64(13);
        let eq_eval_1: Fr = split_eq.current_scalar * split_eq.get_current_w();
        let eq_eval_0 = split_eq.current_scalar - eq_eval_1;
        let s_0 = eq_eval_0 * q_0;
        let s_1 = eq_eval_1 * (q_0 + q_1);
        let claim = s_0 + s_1;

        let poly = split_eq.gruen_poly_deg_2(q_0, claim);
        assert_eq!(poly.evaluate(Fr::zero()) + poly.evaluate(Fr::one()), claim);
        assert_eq!(poly.evaluate(Fr::zero()), s_0);
    }

    #[test]
    fn gruen_poly_deg_3_consistency() {
        let mut rng = ChaCha20Rng::seed_from_u64(108);
        let w: Vec<<Fr as WithChallenge>::Challenge> =
            std::iter::repeat_with(|| <Fr as WithChallenge>::Challenge::random(&mut rng))
                .take(5)
                .collect();

        let split_eq = SplitEqEvaluator::<Fr>::new(&w, BindingOrder::LowToHigh);

        // q(x) = c + d·x + e·x² with known c = 3, e = 5, d = 2
        let c = Fr::from_u64(3);
        let d = Fr::from_u64(2);
        let e = Fr::from_u64(5);
        let eq_eval_1: Fr = split_eq.current_scalar * split_eq.get_current_w();
        let eq_eval_0 = split_eq.current_scalar - eq_eval_1;
        let q_0 = c;
        let q_1 = c + d + e;
        let s_0 = eq_eval_0 * q_0;
        let s_1 = eq_eval_1 * q_1;
        let claim = s_0 + s_1;

        let poly = split_eq.gruen_poly_deg_3(c, e, claim);
        assert_eq!(poly.evaluate(Fr::zero()) + poly.evaluate(Fr::one()), claim);
        assert_eq!(poly.evaluate(Fr::zero()), s_0);
    }
}
