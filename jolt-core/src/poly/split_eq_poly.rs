//! Implements the Dao-Thaler + Gruen optimization for EQ polynomial evaluations
//! https://eprint.iacr.org/2024/1210.pdf

use allocative::Allocative;
use ark_ff::Zero;
use rayon::prelude::*;

use super::dense_mlpoly::DensePolynomial;
use super::multilinear_polynomial::BindingOrder;
use crate::{
    field::JoltField,
    poly::{eq_poly::EqPolynomial, unipoly::UniPoly},
    utils::math::Math,
};

/// Split equality polynomial for efficient sumcheck.
///
/// Factors eq(w, x) into three parts and precomputes prefix tables:
///
/// ```text
/// eq(w, x) = eq(w_out, x_out) · eq(w_in, x_in) · eq(w_last, x_last)
/// ```
///
/// By caching prefix eq tables for w_out and w_in, we avoid redundant
/// computation across sumcheck rounds. The split structure also enables
/// efficient parallel accumulation over (x_out, x_in) pairs.
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
/// `E_out_vec` and `E_in_vec` store prefix eq tables with uniform indexing:
///
/// ```text
/// E_out_vec[k] = eq(w_out[..k], ·) over {0,1}^k    (size 2^k)
/// E_in_vec[k]  = eq(w_in[..k], ·)  over {0,1}^k    (size 2^k)
///
/// Indexing:
///   [0]: [1]           ← eq over 0 vars (always 1)
///   [1]: [1-w_0, w_0]  ← eq over 1 var
///   [2]: size-4 table  ← eq over 2 vars
///   ...
/// ```
///
/// This uniform indexing (k → k vars) eliminates special-casing throughout the code.
///
/// # Binding Progress (LowToHigh)
///
/// ```text
/// Round 0:  current_index = n,   E_out = full,  E_in = full
/// Round i:  current_index = n-i, pop from E_in (then E_out)
/// Final:    current_index = 0,   E_out = [[1]], E_in = [[1]]
/// ```
///
/// The `current_scalar` accumulates eq(w_bound, r_bound) as we bind variables.
#[derive(Debug, Clone, PartialEq, Allocative)]
pub struct GruenSplitEqPolynomial<F: JoltField> {
    /// Number of unbound variables remaining (decrements each round).
    pub(crate) current_index: usize,
    /// Accumulated eq(w_bound, r_bound) from already-bound variables.
    pub(crate) current_scalar: F,
    /// The full challenge vector w.
    pub(crate) w: Vec<F::Challenge>,
    /// Prefix eq tables for w_in. E_in_vec[k] = eq(w_in[..k], ·) over {0,1}^k.
    /// Invariant: always non-empty; E_in_vec[0] = [1].
    pub(crate) E_in_vec: Vec<Vec<F>>,
    /// Prefix eq tables for w_out. E_out_vec[k] = eq(w_out[..k], ·) over {0,1}^k.
    /// Invariant: always non-empty; E_out_vec[0] = [1].
    pub(crate) E_out_vec: Vec<Vec<F>>,
    /// Binding order: LowToHigh (LSB first) or HighToLow (MSB first).
    pub(crate) binding_order: BindingOrder,
}

impl<F: JoltField> GruenSplitEqPolynomial<F> {
    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::new_with_scaling")]
    pub fn new_with_scaling(
        w: &[F::Challenge],
        binding_order: BindingOrder,
        scaling_factor: Option<F>,
    ) -> Self {
        match binding_order {
            BindingOrder::LowToHigh => {
                let m = w.len() / 2;
                //   w = [w_out, w_in, w_last]
                //         ↑      ↑      ↑
                //         |      |      |
                //         |      |      last element
                //         |      second half of remaining elements (for E_in)
                //         first half of remaining elements (for E_out)
                let (_w_last, wprime) = w.split_last().unwrap();
                let (w_out, w_in) = wprime.split_at(m);
                // evals_cached returns (n+1) tables where index k = eq over k vars.
                // E_*_vec[0] = [1] already.
                let (E_out_vec, E_in_vec) = rayon::join(
                    || EqPolynomial::evals_cached(w_out),
                    || EqPolynomial::evals_cached(w_in),
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
                // For high-to-low binding, we bind from MSB (index 0) to LSB (index n-1).
                // The split should be: w_in = first half, w_out = second half
                // [w_first, w_in, w_out]
                let (_, wprime) = w.split_first().unwrap();
                let m = w.len() / 2;
                let (w_in, w_out) = wprime.split_at(m);
                // evals_cached_rev returns (n+1) tables where index k = eq over k vars.
                // E_*_vec[0] = [1] already.
                let (E_in_vec, E_out_vec) = rayon::join(
                    || EqPolynomial::evals_cached_rev(w_in),
                    || EqPolynomial::evals_cached_rev(w_out),
                );

                Self {
                    current_index: 0, // Start from 0 for high-to-low up to w.len() - 1
                    current_scalar: scaling_factor.unwrap_or(F::one()),
                    w: w.to_vec(),
                    E_in_vec,
                    E_out_vec,
                    binding_order,
                }
            }
        }
    }

    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::new")]
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

    /// Number of variables that have already been bound into `current_scalar`.
    /// For LowToHigh this is `w.len() - current_index`; for HighToLow it is
    /// `current_index`.
    pub fn num_challenges(&self) -> usize {
        match self.binding_order {
            BindingOrder::LowToHigh => self.w.len() - self.current_index,
            BindingOrder::HighToLow => self.current_index,
        }
    }

    pub fn E_in_current_len(&self) -> usize {
        // Invariant: E_in_vec always has at least one entry ([1] at index 0)
        self.E_in_vec.last().expect("E_in_vec is never empty").len()
    }

    pub fn E_out_current_len(&self) -> usize {
        // Invariant: E_out_vec always has at least one entry ([1] at index 0)
        self.E_out_vec
            .last()
            .expect("E_out_vec is never empty")
            .len()
    }

    /// Return the last vector from `E_in_vec` as a slice.
    /// Invariant: always returns at least `[1]` when fully bound.
    pub fn E_in_current(&self) -> &[F] {
        self.E_in_vec.last().expect("E_in_vec is never empty")
    }

    /// Return the last vector from `E_out_vec` as a slice.
    /// Invariant: always returns at least `[1]` when fully bound.
    pub fn E_out_current(&self) -> &[F] {
        self.E_out_vec.last().expect("E_out_vec is never empty")
    }

    /// Return the (E_out, E_in) tables corresponding to a streaming window of the
    /// given `window_size`, using an explicit slice-based factorisation of the
    /// current unbound variables.
    ///
    /// Semantics (LowToHigh):
    /// - Let `num_unbound = current_index` and `remaining_w = w[..num_unbound]`.
    /// - For a window of size `window_size >= 1`, define:
    ///     - `w_window` as the last `window_size` bits of `remaining_w`
    ///     - `w_head`   as the prefix before `w_window`
    ///     - within `w_window`, the last bit is the current Gruen variable and the
    ///       preceding `window_size - 1` bits are the "active" window bits
    /// - This function returns eq tables over `w_head`, split into two halves
    ///   `w_out` and `w_in`:
    ///     - `w_head = [w_out || w_in]` with `w_out` = first `⌊|w_head| / 2⌋` bits
    ///     - `eq(w_head, (x_out, x_in)) = E_out[x_out] * E_in[x_in]`.
    ///
    /// The active window bits are handled separately by [`E_active_for_window`].
    /// Together they satisfy, for `BindingOrder::LowToHigh`,
    ///   log2(|E_out|) + log2(|E_in|) + log2(|E_active|) + 1 = #unbound bits,
    /// where the final `+ 1` accounts for the current linear Gruen bit.
    ///
    /// This helper returns slices and represents "no head bits" as
    /// single-entry `[1]` tables, matching `eq((), ()) = 1`.
    pub fn E_out_in_for_window(&self, window_size: usize) -> (&[F], &[F]) {
        match self.binding_order {
            BindingOrder::LowToHigh => {
                let num_unbound = self.current_index;

                // Restrict window size to the actually available unbound bits.
                let window_size = core::cmp::min(window_size, num_unbound);
                let head_len = num_unbound.saturating_sub(window_size);

                // The head prefix consists of the earliest `head_len` bits of `w`.
                // These live entirely in the original `[w_out || w_in] = w[..n-1]`
                // region, so we can factor them via prefixes of `w_out` and `w_in`.
                let n = self.w.len();
                let m = n / 2;

                let head_out_bits = core::cmp::min(head_len, m);
                let head_in_bits = head_len.saturating_sub(head_out_bits);

                // Invariant: head_out_bits + head_in_bits == head_len
                debug_assert_eq!(
                    head_out_bits + head_in_bits,
                    head_len,
                    "head bit split mismatch: {head_out_bits} + {head_in_bits} != {head_len}",
                );
                debug_assert!(
                    head_out_bits <= m,
                    "head_out_bits={head_out_bits} exceeds m={m}",
                );
                debug_assert!(
                    head_in_bits <= n - 1 - m,
                    "head_in_bits={} exceeds available in bits={}",
                    head_in_bits,
                    n - 1 - m
                );

                // With the new invariant, E_*_vec[k] = eq table over k variables,
                // and E_*_vec[0] = [1]. No special cases needed!
                debug_assert!(
                    head_out_bits < self.E_out_vec.len(),
                    "head_out_bits={} out of bounds for E_out_vec.len()={}",
                    head_out_bits,
                    self.E_out_vec.len()
                );
                debug_assert!(
                    head_in_bits < self.E_in_vec.len(),
                    "head_in_bits={} out of bounds for E_in_vec.len()={}",
                    head_in_bits,
                    self.E_in_vec.len()
                );

                let e_out = &self.E_out_vec[head_out_bits];
                let e_in = &self.E_in_vec[head_in_bits];

                (e_out, e_in)
            }
            BindingOrder::HighToLow => {
                // Streaming windows are not defined for HighToLow in the current
                // Spartan code paths; return neutral head tables.
                unimplemented!("Not implemented for high to low");
            }
        }
    }

    /// Return the equality table over the "active" window bits (all but the
    /// last variable in the current streaming window). This is used when
    /// projecting the multiquadratic t'(z_0, ..., z_{w-1}) down to a univariate
    /// in the first variable by summing against eq(tau_active, ·) over the
    /// remaining coordinates.
    ///
    /// We derive the active slice directly from the unbound portion of `w`.
    /// For LowToHigh binding, the unbound variables are `w[..current_index]`;
    /// the last `window_size` of these belong to the current window, and all
    /// but the final one are "active".
    pub fn E_active_for_window(&self, window_size: usize) -> Vec<F> {
        if window_size <= 1 {
            // No active bits in a size-0/1 window; eq over zero vars is [1].
            return vec![F::one()];
        }

        match self.binding_order {
            BindingOrder::LowToHigh => {
                let num_unbound = self.current_index;
                if window_size > num_unbound {
                    // Clamp to the maximum meaningful window size at this round.
                    return vec![F::one()];
                }
                let remaining_w = &self.w[..num_unbound];
                let window_start = remaining_w.len() - window_size;
                let (_w_body, w_window) = remaining_w.split_at(window_start);
                let (w_active, _w_curr_slice) = w_window.split_at(window_size - 1);
                // We only need the full eq table over the active window bits.
                EqPolynomial::<F>::evals(w_active)
            }
            BindingOrder::HighToLow => {
                // Not used for the outer Spartan streaming code.
                unimplemented!("Not implemented for high to low");
            }
        }
    }

    #[tracing::instrument(skip_all, name = "GruenSplitEqPolynomial::bind")]
    pub fn bind(&mut self, r: F::Challenge) {
        match self.binding_order {
            BindingOrder::LowToHigh => {
                // multiply `current_scalar` by `eq(w[i], r) = (1 - w[i]) * (1 - r) + w[i] * r`
                // which is the same as `1 - w[i] - r + 2 * w[i] * r`
                let prod_w_r = self.w[self.current_index - 1] * r;
                self.current_scalar *=
                    F::one() - self.w[self.current_index - 1] - r + prod_w_r + prod_w_r;
                // decrement `current_index`
                self.current_index -= 1;
                // pop the last vector from `E_in_vec` or `E_out_vec` (since we don't need it anymore)
                // Invariant: never pop the [1] at index 0, so check len() > 1
                if self.w.len() / 2 < self.current_index && self.E_in_vec.len() > 1 {
                    self.E_in_vec.pop();
                } else if 0 < self.current_index && self.E_out_vec.len() > 1 {
                    self.E_out_vec.pop();
                }
            }
            BindingOrder::HighToLow => {
                // multiply `current_scalar` by `eq(w[i], r) = (1 - w[i]) * (1 - r) + w[i] * r`
                // which is the same as `1 - w[i] - r + 2 * w[i] * r`
                let prod_w_r = self.w[self.current_index] * r;
                self.current_scalar *=
                    F::one() - self.w[self.current_index] - r + prod_w_r + prod_w_r;

                // increment `current_index` (going from 0 to n-1)
                self.current_index += 1;

                // pop the last vector from `E_in_vec` or `E_out_vec` (since we don't need it anymore)
                // For high-to-low, we bind variables in the first half first (E_in),
                // then variables in the second half (E_out)
                // Invariant: never pop the [1] at index 0, so check len() > 1
                if self.current_index <= self.w.len() / 2 && self.E_in_vec.len() > 1 {
                    // We're binding variables from the first half (E_in)
                    self.E_in_vec.pop();
                } else if self.current_index <= self.w.len() && self.E_out_vec.len() > 1 {
                    // We're binding variables from the second half (E_out)
                    self.E_out_vec.pop();
                }
            }
        }
    }

    /// Compute the cubic polynomial s(X) = l(X) * q(X), where l(X) is the
    /// current (linear) eq polynomial and q(X) = c + dX + eX^2, given the following:
    /// - c, the constant term of q
    /// - e, the quadratic term of q
    /// - the previous round claim, s(0) + s(1)
    pub fn gruen_poly_deg_3(
        &self,
        q_constant: F,
        q_quadratic_coeff: F,
        s_0_plus_s_1: F,
    ) -> UniPoly<F> {
        // We want to compute the evaluations of the cubic polynomial s(X) = l(X) * q(X), where
        // l is linear, and q is quadratic, at the points {0, 2, 3}.
        //
        // At this point, we have
        // - the linear polynomial, l(X) = a + bX
        // - the quadratic polynomial, q(X) = c + dX + eX^2
        // - the previous round's claim s(0) + s(1) = a * c + (a + b) * (c + d + e)
        //
        // Both l and q are represented by their evaluations at 0 and infinity. I.e., we have a, b, c,
        // and e, but not d. We compute s by first computing l and q at points 2 and 3.

        // Evaluations of the linear polynomial
        let eq_eval_1 = self.current_scalar
            * match self.binding_order {
                BindingOrder::LowToHigh => self.w[self.current_index - 1],
                BindingOrder::HighToLow => self.w[self.current_index],
            };
        let eq_eval_0 = self.current_scalar - eq_eval_1;
        let eq_m = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_m;
        let eq_eval_3 = eq_eval_2 + eq_m;

        // Evaluations of the quadratic polynomial
        let quadratic_eval_0 = q_constant;
        let cubic_eval_0 = eq_eval_0 * quadratic_eval_0;
        let cubic_eval_1 = s_0_plus_s_1 - cubic_eval_0;
        // q(1) = c + d + e
        let quadratic_eval_1 = cubic_eval_1 / eq_eval_1;
        // q(2) = c + 2d + 4e = q(1) + q(1) - q(0) + 2e
        let e_times_2 = q_quadratic_coeff + q_quadratic_coeff;
        let quadratic_eval_2 = quadratic_eval_1 + quadratic_eval_1 - quadratic_eval_0 + e_times_2;
        // q(3) = c + 3d + 9e = q(2) + q(1) - q(0) + 4e
        let quadratic_eval_3 =
            quadratic_eval_2 + quadratic_eval_1 - quadratic_eval_0 + e_times_2 + e_times_2;

        UniPoly::from_evals(&[
            cubic_eval_0,
            cubic_eval_1,
            eq_eval_2 * quadratic_eval_2,
            eq_eval_3 * quadratic_eval_3,
        ])
    }

    /// Compute the quadratic polynomial s(X) = l(X) * q(X), where l(X) is the
    /// current (linear) Dao-Thaler eq polynomial and q(X) = c + dx
    /// - c, the constant term of q
    /// - the previous round claim, s(0) + s(1)
    pub fn gruen_poly_deg_2(&self, q_0: F, previous_claim: F) -> UniPoly<F> {
        // We want to compute the evaluations of the quadratic polynomial s(X) = l(X) * q(X), where
        // l is linear, and q is linear, at the points {0, 2}.
        //
        // At this point, we have:
        // - the linear polynomial, l(X) = a + bX
        // - the linear polynomial, q(X) = c + dX
        // - the previous round's claim s(0) + s(1) = a * c + (a + b) * (c + d)
        //
        // We have q(0) = c, and we need to compute q(1) and q(2).

        // Evaluations of the linear eq polynomial
        let eq_eval_1 = self.current_scalar
            * match self.binding_order {
                BindingOrder::LowToHigh => self.w[self.current_index - 1],
                BindingOrder::HighToLow => self.w[self.current_index],
            };
        let eq_eval_0 = self.current_scalar - eq_eval_1;

        // slope for eq
        let eq_m = eq_eval_1 - eq_eval_0;
        let eq_eval_2 = eq_eval_1 + eq_m;

        // Evaluations of the linear q(x) polynomial
        let linear_eval_0 = q_0;
        let quadratic_eval_0 = eq_eval_0 * linear_eval_0;
        let quadratic_eval_1 = previous_claim - quadratic_eval_0;

        // q(1) = c + d
        let linear_eval_1 = quadratic_eval_1 / eq_eval_1;

        // q(2) = c + 2d = 2*q(1) - q(0)
        let linear_eval_2 = linear_eval_1 + linear_eval_1 - linear_eval_0;

        UniPoly::from_evals(&[
            quadratic_eval_0,
            quadratic_eval_1,
            eq_eval_2 * linear_eval_2,
        ])
    }

    pub fn merge(&self) -> DensePolynomial<F> {
        let evals = match self.binding_order {
            BindingOrder::LowToHigh => {
                // For low-to-high, current_index tracks how many variables remain unbound
                // We want eq(w[0..current_index], x)
                EqPolynomial::evals_parallel(
                    &self.w[..self.current_index],
                    Some(self.current_scalar),
                )
            }
            BindingOrder::HighToLow => {
                // For high-to-low, current_index tracks how many variables have been bound
                // We want eq(w[current_index..], x)
                EqPolynomial::evals_parallel(
                    &self.w[self.current_index..],
                    Some(self.current_scalar),
                )
            }
        };
        DensePolynomial::new(evals)
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

    /// Parallel fold over current split-eq weights:
    ///   Σ_{x_out} E_out[x_out] · fold_{x_in}(E_in[x_in] · custom(x_out, x_in))
    ///
    /// The caller supplies how to:
    /// - create an inner accumulator,
    /// - step the inner accumulator with (g, x_in, e_in),
    /// - turn the finished inner accumulator into an outer accumulator item given (x_out, e_out),
    /// - and merge outer accumulator items across x_out in parallel.
    ///
    /// Invariant: E_in and E_out are never empty (at minimum they contain [1]).
    /// When fully bound, E_in = [1], so the inner loop runs once with e_in = 1.
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
        let in_len = e_in.len();

        (0..out_len)
            .into_par_iter()
            .map(|x_out| {
                let mut inner_acc = make_inner();

                // No special case needed: E_in is always at least [1]
                for x_in in 0..in_len {
                    let g = self.group_index(x_out, x_in);
                    inner_step(&mut inner_acc, g, x_in, e_in[x_in]);
                }

                outer_step(x_out, e_out[x_out], inner_acc)
            })
            .reduce_with(merge)
            .expect("par_fold_out_in: empty E_out; invariant violation")
    }

    /// Common delayed reduction with Montgomery reduction pattern:
    /// - inner accumulates with e_in.mul_unreduced over NUM_OUT outputs,
    /// - reduce once with Montgomery reduction,
    /// - outer scales by e_out.mul_unreduced,
    /// - reduce at end and return [F; NUM_OUT] with Montgomery reduction.
    #[inline]
    pub fn par_fold_out_in_unreduced<const LIMBS: usize, const NUM_OUT: usize>(
        &self,
        per_g_values: &(impl Fn(usize) -> [F; NUM_OUT] + Sync + Send),
    ) -> [F; NUM_OUT] {
        self.par_fold_out_in(
            || [F::Unreduced::<LIMBS>::zero(); NUM_OUT],
            |inner, g, _x_in, e_in| {
                let vals = per_g_values(g);
                for k in 0..NUM_OUT {
                    inner[k] += e_in.mul_unreduced::<LIMBS>(vals[k]);
                }
            },
            |_x_out, e_out, inner| {
                let mut outer = [F::Unreduced::<LIMBS>::zero(); NUM_OUT];
                for k in 0..NUM_OUT {
                    let inner_red = F::from_montgomery_reduce::<LIMBS>(inner[k]);
                    outer[k] = e_out.mul_unreduced::<LIMBS>(inner_red);
                }
                outer
            },
            |mut a, b| {
                for k in 0..NUM_OUT {
                    a[k] += b[k];
                }
                a
            },
        )
        .map(F::from_montgomery_reduce::<LIMBS>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::{test_rng, One};

    #[test]
    fn window_out_in() {
        const NUM_VARS: usize = 17;
        let mut rng = test_rng();
        let w: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let split_eq = GruenSplitEqPolynomial::<Fr>::new(&w, BindingOrder::LowToHigh);
        let (e_prime_out, e_prime_in) = split_eq.E_out_in_for_window(1);
        assert_eq!(split_eq.E_out_current_len(), 1 << 8);
        assert_eq!(e_prime_out.len(), split_eq.E_out_current().len());
        assert_eq!(e_prime_in.len(), split_eq.E_in_current().len());
    }

    #[test]
    fn bind_low_high() {
        const NUM_VARS: usize = 10;
        let mut rng = test_rng();
        let w: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let mut regular_eq = DensePolynomial::<Fr>::new(EqPolynomial::evals(&w));
        let mut split_eq = GruenSplitEqPolynomial::new(&w, BindingOrder::LowToHigh);
        assert_eq!(regular_eq, split_eq.merge());

        for _ in 0..NUM_VARS {
            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            regular_eq.bound_poly_var_bot(&r);
            split_eq.bind(r);

            let merged = split_eq.merge();
            assert_eq!(regular_eq.Z[..regular_eq.len()], merged.Z[..merged.len()]);
        }
    }

    #[test]
    fn bind_high_low() {
        const NUM_VARS: usize = 10;
        let mut rng = test_rng();
        let w: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let mut regular_eq = DensePolynomial::<Fr>::new(EqPolynomial::evals(&w));
        let mut split_eq_high_to_low = GruenSplitEqPolynomial::new(&w, BindingOrder::HighToLow);

        // Verify they start equal
        assert_eq!(regular_eq, split_eq_high_to_low.merge());

        // Bind with same random values, but regular_eq uses top and split uses new high-to-low
        for _ in 0..NUM_VARS {
            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            regular_eq.bound_poly_var_top(&r);
            split_eq_high_to_low.bind(r);
            let merged = split_eq_high_to_low.merge();

            assert_eq!(regular_eq.Z[..regular_eq.len()], merged.Z[..merged.len()]);
        }
    }

    /// For window_size = 1, `E_out_in_for_window` should factor the eq polynomial
    /// over the head bits `w[..current_index-1]` into a product of two tables.
    #[test]
    fn window_size_one_matches_current() {
        const NUM_VARS: usize = 10;
        let mut rng = test_rng();
        let w: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let mut split_eq: GruenSplitEqPolynomial<Fr> =
            GruenSplitEqPolynomial::new(&w, BindingOrder::LowToHigh);

        for _round in 0..NUM_VARS {
            let num_unbound = split_eq.current_index;
            if num_unbound <= 1 {
                break;
            }

            // Factor head = w[..num_unbound-1] into (E_out, E_in).
            let (e_out_window, e_in_window) = split_eq.E_out_in_for_window(1);
            let w_head = &split_eq.w[..num_unbound - 1];
            let head_evals = EqPolynomial::evals(w_head);

            let num_x_out = e_out_window.len();
            let num_x_in = e_in_window.len();
            assert_eq!(num_x_out * num_x_in, head_evals.len());

            let x_in_bits = num_x_in.log_2();
            for x_out in 0..num_x_out {
                for x_in in 0..num_x_in {
                    let idx = (x_out << x_in_bits) | x_in;
                    assert_eq!(
                        e_out_window[x_out] * e_in_window[x_in],
                        head_evals[idx],
                        "factorisation mismatch at round={_round}, x_out={x_out}, x_in={x_in}",
                    );
                }
            }

            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            split_eq.bind(r);
        }
    }

    /// Check basic bit-accounting invariants for the streaming factorisation:
    ///   log2(|E_out|) + log2(|E_in|) + log2(|E_active|) + 1 = number of unbound variables
    /// for all window sizes and all rounds (LowToHigh).
    #[test]
    fn window_bit_accounting_invariants() {
        const NUM_VARS: usize = 8;
        let mut rng = test_rng();
        let w: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let mut split_eq: GruenSplitEqPolynomial<Fr> =
            GruenSplitEqPolynomial::new(&w, BindingOrder::LowToHigh);

        // Walk through all rounds, checking all window sizes that are
        // meaningful at that point (at least one unbound variable).
        for _round in 0..NUM_VARS {
            let num_unbound = split_eq.len().log_2();
            if num_unbound == 0 {
                break;
            }

            for window_size in 1..=num_unbound {
                let (e_out, e_in) = split_eq.E_out_in_for_window(window_size);
                let e_active = split_eq.E_active_for_window(window_size);
                // By construction, each side represents at least one entry.
                debug_assert!(!e_out.is_empty());
                debug_assert!(!e_in.is_empty());
                debug_assert!(!e_active.is_empty());

                let bits_out = e_out.len().log_2();
                let bits_in = e_in.len().log_2();
                let bits_active = e_active.len().log_2();

                // One bit is reserved for the current variable in the Gruen
                // cubic (the eq polynomial is linear in that bit).
                assert_eq!(
                    bits_out + bits_in + bits_active + 1,
                    num_unbound,
                    "bit accounting failed for window_size={window_size} (bits_out={bits_out}, bits_in={bits_in}, bits_active={bits_active}, num_unbound={num_unbound})",
                );
            }

            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            split_eq.bind(r);
        }
    }

    /// Verify that evals_cached returns [1] at index 0 (eq over 0 vars).
    #[test]
    fn evals_cached_starts_with_one() {
        use crate::poly::eq_poly::EqPolynomial;

        let mut rng = test_rng();
        for num_vars in 1..=10 {
            let w: Vec<<Fr as JoltField>::Challenge> =
                std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                    .take(num_vars)
                    .collect();

            let tables = EqPolynomial::<Fr>::evals_cached(&w);

            // Should have num_vars + 1 tables
            assert_eq!(tables.len(), num_vars + 1);

            // tables[0] = [1] (eq over 0 vars)
            assert_eq!(tables[0].len(), 1);
            assert_eq!(tables[0][0], Fr::one());

            // tables[k] should have 2^k entries
            for k in 0..=num_vars {
                assert_eq!(
                    tables[k].len(),
                    1 << k,
                    "tables[{k}] should have 2^{k} entries"
                );
            }
        }
    }

    /// Verify the [1] invariant is maintained throughout binding.
    #[test]
    fn e_vec_always_has_one_at_index_zero() {
        const NUM_VARS: usize = 10;
        let mut rng = test_rng();
        let w: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(NUM_VARS)
                .collect();

        let mut split_eq = GruenSplitEqPolynomial::<Fr>::new(&w, BindingOrder::LowToHigh);

        // Check invariant at construction
        assert!(!split_eq.E_out_vec.is_empty());
        assert!(!split_eq.E_in_vec.is_empty());
        assert_eq!(split_eq.E_out_vec[0], vec![Fr::one()]);
        assert_eq!(split_eq.E_in_vec[0], vec![Fr::one()]);

        // Check invariant after each bind
        for _ in 0..NUM_VARS {
            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            split_eq.bind(r);

            // E_*_vec should never be empty
            assert!(!split_eq.E_out_vec.is_empty());
            assert!(!split_eq.E_in_vec.is_empty());
            // Index 0 should always be [1]
            assert_eq!(split_eq.E_out_vec[0], vec![Fr::one()]);
            assert_eq!(split_eq.E_in_vec[0], vec![Fr::one()]);
        }
    }
}
