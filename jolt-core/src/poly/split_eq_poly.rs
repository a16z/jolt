//! Implements the Dao-Thaler + Gruen optimization for EQ polynomial evaluations
//! https://eprint.iacr.org/2024/1210.pdf

use allocative::Allocative;
use ark_ff::Zero;
use rayon::prelude::*;

use super::dense_mlpoly::DensePolynomial;
use super::multilinear_polynomial::BindingOrder;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::{
    field::JoltField,
    poly::{eq_poly::EqPolynomial, unipoly::UniPoly},
    utils::math::Math,
};

#[derive(Debug, Clone, PartialEq, Allocative)]
/// A struct holding the equality polynomial evaluations for use in sum-check, when incorporating
/// both the Gruen and Dao-Thaler optimizations.
///
/// For the `i = 0..n`-th round of sum-check, we want the following invariants (low to high):
///
/// - `current_index = n - i` (where `n = w.len()`)
/// - `current_scalar = eq(w[(n - i)..],r[..i])`
/// - `E_out_vec.last().unwrap() = [eq(w[..min(i, n/2)], x) for all x in {0, 1}^{n - min(i, n/2)}]`
/// - If `i < n/2`, then `E_in_vec.last().unwrap() = [eq(w[n/2..(n/2 + i + 1)], x) for all x in {0,
///   1}^{n/2 - i - 1}]`; else `E_in_vec` is empty
///
/// Implements both LowToHigh ordering and HighToLow ordering.
pub struct GruenSplitEqPolynomial<F: JoltField> {
    pub(crate) current_index: usize,
    pub(crate) current_scalar: F,
    pub(crate) w: Vec<F::Challenge>,
    pub(crate) E_in_vec: Vec<Vec<F>>,
    pub(crate) E_out_vec: Vec<Vec<F>>,
    /// Cached `[1]` table used to represent eq over zero variables when a side
    /// (head, inner, or active) has no bits.
    one_table: Vec<F>,
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
                let (E_out_vec, E_in_vec) = rayon::join(
                    || EqPolynomial::evals_cached(w_out),
                    || EqPolynomial::evals_cached(w_in),
                );
                let one_table = vec![F::one()];
                Self {
                    current_index: w.len(),
                    current_scalar: scaling_factor.unwrap_or(F::one()),
                    w: w.to_vec(),
                    E_in_vec,
                    E_out_vec,
                    one_table,
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
                let (E_in_vec, E_out_vec) = rayon::join(
                    || EqPolynomial::evals_cached_rev(w_in),
                    || EqPolynomial::evals_cached_rev(w_out),
                );
                let one_table = vec![F::one()];

                Self {
                    current_index: 0, // Start from 0 for high-to-low up to w.len() - 1
                    current_scalar: scaling_factor.unwrap_or(F::one()),
                    w: w.to_vec(),
                    E_in_vec,
                    E_out_vec,
                    one_table,
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
        self.E_in_vec.last().map_or(0, |v| v.len())
    }

    pub fn E_out_current_len(&self) -> usize {
        self.E_out_vec.last().map_or(0, |v| v.len())
    }

    /// Return the last vector from `E1` as a slice
    pub fn E_in_current(&self) -> &[F] {
        self.E_in_vec.last().map_or(&[], |v| v.as_slice())
    }

    /// Return the last vector from `E2` as a slice
    pub fn E_out_current(&self) -> &[F] {
        self.E_out_vec.last().map_or(&[], |v| v.as_slice())
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
        if window_size == 0 {
            return (&self.one_table, &self.one_table);
        }

        match self.binding_order {
            BindingOrder::LowToHigh => {
                let num_unbound = self.current_index;
                if num_unbound == 0 {
                    return (&self.one_table, &self.one_table);
                }

                // Restrict window size to the actually available unbound bits.
                let window_size = core::cmp::min(window_size, num_unbound);
                let head_len = num_unbound.saturating_sub(window_size);
                if head_len == 0 {
                    // No head bits: represent as eq over zero vars.
                    return (&self.one_table, &self.one_table);
                }

                // The head prefix consists of the earliest `head_len` bits of `w`.
                // These live entirely in the original `[w_out || w_in] = w[..n-1]`
                // region, so we can factor them via prefixes of `w_out` and `w_in`.
                let n = self.w.len();
                let m = n / 2;

                let head_out_bits = core::cmp::min(head_len, m);
                let head_in_bits = head_len.saturating_sub(head_out_bits);

                let e_out = if head_out_bits == 0 {
                    &self.one_table
                } else {
                    debug_assert!(
                        head_out_bits < self.E_out_vec.len(),
                        "head_out_bits={} E_out_vec.len()={}",
                        head_out_bits,
                        self.E_out_vec.len()
                    );
                    &self.E_out_vec[head_out_bits]
                };
                let e_in = if head_in_bits == 0 {
                    &self.one_table
                } else {
                    debug_assert!(
                        head_in_bits < self.E_in_vec.len(),
                        "head_in_bits={} E_in_vec.len()={}",
                        head_in_bits,
                        self.E_in_vec.len()
                    );
                    &self.E_in_vec[head_in_bits]
                };

                (e_out, e_in)
            }
            BindingOrder::HighToLow => {
                // Streaming windows are not defined for HighToLow in the current
                // Spartan code paths; return neutral head tables.
                (&self.one_table, &self.one_table)
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
                vec![F::one()]
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
                if self.w.len() / 2 < self.current_index {
                    self.E_in_vec.pop();
                } else if 0 < self.current_index {
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
                if self.current_index <= self.w.len() / 2 {
                    // We're binding variables from the first half (E_in)
                    self.E_in_vec.pop();
                } else if self.current_index <= self.w.len() {
                    // We're binding variables from the second half (E_out)
                    self.E_out_vec.pop();
                }
            }
        }
    }

    /// Returns an interleaved vector merging the current bit `w` with `E_in_current()`.
    ///
    /// For each entry `low = E_in_current()[x_in]`, produces the pair:
    ///   [ low * (1 - w), low * w ]
    ///
    /// The returned vector has length `2 * E_in_current_len()`, laid out as
    ///   [low0_0, low0_1, low1_0, low1_1, ...] matching index pairs (j, j+1).
    pub fn merged_in_with_current_w(&self) -> Vec<F> {
        let e_in = self.E_in_current();
        let w = self.get_current_w();
        let mut merged: Vec<F> = unsafe_allocate_zero_vec(2 * e_in.len());
        merged
            .par_chunks_exact_mut(2)
            .zip(e_in.par_iter())
            .for_each(|(chunk, &low)| {
                let eval1 = low * w;
                let eval0 = low - eval1;
                chunk[0] = eval0;
                chunk[1] = eval1;
            });
        merged
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
    /// When E_in is fully bound (len == 0 or 1), we invoke `inner_step` exactly once with e_in = 1 at x_in = 0.
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

                if in_len <= 1 {
                    // Fully bound inner (including zero): single logical contribution with e_in = 1
                    let g = self.group_index(x_out, 0);
                    inner_step(&mut inner_acc, g, 0, F::one());
                } else {
                    for x_in in 0..in_len {
                        let g = self.group_index(x_out, x_in);
                        inner_step(&mut inner_acc, g, x_in, e_in[x_in]);
                    }
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
    use ark_std::test_rng;

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
}
