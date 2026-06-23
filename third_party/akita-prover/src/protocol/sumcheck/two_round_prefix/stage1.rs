use super::common::*;
use akita_algebra::eq_poly::EqPolynomial;
use akita_field::parallel::*;
use akita_field::unreduced::HasUnreducedOps;
use akita_field::{FieldCore, FromPrimitiveInt, Zero};
#[cfg(all(test, not(feature = "zk")))]
use akita_sumcheck::UniPoly;
use akita_sumcheck::{reduce_signed_accum, EqFactoredUniPoly};

/// Candidate stage-1 domain `{1, -1, 2, Infinity}`.
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn stage1_prefix_points<E: FieldCore + FromPrimitiveInt>() -> [PrefixPoint<E>; 4] {
    [
        PrefixPoint::Finite(E::one()),
        PrefixPoint::Finite(E::zero() - E::one()),
        PrefixPoint::Finite(E::from_u64(2)),
        PrefixPoint::Infinity,
    ]
}

/// Safe full stage-1 fallback domain `{0, 1, -1, 2, Infinity}`.
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn stage1_full_prefix_points<E: FieldCore + FromPrimitiveInt>() -> [PrefixPoint<E>; 5] {
    [
        PrefixPoint::Finite(E::zero()),
        PrefixPoint::Finite(E::one()),
        PrefixPoint::Finite(E::zero() - E::one()),
        PrefixPoint::Finite(E::from_u64(2)),
        PrefixPoint::Infinity,
    ]
}

/// Internal stage-1 first-two-round bivariate-skip payload.
///
/// This is built and consumed inside the prover to reconstruct ordinary
/// eq-factored sumcheck round messages; it is not serialized in the Akita proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage1BivariateSkipProof<E: FieldCore> {
    pub evals_except_boolean_core: Vec<E>,
}

#[inline]
pub(crate) fn stage1_full_grid_index(x_idx: usize, y_idx: usize) -> usize {
    x_idx * 5 + y_idx
}

#[inline]
pub(crate) fn stage1_is_boolean_corner(x_idx: usize, y_idx: usize) -> bool {
    x_idx < 2 && y_idx < 2
}

#[inline]
pub(crate) fn stage1_quartic_coeffs_from_prefix_values<E: FieldCore + FromPrimitiveInt>(
    values: [E; 5],
) -> [E; 5] {
    let [at_0, at_1, at_neg_1, at_2, at_inf] = values;
    let two_inv = E::from_u64(2)
        .inverse()
        .expect("stage1 prefix interpolation requires 2 to be invertible");
    let three_inv = E::from_u64(3)
        .inverse()
        .expect("stage1 prefix interpolation requires 3 to be invertible");

    let a0 = at_0;
    let a4 = at_inf;
    let rhs_at_1 = at_1 - a0 - a4;
    let rhs_at_neg_1 = at_neg_1 - a0 - a4;
    let a2 = (rhs_at_1 + rhs_at_neg_1) * two_inv;
    let a1_plus_a3 = (rhs_at_1 - rhs_at_neg_1) * two_inv;
    let rhs_at_2 = at_2 - a0 - E::from_u64(16) * a4;
    let a1_plus_4a3 = rhs_at_2 * two_inv - E::from_u64(2) * a2;
    let a3 = (a1_plus_4a3 - a1_plus_a3) * three_inv;
    let a1 = a1_plus_a3 - a3;
    [a0, a1, a2, a3, a4]
}

#[inline]
pub(crate) fn stage1_eval_quartic_from_prefix_values<E: FieldCore + FromPrimitiveInt>(
    values: [E; 5],
    x: E,
) -> E {
    let [a0, a1, a2, a3, a4] = stage1_quartic_coeffs_from_prefix_values(values);
    a0 + x * (a1 + x * (a2 + x * (a3 + x * a4)))
}

#[inline]
pub(crate) fn eval_stage1_biquartic_from_full_grid<E: FieldCore + FromPrimitiveInt>(
    full_grid: [E; 25],
    x: E,
    y: E,
) -> E {
    let x_rows = std::array::from_fn(|x_idx| {
        stage1_eval_quartic_from_prefix_values(
            [
                full_grid[stage1_full_grid_index(x_idx, 0)],
                full_grid[stage1_full_grid_index(x_idx, 1)],
                full_grid[stage1_full_grid_index(x_idx, 2)],
                full_grid[stage1_full_grid_index(x_idx, 3)],
                full_grid[stage1_full_grid_index(x_idx, 4)],
            ],
            y,
        )
    });
    stage1_eval_quartic_from_prefix_values(x_rows, x)
}

/// Whether stage 1 has enough leading y-rounds to use the 2-round prefix path.
#[inline]
pub(crate) fn can_use_stage1_two_round_prefix(ring_bits: usize, b: usize) -> bool {
    ring_bits >= 2 && matches!(b, 4 | 8)
}

/// Build the stage-1 first-two-round bivariate-skip payload from the compact
/// witness columns at the start of stage 1.
///
/// Returns `None` when there are fewer than two leading y-rounds to batch.
#[tracing::instrument(
    skip_all,
    name = "two_round_prefix::build_stage1_bivariate_skip_proof_from_compact"
)]
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn build_stage1_bivariate_skip_proof_from_compact<
    E: FieldCore + FromPrimitiveInt + HasUnreducedOps,
>(
    w_compact: &[i8],
    tau0: &[E],
    b: usize,
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Option<Stage1BivariateSkipProof<E>> {
    let y_len = 1usize << ring_bits;
    assert_eq!(w_compact.len(), live_x_cols * y_len);
    let s_compact = w_compact
        .iter()
        .map(|&w| {
            let w = i32::from(w);
            (w * (w + 1)) as i16
        })
        .collect::<Vec<_>>();
    build_stage1_bivariate_skip_proof_from_s_compact(
        &s_compact,
        tau0,
        b,
        live_x_cols,
        col_bits,
        ring_bits,
    )
}

/// Build the stage-1 first-two-round bivariate-skip payload from the compact
/// `s = w(w+1)` table already materialized by the prover.
#[tracing::instrument(
    skip_all,
    name = "two_round_prefix::build_stage1_bivariate_skip_proof_from_s_compact"
)]
pub(crate) fn build_stage1_bivariate_skip_proof_from_s_compact<
    E: FieldCore + FromPrimitiveInt + HasUnreducedOps,
>(
    s_compact: &[i16],
    tau0: &[E],
    b: usize,
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Option<Stage1BivariateSkipProof<E>> {
    if !can_use_stage1_two_round_prefix(ring_bits, b) {
        return None;
    }

    let y_len = 1usize << ring_bits;
    assert_eq!(s_compact.len(), live_x_cols * y_len);
    assert_eq!(tau0.len(), col_bits + ring_bits);

    let eq_y_suffix = EqPolynomial::evals(&tau0[2..ring_bits])
        .expect("stage-1 two-round prefix dimensions are prevalidated");
    let eq_x = EqPolynomial::evals(&tau0[ring_bits..])
        .expect("stage-1 x-prefix dimensions are prevalidated");
    let y_quads = y_len / 4;
    debug_assert!(eq_y_suffix.len() >= y_quads);
    debug_assert!(eq_x.len() >= live_x_cols);

    let evals_except_boolean_core = match b {
        4 => {
            let (pos, neg) = cfg_fold_reduce!(
                0..live_x_cols,
                || {
                    (
                        [E::MulU64Accum::zero(); STAGE1_B4_PREFIX_EVAL_COUNT],
                        [E::MulU64Accum::zero(); STAGE1_B4_PREFIX_EVAL_COUNT],
                    )
                },
                |(mut pos, mut neg), x_col| {
                    let col = &s_compact[x_col * y_len..(x_col + 1) * y_len];
                    let eq_x_weight = eq_x[x_col];
                    for (y_quad, &eq_y_weight) in eq_y_suffix.iter().take(y_quads).enumerate() {
                        let base = 4 * y_quad;
                        let lookup_idx = stage1_b4_lookup_index_from_digits([
                            stage1_b4_s_digit_from_compact_s(col[base]),
                            stage1_b4_s_digit_from_compact_s(col[base + 1]),
                            stage1_b4_s_digit_from_compact_s(col[base + 2]),
                            stage1_b4_s_digit_from_compact_s(col[base + 3]),
                        ]);
                        let weight = eq_x_weight * eq_y_weight;
                        accum_lookup_vector_signed(
                            &mut pos,
                            &mut neg,
                            weight,
                            &STAGE1_B4_PREFIX_LOOKUP_TABLE[lookup_idx],
                        );
                    }
                    (pos, neg)
                },
                |(mut pos_a, mut neg_a), (pos_b, neg_b)| {
                    for (dst, src) in pos_a.iter_mut().zip(pos_b.iter()) {
                        *dst += *src;
                    }
                    for (dst, src) in neg_a.iter_mut().zip(neg_b.iter()) {
                        *dst += *src;
                    }
                    (pos_a, neg_a)
                }
            );
            (0..STAGE1_B4_PREFIX_EVAL_COUNT)
                .map(|idx| reduce_signed_accum::<E>(pos[idx], neg[idx]))
                .collect()
        }
        8 => {
            let (pos, neg) = cfg_fold_reduce!(
                0..live_x_cols,
                || {
                    (
                        [E::MulU64Accum::zero(); STAGE1_PREFIX_EVAL_COUNT],
                        [E::MulU64Accum::zero(); STAGE1_PREFIX_EVAL_COUNT],
                    )
                },
                |(mut pos, mut neg), x_col| {
                    let col = &s_compact[x_col * y_len..(x_col + 1) * y_len];
                    let eq_x_weight = eq_x[x_col];
                    for (y_quad, &eq_y_weight) in eq_y_suffix.iter().take(y_quads).enumerate() {
                        let base = 4 * y_quad;
                        let lookup_idx = stage1_b8_lookup_index_from_digits([
                            stage1_b8_s_digit_from_compact_s(col[base]),
                            stage1_b8_s_digit_from_compact_s(col[base + 1]),
                            stage1_b8_s_digit_from_compact_s(col[base + 2]),
                            stage1_b8_s_digit_from_compact_s(col[base + 3]),
                        ]);
                        let weight = eq_x_weight * eq_y_weight;
                        accum_lookup_vector_signed(
                            &mut pos,
                            &mut neg,
                            weight,
                            &STAGE1_B8_PREFIX_LOOKUP_TABLE[lookup_idx],
                        );
                    }
                    (pos, neg)
                },
                |(mut pos_a, mut neg_a), (pos_b, neg_b)| {
                    for (dst, src) in pos_a.iter_mut().zip(pos_b.iter()) {
                        *dst += *src;
                    }
                    for (dst, src) in neg_a.iter_mut().zip(neg_b.iter()) {
                        *dst += *src;
                    }
                    (pos_a, neg_a)
                }
            );
            (0..STAGE1_PREFIX_EVAL_COUNT)
                .map(|idx| reduce_signed_accum::<E>(pos[idx], neg[idx]))
                .collect()
        }
        _ => unreachable!("unsupported stage-1 two-round prefix basis"),
    };

    Some(Stage1BivariateSkipProof {
        evals_except_boolean_core,
    })
}

#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn stage1_storage_vector_from_quad<E: FieldCore + FromPrimitiveInt>(
    quad: [E; 4],
    b: usize,
) -> Vec<E> {
    let points = stage1_full_prefix_points::<E>();
    let mut out = Vec::with_capacity(STAGE1_PREFIX_EVAL_COUNT);
    for x_idx in 0..5 {
        for y_idx in 0..5 {
            if stage1_is_boolean_corner(x_idx, y_idx) {
                continue;
            }
            out.push(stage1_local_norm_raw_eval(
                quad,
                points[x_idx],
                points[y_idx],
                b,
            ));
        }
    }
    out
}

/// State needed to reconstruct the first two ordinary stage-1 round messages
/// from the internal bivariate-skip payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage1B4BivariateSkipState<E: FieldCore> {
    x_row_coeffs: [[E; 3]; 3],
    tau0: E,
    tau1: E,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage1B8BivariateSkipState<E: FieldCore> {
    pub(crate) full_grid: [E; 25],
    pub(crate) tau0: E,
    pub(crate) tau1: E,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Stage1BivariateSkipState<E: FieldCore> {
    B4(Stage1B4BivariateSkipState<E>),
    B8(Stage1B8BivariateSkipState<E>),
}

impl<E: FieldCore + FromPrimitiveInt> Stage1BivariateSkipState<E> {
    pub(crate) fn new(proof: &Stage1BivariateSkipProof<E>, tau0: &[E], b: usize) -> Option<Self> {
        if tau0.len() < 2 {
            return None;
        }

        match b {
            4 => {
                if proof.evals_except_boolean_core.len() != STAGE1_B4_PREFIX_EVAL_COUNT {
                    return None;
                }
                let mut full_grid = [E::zero(); 9];
                for (payload_idx, &grid_idx) in STAGE1_B4_NONBOOLEAN_GRID_INDICES.iter().enumerate()
                {
                    full_grid[grid_idx] = proof.evals_except_boolean_core[payload_idx];
                }
                let x_row_coeffs = std::array::from_fn(|y_idx| {
                    quadratic_coeffs_from_01_inf(
                        full_grid[y_idx],
                        full_grid[3 + y_idx],
                        full_grid[6 + y_idx],
                    )
                });
                Some(Self::B4(Stage1B4BivariateSkipState {
                    x_row_coeffs,
                    tau0: tau0[0],
                    tau1: tau0[1],
                }))
            }
            8 => {
                if proof.evals_except_boolean_core.len() != STAGE1_PREFIX_EVAL_COUNT {
                    return None;
                }

                let mut full_grid = [E::zero(); 25];
                let mut payload_idx = 0usize;
                for x_idx in 0..5 {
                    for y_idx in 0..5 {
                        if stage1_is_boolean_corner(x_idx, y_idx) {
                            continue;
                        }
                        full_grid[stage1_full_grid_index(x_idx, y_idx)] =
                            proof.evals_except_boolean_core[payload_idx];
                        payload_idx += 1;
                    }
                }

                Some(Self::B8(Stage1B8BivariateSkipState {
                    full_grid,
                    tau0: tau0[0],
                    tau1: tau0[1],
                }))
            }
            _ => None,
        }
    }

    #[cfg(all(test, not(feature = "zk")))]
    pub(crate) fn reconstruct_round0_poly(&self) -> UniPoly<E> {
        match self {
            Self::B4(state) => state.reconstruct_round0_poly(),
            Self::B8(state) => state.reconstruct_round0_poly(),
        }
    }

    #[cfg(all(test, not(feature = "zk")))]
    pub(crate) fn reconstruct_round1_poly(&self, r0: E) -> UniPoly<E> {
        match self {
            Self::B4(state) => state.reconstruct_round1_poly(r0),
            Self::B8(state) => state.reconstruct_round1_poly(r0),
        }
    }

    pub(crate) fn reconstruct_round0_eq_poly(&self) -> EqFactoredUniPoly<E> {
        match self {
            Self::B4(state) => state.reconstruct_round0_eq_poly(),
            Self::B8(state) => state.reconstruct_round0_eq_poly(),
        }
    }

    pub(crate) fn reconstruct_round1_eq_poly(&self, r0: E) -> EqFactoredUniPoly<E> {
        match self {
            Self::B4(state) => state.reconstruct_round1_eq_poly(r0),
            Self::B8(state) => state.reconstruct_round1_eq_poly(r0),
        }
    }
}

impl<E: FieldCore + FromPrimitiveInt> Stage1B4BivariateSkipState<E> {
    #[cfg(all(test, not(feature = "zk")))]
    fn reconstruct_round0_poly(&self) -> UniPoly<E> {
        let q_x = add_quadratic_coeffs(
            scale_quadratic_coeffs(self.x_row_coeffs[0], E::one() - self.tau1),
            scale_quadratic_coeffs(self.x_row_coeffs[1], self.tau1),
        );
        coeff_array_to_poly(mul_linear_by_quadratic_coeffs(self.tau0, q_x))
    }

    #[cfg(all(test, not(feature = "zk")))]
    fn reconstruct_round1_poly(&self, r0: E) -> UniPoly<E> {
        let y_values: [E; 3] =
            std::array::from_fn(|y_idx| eval_quadratic_from_coeffs(self.x_row_coeffs[y_idx], r0));
        let q_y = quadratic_coeffs_from_01_inf(y_values[0], y_values[1], y_values[2]);
        let round0_eq = linear_eq_eval(self.tau0, r0);
        let coeffs = mul_linear_by_quadratic_coeffs(self.tau1, q_y).map(|coeff| round0_eq * coeff);
        coeff_array_to_poly(coeffs)
    }

    pub(crate) fn reconstruct_round0_eq_poly(&self) -> EqFactoredUniPoly<E> {
        let q_x = add_quadratic_coeffs(
            scale_quadratic_coeffs(self.x_row_coeffs[0], E::one() - self.tau1),
            scale_quadratic_coeffs(self.x_row_coeffs[1], self.tau1),
        );
        EqFactoredUniPoly::from_q_coeffs(q_x.into())
    }

    pub(crate) fn reconstruct_round1_eq_poly(&self, r0: E) -> EqFactoredUniPoly<E> {
        let y_values: [E; 3] =
            std::array::from_fn(|y_idx| eval_quadratic_from_coeffs(self.x_row_coeffs[y_idx], r0));
        let q_y = quadratic_coeffs_from_01_inf(y_values[0], y_values[1], y_values[2]);
        EqFactoredUniPoly::from_q_coeffs(q_y.into())
    }
}

impl<E: FieldCore + FromPrimitiveInt> Stage1B8BivariateSkipState<E> {
    #[cfg(all(test, not(feature = "zk")))]
    fn reconstruct_round0_poly(&self) -> UniPoly<E> {
        let l1_at_0 = E::one() - self.tau1;
        let l1_at_1 = self.tau1;
        let evals: Vec<E> = (0..=5u64)
            .map(|x_raw| {
                let x = E::from_u64(x_raw);
                let q_x0 = eval_stage1_biquartic_from_full_grid(self.full_grid, x, E::zero());
                let q_x1 = eval_stage1_biquartic_from_full_grid(self.full_grid, x, E::one());
                linear_eq_eval(self.tau0, x) * (l1_at_0 * q_x0 + l1_at_1 * q_x1)
            })
            .collect();
        UniPoly::from_evals(&evals)
    }

    #[cfg(all(test, not(feature = "zk")))]
    fn reconstruct_round1_poly(&self, r0: E) -> UniPoly<E> {
        let l0_at_r0 = linear_eq_eval(self.tau0, r0);
        let evals: Vec<E> = (0..=5u64)
            .map(|y_raw| {
                let y = E::from_u64(y_raw);
                l0_at_r0
                    * linear_eq_eval(self.tau1, y)
                    * eval_stage1_biquartic_from_full_grid(self.full_grid, r0, y)
            })
            .collect();
        UniPoly::from_evals(&evals)
    }

    pub(crate) fn reconstruct_round0_eq_poly(&self) -> EqFactoredUniPoly<E> {
        let l1_at_0 = E::one() - self.tau1;
        let l1_at_1 = self.tau1;
        let evals: Vec<E> = (0..=4u64)
            .map(|x_raw| {
                let x = E::from_u64(x_raw);
                let q_x0 = eval_stage1_biquartic_from_full_grid(self.full_grid, x, E::zero());
                let q_x1 = eval_stage1_biquartic_from_full_grid(self.full_grid, x, E::one());
                l1_at_0 * q_x0 + l1_at_1 * q_x1
            })
            .collect();
        interpolate_eq_factored_q_poly(&evals, STAGE1_B8_Q_POLY_DEGREE)
    }

    pub(crate) fn reconstruct_round1_eq_poly(&self, r0: E) -> EqFactoredUniPoly<E> {
        let evals: Vec<E> = (0..=4u64)
            .map(|y_raw| {
                let y = E::from_u64(y_raw);
                eval_stage1_biquartic_from_full_grid(self.full_grid, r0, y)
            })
            .collect();
        interpolate_eq_factored_q_poly(&evals, STAGE1_B8_Q_POLY_DEGREE)
    }
}
