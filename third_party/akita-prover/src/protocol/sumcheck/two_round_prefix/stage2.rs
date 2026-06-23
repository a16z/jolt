use super::common::*;
use akita_algebra::eq_poly::EqPolynomial;
use akita_field::parallel::*;
use akita_field::unreduced::HasUnreducedOps;
use akita_field::{FieldCore, FromPrimitiveInt, Zero};
use akita_sumcheck::{reduce_signed_accum, UniPoly};
use akita_types::TraceTable;

/// Boolean corner in the `{0, 1}^2` sub-grid of the stage-2 full domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BooleanCorner {
    ZeroZero,
    ZeroOne,
    OneZero,
    OneOne,
}

impl BooleanCorner {
    pub(crate) const ALL: [Self; 4] = [Self::ZeroZero, Self::ZeroOne, Self::OneZero, Self::OneOne];
    #[cfg(all(test, not(feature = "zk")))]
    pub(crate) const DEFAULT_STAGE2_NORM: Self = Self::ZeroZero;
    pub(crate) const DEFAULT_STAGE2_RELATION: Self = Self::ZeroZero;

    #[inline]
    pub(crate) fn default_norm_order() -> [Self; 4] {
        Self::ALL
    }

    #[inline]
    pub(crate) fn boolean_index(self) -> usize {
        match self {
            Self::ZeroZero => 0,
            Self::ZeroOne => 1,
            Self::OneZero => 2,
            Self::OneOne => 3,
        }
    }

    #[inline]
    pub(crate) fn grid_index(self) -> usize {
        match self {
            Self::ZeroZero => 0,
            Self::ZeroOne => 1,
            Self::OneZero => 3,
            Self::OneOne => 4,
        }
    }
}

/// Internal compressed stage-2 `{0, 1, Infinity}^2` grid with one omitted
/// Boolean corner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage2CompressedGrid<E: FieldCore> {
    pub omitted_corner: BooleanCorner,
    pub evals_except_corner: [E; 8],
}

impl<E: FieldCore> Stage2CompressedGrid<E> {
    #[cfg(all(test, not(feature = "zk")))]
    pub(crate) fn from_full_grid(full_grid: [E; 9], omitted_corner: BooleanCorner) -> Self {
        let omitted_idx = omitted_corner.grid_index();
        let mut out_idx = 0usize;
        let evals_except_corner = std::array::from_fn(|_| {
            while out_idx == omitted_idx {
                out_idx += 1;
            }
            let value = full_grid[out_idx];
            out_idx += 1;
            value
        });
        Self {
            omitted_corner,
            evals_except_corner,
        }
    }

    pub(crate) fn reconstruct_with_corner_value(&self, omitted_value: E) -> [E; 9] {
        let omitted_idx = self.omitted_corner.grid_index();
        let mut src_idx = 0usize;
        std::array::from_fn(|dst_idx| {
            if dst_idx == omitted_idx {
                omitted_value
            } else {
                let value = self.evals_except_corner[src_idx];
                src_idx += 1;
                value
            }
        })
    }
}

/// Internal stage-2 first-two-round bivariate-skip payload.
///
/// This payload is built and consumed inside the prover to reconstruct ordinary
/// stage-2 sumcheck round messages; it is not serialized in the Akita proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage2BivariateSkipProof<E: FieldCore> {
    pub norm: Stage2CompressedGrid<E>,
    pub relation: Stage2CompressedGrid<E>,
}

/// Return the stage-2 full-domain grid in row-major `x`-major order over
/// `{0, 1, Infinity}^2`.
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn stage2_full_grid_values<E: FieldCore + FromPrimitiveInt>(
    mut eval: impl FnMut(PrefixPoint<E>, PrefixPoint<E>) -> E,
) -> [E; 9] {
    let points = stage2_full_prefix_points::<E>();
    std::array::from_fn(|idx| {
        let x = points[idx / 3];
        let y = points[idx % 3];
        eval(x, y)
    })
}

/// Evaluate a biquadratic from its full `{0, 1, Infinity}^2` grid.
#[inline]
#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn eval_biquadratic_from_full_grid<E: FieldCore>(
    full_grid: [E; 9],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
) -> E {
    let q_y0 = eval_quadratic_from_01_inf(full_grid[0], full_grid[3], full_grid[6], x);
    let q_y1 = eval_quadratic_from_01_inf(full_grid[1], full_grid[4], full_grid[7], x);
    let q_yinf = eval_quadratic_from_01_inf(full_grid[2], full_grid[5], full_grid[8], x);
    eval_quadratic_from_01_inf(q_y0, q_y1, q_yinf, y)
}

/// Return the local claim weights for the four Boolean corners of the stage-2
/// norm half, ordered as `[(0,0), (0,1), (1,0), (1,1)]`.
#[inline]
pub(crate) fn stage2_norm_corner_weights_from_linear_evals<E: FieldCore>(
    l0_at_0: E,
    l0_at_1: E,
    l1_at_0: E,
    l1_at_1: E,
) -> [E; 4] {
    [
        l0_at_0 * l1_at_0,
        l0_at_0 * l1_at_1,
        l0_at_1 * l1_at_0,
        l0_at_1 * l1_at_1,
    ]
}

/// Return the local claim weights for the four Boolean corners of the stage-2
/// norm half when the two local eq factors are `eq(tau0, X)` and `eq(tau1, Y)`.
#[inline]
pub(crate) fn stage2_norm_corner_weights_from_taus<E: FieldCore>(tau0: E, tau1: E) -> [E; 4] {
    stage2_norm_corner_weights_from_linear_evals(E::one() - tau0, tau0, E::one() - tau1, tau1)
}

/// Choose the default omitted corner for stage-2 norm compression, preferring
/// `(0,0)` when its claim weight is nonzero.
#[inline]
pub(crate) fn default_stage2_norm_omitted_corner<E: FieldCore>(
    corner_weights: [E; 4],
) -> BooleanCorner {
    for corner in BooleanCorner::default_norm_order() {
        if !corner_weights[corner.boolean_index()].is_zero() {
            return corner;
        }
    }
    unreachable!("at least one Boolean-corner weight must be nonzero");
}

/// Recover a full stage-2 grid from an omitted-corner compression and a
/// weighted Boolean-corner claim relation.
pub(crate) fn recover_stage2_grid_from_corner_claim<E: FieldCore>(
    compressed: &Stage2CompressedGrid<E>,
    corner_weights: [E; 4],
    claim: E,
) -> Option<[E; 9]> {
    let omitted_weight = corner_weights[compressed.omitted_corner.boolean_index()];
    let omitted_weight_inv = omitted_weight.inverse()?;
    let mut full_grid = compressed.reconstruct_with_corner_value(E::zero());
    let known_sum = BooleanCorner::ALL
        .iter()
        .copied()
        .filter(|corner| *corner != compressed.omitted_corner)
        .fold(E::zero(), |acc, corner| {
            acc + corner_weights[corner.boolean_index()] * full_grid[corner.grid_index()]
        });
    let omitted_value = (claim - known_sum) * omitted_weight_inv;
    full_grid[compressed.omitted_corner.grid_index()] = omitted_value;
    Some(full_grid)
}

/// Recover a full stage-2 relation grid from its default `(0,0)` omission.
#[inline]
pub(crate) fn recover_stage2_relation_grid_from_claim<E: FieldCore>(
    compressed: &Stage2CompressedGrid<E>,
    relation_claim: E,
) -> [E; 9] {
    recover_stage2_grid_from_corner_claim(compressed, [E::one(); 4], relation_claim)
        .expect("relation corner weights are all one")
}

/// Recover a full stage-2 norm grid from an omitted Boolean corner and the
/// weighted local norm claim.
#[inline]
pub(crate) fn recover_stage2_norm_grid_from_claim<E: FieldCore>(
    compressed: &Stage2CompressedGrid<E>,
    corner_weights: [E; 4],
    norm_claim: E,
) -> Option<[E; 9]> {
    recover_stage2_grid_from_corner_claim(compressed, corner_weights, norm_claim)
}

/// Whether stage 2 has enough y-rounds to use the 2-round prefix path.
#[inline]
pub(crate) fn can_use_stage2_two_round_prefix(ring_bits: usize, b: usize) -> bool {
    ring_bits >= 2 && matches!(b, 4 | 8)
}

/// Build the stage-2 first-two-round bivariate-skip payload from the compact
/// witness table at the start of stage 2.
///
/// Returns `None` when there are fewer than two y-rounds to batch.
#[tracing::instrument(
    skip_all,
    name = "two_round_prefix::build_stage2_bivariate_skip_proof_from_compact"
)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_stage2_bivariate_skip_proof_from_compact<
    E: FieldCore + FromPrimitiveInt + HasUnreducedOps,
>(
    w_compact: &[i8],
    alpha_evals_y: &[E],
    m_evals_x: &[E],
    trace_table: Option<&TraceTable<E>>,
    stage1_point: &[E],
    b: usize,
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Option<Stage2BivariateSkipProof<E>> {
    if !can_use_stage2_two_round_prefix(ring_bits, b) {
        return None;
    }

    let y_len = 1usize << ring_bits;
    assert_eq!(alpha_evals_y.len(), y_len);
    assert_eq!(w_compact.len(), live_x_cols * y_len);
    if let Some(TraceTable::RingDense(trace)) = trace_table {
        assert_eq!(trace.len(), live_x_cols * y_len);
    }
    assert_eq!(m_evals_x.len(), 1usize << col_bits);
    assert_eq!(stage1_point.len(), col_bits + ring_bits);

    let eq_y_suffix = EqPolynomial::evals(&stage1_point[2..ring_bits])
        .expect("stage-2 two-round prefix dimensions are prevalidated");
    let eq_x = EqPolynomial::evals(&stage1_point[ring_bits..])
        .expect("stage-2 x-prefix dimensions are prevalidated");
    let y_quads = y_len >> 2;
    debug_assert_eq!(eq_y_suffix.len(), y_quads);
    let norm_omitted_corner = default_stage2_norm_omitted_corner(
        stage2_norm_corner_weights_from_taus(stage1_point[0], stage1_point[1]),
    );
    let norm_point_indices =
        &STAGE2_COMPRESSED_POINT_INDICES_BY_OMITTED_CORNER[norm_omitted_corner.boolean_index()];
    let alpha_point_values_by_quad: Vec<[E; STAGE2_COMPRESSED_POINT_COUNT]> = (0..y_quads)
        .map(|y_quad| {
            let base = 4 * y_quad;
            let alpha_quad = std::array::from_fn(|offset| alpha_evals_y[base + offset]);
            stage2_relation_m_point_values_compressed(alpha_quad)
        })
        .collect();

    let w_digit_fn: fn(i8) -> usize = match b {
        4 => stage2_b4_w_digit,
        8 => stage2_b8_w_digit,
        _ => unreachable!("unsupported stage-2 two-round prefix basis"),
    };
    let lookup_index_fn: fn([usize; 4]) -> usize = match b {
        4 => stage2_b4_lookup_index_from_digits,
        8 => stage2_b8_lookup_index_from_digits,
        _ => unreachable!(),
    };
    let norm_table: &[[i64; STAGE2_PREFIX_POINT_COUNT]] = match b {
        4 => &STAGE2_B4_NORM_LOOKUP_TABLE,
        8 => &STAGE2_B8_NORM_LOOKUP_TABLE,
        _ => unreachable!(),
    };
    let rel_table: &[[i64; STAGE2_COMPRESSED_POINT_COUNT]] = match b {
        4 => &STAGE2_B4_RELATION_WEIGHT_COMPRESSED_TABLE,
        8 => &STAGE2_B8_RELATION_WEIGHT_COMPRESSED_TABLE,
        _ => unreachable!(),
    };

    let (norm_pos, norm_neg, rel_accum, trace_pos, trace_neg) = cfg_fold_reduce!(
        0..live_x_cols,
        || {
            (
                [E::MulU64Accum::zero(); STAGE2_COMPRESSED_POINT_COUNT],
                [E::MulU64Accum::zero(); STAGE2_COMPRESSED_POINT_COUNT],
                [E::ProductAccum::zero(); STAGE2_COMPRESSED_POINT_COUNT],
                [E::MulU64Accum::zero(); STAGE2_COMPRESSED_POINT_COUNT],
                [E::MulU64Accum::zero(); STAGE2_COMPRESSED_POINT_COUNT],
            )
        },
        |(mut norm_pos, mut norm_neg, mut rel_accum, mut trace_pos, mut trace_neg), x_idx| {
            let column = &w_compact[x_idx * y_len..(x_idx + 1) * y_len];
            let eq_x_weight = eq_x[x_idx];
            let row_val = m_evals_x[x_idx];
            let mut x_rel_pos = [E::MulU64Accum::zero(); STAGE2_COMPRESSED_POINT_COUNT];
            let mut x_rel_neg = [E::MulU64Accum::zero(); STAGE2_COMPRESSED_POINT_COUNT];
            for (y_quad, &eq_y_weight) in eq_y_suffix.iter().enumerate() {
                let base = 4 * y_quad;
                let lookup_idx = lookup_index_fn([
                    w_digit_fn(column[base]),
                    w_digit_fn(column[base + 1]),
                    w_digit_fn(column[base + 2]),
                    w_digit_fn(column[base + 3]),
                ]);
                let norm_weight = eq_y_weight * eq_x_weight;
                accum_lookup_vector_signed_selected(
                    &mut norm_pos,
                    &mut norm_neg,
                    norm_weight,
                    &norm_table[lookup_idx],
                    norm_point_indices,
                );
                accum_pointwise_signed(
                    &mut x_rel_pos,
                    &mut x_rel_neg,
                    &alpha_point_values_by_quad[y_quad],
                    &rel_table[lookup_idx],
                );
                if let Some(trace_table) = trace_table {
                    let trace_quad = trace_table.quad_at(x_idx, base, y_len);
                    let trace_point_values = stage2_relation_m_point_values_compressed(trace_quad);
                    accum_pointwise_signed(
                        &mut trace_pos,
                        &mut trace_neg,
                        &trace_point_values,
                        &rel_table[lookup_idx],
                    );
                }
            }
            for idx in 0..STAGE2_COMPRESSED_POINT_COUNT {
                let x_rel = reduce_signed_accum::<E>(x_rel_pos[idx], x_rel_neg[idx]);
                rel_accum[idx] += row_val.mul_to_product_accum(x_rel);
            }
            (norm_pos, norm_neg, rel_accum, trace_pos, trace_neg)
        },
        |(mut norm_pos_a, mut norm_neg_a, mut rel_accum_a, mut trace_pos_a, mut trace_neg_a),
         (norm_pos_b, norm_neg_b, rel_accum_b, trace_pos_b, trace_neg_b)| {
            for (dst, src) in norm_pos_a.iter_mut().zip(norm_pos_b.iter()) {
                *dst += *src;
            }
            for (dst, src) in norm_neg_a.iter_mut().zip(norm_neg_b.iter()) {
                *dst += *src;
            }
            for (dst, src) in rel_accum_a.iter_mut().zip(rel_accum_b.iter()) {
                *dst += *src;
            }
            for (dst, src) in trace_pos_a.iter_mut().zip(trace_pos_b.iter()) {
                *dst += *src;
            }
            for (dst, src) in trace_neg_a.iter_mut().zip(trace_neg_b.iter()) {
                *dst += *src;
            }
            (
                norm_pos_a,
                norm_neg_a,
                rel_accum_a,
                trace_pos_a,
                trace_neg_a,
            )
        }
    );
    let norm_evals_except_corner: [E; STAGE2_COMPRESSED_POINT_COUNT] =
        std::array::from_fn(|idx| reduce_signed_accum::<E>(norm_pos[idx], norm_neg[idx]));
    let relation_evals_except_corner: [E; STAGE2_COMPRESSED_POINT_COUNT] =
        std::array::from_fn(|idx| {
            E::reduce_product_accum(rel_accum[idx])
                + reduce_signed_accum::<E>(trace_pos[idx], trace_neg[idx])
        });
    Some(Stage2BivariateSkipProof {
        norm: Stage2CompressedGrid {
            omitted_corner: norm_omitted_corner,
            evals_except_corner: norm_evals_except_corner,
        },
        relation: Stage2CompressedGrid {
            omitted_corner: BooleanCorner::DEFAULT_STAGE2_RELATION,
            evals_except_corner: relation_evals_except_corner,
        },
    })
}

/// State needed to reconstruct the first two ordinary stage-2 round messages
/// from the internal bivariate-skip payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage2BivariateSkipState<E: FieldCore> {
    norm_x_row_coeffs: [[E; 3]; 3],
    relation_x_row_coeffs: [[E; 3]; 3],
    tau0: E,
    tau1: E,
    batching_coeff: E,
}

impl<E: FieldCore> Stage2BivariateSkipState<E> {
    pub(crate) fn new(
        proof: &Stage2BivariateSkipProof<E>,
        stage1_point: &[E],
        s_claim: E,
        relation_claim: E,
        batching_coeff: E,
    ) -> Option<Self> {
        if stage1_point.len() < 2 {
            return None;
        }
        let tau0 = stage1_point[0];
        let tau1 = stage1_point[1];
        let norm_full_grid = recover_stage2_norm_grid_from_claim(
            &proof.norm,
            stage2_norm_corner_weights_from_taus(tau0, tau1),
            s_claim,
        )?;
        let relation_full_grid =
            recover_stage2_relation_grid_from_claim(&proof.relation, relation_claim);
        let norm_x_row_coeffs = std::array::from_fn(|y_idx| {
            quadratic_coeffs_from_01_inf(
                norm_full_grid[y_idx],
                norm_full_grid[3 + y_idx],
                norm_full_grid[6 + y_idx],
            )
        });
        let relation_x_row_coeffs = std::array::from_fn(|y_idx| {
            quadratic_coeffs_from_01_inf(
                relation_full_grid[y_idx],
                relation_full_grid[3 + y_idx],
                relation_full_grid[6 + y_idx],
            )
        });
        Some(Self {
            norm_x_row_coeffs,
            relation_x_row_coeffs,
            tau0,
            tau1,
            batching_coeff,
        })
    }
}

impl<E: FieldCore + FromPrimitiveInt> Stage2BivariateSkipState<E> {
    #[inline]
    pub(crate) fn reconstruct_round0_polys(&self) -> (UniPoly<E>, UniPoly<E>) {
        let norm_q = add_quadratic_coeffs(
            scale_quadratic_coeffs(self.norm_x_row_coeffs[0], E::one() - self.tau1),
            scale_quadratic_coeffs(self.norm_x_row_coeffs[1], self.tau1),
        );
        let mut norm_coeffs = mul_linear_by_quadratic_coeffs(self.tau0, norm_q);
        for coeff in &mut norm_coeffs {
            *coeff = self.batching_coeff * *coeff;
        }
        let relation_coeffs =
            add_quadratic_coeffs(self.relation_x_row_coeffs[0], self.relation_x_row_coeffs[1]);
        (
            UniPoly::from_coeffs(norm_coeffs.to_vec()),
            UniPoly::from_coeffs(relation_coeffs.to_vec()),
        )
    }

    #[inline]
    pub(crate) fn reconstruct_round1_polys(&self, r0: E) -> (UniPoly<E>, UniPoly<E>) {
        let norm_y_values: [E; 3] = std::array::from_fn(|y_idx| {
            eval_quadratic_from_coeffs(self.norm_x_row_coeffs[y_idx], r0)
        });
        let norm_q =
            quadratic_coeffs_from_01_inf(norm_y_values[0], norm_y_values[1], norm_y_values[2]);
        let round0_eq = linear_eq_eval(self.tau0, r0);
        let mut norm_coeffs = mul_linear_by_quadratic_coeffs(self.tau1, norm_q);
        for coeff in &mut norm_coeffs {
            *coeff = self.batching_coeff * round0_eq * *coeff;
        }
        let relation_y_values: [E; 3] = std::array::from_fn(|y_idx| {
            eval_quadratic_from_coeffs(self.relation_x_row_coeffs[y_idx], r0)
        });
        let relation_coeffs = quadratic_coeffs_from_01_inf(
            relation_y_values[0],
            relation_y_values[1],
            relation_y_values[2],
        );
        (
            UniPoly::from_coeffs(norm_coeffs.to_vec()),
            UniPoly::from_coeffs(relation_coeffs.to_vec()),
        )
    }
}
