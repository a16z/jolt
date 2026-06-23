use super::common::*;
use super::stage1::*;
use super::stage2::*;
use crate::protocol::sumcheck::akita_stage1::advance_stage1_claim;
use crate::protocol::sumcheck::akita_stage1::AkitaStage1Prover;
use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{FieldCore, Prime128Offset275};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_sumcheck::{EqFactoredSumcheckInstanceProver, EqFactoredUniPoly, UniPoly};
use akita_types::{range_check_eval_from_s, reorder_stage1_coords};
use akita_types::{TraceSparseColumn, TraceTable};
use std::collections::HashMap;

type F = Prime128Offset275;

fn gaussian_rank(mut rows: Vec<Vec<F>>) -> usize {
    rows.retain(|row| row.iter().any(|x| !x.is_zero()));
    if rows.is_empty() {
        return 0;
    }

    let num_cols = rows[0].len();
    let mut rank = 0usize;
    let mut col = 0usize;
    while rank < rows.len() && col < num_cols {
        let Some(pivot_row) = (rank..rows.len()).find(|&r| !rows[r][col].is_zero()) else {
            col += 1;
            continue;
        };
        rows.swap(rank, pivot_row);
        let pivot_inv = rows[rank][col].inverse().expect("pivot must be invertible");
        for entry in &mut rows[rank] {
            *entry *= pivot_inv;
        }
        let pivot_snapshot = rows[rank].clone();
        for (row_idx, row) in rows.iter_mut().enumerate() {
            if row_idx == rank || row[col].is_zero() {
                continue;
            }
            let factor = row[col];
            for (entry, &pivot_entry) in row.iter_mut().zip(pivot_snapshot.iter()) {
                *entry -= factor * pivot_entry;
            }
        }
        rank += 1;
        col += 1;
    }
    rank
}

fn vec_key(vals: &[F]) -> String {
    format!("{vals:?}")
}

fn stage2_norm_round_values(w_quad: [F; 4], tau0: F, tau1: F, r0: F) -> Vec<F> {
    let l0 = |x: F| tau0 * x + (F::one() - tau0) * (F::one() - x);
    let l1 = |y: F| tau1 * y + (F::one() - tau1) * (F::one() - y);
    let q = |x: F, y: F| {
        let w = bilinear_eval(w_quad, x, y);
        w * (w + F::one())
    };

    let mut out = Vec::new();
    for x in 0..=3u64 {
        let x = F::from_u64(x);
        out.push(l0(x) * (l1(F::zero()) * q(x, F::zero()) + l1(F::one()) * q(x, F::one())));
    }
    for y in 0..=3u64 {
        let y = F::from_u64(y);
        out.push(l1(y) * l0(r0) * q(r0, y));
    }
    out
}

fn stage2_relation_round_values(w_quad: [F; 4], m_quad: [F; 4], r0: F) -> Vec<F> {
    let relation = |x: F, y: F| bilinear_eval(w_quad, x, y) * bilinear_eval(m_quad, x, y);
    let mut out = Vec::new();
    for x in 0..=2u64 {
        let x = F::from_u64(x);
        out.push(relation(x, F::zero()) + relation(x, F::one()));
    }
    for y in 0..=2u64 {
        let y = F::from_u64(y);
        out.push(relation(r0, y));
    }
    out
}

fn stage2_norm_claim_from_full_grid(full_grid: [F; 9], corner_weights: [F; 4]) -> F {
    BooleanCorner::ALL
        .iter()
        .copied()
        .fold(F::zero(), |acc, corner| {
            acc + corner_weights[corner.boolean_index()] * full_grid[corner.grid_index()]
        })
}

fn stage2_relation_claim_from_full_grid(full_grid: [F; 9]) -> F {
    stage2_norm_claim_from_full_grid(full_grid, [F::one(); 4])
}

fn stage2_norm_round_values_from_full_grid(full_grid: [F; 9], tau0: F, tau1: F, r0: F) -> Vec<F> {
    let l0_at = |x: PrefixPoint<F>| match x {
        PrefixPoint::Finite(x) => tau0 * x + (F::one() - tau0) * (F::one() - x),
        PrefixPoint::Infinity => tau0,
    };
    let l1_0 = F::one() - tau1;
    let l1_1 = tau1;
    let mut out = Vec::new();
    for x in [F::zero(), F::one(), F::from_u64(2), F::from_u64(3)] {
        let x_point = PrefixPoint::Finite(x);
        let q_x0 =
            eval_biquadratic_from_full_grid(full_grid, x_point, PrefixPoint::Finite(F::zero()));
        let q_x1 =
            eval_biquadratic_from_full_grid(full_grid, x_point, PrefixPoint::Finite(F::one()));
        out.push(l0_at(x_point) * (l1_0 * q_x0 + l1_1 * q_x1));
    }
    for y in [F::zero(), F::one(), F::from_u64(2), F::from_u64(3)] {
        let y_point = PrefixPoint::Finite(y);
        let q_r0_y = eval_biquadratic_from_full_grid(full_grid, PrefixPoint::Finite(r0), y_point);
        let l1_y = tau1 * y + (F::one() - tau1) * (F::one() - y);
        out.push(l1_y * l0_at(PrefixPoint::Finite(r0)) * q_r0_y);
    }
    out
}

fn stage2_relation_round_values_from_full_grid(full_grid: [F; 9], r0: F) -> Vec<F> {
    let mut out = Vec::new();
    for x in [F::zero(), F::one(), F::from_u64(2)] {
        let q_x0 = eval_biquadratic_from_full_grid(
            full_grid,
            PrefixPoint::Finite(x),
            PrefixPoint::Finite(F::zero()),
        );
        let q_x1 = eval_biquadratic_from_full_grid(
            full_grid,
            PrefixPoint::Finite(x),
            PrefixPoint::Finite(F::one()),
        );
        out.push(q_x0 + q_x1);
    }
    for y in [F::zero(), F::one(), F::from_u64(2)] {
        out.push(eval_biquadratic_from_full_grid(
            full_grid,
            PrefixPoint::Finite(r0),
            PrefixPoint::Finite(y),
        ));
    }
    out
}

fn tensor_values<E: FieldCore, const NX: usize, const NY: usize>(
    xs: [PrefixPoint<E>; NX],
    ys: [PrefixPoint<E>; NY],
    mut eval: impl FnMut(PrefixPoint<E>, PrefixPoint<E>) -> E,
) -> Vec<E> {
    let mut out = Vec::with_capacity(NX * NY);
    for &x in &xs {
        for &y in &ys {
            out.push(eval(x, y));
        }
    }
    out
}

fn stage1_norm_round_values(s_quad: [F; 4], tau0: F, tau1: F, r0: F, b: usize) -> Vec<F> {
    let l0 = |x: F| tau0 * x + (F::one() - tau0) * (F::one() - x);
    let l1 = |y: F| tau1 * y + (F::one() - tau1) * (F::one() - y);
    let q = |x: F, y: F| range_check_eval_from_s(bilinear_eval(s_quad, x, y), b);

    let mut out = Vec::new();
    for x in 0..=5u64 {
        let x = F::from_u64(x);
        out.push(l0(x) * (l1(F::zero()) * q(x, F::zero()) + l1(F::one()) * q(x, F::one())));
    }
    for y in 0..=5u64 {
        let y = F::from_u64(y);
        out.push(l0(r0) * l1(y) * q(r0, y));
    }
    out
}

fn build_stage1_bivariate_skip_proof_from_compact_reference(
    w_compact: &[i8],
    tau0: &[F],
    b: usize,
    live_x_cols: usize,
    _col_bits: usize,
    ring_bits: usize,
) -> Option<Stage1BivariateSkipProof<F>> {
    if !can_use_stage1_two_round_prefix(ring_bits, b) {
        return None;
    }

    let y_len = 1usize << ring_bits;
    let eq_y_suffix = EqPolynomial::evals(&tau0[2..ring_bits])
        .expect("stage-1 reference two-round prefix dimensions are prevalidated");
    let eq_x = EqPolynomial::evals(&tau0[ring_bits..])
        .expect("stage-1 reference x-prefix dimensions are prevalidated");
    let points = stage1_full_prefix_points::<F>();
    let y_quads = y_len / 4;
    let mut evals_except_boolean_core = Vec::with_capacity(STAGE1_PREFIX_EVAL_COUNT);

    for x_idx in 0..5 {
        for y_idx in 0..5 {
            if stage1_is_boolean_corner(x_idx, y_idx) {
                continue;
            }
            let mut accum = F::zero();
            let x = points[x_idx];
            let y = points[y_idx];
            for x_col in 0..live_x_cols {
                let col = &w_compact[x_col * y_len..(x_col + 1) * y_len];
                let eq_x_weight = eq_x[x_col];
                for (y_quad, &eq_y_weight) in eq_y_suffix.iter().enumerate().take(y_quads) {
                    let base = 4 * y_quad;
                    let s_quad = std::array::from_fn(|offset| {
                        let w = i64::from(col[base + offset]);
                        F::from_i64(w * (w + 1))
                    });
                    accum +=
                        eq_x_weight * eq_y_weight * stage1_local_norm_raw_eval(s_quad, x, y, b);
                }
            }
            evals_except_boolean_core.push(accum);
        }
    }

    Some(Stage1BivariateSkipProof {
        evals_except_boolean_core,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_stage2_bivariate_skip_proof_from_compact_reference(
    w_compact: &[i8],
    alpha_evals_y: &[F],
    m_evals_x: &[F],
    trace_compact: Option<&[F]>,
    stage1_point: &[F],
    b: usize,
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Option<Stage2BivariateSkipProof<F>> {
    if !can_use_stage2_two_round_prefix(ring_bits, b) {
        return None;
    }

    let y_len = 1usize << ring_bits;
    assert_eq!(m_evals_x.len(), 1usize << col_bits);
    if let Some(trace) = trace_compact {
        assert_eq!(trace.len(), live_x_cols * y_len);
    }
    let eq_y_suffix = EqPolynomial::evals(&stage1_point[2..ring_bits])
        .expect("stage-2 reference two-round prefix dimensions are prevalidated");
    let eq_x = EqPolynomial::evals(&stage1_point[ring_bits..])
        .expect("stage-2 reference x-prefix dimensions are prevalidated");
    let points = stage2_full_prefix_points::<F>();
    let y_quads = y_len >> 2;
    let mut norm_full = [F::zero(); 9];
    let mut relation_full = [F::zero(); 9];

    for x_idx in 0..live_x_cols {
        let column = &w_compact[x_idx * y_len..(x_idx + 1) * y_len];
        let trace_column = trace_compact.map(|trace| &trace[x_idx * y_len..(x_idx + 1) * y_len]);
        let row_val = m_evals_x[x_idx];
        let eq_x_weight = eq_x[x_idx];
        for (y_quad, &eq_y_weight) in eq_y_suffix.iter().enumerate().take(y_quads) {
            let base = 4 * y_quad;
            let w_quad = std::array::from_fn(|offset| F::from_i64(column[base + offset] as i64));
            let alpha_quad = std::array::from_fn(|offset| alpha_evals_y[base + offset]);
            let trace_quad = trace_column
                .map(|trace_column| std::array::from_fn(|offset| trace_column[base + offset]));
            let norm_weight = eq_y_weight * eq_x_weight;
            for idx in 0..9 {
                let x = points[idx / 3];
                let y = points[idx % 3];
                norm_full[idx] += norm_weight * stage2_local_norm_raw_eval(w_quad, x, y);
                relation_full[idx] += stage2_local_relation_eval(w_quad, alpha_quad, row_val, x, y);
                if let Some(trace_quad) = trace_quad {
                    relation_full[idx] +=
                        stage2_local_relation_eval(w_quad, trace_quad, F::one(), x, y);
                }
            }
        }
    }

    let norm_omitted_corner = default_stage2_norm_omitted_corner(
        stage2_norm_corner_weights_from_taus(stage1_point[0], stage1_point[1]),
    );
    Some(Stage2BivariateSkipProof {
        norm: Stage2CompressedGrid::from_full_grid(norm_full, norm_omitted_corner),
        relation: Stage2CompressedGrid::from_full_grid(
            relation_full,
            BooleanCorner::DEFAULT_STAGE2_RELATION,
        ),
    })
}

#[test]
fn stage1_b8_lookup_table_matches_raw_evals() {
    let points = stage1_full_prefix_points::<F>();
    for (d0, &s00) in STAGE1_B8_S_VALUES.iter().enumerate() {
        for (d1, &s10) in STAGE1_B8_S_VALUES.iter().enumerate() {
            for (d2, &s01) in STAGE1_B8_S_VALUES.iter().enumerate() {
                for (d3, &s11) in STAGE1_B8_S_VALUES.iter().enumerate() {
                    let lookup = &STAGE1_B8_PREFIX_LOOKUP_TABLE
                        [stage1_b8_lookup_index_from_digits([d0, d1, d2, d3])];
                    let quad = [
                        F::from_i64(s00),
                        F::from_i64(s10),
                        F::from_i64(s01),
                        F::from_i64(s11),
                    ];
                    let mut point_idx = 0usize;
                    for x_idx in 0..5 {
                        for y_idx in 0..5 {
                            if stage1_is_boolean_corner(x_idx, y_idx) {
                                continue;
                            }
                            assert_eq!(
                                F::from_i64(lookup[point_idx]),
                                stage1_local_norm_raw_eval(quad, points[x_idx], points[y_idx], 8,),
                            );
                            point_idx += 1;
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn stage2_b8_norm_lookup_table_matches_raw_evals() {
    let points = stage2_full_prefix_points::<F>();
    for w00 in -4i64..=3 {
        for w10 in -4i64..=3 {
            for w01 in -4i64..=3 {
                for w11 in -4i64..=3 {
                    let lookup =
                        &STAGE2_B8_NORM_LOOKUP_TABLE[stage2_b8_lookup_index_from_digits([
                            (w00 + 4) as usize,
                            (w10 + 4) as usize,
                            (w01 + 4) as usize,
                            (w11 + 4) as usize,
                        ])];
                    let quad = [
                        F::from_i64(w00),
                        F::from_i64(w10),
                        F::from_i64(w01),
                        F::from_i64(w11),
                    ];
                    for point_idx in 0..STAGE2_PREFIX_POINT_COUNT {
                        let x = points[point_idx / 3];
                        let y = points[point_idx % 3];
                        assert_eq!(
                            F::from_i64(lookup[point_idx]),
                            stage2_local_norm_raw_eval(quad, x, y),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn stage2_b8_relation_weight_table_matches_prefix_w_evals() {
    let points = stage2_full_prefix_points::<F>();
    for w00 in -4i64..=3 {
        for w10 in -4i64..=3 {
            for w01 in -4i64..=3 {
                for w11 in -4i64..=3 {
                    let lookup = &STAGE2_B8_RELATION_WEIGHT_TABLE
                        [stage2_b8_lookup_index_from_digits([
                            (w00 + 4) as usize,
                            (w10 + 4) as usize,
                            (w01 + 4) as usize,
                            (w11 + 4) as usize,
                        ])];
                    let quad = [
                        F::from_i64(w00),
                        F::from_i64(w10),
                        F::from_i64(w01),
                        F::from_i64(w11),
                    ];
                    for point_idx in 0..STAGE2_PREFIX_POINT_COUNT {
                        let x = points[point_idx / 3];
                        let y = points[point_idx % 3];
                        assert_eq!(
                            F::from_i64(lookup[point_idx]),
                            bilinear_eval_on_prefix_points(quad, x, y),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn stage1_bivariate_skip_proof_builder_matches_reference() {
    let col_bits = 3;
    let ring_bits = 2;
    let w_compact: Vec<i8> = (0..(5usize << ring_bits))
        .map(|i| ((3 * i + 1) % 8) as i8 - 4)
        .collect();
    let tau0_raw = vec![
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
    ];
    let tau0 = reorder_stage1_coords(&tau0_raw, col_bits, ring_bits);
    assert_eq!(
        build_stage1_bivariate_skip_proof_from_compact(
            &w_compact, &tau0, 8, 5, col_bits, ring_bits
        ),
        build_stage1_bivariate_skip_proof_from_compact_reference(
            &w_compact, &tau0, 8, 5, col_bits, ring_bits,
        ),
    );
}

#[test]
fn stage2_bivariate_skip_proof_builder_matches_reference() {
    let w_compact = vec![1, -2, 0, 2, 1, -1, 2, 1, 0, 2];
    let alpha_evals_y = [F::from_u64(3), F::from_u64(5)];
    let m_evals_x = [
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
        F::from_u64(17),
        F::from_u64(19),
        F::from_u64(23),
        F::from_u64(29),
        F::from_u64(31),
    ];
    let stage1_point = [
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
    ];
    assert_eq!(
        build_stage2_bivariate_skip_proof_from_compact(
            &w_compact,
            &alpha_evals_y,
            &m_evals_x,
            None,
            &stage1_point,
            8,
            5,
            3,
            1,
        ),
        build_stage2_bivariate_skip_proof_from_compact_reference(
            &w_compact,
            &alpha_evals_y,
            &m_evals_x,
            None,
            &stage1_point,
            8,
            5,
            3,
            1,
        ),
    );
}

#[test]
fn stage2_bivariate_skip_proof_builder_with_trace_matches_reference() {
    let live_x_cols = 5usize;
    let col_bits = 3usize;
    let ring_bits = 2usize;
    let y_len = 1usize << ring_bits;
    let w_compact: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| ((5 * i + 3) % 8) as i8 - 4)
        .collect();
    let trace_compact: Vec<F> = (0..(live_x_cols * y_len))
        .map(|i| F::from_u64((7 * i as u64) + 11))
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((13 * i as u64) + 17))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((19 * i as u64) + 23))
        .collect();
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((29 * i as u64) + 31))
        .collect();

    assert_eq!(
        {
            let trace_table = TraceTable::ring_dense(trace_compact.clone());
            build_stage2_bivariate_skip_proof_from_compact(
                &w_compact,
                &alpha_evals_y,
                &m_evals_x,
                Some(&trace_table),
                &stage1_point,
                8,
                live_x_cols,
                col_bits,
                ring_bits,
            )
        },
        build_stage2_bivariate_skip_proof_from_compact_reference(
            &w_compact,
            &alpha_evals_y,
            &m_evals_x,
            Some(&trace_compact),
            &stage1_point,
            8,
            live_x_cols,
            col_bits,
            ring_bits,
        ),
    );
}

#[test]
fn stage2_bivariate_skip_proof_builder_with_sparse_trace_matches_dense() {
    let live_x_cols = 5usize;
    let col_bits = 3usize;
    let ring_bits = 2usize;
    let y_len = 1usize << ring_bits;
    let w_compact: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| ((7 * i + 5) % 8) as i8 - 4)
        .collect();
    let trace_compact: Vec<F> = (0..(live_x_cols * y_len))
        .map(|i| F::from_u64((11 * i as u64) + 13))
        .collect();
    let sparse_trace = TraceTable::field_sparse(
        (0..live_x_cols)
            .map(|col| TraceSparseColumn {
                col,
                values: trace_compact[col * y_len..(col + 1) * y_len].to_vec(),
            })
            .collect(),
        live_x_cols,
        y_len,
    );
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((17 * i as u64) + 19))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((23 * i as u64) + 29))
        .collect();
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((31 * i as u64) + 37))
        .collect();
    assert_eq!(
        build_stage2_bivariate_skip_proof_from_compact(
            &w_compact,
            &alpha_evals_y,
            &m_evals_x,
            Some(&sparse_trace),
            &stage1_point,
            8,
            live_x_cols,
            col_bits,
            ring_bits,
        ),
        build_stage2_bivariate_skip_proof_from_compact_reference(
            &w_compact,
            &alpha_evals_y,
            &m_evals_x,
            Some(&trace_compact),
            &stage1_point,
            8,
            live_x_cols,
            col_bits,
            ring_bits,
        ),
    );
}

#[test]
fn stage2_bivariate_skip_proof_builder_matches_reference_large_odd_randomized() {
    let live_x_cols = 34_519usize;
    let col_bits = 16usize;
    let ring_bits = 6usize;
    let y_len = 1usize << ring_bits;
    let w_compact: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| ((i * 37 + 11) % 8) as i8 - 4)
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| {
            F::from_u64(
                (i as u64)
                    .wrapping_mul(0x9e37_79b9)
                    .wrapping_add(0x1234_5678),
            )
        })
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| {
            F::from_u64(
                (i as u64)
                    .wrapping_mul(0x85eb_ca6b)
                    .wrapping_add(0xc2b2_ae35),
            )
        })
        .collect();
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| {
            F::from_u64(
                (i as u64)
                    .wrapping_mul(0x27d4_eb2d)
                    .wrapping_add(0x1656_67b1),
            )
        })
        .collect();
    assert_eq!(
        build_stage2_bivariate_skip_proof_from_compact(
            &w_compact,
            &alpha_evals_y,
            &m_evals_x,
            None,
            &stage1_point,
            8,
            live_x_cols,
            col_bits,
            ring_bits,
        ),
        build_stage2_bivariate_skip_proof_from_compact_reference(
            &w_compact,
            &alpha_evals_y,
            &m_evals_x,
            None,
            &stage1_point,
            8,
            live_x_cols,
            col_bits,
            ring_bits,
        ),
    );
}

#[test]
fn stage1_candidate_omits_11_via_zero_check() {
    let points = stage1_prefix_points::<F>();
    let one = points[0];
    let valid_s = [0i64, 2, 6, 12];
    for &s00 in &valid_s {
        for &s10 in &valid_s {
            for &s01 in &valid_s {
                for &s11 in &valid_s {
                    let quad = [
                        F::from_i64(s00),
                        F::from_i64(s10),
                        F::from_i64(s01),
                        F::from_i64(s11),
                    ];
                    assert_eq!(
                        stage1_local_norm_eval(quad, one, one, 8),
                        F::zero(),
                        "stage1 local zero-check should vanish at (1,1)"
                    );
                }
            }
        }
    }
}

#[test]
fn stage1_candidate_storage_family_has_rank_15() {
    let [one, neg_one, two, inf] = stage1_prefix_points::<F>();
    let storage_points = [
        (one, neg_one),
        (one, two),
        (one, inf),
        (neg_one, one),
        (neg_one, neg_one),
        (neg_one, two),
        (neg_one, inf),
        (two, one),
        (two, neg_one),
        (two, two),
        (two, inf),
        (inf, one),
        (inf, neg_one),
        (inf, two),
        (inf, inf),
    ];
    let valid_s = [0i64, 2, 6, 12];
    let mut rows = Vec::new();
    for &s00 in &valid_s {
        for &s10 in &valid_s {
            for &s01 in &valid_s {
                for &s11 in &valid_s {
                    let quad = [
                        F::from_i64(s00),
                        F::from_i64(s10),
                        F::from_i64(s01),
                        F::from_i64(s11),
                    ];
                    rows.push(
                        storage_points
                            .iter()
                            .map(|&(x, y)| stage1_local_norm_eval(quad, x, y, 8))
                            .collect(),
                    );
                }
            }
        }
    }
    assert_eq!(gaussian_rank(rows), 15);
}

#[test]
fn stage1_full_domain_omits_boolean_core_via_zero_check() {
    let points = stage1_full_prefix_points::<F>();
    let valid_s = [0i64, 2, 6, 12];
    for &s00 in &valid_s {
        for &s10 in &valid_s {
            for &s01 in &valid_s {
                for &s11 in &valid_s {
                    let quad = [
                        F::from_i64(s00),
                        F::from_i64(s10),
                        F::from_i64(s01),
                        F::from_i64(s11),
                    ];
                    for &(x_idx, y_idx) in &[(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
                        assert_eq!(
                            stage1_local_norm_raw_eval(quad, points[x_idx], points[y_idx], 8),
                            F::zero(),
                            "stage1 local zero-check should vanish on the Boolean core",
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn stage1_full_storage_family_has_rank_21() {
    let points = stage1_full_prefix_points::<F>();
    let mut storage_points = Vec::new();
    for x_idx in 0..5 {
        for y_idx in 0..5 {
            if stage1_is_boolean_corner(x_idx, y_idx) {
                continue;
            }
            storage_points.push((points[x_idx], points[y_idx]));
        }
    }

    let valid_s = [0i64, 2, 6, 12];
    let mut rows = Vec::new();
    for &s00 in &valid_s {
        for &s10 in &valid_s {
            for &s01 in &valid_s {
                for &s11 in &valid_s {
                    let quad = [
                        F::from_i64(s00),
                        F::from_i64(s10),
                        F::from_i64(s01),
                        F::from_i64(s11),
                    ];
                    rows.push(
                        storage_points
                            .iter()
                            .map(|&(x, y)| stage1_local_norm_raw_eval(quad, x, y, 8))
                            .collect(),
                    );
                }
            }
        }
    }
    assert_eq!(gaussian_rank(rows), 21);
}

#[test]
fn stage1_storage_domain_matches_local_round_messages() {
    let tau0 = F::from_u64(7);
    let tau1 = F::from_u64(11);
    let r0 = F::from_u64(13);
    let valid_s = [0i64, 2, 6, 12];

    for &s00 in &valid_s {
        for &s10 in &valid_s {
            for &s01 in &valid_s {
                for &s11 in &valid_s {
                    let quad = [
                        F::from_i64(s00),
                        F::from_i64(s10),
                        F::from_i64(s01),
                        F::from_i64(s11),
                    ];
                    let proof = Stage1BivariateSkipProof {
                        evals_except_boolean_core: stage1_storage_vector_from_quad(quad, 8),
                    };
                    let skip_state = Stage1BivariateSkipState::new(&proof, &[tau0, tau1], 8)
                        .expect("stage1 bivariate-skip state should build");
                    let round_values = stage1_norm_round_values(quad, tau0, tau1, r0, 8);
                    assert_eq!(
                        skip_state.reconstruct_round0_poly(),
                        UniPoly::from_evals(&round_values[..6])
                    );
                    assert_eq!(
                        skip_state.reconstruct_round1_poly(r0),
                        UniPoly::from_evals(&round_values[6..])
                    );
                }
            }
        }
    }
}

#[test]
fn stage1_bivariate_skip_proof_reconstructs_first_two_rounds() {
    let b = 8;
    let live_x_cols = 5;
    let col_bits = 3;
    let ring_bits = 2;
    let w_compact: Vec<i8> = (0..(live_x_cols << ring_bits))
        .map(|i| ((5 * i + 3) % b) as i8 - (b / 2) as i8)
        .collect();
    let tau0_raw = vec![
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
    ];
    let tau0 = reorder_stage1_coords(&tau0_raw, col_bits, ring_bits);

    let proof = build_stage1_bivariate_skip_proof_from_compact(
        &w_compact,
        &tau0,
        b,
        live_x_cols,
        col_bits,
        ring_bits,
    )
    .expect("stage1 bivariate-skip payload should be available");
    let skip_state = Stage1BivariateSkipState::new(&proof, &tau0, b)
        .expect("stage1 bivariate-skip state should build");

    let mut prover =
        AkitaStage1Prover::<F>::new(&w_compact, &tau0, b, live_x_cols, col_bits, ring_bits)
            .unwrap();
    let round0 = prover.compute_round_eq_factored(0);
    assert_eq!(skip_state.reconstruct_round0_eq_poly(), round0);

    let r0 = F::from_u64(9);
    let _ = advance_stage1_claim(&prover, F::zero(), F::one(), &round0, r0);
    prover.ingest_challenge(0, r0);

    let round1 = prover.compute_round_eq_factored(1);
    assert_eq!(skip_state.reconstruct_round1_eq_poly(r0), round1);
}

#[test]
fn stage1_b8_reconstructed_eq_polys_keep_degree4_storage_width() {
    let state = Stage1B8BivariateSkipState {
        full_grid: [F::zero(); 25],
        tau0: F::from_u64(3),
        tau1: F::from_u64(5),
    };

    for poly in [
        state.reconstruct_round0_eq_poly(),
        state.reconstruct_round1_eq_poly(F::from_u64(7)),
    ] {
        assert_eq!(
            poly.coeffs_except_linear_term.len(),
            EqFactoredUniPoly::<F>::stored_coeff_count_for_degree(STAGE1_B8_Q_POLY_DEGREE)
        );
        assert_eq!(
            poly.coeffs_except_linear_term,
            vec![
                F::zero();
                EqFactoredUniPoly::<F>::stored_coeff_count_for_degree(STAGE1_B8_Q_POLY_DEGREE)
            ]
        );

        let mut bytes = Vec::new();
        poly.serialize_uncompressed(&mut bytes)
            .expect("eq-factored poly should serialize");
        let decoded =
            EqFactoredUniPoly::<F>::deserialize_uncompressed(&bytes[..], &STAGE1_B8_Q_POLY_DEGREE)
                .expect("eq-factored poly should deserialize at degree 4");
        assert_eq!(decoded, poly);
    }
}

#[test]
fn stage2_default_norm_omitted_corner_prefers_00() {
    let weights = stage2_norm_corner_weights_from_taus(F::from_u64(7), F::from_u64(11));
    assert_eq!(
        default_stage2_norm_omitted_corner(weights),
        BooleanCorner::DEFAULT_STAGE2_NORM
    );
}

#[test]
fn stage2_default_norm_omitted_corner_falls_back_when_00_is_zero() {
    let weights = stage2_norm_corner_weights_from_taus(F::one(), F::from_u64(11));
    assert_eq!(
        default_stage2_norm_omitted_corner(weights),
        BooleanCorner::OneZero
    );

    let weights = stage2_norm_corner_weights_from_taus(F::from_u64(7), F::one());
    assert_eq!(
        default_stage2_norm_omitted_corner(weights),
        BooleanCorner::ZeroOne
    );

    let weights = stage2_norm_corner_weights_from_taus(F::one(), F::one());
    assert_eq!(
        default_stage2_norm_omitted_corner(weights),
        BooleanCorner::OneOne
    );
}

#[test]
fn stage2_norm_reduced_domain_has_round_message_collision() {
    let reduced = stage2_reduced_prefix_points::<F>();
    let tau0 = F::from_u64(7);
    let tau1 = F::from_u64(11);
    let r0 = F::from_u64(13);

    let mut seen: HashMap<String, Vec<F>> = HashMap::new();
    let mut found_collision = false;
    for w00 in -4i64..=3 {
        for w10 in -4i64..=3 {
            for w01 in -4i64..=3 {
                for w11 in -4i64..=3 {
                    let quad = [
                        F::from_i64(w00),
                        F::from_i64(w10),
                        F::from_i64(w01),
                        F::from_i64(w11),
                    ];
                    let storage = tensor_values(reduced, reduced, |x, y| {
                        stage2_local_norm_candidate_eval(quad, x, y)
                    });
                    let target = stage2_norm_round_values(quad, tau0, tau1, r0);
                    let key = vec_key(&storage);
                    if let Some(existing) = seen.get(&key) {
                        if *existing != target {
                            found_collision = true;
                            break;
                        }
                    } else {
                        seen.insert(key, target);
                    }
                }
                if found_collision {
                    break;
                }
            }
            if found_collision {
                break;
            }
        }
        if found_collision {
            break;
        }
    }
    assert!(
        found_collision,
        "reduced stage-2 norm domain should not uniquely determine local round messages"
    );
}

#[test]
fn stage2_relation_reduced_domain_has_round_message_collision() {
    let reduced = stage2_reduced_prefix_points::<F>();
    let r0 = F::from_u64(13);
    let alpha = F::one();
    let bit = [F::zero(), F::one()];

    let mut seen: HashMap<String, Vec<F>> = HashMap::new();
    let mut found_collision = false;
    for &w00 in &bit {
        for &w10 in &bit {
            for &w01 in &bit {
                for &w11 in &bit {
                    let w_quad = [w00, w10, w01, w11];
                    for &m00 in &bit {
                        for &m10 in &bit {
                            for &m01 in &bit {
                                for &m11 in &bit {
                                    let m_quad = [m00, m10, m01, m11];
                                    let storage = tensor_values(reduced, reduced, |x, y| {
                                        stage2_local_relation_eval(w_quad, m_quad, alpha, x, y)
                                    });
                                    let target = stage2_relation_round_values(w_quad, m_quad, r0);
                                    let key = vec_key(&storage);
                                    if let Some(existing) = seen.get(&key) {
                                        if *existing != target {
                                            found_collision = true;
                                            break;
                                        }
                                    } else {
                                        seen.insert(key, target);
                                    }
                                }
                                if found_collision {
                                    break;
                                }
                            }
                            if found_collision {
                                break;
                            }
                        }
                        if found_collision {
                            break;
                        }
                    }
                    if found_collision {
                        break;
                    }
                }
                if found_collision {
                    break;
                }
            }
            if found_collision {
                break;
            }
        }
        if found_collision {
            break;
        }
    }
    assert!(
        found_collision,
        "reduced stage-2 relation domain should not uniquely determine local round messages"
    );
}

#[test]
fn stage2_norm_full_domain_matches_local_round_messages() {
    let full = stage2_full_prefix_points::<F>();
    let tau0 = F::from_u64(7);
    let tau1 = F::from_u64(11);
    let r0 = F::from_u64(13);

    let mut seen: HashMap<String, Vec<F>> = HashMap::new();
    for w00 in -4i64..=3 {
        for w10 in -4i64..=3 {
            for w01 in -4i64..=3 {
                for w11 in -4i64..=3 {
                    let quad = [
                        F::from_i64(w00),
                        F::from_i64(w10),
                        F::from_i64(w01),
                        F::from_i64(w11),
                    ];
                    let storage =
                        tensor_values(full, full, |x, y| stage2_local_norm_raw_eval(quad, x, y));
                    let target = stage2_norm_round_values(quad, tau0, tau1, r0);
                    let key = vec_key(&storage);
                    if let Some(existing) = seen.get(&key) {
                        assert_eq!(
                            existing, &target,
                            "full stage-2 norm domain lost information for a compact quad"
                        );
                    } else {
                        seen.insert(key, target);
                    }
                }
            }
        }
    }
}

#[test]
fn stage2_norm_8_point_reconstruction_matches_full_grid_and_round_messages() {
    let tau_choices = [F::zero(), F::one(), F::from_u64(2), F::from_u64(7)];
    let r0 = F::from_u64(13);

    for &tau0 in &tau_choices {
        for &tau1 in &tau_choices {
            let corner_weights = stage2_norm_corner_weights_from_taus(tau0, tau1);
            for w00 in -4i64..=3 {
                for w10 in -4i64..=3 {
                    for w01 in -4i64..=3 {
                        for w11 in -4i64..=3 {
                            let quad = [
                                F::from_i64(w00),
                                F::from_i64(w10),
                                F::from_i64(w01),
                                F::from_i64(w11),
                            ];
                            let full_grid = stage2_full_grid_values(|x, y| {
                                stage2_local_norm_raw_eval(quad, x, y)
                            });
                            let norm_claim =
                                stage2_norm_claim_from_full_grid(full_grid, corner_weights);
                            let omitted_corner = default_stage2_norm_omitted_corner(corner_weights);
                            let compressed =
                                Stage2CompressedGrid::from_full_grid(full_grid, omitted_corner);
                            let recovered = recover_stage2_norm_grid_from_claim(
                                &compressed,
                                corner_weights,
                                norm_claim,
                            )
                            .expect("selected norm corner should be recoverable");

                            assert_eq!(
                                    recovered, full_grid,
                                    "norm full-grid reconstruction mismatch for quad={quad:?}, tau0={tau0:?}, tau1={tau1:?}"
                                );
                            assert_eq!(
                                    stage2_norm_round_values_from_full_grid(recovered, tau0, tau1, r0),
                                    stage2_norm_round_values(quad, tau0, tau1, r0),
                                    "norm round reconstruction mismatch for quad={quad:?}, tau0={tau0:?}, tau1={tau1:?}"
                                );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn stage2_relation_full_domain_matches_local_round_messages() {
    let full = stage2_full_prefix_points::<F>();
    let r0 = F::from_u64(13);
    let alpha = F::one();
    let bit = [F::zero(), F::one()];

    let mut seen: HashMap<String, Vec<F>> = HashMap::new();
    for &w00 in &bit {
        for &w10 in &bit {
            for &w01 in &bit {
                for &w11 in &bit {
                    let w_quad = [w00, w10, w01, w11];
                    for &m00 in &bit {
                        for &m10 in &bit {
                            for &m01 in &bit {
                                for &m11 in &bit {
                                    let m_quad = [m00, m10, m01, m11];
                                    let storage = tensor_values(full, full, |x, y| {
                                        stage2_local_relation_eval(w_quad, m_quad, alpha, x, y)
                                    });
                                    let target = stage2_relation_round_values(w_quad, m_quad, r0);
                                    let key = vec_key(&storage);
                                    if let Some(existing) = seen.get(&key) {
                                        assert_eq!(
                                            existing, &target,
                                            "full stage-2 relation domain lost information"
                                        );
                                    } else {
                                        seen.insert(key, target);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn stage2_relation_8_point_reconstruction_matches_full_grid_and_round_messages() {
    let r0 = F::from_u64(13);
    let alpha_choices = [F::zero(), F::one(), F::from_u64(3)];
    let bit = [F::zero(), F::one()];

    for &alpha in &alpha_choices {
        for &w00 in &bit {
            for &w10 in &bit {
                for &w01 in &bit {
                    for &w11 in &bit {
                        let w_quad = [w00, w10, w01, w11];
                        for &m00 in &bit {
                            for &m10 in &bit {
                                for &m01 in &bit {
                                    for &m11 in &bit {
                                        let m_quad = [m00, m10, m01, m11];
                                        let full_grid = stage2_full_grid_values(|x, y| {
                                            stage2_local_relation_eval(w_quad, m_quad, alpha, x, y)
                                        });
                                        let relation_claim =
                                            stage2_relation_claim_from_full_grid(full_grid);
                                        let compressed = Stage2CompressedGrid::from_full_grid(
                                            full_grid,
                                            BooleanCorner::DEFAULT_STAGE2_RELATION,
                                        );
                                        let recovered = recover_stage2_relation_grid_from_claim(
                                            &compressed,
                                            relation_claim,
                                        );

                                        assert_eq!(
                                            recovered, full_grid,
                                            "relation full-grid reconstruction mismatch"
                                        );
                                        assert_eq!(
                                            stage2_relation_round_values_from_full_grid(
                                                recovered, r0
                                            ),
                                            stage2_relation_round_values(w_quad, m_quad, r0)
                                                .into_iter()
                                                .map(|value| alpha * value)
                                                .collect::<Vec<_>>(),
                                            "relation round reconstruction mismatch"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
