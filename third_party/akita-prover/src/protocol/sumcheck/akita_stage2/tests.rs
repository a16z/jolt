mod trace_prefix;

use super::*;
use crate::protocol::sumcheck::akita_stage1::pad_compact_witness;
use akita_field::Prime128Offset275;
use akita_sumcheck::multilinear_eval;
use akita_types::{TraceSparseColumn, TraceTable};

type F = Prime128Offset275;

#[derive(Clone, Copy)]
pub(super) struct Stage2Params<'a> {
    stage1_point: &'a [F],
    b: usize,
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
}

fn s_claim_from_compact_rows(w_compact: &[i8], params: &Stage2Params<'_>) -> F {
    let padded = if params.live_x_cols == (1usize << params.col_bits) {
        w_compact.to_vec()
    } else {
        pad_compact_witness(
            w_compact,
            params.live_x_cols,
            params.col_bits,
            params.ring_bits,
        )
    };
    let s_evals: Vec<F> = padded
        .iter()
        .map(|&w| {
            let w = F::from_i64(w as i64);
            w * (w + F::one())
        })
        .collect();
    multilinear_eval(&s_evals, params.stage1_point).expect("valid stage-2 witness shape")
}

fn relation_claim_from_compact_rows(
    w_compact: &[i8],
    alpha_evals_y: &[F],
    m_evals_x: &[F],
    params: &Stage2Params<'_>,
) -> F {
    let mut claim = F::zero();
    let y_len = 1usize << params.ring_bits;
    for (x, &m_eval_x) in m_evals_x.iter().enumerate().take(params.live_x_cols) {
        let column = &w_compact[x * y_len..(x + 1) * y_len];
        for (y, &alpha) in alpha_evals_y.iter().enumerate() {
            claim += F::from_i64(column[y] as i64) * alpha * m_eval_x;
        }
    }
    claim
}

fn trace_claim_from_compact_rows(
    w_compact: &[i8],
    trace_compact: &[F],
    params: &Stage2Params<'_>,
) -> F {
    let mut claim = F::zero();
    let y_len = 1usize << params.ring_bits;
    assert_eq!(trace_compact.len(), params.live_x_cols * y_len);
    for x in 0..params.live_x_cols {
        let column = &w_compact[x * y_len..(x + 1) * y_len];
        let trace_column = &trace_compact[x * y_len..(x + 1) * y_len];
        for (w, trace) in column.iter().zip(trace_column.iter()) {
            claim += F::from_i64(*w as i64) * *trace;
        }
    }
    claim
}

fn new_stage2_test_prover(
    batching_coeff: F,
    w_compact: Vec<i8>,
    alpha_evals_y: Vec<F>,
    m_evals_x: Vec<F>,
    params: Stage2Params<'_>,
) -> AkitaStage2Prover<F> {
    let s_claim = s_claim_from_compact_rows(&w_compact, &params);
    let relation_claim =
        relation_claim_from_compact_rows(&w_compact, &alpha_evals_y, &m_evals_x, &params);
    AkitaStage2Prover::new(
        batching_coeff,
        w_compact,
        params.stage1_point,
        s_claim,
        params.b,
        alpha_evals_y,
        m_evals_x,
        params.live_x_cols,
        params.col_bits,
        params.ring_bits,
        relation_claim,
        None,
        F::zero(),
    )
    .unwrap()
}

pub(super) fn new_stage2_test_prover_with_trace(
    batching_coeff: F,
    w_compact: Vec<i8>,
    alpha_evals_y: Vec<F>,
    m_evals_x: Vec<F>,
    trace_compact: Vec<F>,
    params: Stage2Params<'_>,
) -> AkitaStage2Prover<F> {
    let s_claim = s_claim_from_compact_rows(&w_compact, &params);
    let relation_claim =
        relation_claim_from_compact_rows(&w_compact, &alpha_evals_y, &m_evals_x, &params);
    let trace_opening_claim = trace_claim_from_compact_rows(&w_compact, &trace_compact, &params);
    AkitaStage2Prover::new(
        batching_coeff,
        w_compact,
        params.stage1_point,
        s_claim,
        params.b,
        alpha_evals_y,
        m_evals_x,
        params.live_x_cols,
        params.col_bits,
        params.ring_bits,
        relation_claim,
        Some(TraceTable::ring_dense(trace_compact)),
        trace_opening_claim,
    )
    .unwrap()
}

pub(super) fn new_stage2_test_prover_with_trace_table(
    batching_coeff: F,
    w_compact: Vec<i8>,
    alpha_evals_y: Vec<F>,
    m_evals_x: Vec<F>,
    trace_table: TraceTable<F>,
    trace_claim_table: &[F],
    params: Stage2Params<'_>,
) -> AkitaStage2Prover<F> {
    let s_claim = s_claim_from_compact_rows(&w_compact, &params);
    let relation_claim =
        relation_claim_from_compact_rows(&w_compact, &alpha_evals_y, &m_evals_x, &params);
    let trace_opening_claim = trace_claim_from_compact_rows(&w_compact, trace_claim_table, &params);
    AkitaStage2Prover::new(
        batching_coeff,
        w_compact,
        params.stage1_point,
        s_claim,
        params.b,
        alpha_evals_y,
        m_evals_x,
        params.live_x_cols,
        params.col_bits,
        params.ring_bits,
        relation_claim,
        Some(trace_table),
        trace_opening_claim,
    )
    .unwrap()
}

pub(super) fn pad_trace_compact(
    trace_compact: &[F],
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Vec<F> {
    let y_len = 1usize << ring_bits;
    let x_len = 1usize << col_bits;
    assert_eq!(trace_compact.len(), live_x_cols * y_len);
    let mut padded = vec![F::zero(); x_len * y_len];
    for x in 0..live_x_cols {
        let src = x * y_len;
        let dst = x * y_len;
        padded[dst..dst + y_len].copy_from_slice(&trace_compact[src..src + y_len]);
    }
    padded
}

fn relation_round_reference(
    w_compact: &[i8],
    alpha_compact: &[F],
    m_compact: &[F],
    ring_bits: usize,
) -> UniPoly<F> {
    let half = w_compact.len() / 2;
    let current_y_mask = (1usize << ring_bits).wrapping_sub(1);
    let mut evals = [F::zero(); 3];
    for j in 0..half {
        let w_0 = F::from_i64(w_compact[2 * j] as i64);
        let w_1 = F::from_i64(w_compact[2 * j + 1] as i64);
        let a_0 = alpha_compact[(2 * j) & current_y_mask];
        let a_1 = alpha_compact[(2 * j + 1) & current_y_mask];
        let m_0 = m_compact[(2 * j) >> ring_bits];
        let m_1 = m_compact[(2 * j + 1) >> ring_bits];
        evals[0] += w_0 * a_0 * m_0;
        evals[1] += w_1 * a_1 * m_1;
        let w_2 = w_1 + w_1 - w_0;
        let a_2 = a_1 + a_1 - a_0;
        let m_2 = m_1 + m_1 - m_0;
        evals[2] += w_2 * a_2 * m_2;
    }
    UniPoly::from_evals(&evals)
}

fn virtual_round_reference(split_eq: &GruenSplitEq<F>, w_compact: &[i8]) -> UniPoly<F> {
    let half = w_compact.len() / 2;
    let (e_first, e_second) = split_eq.remaining_eq_tables();
    let num_first = e_first.len();
    let first_bits = num_first.trailing_zeros();
    let mut evals = [F::zero(); 3];
    for j in 0..half {
        let j_low = j & (num_first - 1);
        let j_high = j >> first_bits;
        let eq_rem = e_first[j_low] * e_second[j_high];
        let w_0 = F::from_i64(w_compact[2 * j] as i64);
        let w_1 = F::from_i64(w_compact[2 * j + 1] as i64);
        let w_2 = w_1 + w_1 - w_0;
        evals[0] += eq_rem * w_0 * (w_0 + F::one());
        evals[1] += eq_rem * w_1 * (w_1 + F::one());
        evals[2] += eq_rem * w_2 * (w_2 + F::one());
    }
    split_eq.gruen_mul(&UniPoly::from_evals(&evals))
}

fn fold_compact_prefix_x_reference(
    w_compact: &[i8],
    live_x_cols: usize,
    y_len: usize,
    r: F,
) -> Vec<F> {
    let next_live_x_cols = live_x_cols.div_ceil(2);
    let mut out = vec![F::zero(); y_len * next_live_x_cols];
    for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
        let row_start = y * live_x_cols;
        let row = &w_compact[row_start..row_start + live_x_cols];
        for (pair_x, dst) in row_out.iter_mut().enumerate() {
            let left = 2 * pair_x;
            let w_0 = F::from_i64(row[left] as i64);
            let w_1 = if left + 1 < live_x_cols {
                F::from_i64(row[left + 1] as i64)
            } else {
                F::zero()
            };
            *dst = w_0 + r * (w_1 - w_0);
        }
    }
    out
}

fn fold_compact_to_full_reference(w_compact: &[i8], r: F) -> Vec<F> {
    (0..w_compact.len() / 2)
        .map(|j| {
            let w_0 = F::from_i64(w_compact[2 * j] as i64);
            let w_1 = F::from_i64(w_compact[2 * j + 1] as i64);
            w_0 + r * (w_1 - w_0)
        })
        .collect()
}

#[test]
fn stage2_compact_fold_lookup_matches_direct_formula() {
    let r = F::from_u64(53);

    let w_prefix = vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1];
    let fold_lut = AkitaStage2Prover::<F>::build_compact_w_fold_lut(&w_prefix, r);
    assert_eq!(
        AkitaStage2Prover::<F>::fold_compact_prefix_x(&w_prefix, 5, 2, &fold_lut),
        fold_compact_prefix_x_reference(&w_prefix, 5, 2, r)
    );

    let w_dense = vec![1, 2, 3, 1, 2, 3];
    let dense_lut = AkitaStage2Prover::<F>::build_compact_w_fold_lut(&w_dense, r);
    assert_eq!(
        AkitaStage2Prover::<F>::fold_compact_to_full(&w_dense, &dense_lut),
        fold_compact_to_full_reference(&w_dense, r)
    );
}

#[test]
fn stage2_compact_round0_matches_unfused_reference() {
    let col_bits = 3usize;
    let ring_bits = 2usize;
    let n = 1usize << (col_bits + ring_bits);
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((i as u64) + 2))
        .collect();
    let alpha_evals_y: Vec<F> = (0..(1usize << ring_bits))
        .map(|i| F::from_u64((3 * i as u64) + 5))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((7 * i as u64) + 11))
        .collect();

    for b in [4usize, 8, 16, 32] {
        let half = (b / 2) as i8;
        let w_compact: Vec<i8> = (0..n).map(|i| ((i * 5 + 3) % b) as i8 - half).collect();
        let prover = new_stage2_test_prover(
            F::from_u64(13),
            w_compact.clone(),
            alpha_evals_y.clone(),
            m_evals_x.clone(),
            Stage2Params {
                stage1_point: &stage1_point,
                b,
                live_x_cols: 1usize << col_bits,
                col_bits,
                ring_bits,
            },
        );
        let (virt_poly, relation_poly) = prover.compute_round_compact_dense_polys(&w_compact);
        let virt_ref = virtual_round_reference(&prover.split_eq, &w_compact);
        let relation_ref =
            relation_round_reference(&w_compact, &alpha_evals_y, &m_evals_x, ring_bits);

        assert_eq!(
            virt_poly, virt_ref,
            "compact virtual round mismatch for b={b}"
        );
        assert_eq!(
            relation_poly, relation_ref,
            "compact relation round mismatch for b={b}"
        );
    }
}

#[test]
fn stage2_prefix_aware_rounds_match_explicit_full_m_table() {
    let ring_bits = 2usize;
    for b in [4usize, 8, 16, 32] {
        let half = (b / 2) as i8;
        for live_x_cols in [5usize, 6usize] {
            let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
            let x_len = 1usize << col_bits;
            let y_len = 1usize << ring_bits;
            let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
                .map(|i| ((i * 7 + 5) % b) as i8 - half)
                .collect();
            let w_padded = pad_compact_witness(&w_prefix, live_x_cols, col_bits, ring_bits);
            let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
                .map(|i| F::from_u64((i as u64) + 31))
                .collect();
            let alpha_evals_y: Vec<F> = (0..y_len)
                .map(|i| F::from_u64((5 * i as u64) + 7))
                .collect();
            let m_evals_x: Vec<F> = (0..x_len)
                .map(|i| F::from_u64((11 * i as u64) + 13))
                .collect();

            let mut prefix_prover = new_stage2_test_prover(
                F::from_u64(17),
                w_prefix.clone(),
                alpha_evals_y.clone(),
                m_evals_x.clone(),
                Stage2Params {
                    stage1_point: &stage1_point,
                    b,
                    live_x_cols,
                    col_bits,
                    ring_bits,
                },
            );
            let mut padded_prover = new_stage2_test_prover(
                F::from_u64(17),
                w_padded.clone(),
                alpha_evals_y.clone(),
                m_evals_x.clone(),
                Stage2Params {
                    stage1_point: &stage1_point,
                    b,
                    live_x_cols: 1usize << col_bits,
                    col_bits,
                    ring_bits,
                },
            );
            let mut prefix_claim = prefix_prover.input_claim();
            let mut padded_claim = padded_prover.input_claim();

            for round in 0..(col_bits + ring_bits) {
                let prefix_poly = prefix_prover.compute_round_univariate(round, prefix_claim);
                let padded_poly = padded_prover.compute_round_univariate(round, padded_claim);
                assert_eq!(
                    prefix_poly, padded_poly,
                    "round {round} polynomial mismatch live_x_cols={live_x_cols} b={b}"
                );

                let challenge = F::from_u64((round as u64) + 37);
                prefix_claim = prefix_poly.evaluate(&challenge);
                padded_claim = padded_poly.evaluate(&challenge);
                prefix_prover.ingest_challenge(round, challenge);
                padded_prover.ingest_challenge(round, challenge);
            }

            assert_eq!(prefix_prover.final_w_eval(), padded_prover.final_w_eval());
            assert_eq!(prefix_claim, padded_claim);
        }
    }
}

#[test]
fn stage2_zero_gated_round0_matches_reference() {
    let col_bits = 3usize;
    let ring_bits = 1usize;
    let w_compact = vec![-1, 0, -1, 0, 0, -1, 0, -1, -1, 0, -1, 0, 0, -1, 0, -1];
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((i as u64) + 41))
        .collect();
    let alpha_evals_y: Vec<F> = (0..(1usize << ring_bits))
        .map(|i| F::from_u64((3 * i as u64) + 43))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((5 * i as u64) + 47))
        .collect();

    let prover = new_stage2_test_prover(
        F::from_u64(19),
        w_compact.clone(),
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        Stage2Params {
            stage1_point: &stage1_point,
            b: 8,
            live_x_cols: 1usize << col_bits,
            col_bits,
            ring_bits,
        },
    );
    let (virt_poly, relation_poly) = prover.compute_round_compact_dense_polys(&w_compact);
    assert_eq!(
        virt_poly,
        virtual_round_reference(&prover.split_eq, &w_compact)
    );
    assert_eq!(
        relation_poly,
        relation_round_reference(&w_compact, &alpha_evals_y, &m_evals_x, ring_bits)
    );
}

#[test]
fn stage2_fused_round2_transition_matches_two_pass_reference() {
    let col_bits = 3usize;
    let ring_bits = 2usize;
    let live_x_cols = 6usize;
    let b = 8usize;
    let half = (b / 2) as i8;
    let y_len = 1usize << ring_bits;
    let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| ((i * 11 + 7) % b) as i8 - half)
        .collect();
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((i as u64) + 71))
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((5 * i as u64) + 73))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((13 * i as u64) + 79))
        .collect();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b,
        live_x_cols,
        col_bits,
        ring_bits,
    };

    let mut prover = new_stage2_test_prover(
        F::from_u64(83),
        w_prefix.clone(),
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        params,
    );
    let round0 = prover.compute_round_univariate(0, prover.input_claim());
    let r0 = F::from_u64(89);
    prover.ingest_challenge(0, r0);
    let round1 = prover.compute_round_univariate(1, round0.evaluate(&r0));
    let r1 = F::from_u64(97);

    let expected_w_full =
        AkitaStage2Prover::<F>::fold_compact_to_round2(&w_prefix, live_x_cols, y_len, r0, r1);
    let expected_alpha_round2 =
        AkitaStage2Prover::<F>::fold_alpha_to_round2(&alpha_evals_y, r0, r1);
    let expected_m_compact = prover.m_compact.clone();

    let mut expected = new_stage2_test_prover(
        F::from_u64(83),
        w_prefix.clone(),
        alpha_evals_y,
        m_evals_x,
        params,
    );
    let expected_round0 = expected.compute_round_univariate(0, expected.input_claim());
    assert_eq!(expected_round0, round0);
    expected.ingest_challenge(0, r0);
    let expected_round1 = expected.compute_round_univariate(1, expected_round0.evaluate(&r0));
    assert_eq!(expected_round1, round1);
    expected.prev_norm_claim = expected
        .prev_norm_poly
        .as_ref()
        .expect("round1 norm poly should be cached")
        .evaluate(&r1);
    expected.split_eq.bind(r1);
    expected.w_table = WTable::Full(expected_w_full.clone());
    expected.alpha_compact = expected_alpha_round2.clone();
    expected.rounds_completed = 2;
    expected.m_compact = expected_m_compact.clone();
    let expected_round2 = expected.compute_current_round_poly_from_state();

    prover.ingest_challenge(1, r1);

    match &prover.w_table {
        WTable::Full(w_full) => assert_eq!(w_full, &expected_w_full),
        WTable::Compact(_) => {
            panic!("expected fused stage2 transition to materialize full table")
        }
    }
    assert_eq!(prover.alpha_compact, expected_alpha_round2);
    assert_eq!(prover.m_compact, expected_m_compact);
    assert!(!prover.can_use_two_round_prefix());
    assert!(!prover.using_two_round_prefix());
    assert!(prover.prefix_r_stage1.is_none());
    assert!(prover.two_round_prefix.is_none());
    assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round2));
}

#[test]
fn stage2_fused_round2_y_round_transition_matches_two_pass_reference() {
    let col_bits = 3usize;
    let ring_bits = 4usize;
    let live_x_cols = 6usize;
    let b = 8usize;
    let half = (b / 2) as i8;
    let y_len = 1usize << ring_bits;
    let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| ((i * 13 + 9) % b) as i8 - half)
        .collect();
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((i as u64) + 101))
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((7 * i as u64) + 103))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((17 * i as u64) + 107))
        .collect();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b,
        live_x_cols,
        col_bits,
        ring_bits,
    };

    let mut prover = new_stage2_test_prover(
        F::from_u64(109),
        w_prefix.clone(),
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        params,
    );
    let round0 = prover.compute_round_univariate(0, prover.input_claim());
    let r0 = F::from_u64(113);
    prover.ingest_challenge(0, r0);
    let round1 = prover.compute_round_univariate(1, round0.evaluate(&r0));
    let r1 = F::from_u64(127);

    let expected_w_full =
        AkitaStage2Prover::<F>::fold_compact_to_round2(&w_prefix, live_x_cols, y_len, r0, r1);
    let expected_alpha_round2 =
        AkitaStage2Prover::<F>::fold_alpha_to_round2(&alpha_evals_y, r0, r1);
    let expected_m_compact = prover.m_compact.clone();

    let mut expected =
        new_stage2_test_prover(F::from_u64(109), w_prefix, alpha_evals_y, m_evals_x, params);
    let expected_round0 = expected.compute_round_univariate(0, expected.input_claim());
    assert_eq!(expected_round0, round0);
    expected.ingest_challenge(0, r0);
    let expected_round1 = expected.compute_round_univariate(1, expected_round0.evaluate(&r0));
    assert_eq!(expected_round1, round1);
    expected.prev_norm_claim = expected
        .prev_norm_poly
        .as_ref()
        .expect("round1 norm poly should be cached")
        .evaluate(&r1);
    expected.split_eq.bind(r1);
    expected.w_table = WTable::Full(expected_w_full.clone());
    expected.alpha_compact = expected_alpha_round2.clone();
    expected.rounds_completed = 2;
    expected.m_compact = expected_m_compact.clone();
    let expected_round2 = expected.compute_current_round_poly_from_state();

    prover.ingest_challenge(1, r1);

    match &prover.w_table {
        WTable::Full(w_full) => assert_eq!(w_full, &expected_w_full),
        WTable::Compact(_) => {
            panic!("expected fused stage2 transition to materialize full table")
        }
    }
    assert_eq!(prover.alpha_compact, expected_alpha_round2);
    assert_eq!(prover.m_compact, expected_m_compact);
    assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round2));
}

#[test]
fn stage2_later_full_prefix_fusion_matches_two_pass_reference() {
    let col_bits = 5usize;
    let ring_bits = 2usize;
    let live_x_cols = 12usize;
    let b = 8usize;
    let half = (b / 2) as i8;
    let y_len = 1usize << ring_bits;
    let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| ((i * 9 + 7) % b) as i8 - half)
        .collect();
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((i as u64) + 131))
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((7 * i as u64) + 137))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((11 * i as u64) + 139))
        .collect();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b,
        live_x_cols,
        col_bits,
        ring_bits,
    };

    let mut prover = new_stage2_test_prover(
        F::from_u64(149),
        w_prefix.clone(),
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        params,
    );
    let round0 = prover.compute_round_univariate(0, prover.input_claim());
    let r0 = F::from_u64(151);
    prover.ingest_challenge(0, r0);
    let round1 = prover.compute_round_univariate(1, round0.evaluate(&r0));
    let r1 = F::from_u64(157);
    prover.ingest_challenge(1, r1);
    let round2 = prover.compute_round_univariate(2, round1.evaluate(&r0));
    let r2 = F::from_u64(163);

    let mut expected =
        new_stage2_test_prover(F::from_u64(149), w_prefix, alpha_evals_y, m_evals_x, params);
    let expected_round0 = expected.compute_round_univariate(0, expected.input_claim());
    assert_eq!(expected_round0, round0);
    expected.ingest_challenge(0, r0);
    let expected_round1 = expected.compute_round_univariate(1, expected_round0.evaluate(&r0));
    assert_eq!(expected_round1, round1);
    expected.ingest_challenge(1, r1);
    let expected_round2 = expected.compute_round_univariate(2, expected_round1.evaluate(&r0));
    assert_eq!(expected_round2, round2);

    let current_w_full = match &expected.w_table {
        WTable::Full(w_full) => w_full.clone(),
        WTable::Compact(_) => panic!("expected later prefix state to be full"),
    };
    let current_m_compact = expected.m_compact.clone();
    let current_y_len = expected.alpha_compact.len();
    let expected_next_w_full = AkitaStage2Prover::<F>::fold_full_prefix_x(
        &current_w_full,
        expected.live_x_cols,
        current_y_len,
        r2,
    );
    let expected_next_m_compact = AkitaStage2Prover::<F>::fold_m_prefix(&current_m_compact, r2);
    expected.prev_norm_claim = expected
        .prev_norm_poly
        .as_ref()
        .expect("round2 norm poly should be cached")
        .evaluate(&r2);
    expected.split_eq.bind(r2);
    expected.live_x_cols = expected.live_x_cols.div_ceil(2);
    expected.rounds_completed += 1;
    expected.m_compact = expected_next_m_compact.clone();
    let (virt_terms, rel_coeffs) =
        expected.compute_round_full_prefix_x_terms(&expected_next_w_full);
    let expected_round3 = expected.combine_terms(virt_terms, rel_coeffs);

    prover.ingest_challenge(2, r2);

    match &prover.w_table {
        WTable::Full(w_full) => assert_eq!(w_full, &expected_next_w_full),
        WTable::Compact(_) => panic!("expected fused later prefix stage to stay full"),
    }
    assert_eq!(prover.m_compact, expected_next_m_compact);
    assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round3));
}

#[test]
fn stage2_large_odd_sparse_boolean_two_round_prefix_matches_direct_path() {
    let col_bits = 16usize;
    let ring_bits = 6usize;
    let live_x_cols = 34_519usize;
    let b = 8usize;
    let y_len = 1usize << ring_bits;
    let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| if (i * 73 + 19) % 17 == 0 { -1 } else { 0 })
        .collect();
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((3 * i as u64) + 167))
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((5 * i as u64) + 173))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((7 * i as u64) + 179))
        .collect();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b,
        live_x_cols,
        col_bits,
        ring_bits,
    };

    let mut prover = new_stage2_test_prover(
        F::from_u64(191),
        w_prefix.clone(),
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        params,
    );
    let mut direct =
        new_stage2_test_prover(F::from_u64(191), w_prefix, alpha_evals_y, m_evals_x, params);
    direct.prefix_r_stage1 = None;

    let mut prover_claim = prover.input_claim();
    let mut direct_claim = direct.input_claim();

    for round in 0..(col_bits + ring_bits) {
        let prover_poly = prover.compute_round_univariate(round, prover_claim);
        let direct_poly = direct.compute_round_univariate(round, direct_claim);
        assert_eq!(
            prover_poly, direct_poly,
            "round {round} polynomial mismatch for large odd sparse boolean witness"
        );

        let challenge = F::from_u64((11 * round as u64) + 197);
        prover_claim = prover_poly.evaluate(&challenge);
        direct_claim = direct_poly.evaluate(&challenge);
        prover.ingest_challenge(round, challenge);
        direct.ingest_challenge(round, challenge);
    }

    assert_eq!(prover_claim, direct_claim);
    assert_eq!(prover.final_w_eval(), direct.final_w_eval());
}

#[test]
fn stage2_large_odd_sparse_boolean_prefix_matches_padded_reference() {
    let col_bits = 16usize;
    let ring_bits = 6usize;
    let live_x_cols = 34_519usize;
    let b = 8usize;
    let y_len = 1usize << ring_bits;
    let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| if (i * 73 + 19) % 17 == 0 { -1 } else { 0 })
        .collect();
    let w_padded = pad_compact_witness(&w_prefix, live_x_cols, col_bits, ring_bits);
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((3 * i as u64) + 223))
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((5 * i as u64) + 227))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((7 * i as u64) + 229))
        .collect();

    let mut prefix_prover = new_stage2_test_prover(
        F::from_u64(233),
        w_prefix,
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        Stage2Params {
            stage1_point: &stage1_point,
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        },
    );
    let mut padded_prover = new_stage2_test_prover(
        F::from_u64(233),
        w_padded,
        alpha_evals_y,
        m_evals_x,
        Stage2Params {
            stage1_point: &stage1_point,
            b,
            live_x_cols: 1usize << col_bits,
            col_bits,
            ring_bits,
        },
    );

    let mut prefix_claim = prefix_prover.input_claim();
    let mut padded_claim = padded_prover.input_claim();

    for round in 0..(col_bits + ring_bits) {
        let prefix_poly = prefix_prover.compute_round_univariate(round, prefix_claim);
        let padded_poly = padded_prover.compute_round_univariate(round, padded_claim);
        assert_eq!(
            prefix_poly, padded_poly,
            "round {round} polynomial mismatch for padded large odd sparse boolean witness"
        );

        let challenge = F::from_u64((13 * round as u64) + 239);
        prefix_claim = prefix_poly.evaluate(&challenge);
        padded_claim = padded_poly.evaluate(&challenge);
        prefix_prover.ingest_challenge(round, challenge);
        padded_prover.ingest_challenge(round, challenge);
    }

    assert_eq!(prefix_claim, padded_claim);
    assert_eq!(prefix_prover.final_w_eval(), padded_prover.final_w_eval());
}

#[test]
fn stage2_large_odd_dense_two_round_prefix_matches_direct_path() {
    let col_bits = 16usize;
    let ring_bits = 6usize;
    let live_x_cols = 34_519usize;
    let b = 8usize;
    let half = (b / 2) as i8;
    let y_len = 1usize << ring_bits;
    let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| ((i * 29 + 17) % b) as i8 - half)
        .collect();
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((17 * i as u64) + 241))
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((19 * i as u64) + 251))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((23 * i as u64) + 257))
        .collect();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b,
        live_x_cols,
        col_bits,
        ring_bits,
    };

    let mut prover = new_stage2_test_prover(
        F::from_u64(263),
        w_prefix.clone(),
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        params,
    );
    let mut direct =
        new_stage2_test_prover(F::from_u64(263), w_prefix, alpha_evals_y, m_evals_x, params);
    direct.prefix_r_stage1 = None;

    let mut prover_claim = prover.input_claim();
    let mut direct_claim = direct.input_claim();

    for round in 0..(col_bits + ring_bits) {
        let prover_poly = prover.compute_round_univariate(round, prover_claim);
        let direct_poly = direct.compute_round_univariate(round, direct_claim);
        assert_eq!(
            prover_poly.evaluate(&F::zero()) + prover_poly.evaluate(&F::one()),
            prover_claim,
            "prefix path sumcheck invariant mismatch at round {round}"
        );
        assert_eq!(
            direct_poly.evaluate(&F::zero()) + direct_poly.evaluate(&F::one()),
            direct_claim,
            "direct path sumcheck invariant mismatch at round {round}"
        );
        assert_eq!(
            prover_poly, direct_poly,
            "round {round} polynomial mismatch for large odd dense witness"
        );

        let challenge = F::from_u64((29 * round as u64) + 269);
        prover_claim = prover_poly.evaluate(&challenge);
        direct_claim = direct_poly.evaluate(&challenge);
        prover.ingest_challenge(round, challenge);
        direct.ingest_challenge(round, challenge);
    }

    assert_eq!(prover_claim, direct_claim);
    assert_eq!(prover.final_w_eval(), direct.final_w_eval());
}

#[test]
fn stage2_large_odd_dense_prefix_matches_padded_reference() {
    let col_bits = 16usize;
    let ring_bits = 6usize;
    let live_x_cols = 34_519usize;
    let b = 8usize;
    let half = (b / 2) as i8;
    let y_len = 1usize << ring_bits;
    let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| ((i * 31 + 11) % b) as i8 - half)
        .collect();
    let w_padded = pad_compact_witness(&w_prefix, live_x_cols, col_bits, ring_bits);
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((31 * i as u64) + 271))
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((37 * i as u64) + 277))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((41 * i as u64) + 281))
        .collect();

    let mut prefix_prover = new_stage2_test_prover(
        F::from_u64(283),
        w_prefix,
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        Stage2Params {
            stage1_point: &stage1_point,
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        },
    );
    let mut padded_prover = new_stage2_test_prover(
        F::from_u64(283),
        w_padded,
        alpha_evals_y,
        m_evals_x,
        Stage2Params {
            stage1_point: &stage1_point,
            b,
            live_x_cols: 1usize << col_bits,
            col_bits,
            ring_bits,
        },
    );

    let mut prefix_claim = prefix_prover.input_claim();
    let mut padded_claim = padded_prover.input_claim();

    for round in 0..(col_bits + ring_bits) {
        let prefix_poly = prefix_prover.compute_round_univariate(round, prefix_claim);
        let padded_poly = padded_prover.compute_round_univariate(round, padded_claim);
        assert_eq!(
            prefix_poly, padded_poly,
            "round {round} polynomial mismatch for padded large odd dense witness"
        );

        let challenge = F::from_u64((43 * round as u64) + 293);
        prefix_claim = prefix_poly.evaluate(&challenge);
        padded_claim = padded_poly.evaluate(&challenge);
        prefix_prover.ingest_challenge(round, challenge);
        padded_prover.ingest_challenge(round, challenge);
    }

    assert_eq!(prefix_claim, padded_claim);
    assert_eq!(prefix_prover.final_w_eval(), padded_prover.final_w_eval());
}
