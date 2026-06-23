use super::*;

#[test]
fn stage2_trace_two_round_prefix_matches_direct_path() {
    let col_bits = 5usize;
    let ring_bits = 4usize;
    let live_x_cols = 19usize;
    let b = 8usize;
    let half = (b / 2) as i8;
    let y_len = 1usize << ring_bits;
    let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| ((i * 17 + 5) % b) as i8 - half)
        .collect();
    let trace_compact: Vec<F> = (0..(live_x_cols * y_len))
        .map(|i| F::from_u64((19 * i as u64) + 23))
        .collect();
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((3 * i as u64) + 31))
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((5 * i as u64) + 37))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((7 * i as u64) + 41))
        .collect();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b,
        live_x_cols,
        col_bits,
        ring_bits,
    };

    let mut prover = new_stage2_test_prover_with_trace(
        F::from_u64(43),
        w_prefix.clone(),
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        trace_compact.clone(),
        params,
    );
    assert!(prover.can_use_two_round_prefix());
    let mut direct = new_stage2_test_prover_with_trace(
        F::from_u64(43),
        w_prefix,
        alpha_evals_y,
        m_evals_x,
        trace_compact,
        params,
    );
    direct.prefix_r_stage1 = None;
    assert!(!direct.can_use_two_round_prefix());

    let mut prover_claim = prover.input_claim();
    let mut direct_claim = direct.input_claim();
    assert_eq!(prover_claim, direct_claim);
    for round in 0..(col_bits + ring_bits) {
        let prover_poly = prover.compute_round_univariate(round, prover_claim);
        let direct_poly = direct.compute_round_univariate(round, direct_claim);
        assert_eq!(
            prover_poly, direct_poly,
            "trace two-round prefix mismatch at round {round}"
        );

        let challenge = F::from_u64((11 * round as u64) + 47);
        prover_claim = prover_poly.evaluate(&challenge);
        direct_claim = direct_poly.evaluate(&challenge);
        prover.ingest_challenge(round, challenge);
        direct.ingest_challenge(round, challenge);
    }

    assert_eq!(prover_claim, direct_claim);
    assert_eq!(prover.final_w_eval(), direct.final_w_eval());
}

#[test]
fn stage2_sparse_trace_table_matches_dense_trace_table() {
    let col_bits = 5usize;
    let ring_bits = 4usize;
    let live_x_cols = 19usize;
    let b = 8usize;
    let half = (b / 2) as i8;
    let y_len = 1usize << ring_bits;
    let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| ((i * 41 + 17) % b) as i8 - half)
        .collect();
    let active_cols = [1usize, 4, 11, 17];
    let mut trace_compact = vec![F::zero(); live_x_cols * y_len];
    for &col in &active_cols {
        for y in 0..y_len {
            trace_compact[col * y_len + y] = F::from_u64((43 * (col * y_len + y) as u64) + 109);
        }
    }
    let sparse_trace = TraceTable::field_sparse(
        active_cols
            .iter()
            .map(|&col| TraceSparseColumn {
                col,
                values: trace_compact[col * y_len..(col + 1) * y_len].to_vec(),
            })
            .collect(),
        live_x_cols,
        y_len,
    );
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((47 * i as u64) + 113))
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((53 * i as u64) + 127))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((59 * i as u64) + 131))
        .collect();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b,
        live_x_cols,
        col_bits,
        ring_bits,
    };

    let mut dense = new_stage2_test_prover_with_trace_table(
        F::from_u64(137),
        w_prefix.clone(),
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        TraceTable::ring_dense(trace_compact.clone()),
        &trace_compact,
        params,
    );
    let mut sparse = new_stage2_test_prover_with_trace_table(
        F::from_u64(137),
        w_prefix,
        alpha_evals_y,
        m_evals_x,
        sparse_trace,
        &trace_compact,
        params,
    );

    let mut dense_claim = dense.input_claim();
    let mut sparse_claim = sparse.input_claim();
    assert_eq!(dense_claim, sparse_claim);
    for round in 0..(col_bits + ring_bits) {
        let dense_poly = dense.compute_round_univariate(round, dense_claim);
        let sparse_poly = sparse.compute_round_univariate(round, sparse_claim);
        assert_eq!(
            dense_poly, sparse_poly,
            "sparse trace mismatch at round {round}"
        );

        let challenge = F::from_u64((61 * round as u64) + 139);
        dense_claim = dense_poly.evaluate(&challenge);
        sparse_claim = sparse_poly.evaluate(&challenge);
        dense.ingest_challenge(round, challenge);
        sparse.ingest_challenge(round, challenge);
    }
    assert_eq!(dense_claim, sparse_claim);
    assert_eq!(dense.final_w_eval(), sparse.final_w_eval());
}

#[test]
fn stage2_trace_two_round_prefix_matches_padded_reference() {
    let col_bits = 5usize;
    let ring_bits = 4usize;
    let live_x_cols = 19usize;
    let b = 8usize;
    let half = (b / 2) as i8;
    let y_len = 1usize << ring_bits;
    let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| ((i * 23 + 7) % b) as i8 - half)
        .collect();
    let trace_compact: Vec<F> = (0..(live_x_cols * y_len))
        .map(|i| F::from_u64((29 * i as u64) + 53))
        .collect();
    let w_padded = pad_compact_witness(&w_prefix, live_x_cols, col_bits, ring_bits);
    let trace_padded = pad_trace_compact(&trace_compact, live_x_cols, col_bits, ring_bits);
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((13 * i as u64) + 59))
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((17 * i as u64) + 61))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((19 * i as u64) + 67))
        .collect();

    let mut prefix_prover = new_stage2_test_prover_with_trace(
        F::from_u64(71),
        w_prefix,
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        trace_compact,
        Stage2Params {
            stage1_point: &stage1_point,
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        },
    );
    let mut padded_prover = new_stage2_test_prover_with_trace(
        F::from_u64(71),
        w_padded,
        alpha_evals_y,
        m_evals_x,
        trace_padded,
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
    assert_eq!(prefix_claim, padded_claim);
    for round in 0..(col_bits + ring_bits) {
        let prefix_poly = prefix_prover.compute_round_univariate(round, prefix_claim);
        let padded_poly = padded_prover.compute_round_univariate(round, padded_claim);
        assert_eq!(
            prefix_poly, padded_poly,
            "trace prefix/padded mismatch at round {round}"
        );

        let challenge = F::from_u64((23 * round as u64) + 73);
        prefix_claim = prefix_poly.evaluate(&challenge);
        padded_claim = padded_poly.evaluate(&challenge);
        prefix_prover.ingest_challenge(round, challenge);
        padded_prover.ingest_challenge(round, challenge);
    }

    assert_eq!(prefix_claim, padded_claim);
    assert_eq!(prefix_prover.final_w_eval(), padded_prover.final_w_eval());
}

#[test]
fn stage2_trace_round2_cached_poly_matches_reference() {
    let col_bits = 4usize;
    let ring_bits = 4usize;
    let live_x_cols = 11usize;
    let b = 8usize;
    let half = (b / 2) as i8;
    let y_len = 1usize << ring_bits;
    let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| ((i * 31 + 11) % b) as i8 - half)
        .collect();
    let trace_compact: Vec<F> = (0..(live_x_cols * y_len))
        .map(|i| F::from_u64((37 * i as u64) + 79))
        .collect();
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((29 * i as u64) + 83))
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((31 * i as u64) + 89))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((37 * i as u64) + 97))
        .collect();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b,
        live_x_cols,
        col_bits,
        ring_bits,
    };

    let mut prover = new_stage2_test_prover_with_trace(
        F::from_u64(101),
        w_prefix.clone(),
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        trace_compact.clone(),
        params,
    );
    let round0 = prover.compute_round_univariate(0, prover.input_claim());
    let r0 = F::from_u64(103);
    prover.ingest_challenge(0, r0);
    let round1 = prover.compute_round_univariate(1, round0.evaluate(&r0));
    let r1 = F::from_u64(107);

    let expected_w_full =
        AkitaStage2Prover::<F>::fold_compact_to_round2(&w_prefix, live_x_cols, y_len, r0, r1);
    let expected_alpha_round2 =
        AkitaStage2Prover::<F>::fold_alpha_to_round2(&alpha_evals_y, r0, r1);
    let mut expected_trace = TraceTable::ring_dense(trace_compact.clone());
    expected_trace.fold_y2(live_x_cols, y_len, r0, r1);
    let TraceTable::RingDense(expected_trace_round2) = expected_trace else {
        panic!("expected ring-dense trace table");
    };
    let expected_m_compact = prover.m_compact.clone();

    let mut expected = new_stage2_test_prover_with_trace(
        F::from_u64(101),
        w_prefix,
        alpha_evals_y,
        m_evals_x,
        trace_compact,
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
    expected.trace_table = Some(TraceTable::ring_dense(expected_trace_round2.clone()));
    expected.rounds_completed = 2;
    expected.m_compact = expected_m_compact.clone();
    let expected_round2 = expected.compute_current_round_poly_from_state();

    prover.ingest_challenge(1, r1);

    match &prover.w_table {
        WTable::Full(w_full) => assert_eq!(w_full, &expected_w_full),
        WTable::Compact(_) => {
            panic!("expected fused trace stage2 transition to materialize full table")
        }
    }
    assert_eq!(prover.alpha_compact, expected_alpha_round2);
    assert_eq!(
        prover.trace_table.as_ref(),
        Some(&TraceTable::ring_dense(expected_trace_round2)),
        "two-round handoff must preserve the folded trace table"
    );
    assert_eq!(prover.m_compact, expected_m_compact);
    assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round2));
}
