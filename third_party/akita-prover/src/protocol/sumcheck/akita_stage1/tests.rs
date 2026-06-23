use super::*;
use akita_field::Prime128Offset275;
use akita_sumcheck::multilinear_eval;
use akita_types::reorder_stage1_coords;

type F = Prime128Offset275;

#[test]
fn stage1_new_rejects_malformed_shapes_without_panicking() {
    let tau = vec![F::zero(); usize::BITS as usize];
    assert!(AkitaStage1Prover::<F>::new(&[], &tau, 4, 1, 0, usize::BITS as usize).is_err());

    let tau = vec![F::zero(); usize::BITS as usize + 1];
    assert!(AkitaStage1Prover::<F>::new(&[], &tau, 4, 3, 2, usize::BITS as usize - 1).is_err());

    assert!(AkitaStage1Prover::<F>::new(&[], &[], 1, 1, 0, 0).is_err());
}

fn fold_s_compact_prefix_x_reference(
    s_compact: &[i16],
    live_x_cols: usize,
    y_len: usize,
    r: F,
) -> Vec<F> {
    let next_live_x_cols = live_x_cols.div_ceil(2);
    let mut out = vec![F::zero(); y_len * next_live_x_cols];
    for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
        let row_start = y * live_x_cols;
        let row = &s_compact[row_start..row_start + live_x_cols];
        for (pair_x, dst) in row_out.iter_mut().enumerate() {
            let left = 2 * pair_x;
            let s_0 = F::from_i64(i64::from(row[left]));
            let s_1 = if left + 1 < live_x_cols {
                F::from_i64(i64::from(row[left + 1]))
            } else {
                F::zero()
            };
            *dst = s_0 + r * (s_1 - s_0);
        }
    }
    out
}

fn fold_s_compact_to_full_reference(s_compact: &[i16], r: F) -> Vec<F> {
    (0..s_compact.len() / 2)
        .map(|j| {
            let s_0 = F::from_i64(i64::from(s_compact[2 * j]));
            let s_1 = F::from_i64(i64::from(s_compact[2 * j + 1]));
            s_0 + r * (s_1 - s_0)
        })
        .collect()
}

#[test]
fn stage1_compact_fold_lookup_matches_direct_formula() {
    let b = 8usize;
    let r = F::from_u64(41);

    let s_prefix = vec![2, 6, 12, 2, 6, 12, 2, 6, 12, 2];
    let fold_lut = AkitaStage1Prover::<F>::build_compact_s_fold_lut(b, r);
    assert_eq!(
        AkitaStage1Prover::<F>::fold_s_compact_prefix_x(&s_prefix, 5, 2, &fold_lut),
        fold_s_compact_prefix_x_reference(&s_prefix, 5, 2, r)
    );

    let s_dense = vec![2, 6, 12, 2, 6, 12];
    let dense_lut = AkitaStage1Prover::<F>::build_compact_s_fold_lut(b, r);
    assert_eq!(
        AkitaStage1Prover::<F>::fold_s_compact_to_full(&s_dense, &dense_lut),
        fold_s_compact_to_full_reference(&s_dense, r)
    );
}

#[test]
fn stage1_round0_matches_dense_reference() {
    let col_bits = 3usize;
    let ring_bits = 2usize;
    let n = 1usize << (col_bits + ring_bits);
    let tau0: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((i as u64) + 2))
        .collect();
    let tau0 = reorder_stage1_coords(&tau0, col_bits, ring_bits);

    for b in [4usize, 8, 16, 32] {
        let half = (b / 2) as i8;
        let w_compact: Vec<i8> = (0..n).map(|i| ((i * 5 + 3) % b) as i8 - half).collect();

        let mut prover = AkitaStage1Prover::new(
            &w_compact,
            &tau0,
            b,
            1usize << col_bits,
            col_bits,
            ring_bits,
        )
        .unwrap();
        let stage1_poly = prover.compute_round_eq_factored(0);
        let s_compact = build_compact_s_table(&w_compact);
        let reference = compute_norm_round_eq_poly_from_s_compact(
            &prover.split_eq,
            &s_compact,
            &prover.range_precomp,
        );

        assert_eq!(stage1_poly, reference, "stage1 round0 mismatch for b={b}");
    }
}

#[test]
fn stage1_compact_coeff_lut_reaches_b16() {
    for b in [4usize, 8, 16] {
        let precomp = RangeAffineFromSPrecomp::<F>::new(b);
        assert!(
            precomp.compact_coeffs_lut(0, 0).is_some(),
            "expected compact coefficient LUT for b={b}"
        );
    }

    let precomp = RangeAffineFromSPrecomp::<F>::new(32);
    assert!(precomp.compact_coeffs_lut(0, 0).is_none());
}

#[test]
fn stage1_field_coeff_lut_reaches_b32() {
    for b in [4usize, 8, 16] {
        let precomp = RangeAffineFromSPrecomp::<F>::new(b);
        assert!(precomp.field_coeffs_lut(0, 0).is_none());
    }

    let precomp = RangeAffineFromSPrecomp::<F>::new(32);
    assert!(
        precomp.field_coeffs_lut(0, 0).is_some(),
        "expected field coefficient LUT for b=32"
    );
}

#[test]
fn stage1_prefix_aware_rounds_match_explicit_zero_padding() {
    let ring_bits = 2usize;
    for b in [4usize, 8, 16, 32] {
        let half = (b / 2) as i8;
        for live_x_cols in [5usize, 6usize] {
            let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
            let y_len = 1usize << ring_bits;
            let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
                .map(|i| ((i * 7 + 5) % b) as i8 - half)
                .collect();
            let w_padded = pad_compact_witness(&w_prefix, live_x_cols, col_bits, ring_bits);
            let tau0: Vec<F> = (0..(col_bits + ring_bits))
                .map(|i| F::from_u64((i as u64) + 19))
                .collect();
            let tau0 = reorder_stage1_coords(&tau0, col_bits, ring_bits);
            let mut prefix_prover =
                AkitaStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, col_bits, ring_bits)
                    .unwrap();
            let mut padded_prover = AkitaStage1Prover::new(
                &w_padded,
                &tau0,
                b,
                1usize << col_bits,
                col_bits,
                ring_bits,
            )
            .unwrap();
            let mut challenges = Vec::new();
            let mut prefix_claim = F::zero();
            let mut prefix_scale = F::one();
            let mut padded_claim = F::zero();
            let mut padded_scale = F::one();

            for round in 0..(col_bits + ring_bits) {
                let prefix_poly = prefix_prover.compute_round_eq_factored(round);
                let padded_poly = padded_prover.compute_round_eq_factored(round);
                assert_eq!(
                    prefix_poly, padded_poly,
                    "round {round} polynomial mismatch live_x_cols={live_x_cols} b={b}"
                );

                let challenge = F::from_u64((round as u64) + 29);
                challenges.push(challenge);
                (prefix_claim, prefix_scale) = advance_stage1_claim(
                    &prefix_prover,
                    prefix_claim,
                    prefix_scale,
                    &prefix_poly,
                    challenge,
                );
                (padded_claim, padded_scale) = advance_stage1_claim(
                    &padded_prover,
                    padded_claim,
                    padded_scale,
                    &padded_poly,
                    challenge,
                );
                prefix_prover.ingest_challenge(round, challenge);
                padded_prover.ingest_challenge(round, challenge);
            }

            assert_eq!(prefix_prover.final_s_claim(), padded_prover.final_s_claim());
            assert_eq!(prefix_claim, padded_claim);
            assert_eq!(prefix_scale, padded_scale);
            let s_padded: Vec<F> = build_compact_s_table(&w_padded)
                .into_iter()
                .map(|s| F::from_i64(i64::from(s)))
                .collect();
            assert_eq!(
                prefix_prover.final_s_claim(),
                multilinear_eval(&s_padded, &challenges).unwrap(),
                "final s-claim mismatch live_x_cols={live_x_cols} b={b}"
            );
        }
    }
}

#[test]
fn stage1_fused_round2_transition_matches_two_pass_reference() {
    let col_bits = 3usize;
    let ring_bits = 2usize;
    let live_x_cols = 6usize;
    let y_len = 1usize << ring_bits;
    for b in [4usize, 8] {
        let half = (b / 2) as i8;
        let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
            .map(|i| ((i * 9 + 5) % b) as i8 - half)
            .collect();
        let s_compact = build_compact_s_table(&w_prefix);
        let tau0: Vec<F> = (0..(col_bits + ring_bits))
            .map(|i| F::from_u64((i as u64) + 53))
            .collect();
        let tau0 = reorder_stage1_coords(&tau0, col_bits, ring_bits);

        let mut prover =
            AkitaStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, col_bits, ring_bits).unwrap();
        let round0 = prover.compute_round_eq_factored(0);
        let r0 = F::from_u64(61);
        let (claim1, scale1) = advance_stage1_claim(&prover, F::zero(), F::one(), &round0, r0);
        prover.ingest_challenge(0, r0);
        let round1 = prover.compute_round_eq_factored(1);
        let r1 = F::from_u64(67);
        let (_claim2, _scale2) = advance_stage1_claim(&prover, claim1, scale1, &round1, r1);

        let expected_s_full = AkitaStage1Prover::<F>::fold_s_compact_to_round2(
            &s_compact,
            live_x_cols,
            y_len,
            r0,
            r1,
        );
        let mut expected =
            AkitaStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, col_bits, ring_bits).unwrap();
        expected.split_eq.bind(r0);
        expected.split_eq.bind(r1);
        expected.rounds_completed = 2;
        let expected_round2 = expected.compute_round_full_prefix_x(&expected_s_full);

        prover.ingest_challenge(1, r1);

        match &prover.s_table {
            STable::Full(s_full) => assert_eq!(s_full, &expected_s_full),
            STable::Compact(_) => {
                panic!("expected fused stage1 transition to materialize full table")
            }
        }
        assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round2));
    }
}

#[test]
fn stage1_later_full_prefix_fusion_matches_two_pass_reference() {
    let col_bits = 5usize;
    let ring_bits = 2usize;
    let live_x_cols = 12usize;
    let y_len = 1usize << ring_bits;
    for b in [4usize, 8] {
        let half = (b / 2) as i8;
        let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
            .map(|i| ((i * 5 + 11) % b) as i8 - half)
            .collect();
        let tau0: Vec<F> = (0..(col_bits + ring_bits))
            .map(|i| F::from_u64((i as u64) + 101))
            .collect();
        let tau0 = reorder_stage1_coords(&tau0, col_bits, ring_bits);

        let mut prover =
            AkitaStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, col_bits, ring_bits).unwrap();
        let round0 = prover.compute_round_eq_factored(0);
        let r0 = F::from_u64(107);
        let (claim1, scale1) = advance_stage1_claim(&prover, F::zero(), F::one(), &round0, r0);
        prover.ingest_challenge(0, r0);

        let round1 = prover.compute_round_eq_factored(1);
        let r1 = F::from_u64(109);
        let (claim2, scale2) = advance_stage1_claim(&prover, claim1, scale1, &round1, r1);
        prover.ingest_challenge(1, r1);

        let round2 = prover.compute_round_eq_factored(2);
        let r2 = F::from_u64(113);
        let (claim3, _scale3) = advance_stage1_claim(&prover, claim2, scale2, &round2, r2);

        let mut expected =
            AkitaStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, col_bits, ring_bits).unwrap();
        let expected_round0 = expected.compute_round_eq_factored(0);
        assert_eq!(expected_round0, round0);
        expected.ingest_challenge(0, r0);
        let expected_round1 = expected.compute_round_eq_factored(1);
        assert_eq!(expected_round1, round1);
        expected.ingest_challenge(1, r1);
        let expected_round2 = expected.compute_round_eq_factored(2);
        assert_eq!(expected_round2, round2);

        let current_s_full = match &expected.s_table {
            STable::Full(s_full) => s_full.clone(),
            STable::Compact(_) => panic!("expected later prefix state to be full"),
        };
        let current_y_len = current_s_full.len() / expected.live_x_cols;
        let expected_next_s_full = AkitaStage1Prover::<F>::fold_s_full_prefix_x(
            &current_s_full,
            expected.live_x_cols,
            current_y_len,
            r2,
        );
        expected.split_eq.bind(r2);
        expected.live_x_cols = expected.live_x_cols.div_ceil(2);
        expected.rounds_completed += 1;
        let _ = claim3;
        let expected_round3 = expected.compute_round_full_prefix_x(&expected_next_s_full);

        prover.ingest_challenge(2, r2);

        match &prover.s_table {
            STable::Full(s_full) => assert_eq!(s_full, &expected_next_s_full),
            STable::Compact(_) => panic!("expected fused later prefix stage to stay full"),
        }
        assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round3));
    }
}

#[test]
fn stage1_sparse_x_y_fusion_matches_two_pass_reference() {
    let col_bits = 3usize;
    let ring_bits = 4usize;
    let live_x_cols = 6usize;
    let y_len = 1usize << ring_bits;
    for b in [4usize, 8] {
        let half = (b / 2) as i8;
        let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
            .map(|i| ((i * 7 + 9) % b) as i8 - half)
            .collect();
        let tau0: Vec<F> = (0..(col_bits + ring_bits))
            .map(|i| F::from_u64((i as u64) + 131))
            .collect();
        let tau0 = reorder_stage1_coords(&tau0, col_bits, ring_bits);

        let mut prover =
            AkitaStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, col_bits, ring_bits).unwrap();
        let round0 = prover.compute_round_eq_factored(0);
        let r0 = F::from_u64(137);
        let (claim1, scale1) = advance_stage1_claim(&prover, F::zero(), F::one(), &round0, r0);
        prover.ingest_challenge(0, r0);

        let round1 = prover.compute_round_eq_factored(1);
        let r1 = F::from_u64(139);
        let (claim2, scale2) = advance_stage1_claim(&prover, claim1, scale1, &round1, r1);
        prover.ingest_challenge(1, r1);

        let round2 = prover.compute_round_eq_factored(2);
        let r2 = F::from_u64(149);
        let (_claim3, _scale3) = advance_stage1_claim(&prover, claim2, scale2, &round2, r2);

        let mut expected =
            AkitaStage1Prover::new(&w_prefix, &tau0, b, live_x_cols, col_bits, ring_bits).unwrap();
        let expected_round0 = expected.compute_round_eq_factored(0);
        assert_eq!(expected_round0, round0);
        expected.ingest_challenge(0, r0);
        let expected_round1 = expected.compute_round_eq_factored(1);
        assert_eq!(expected_round1, round1);
        expected.ingest_challenge(1, r1);
        let expected_round2 = expected.compute_round_eq_factored(2);
        assert_eq!(expected_round2, round2);

        let current_s_full = match &expected.s_table {
            STable::Full(s_full) => s_full.clone(),
            STable::Compact(_) => panic!("expected sparse-x/y state to be full"),
        };
        let current_y_len = current_s_full.len() / expected.live_x_cols;
        let expected_next_s_full = AkitaStage1Prover::<F>::fold_s_full_sparse_x_y(
            &current_s_full,
            expected.live_x_cols,
            current_y_len,
            r2,
        );
        expected.split_eq.bind(r2);
        expected.rounds_completed += 1;
        let expected_round3 = expected.compute_round_full_sparse_x_y(&expected_next_s_full);

        prover.ingest_challenge(2, r2);

        match &prover.s_table {
            STable::Full(s_full) => assert_eq!(s_full, &expected_next_s_full),
            STable::Compact(_) => panic!("expected sparse-x/y fusion to stay full"),
        }
        assert_eq!(prover.cached_round_poly.as_ref(), Some(&expected_round3));
    }
}
