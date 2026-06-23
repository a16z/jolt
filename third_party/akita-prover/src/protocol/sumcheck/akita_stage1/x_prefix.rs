use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage1Prover<E> {
    #[tracing::instrument(
        skip_all,
        name = "AkitaStage1Prover::fuse_full_prefix_x_and_compute_round"
    )]
    pub(super) fn fuse_full_prefix_x_and_compute_round(
        &self,
        s_full: &[E],
        r: E,
    ) -> (Vec<E>, EqFactoredUniPoly<E>) {
        debug_assert!(self.next_use_prefix_x_round_after_current());
        debug_assert!(self.current_x_width() >= 2);

        let old_live_x_cols = self.live_x_cols;
        let next_live_x_cols = old_live_x_cols.div_ceil(2);
        let y_len = s_full.len() / old_live_x_cols;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let next_current_x_half = 1usize << (self.current_x_width() - 2);
        let live_pairs = next_live_x_cols.div_ceil(2);
        let block_size = num_first.min(live_pairs);

        let range_pc = &self.range_precomp;
        let full_num_coeffs_q = range_pc.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let mut out = vec![E::zero(); y_len * next_live_x_cols];

        #[cfg(feature = "parallel")]
        let q_coeffs = out
            .par_chunks_mut(next_live_x_cols)
            .enumerate()
            .map(|(y, row_out)| {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let row = &s_full[y * old_live_x_cols..(y + 1) * old_live_x_cols];
                let j_base = y * next_current_x_half;
                let mut outer_accum = vec![E::ProductAccum::zero(); num_coeffs_q];
                let mut batch_out = [[E::zero(); MAX_AFFINE_COEFFS]; 4];
                let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];
                    let blk_len = blk_end - blk;
                    let full_chunks = blk_len / 4;

                    for chunk in 0..full_chunks {
                        let pair_base = blk + chunk * 4;
                        let mut pairs = [(E::zero(), E::zero()); 4];
                        for (slot, pair_x) in (pair_base..pair_base + 4).enumerate() {
                            let left_next = 2 * pair_x;
                            let left_old = 4 * pair_x;
                            let s0 = fold_full_prefix_pair(row, left_old, r);
                            row_out[left_next] = s0;
                            let s1 = if left_next + 1 < next_live_x_cols {
                                let s1 = fold_full_prefix_pair(row, left_old + 2, r);
                                row_out[left_next + 1] = s1;
                                s1
                            } else {
                                E::zero()
                            };
                            pairs[slot] = (s0, s1);
                        }

                        compute_entry_coeffs_from_s_x4(
                            &mut batch_out,
                            range_pc,
                            [pairs[0].0, pairs[1].0, pairs[2].0, pairs[3].0],
                            [
                                pairs[0].1 - pairs[0].0,
                                pairs[1].1 - pairs[1].0,
                                pairs[2].1 - pairs[2].0,
                                pairs[3].1 - pairs[3].0,
                            ],
                        );

                        for (slot, _) in pairs.iter().enumerate() {
                            let pair_x = pair_base + slot;
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            accumulate_dense_entry_coeffs(
                                &mut inner_accum[..num_coeffs_q],
                                &batch_out[slot][..full_num_coeffs_q],
                                e_in,
                            );
                        }
                    }

                    for pair_x in blk + full_chunks * 4..blk_end {
                        let left_next = 2 * pair_x;
                        let left_old = 4 * pair_x;
                        let s_0 = fold_full_prefix_pair(row, left_old, r);
                        row_out[left_next] = s_0;
                        let s_1 = if left_next + 1 < next_live_x_cols {
                            let s_1 = fold_full_prefix_pair(row, left_old + 2, r);
                            row_out[left_next + 1] = s_1;
                            s_1
                        } else {
                            E::zero()
                        };
                        compute_entry_coeffs_from_s(
                            &mut entry_buf,
                            &mut s_pows_buf,
                            range_pc,
                            s_0,
                            s_1 - s_0,
                        );
                        let j_low = (j_base + pair_x) & (num_first - 1);
                        let e_in = e_first[j_low];
                        accumulate_dense_entry_coeffs(
                            &mut inner_accum[..num_coeffs_q],
                            &entry_buf[..full_num_coeffs_q],
                            e_in,
                        );
                    }

                    let e_out = e_second[j_high];
                    for k in 0..num_coeffs_q {
                        let inner_reduced = E::reduce_product_accum(inner_accum[k]);
                        outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
                    }
                    blk = blk_end;
                }

                outer_accum
            })
            .reduce(
                || vec![E::ProductAccum::zero(); num_coeffs_q],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += *bi;
                    }
                    a
                },
            )
            .into_iter()
            .map(E::reduce_product_accum)
            .collect::<Vec<_>>();

        #[cfg(not(feature = "parallel"))]
        let q_coeffs = {
            let mut outer_accum = vec![E::ProductAccum::zero(); num_coeffs_q];
            for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let row = &s_full[y * old_live_x_cols..(y + 1) * old_live_x_cols];
                let j_base = y * next_current_x_half;
                let mut batch_out = [[E::zero(); MAX_AFFINE_COEFFS]; 4];
                let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];
                    let blk_len = blk_end - blk;
                    let full_chunks = blk_len / 4;

                    for chunk in 0..full_chunks {
                        let pair_base = blk + chunk * 4;
                        let mut pairs = [(E::zero(), E::zero()); 4];
                        for (slot, pair_x) in (pair_base..pair_base + 4).enumerate() {
                            let left_next = 2 * pair_x;
                            let left_old = 4 * pair_x;
                            let s0 = fold_full_prefix_pair(row, left_old, r);
                            row_out[left_next] = s0;
                            let s1 = if left_next + 1 < next_live_x_cols {
                                let s1 = fold_full_prefix_pair(row, left_old + 2, r);
                                row_out[left_next + 1] = s1;
                                s1
                            } else {
                                E::zero()
                            };
                            pairs[slot] = (s0, s1);
                        }

                        compute_entry_coeffs_from_s_x4(
                            &mut batch_out,
                            range_pc,
                            [pairs[0].0, pairs[1].0, pairs[2].0, pairs[3].0],
                            [
                                pairs[0].1 - pairs[0].0,
                                pairs[1].1 - pairs[1].0,
                                pairs[2].1 - pairs[2].0,
                                pairs[3].1 - pairs[3].0,
                            ],
                        );

                        for (slot, _) in pairs.iter().enumerate() {
                            let pair_x = pair_base + slot;
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            accumulate_dense_entry_coeffs(
                                &mut inner_accum[..num_coeffs_q],
                                &batch_out[slot][..full_num_coeffs_q],
                                e_in,
                            );
                        }
                    }

                    for pair_x in blk + full_chunks * 4..blk_end {
                        let left_next = 2 * pair_x;
                        let left_old = 4 * pair_x;
                        let s_0 = fold_full_prefix_pair(row, left_old, r);
                        row_out[left_next] = s_0;
                        let s_1 = if left_next + 1 < next_live_x_cols {
                            let s_1 = fold_full_prefix_pair(row, left_old + 2, r);
                            row_out[left_next + 1] = s_1;
                            s_1
                        } else {
                            E::zero()
                        };
                        compute_entry_coeffs_from_s(
                            &mut entry_buf,
                            &mut s_pows_buf,
                            range_pc,
                            s_0,
                            s_1 - s_0,
                        );
                        let j_low = (j_base + pair_x) & (num_first - 1);
                        let e_in = e_first[j_low];
                        accumulate_dense_entry_coeffs(
                            &mut inner_accum[..num_coeffs_q],
                            &entry_buf[..full_num_coeffs_q],
                            e_in,
                        );
                    }

                    let e_out = e_second[j_high];
                    for k in 0..num_coeffs_q {
                        let inner_reduced = E::reduce_product_accum(inner_accum[k]);
                        outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
                    }
                    blk = blk_end;
                }
            }

            outer_accum
                .into_iter()
                .map(E::reduce_product_accum)
                .collect::<Vec<_>>()
        };

        let poly = EqFactoredUniPoly::from_q_coeffs(q_coeffs);
        (out, poly)
    }

    #[inline]
    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::compute_round_compact_prefix_x")]
    pub(super) fn compute_round_compact_prefix_x(&self, s_compact: &[i16]) -> EqFactoredUniPoly<E> {
        debug_assert!(self.rounds_completed < self.col_bits);
        debug_assert_eq!(
            s_compact.len(),
            self.live_x_cols * (1usize << (self.num_vars - self.col_bits))
        );

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let block_size = num_first.min(live_pairs);

        let rp = &self.range_precomp;
        let full_num_coeffs_q = rp.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let q_coeffs = if rp.compact_coeffs_lut(0, 0).is_some() {
            cfg_fold_reduce!(
                0..(1usize << (self.num_vars - self.col_bits)),
                || vec![E::ProductAccum::zero(); num_coeffs_q],
                |mut outer_accum, y| {
                    let row_start = y * self.live_x_cols;
                    let row = &s_compact[row_start..row_start + self.live_x_cols];
                    let j_base = y * current_x_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let blk_end = (blk + block_size).min(live_pairs);
                        let j_high = (j_base + blk) >> first_bits;
                        let mut inner_pos = [E::MulU64Accum::zero(); MAX_AFFINE_COEFFS];
                        let mut inner_neg = [E::MulU64Accum::zero(); MAX_AFFINE_COEFFS];

                        for pair_x in blk..blk_end {
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_x;
                            let s0_i = row[left];
                            let s1_i = if left + 1 < self.live_x_cols {
                                row[left + 1]
                            } else {
                                0
                            };
                            let coeffs = rp
                                .compact_coeffs_lut(s0_i, s1_i)
                                .expect("missing compact coefficient LUT");
                            accumulate_compact_coeffs(
                                &mut inner_pos[..num_coeffs_q],
                                &mut inner_neg[..num_coeffs_q],
                                e_in,
                                coeffs,
                            );
                        }

                        let e_out = e_second[j_high];
                        for k in 0..num_coeffs_q {
                            let inner_reduced =
                                reduce_small_coeff_accum(inner_pos[k], inner_neg[k]);
                            outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
                        }
                        blk = blk_end;
                    }
                    outer_accum
                },
                |mut a, b_vec| {
                    for (ai, bi) in a.iter_mut().zip(b_vec.iter()) {
                        *ai += *bi;
                    }
                    a
                }
            )
            .into_iter()
            .map(E::reduce_product_accum)
            .collect()
        } else if rp.field_coeffs_lut(0, 0).is_some() {
            cfg_fold_reduce!(
                0..(1usize << (self.num_vars - self.col_bits)),
                || vec![E::ProductAccum::zero(); num_coeffs_q],
                |mut outer_accum, y| {
                    let row_start = y * self.live_x_cols;
                    let row = &s_compact[row_start..row_start + self.live_x_cols];
                    let j_base = y * current_x_half;

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let blk_end = (blk + block_size).min(live_pairs);
                        let j_high = (j_base + blk) >> first_bits;
                        let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];

                        for pair_x in blk..blk_end {
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_x;
                            let s0_i = row[left];
                            let s1_i = if left + 1 < self.live_x_cols {
                                row[left + 1]
                            } else {
                                0
                            };
                            let coeffs = rp
                                .field_coeffs_lut(s0_i, s1_i)
                                .expect("missing field coefficient LUT");
                            accumulate_dense_entry_coeffs(
                                &mut inner_accum[..num_coeffs_q],
                                coeffs,
                                e_in,
                            );
                        }

                        let e_out = e_second[j_high];
                        for k in 0..num_coeffs_q {
                            let inner_reduced = E::reduce_product_accum(inner_accum[k]);
                            outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
                        }
                        blk = blk_end;
                    }
                    outer_accum
                },
                |mut ca, cb| {
                    for (ai, bi) in ca.iter_mut().zip(cb.iter()) {
                        *ai += *bi;
                    }
                    ca
                }
            )
            .into_iter()
            .map(E::reduce_product_accum)
            .collect()
        } else {
            cfg_fold_reduce!(
                0..(1usize << (self.num_vars - self.col_bits)),
                || vec![E::ProductAccum::zero(); num_coeffs_q],
                |mut outer_accum, y| {
                    let row_start = y * self.live_x_cols;
                    let row = &s_compact[row_start..row_start + self.live_x_cols];
                    let j_base = y * current_x_half;
                    let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                    let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];

                    let mut blk = 0usize;
                    while blk < live_pairs {
                        let blk_end = (blk + block_size).min(live_pairs);
                        let j_high = (j_base + blk) >> first_bits;
                        let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];

                        for pair_x in blk..blk_end {
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_x;
                            let s0_i = row[left];
                            let s1_i = if left + 1 < self.live_x_cols {
                                row[left + 1]
                            } else {
                                0
                            };
                            compute_entry_coeffs_from_s(
                                &mut entry_buf,
                                &mut s_pows_buf,
                                rp,
                                E::from_i64(i64::from(s0_i)),
                                E::from_i64(i64::from(s1_i - s0_i)),
                            );
                            accumulate_dense_entry_coeffs(
                                &mut inner_accum[..num_coeffs_q],
                                &entry_buf[..full_num_coeffs_q],
                                e_in,
                            );
                        }

                        let e_out = e_second[j_high];
                        for k in 0..num_coeffs_q {
                            let inner_reduced = E::reduce_product_accum(inner_accum[k]);
                            outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
                        }
                        blk = blk_end;
                    }
                    outer_accum
                },
                |mut ca, cb| {
                    for (ai, bi) in ca.iter_mut().zip(cb.iter()) {
                        *ai += *bi;
                    }
                    ca
                }
            )
            .into_iter()
            .map(E::reduce_product_accum)
            .collect()
        };

        EqFactoredUniPoly::from_q_coeffs(q_coeffs)
    }

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::compute_round_full_prefix_x")]
    pub(super) fn compute_round_full_prefix_x(&self, s_full: &[E]) -> EqFactoredUniPoly<E> {
        debug_assert!(self.rounds_completed < self.col_bits);
        let y_len = s_full.len() / self.live_x_cols;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let current_x_half = 1usize << (self.current_x_width() - 1);
        let live_pairs = self.live_x_cols.div_ceil(2);
        let block_size = num_first.min(live_pairs);

        let range_pc = &self.range_precomp;
        let full_num_coeffs_q = range_pc.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let q_coeffs = cfg_fold_reduce!(
            0..y_len,
            || vec![E::ProductAccum::zero(); num_coeffs_q],
            |mut outer_accum, y| {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let row_start = y * self.live_x_cols;
                let row = &s_full[row_start..row_start + self.live_x_cols];
                let j_base = y * current_x_half;
                let mut batch_out = [[E::zero(); MAX_AFFINE_COEFFS]; 4];
                let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];
                    let blk_len = blk_end - blk;
                    let full_chunks = blk_len / 4;

                    for chunk in 0..full_chunks {
                        let pair_base = blk + chunk * 4;
                        let mut pairs = [(E::zero(), E::zero()); 4];
                        for (slot, pair_x) in (pair_base..pair_base + 4).enumerate() {
                            let left = 2 * pair_x;
                            let s_0 = row[left];
                            let s_1 = if left + 1 < self.live_x_cols {
                                row[left + 1]
                            } else {
                                E::zero()
                            };
                            pairs[slot] = (s_0, s_1);
                        }

                        compute_entry_coeffs_from_s_x4(
                            &mut batch_out,
                            range_pc,
                            [pairs[0].0, pairs[1].0, pairs[2].0, pairs[3].0],
                            [
                                pairs[0].1 - pairs[0].0,
                                pairs[1].1 - pairs[1].0,
                                pairs[2].1 - pairs[2].0,
                                pairs[3].1 - pairs[3].0,
                            ],
                        );

                        for (slot, _) in pairs.iter().enumerate() {
                            let pair_x = pair_base + slot;
                            let j_low = (j_base + pair_x) & (num_first - 1);
                            let e_in = e_first[j_low];
                            accumulate_dense_entry_coeffs(
                                &mut inner_accum[..num_coeffs_q],
                                &batch_out[slot][..full_num_coeffs_q],
                                e_in,
                            );
                        }
                    }

                    for pair_x in blk + full_chunks * 4..blk_end {
                        let left = 2 * pair_x;
                        let s_0 = row[left];
                        let s_1 = if left + 1 < self.live_x_cols {
                            row[left + 1]
                        } else {
                            E::zero()
                        };
                        compute_entry_coeffs_from_s(
                            &mut entry_buf,
                            &mut s_pows_buf,
                            range_pc,
                            s_0,
                            s_1 - s_0,
                        );
                        let j_low = (j_base + pair_x) & (num_first - 1);
                        let e_in = e_first[j_low];
                        accumulate_dense_entry_coeffs(
                            &mut inner_accum[..num_coeffs_q],
                            &entry_buf[..full_num_coeffs_q],
                            e_in,
                        );
                    }

                    let e_out = e_second[j_high];
                    for k in 0..num_coeffs_q {
                        let inner_reduced = E::reduce_product_accum(inner_accum[k]);
                        outer_accum[k] += e_out.mul_to_product_accum(inner_reduced);
                    }
                    blk = blk_end;
                }

                outer_accum
            },
            |mut ca, cb| {
                for (ai, bi) in ca.iter_mut().zip(cb.iter()) {
                    *ai += *bi;
                }
                ca
            }
        );

        let q_coeffs: Vec<E> = q_coeffs.into_iter().map(E::reduce_product_accum).collect();
        EqFactoredUniPoly::from_q_coeffs(q_coeffs)
    }

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::fold_s_compact_prefix_x")]
    pub(super) fn fold_s_compact_prefix_x(
        s_compact: &[i16],
        live_x_cols: usize,
        y_len: usize,
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![E::zero(); y_len * next_live_x_cols];

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_live_x_cols)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row_start = y * live_x_cols;
                let row = &s_compact[row_start..row_start + live_x_cols];
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let s_1 = if left + 1 < live_x_cols {
                        row[left + 1]
                    } else {
                        0
                    };
                    *dst = fold_lut.fold(row[left], s_1);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
            let row_start = y * live_x_cols;
            let row = &s_compact[row_start..row_start + live_x_cols];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let s_1 = if left + 1 < live_x_cols {
                    row[left + 1]
                } else {
                    0
                };
                *dst = fold_lut.fold(row[left], s_1);
            }
        }

        out
    }

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::fold_s_full_prefix_x")]
    pub(super) fn fold_s_full_prefix_x(
        s_full: &[E],
        live_x_cols: usize,
        y_len: usize,
        r: E,
    ) -> Vec<E> {
        let next_live_x_cols = live_x_cols.div_ceil(2);
        let mut out = vec![E::zero(); y_len * next_live_x_cols];

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_live_x_cols)
            .enumerate()
            .for_each(|(y, row_out)| {
                let row_start = y * live_x_cols;
                let row = &s_full[row_start..row_start + live_x_cols];
                for (pair_x, dst) in row_out.iter_mut().enumerate() {
                    let left = 2 * pair_x;
                    let s_0 = row[left];
                    let s_1 = if left + 1 < live_x_cols {
                        row[left + 1]
                    } else {
                        E::zero()
                    };
                    *dst = s_0 + r * (s_1 - s_0);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (y, row_out) in out.chunks_mut(next_live_x_cols).enumerate() {
            let row_start = y * live_x_cols;
            let row = &s_full[row_start..row_start + live_x_cols];
            for (pair_x, dst) in row_out.iter_mut().enumerate() {
                let left = 2 * pair_x;
                let s_0 = row[left];
                let s_1 = if left + 1 < live_x_cols {
                    row[left + 1]
                } else {
                    E::zero()
                };
                *dst = s_0 + r * (s_1 - s_0);
            }
        }

        out
    }
}
