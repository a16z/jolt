use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage1Prover<E> {
    #[inline]
    pub(super) fn direct_fold_s_quad_to_round2(
        s00: i16,
        s10: i16,
        s01: i16,
        s11: i16,
        r0: E,
        r1: E,
    ) -> E {
        let s00 = E::from_i64(i64::from(s00));
        let s10 = E::from_i64(i64::from(s10));
        let s01 = E::from_i64(i64::from(s01));
        let s11 = E::from_i64(i64::from(s11));
        let x0 = s00 + r0 * (s10 - s00);
        let x1 = s01 + r0 * (s11 - s01);
        x0 + r1 * (x1 - x0)
    }

    #[inline(always)]
    pub(super) fn stage1_b4_quad_lookup_index_from_row(row: &[i16], base: usize) -> usize {
        let d0 = row
            .get(base)
            .copied()
            .map(stage1_b4_s_digit_from_compact_s)
            .unwrap_or(0);
        let d1 = row
            .get(base + 1)
            .copied()
            .map(stage1_b4_s_digit_from_compact_s)
            .unwrap_or(0);
        let d2 = row
            .get(base + 2)
            .copied()
            .map(stage1_b4_s_digit_from_compact_s)
            .unwrap_or(0);
        let d3 = row
            .get(base + 3)
            .copied()
            .map(stage1_b4_s_digit_from_compact_s)
            .unwrap_or(0);
        d0 | (d1 << 1) | (d2 << 2) | (d3 << 3)
    }

    pub(super) fn build_round2_s_lookup_b4(r0: E, r1: E) -> Vec<E> {
        const S_VALUES: [i16; 2] = [0, 2];
        (0..16usize)
            .map(|idx| {
                let d0 = idx & 0b1;
                let d1 = (idx >> 1) & 0b1;
                let d2 = (idx >> 2) & 0b1;
                let d3 = (idx >> 3) & 0b1;
                Self::direct_fold_s_quad_to_round2(
                    S_VALUES[d0],
                    S_VALUES[d1],
                    S_VALUES[d2],
                    S_VALUES[d3],
                    r0,
                    r1,
                )
            })
            .collect()
    }

    #[inline(always)]
    pub(super) fn stage1_b8_quad_lookup_index_from_row(row: &[i16], base: usize) -> usize {
        let d0 = row
            .get(base)
            .copied()
            .map(stage1_b8_s_digit_from_compact_s)
            .unwrap_or(0);
        let d1 = row
            .get(base + 1)
            .copied()
            .map(stage1_b8_s_digit_from_compact_s)
            .unwrap_or(0);
        let d2 = row
            .get(base + 2)
            .copied()
            .map(stage1_b8_s_digit_from_compact_s)
            .unwrap_or(0);
        let d3 = row
            .get(base + 3)
            .copied()
            .map(stage1_b8_s_digit_from_compact_s)
            .unwrap_or(0);
        d0 | (d1 << 2) | (d2 << 4) | (d3 << 6)
    }

    pub(super) fn build_round2_s_lookup_b8(r0: E, r1: E) -> Vec<E> {
        const S_VALUES: [i16; 4] = [0, 2, 6, 12];
        (0..256usize)
            .map(|idx| {
                let d0 = idx & 0b11;
                let d1 = (idx >> 2) & 0b11;
                let d2 = (idx >> 4) & 0b11;
                let d3 = (idx >> 6) & 0b11;
                Self::direct_fold_s_quad_to_round2(
                    S_VALUES[d0],
                    S_VALUES[d1],
                    S_VALUES[d2],
                    S_VALUES[d3],
                    r0,
                    r1,
                )
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::fold_s_compact_to_round2")]
    pub(super) fn fold_s_compact_to_round2(
        s_compact: &[i16],
        live_x_cols: usize,
        y_len: usize,
        r0: E,
        r1: E,
    ) -> Vec<E> {
        debug_assert_eq!(y_len % 4, 0);
        let next_y_len = y_len / 4;
        let mut out = vec![E::zero(); live_x_cols * next_y_len];
        for (x, col_out) in out.chunks_mut(next_y_len).enumerate() {
            let col = &s_compact[x * y_len..(x + 1) * y_len];
            for (quad_y, dst) in col_out.iter_mut().enumerate() {
                let base = 4 * quad_y;
                *dst = Self::direct_fold_s_quad_to_round2(
                    col[base],
                    col[base + 1],
                    col[base + 2],
                    col[base + 3],
                    r0,
                    r1,
                );
            }
        }
        out
    }

    #[tracing::instrument(
        skip_all,
        name = "AkitaStage1Prover::fuse_compact_to_round2_and_compute_round"
    )]
    pub(super) fn fuse_compact_to_round2_and_compute_round(
        &self,
        s_compact: &[i16],
        r0: E,
        r1: E,
    ) -> (Vec<E>, EqFactoredUniPoly<E>) {
        debug_assert!(self.ring_bits() > 2);
        let live_x_cols = self.live_x_cols;
        let y_len = s_compact.len() / live_x_cols;
        debug_assert_eq!(y_len % 4, 0);
        let next_y_len = y_len / 4;
        let live_pairs = next_y_len / 2;
        let current_y_half = next_y_len / 2;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let block_size = num_first.min(live_pairs);
        let quad_fold_lut = match self.b {
            4 => Self::build_round2_s_lookup_b4(r0, r1),
            _ => Self::build_round2_s_lookup_b8(r0, r1),
        };
        let quad_index_fn: fn(&[i16], usize) -> usize = match self.b {
            4 => Self::stage1_b4_quad_lookup_index_from_row,
            _ => Self::stage1_b8_quad_lookup_index_from_row,
        };

        let range_pc = &self.range_precomp;
        let full_num_coeffs_q = range_pc.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let mut out = vec![E::zero(); live_x_cols * next_y_len];

        #[cfg(feature = "parallel")]
        let q_coeffs = out
            .par_chunks_mut(next_y_len)
            .enumerate()
            .map(|(x, col_out)| {
                let col = &s_compact[x * y_len..(x + 1) * y_len];
                let j_base = x * current_y_half;
                let mut outer_accum = vec![E::ProductAccum::zero(); num_coeffs_q];
                let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];

                    for pair_y in blk..blk_end {
                        let j_low = (j_base + pair_y) & (num_first - 1);
                        let e_in = e_first[j_low];
                        let top_y = 2 * pair_y;
                        let top_base = 8 * pair_y;
                        let s0 = quad_fold_lut[quad_index_fn(col, top_base)];
                        let s1 = quad_fold_lut[quad_index_fn(col, top_base + 4)];
                        col_out[top_y] = s0;
                        col_out[top_y + 1] = s1;
                        compute_entry_coeffs_from_s(
                            &mut entry_buf,
                            &mut s_pows_buf,
                            range_pc,
                            s0,
                            s1 - s0,
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
            let mut outer = vec![E::ProductAccum::zero(); num_coeffs_q];
            for (x, col_out) in out.chunks_mut(next_y_len).enumerate() {
                let col = &s_compact[x * y_len..(x + 1) * y_len];
                let j_base = x * current_y_half;
                let mut entry_buf = [E::zero(); MAX_AFFINE_COEFFS];
                let mut s_pows_buf = [E::zero(); MAX_AFFINE_COEFFS];

                let mut blk = 0usize;
                while blk < live_pairs {
                    let blk_end = (blk + block_size).min(live_pairs);
                    let j_high = (j_base + blk) >> first_bits;
                    let mut inner_accum = [E::ProductAccum::zero(); MAX_AFFINE_COEFFS];

                    for pair_y in blk..blk_end {
                        let j_low = (j_base + pair_y) & (num_first - 1);
                        let e_in = e_first[j_low];
                        let top_y = 2 * pair_y;
                        let top_base = 8 * pair_y;
                        let s0 = quad_fold_lut[quad_index_fn(col, top_base)];
                        let s1 = quad_fold_lut[quad_index_fn(col, top_base + 4)];
                        col_out[top_y] = s0;
                        col_out[top_y + 1] = s1;
                        compute_entry_coeffs_from_s(
                            &mut entry_buf,
                            &mut s_pows_buf,
                            range_pc,
                            s0,
                            s1 - s0,
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
                        outer[k] += e_out.mul_to_product_accum(inner_reduced);
                    }
                    blk = blk_end;
                }
            }
            outer
                .into_iter()
                .map(E::reduce_product_accum)
                .collect::<Vec<_>>()
        };

        let poly = EqFactoredUniPoly::from_q_coeffs(q_coeffs);
        (out, poly)
    }
}
