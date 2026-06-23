use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage1Prover<E> {
    #[inline]
    pub(super) fn use_sparse_x_y_round(&self) -> bool {
        !self.in_x_phase() && self.live_x_cols < (1usize << self.col_bits)
    }

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::compute_round_compact_sparse_x_y")]
    pub(super) fn compute_round_compact_sparse_x_y(
        &self,
        s_compact: &[i16],
    ) -> EqFactoredUniPoly<E> {
        debug_assert!(self.use_sparse_x_y_round());
        let y_len = s_compact.len() / self.live_x_cols;
        let y_pairs = y_len / 2;
        compute_norm_round_eq_poly_from_s_compact_with_pairs(
            &self.split_eq,
            &self.range_precomp,
            |j| {
                let x = j / y_pairs;
                if x >= self.live_x_cols {
                    return (0, 0);
                }
                let y_pair = j % y_pairs;
                let top = x * y_len + 2 * y_pair;
                (s_compact[top], s_compact[top + 1])
            },
        )
    }

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::compute_round_full_sparse_x_y")]
    pub(super) fn compute_round_full_sparse_x_y(&self, s_full: &[E]) -> EqFactoredUniPoly<E> {
        debug_assert!(self.use_sparse_x_y_round());
        let y_len = s_full.len() / self.live_x_cols;
        let y_pairs = y_len / 2;
        compute_norm_round_eq_poly_from_s(&self.split_eq, &self.range_precomp, |j| {
            let x = j / y_pairs;
            if x >= self.live_x_cols {
                return (E::zero(), E::zero());
            }
            let y_pair = j % y_pairs;
            let top = x * y_len + 2 * y_pair;
            (s_full[top], s_full[top + 1])
        })
    }

    #[tracing::instrument(
        skip_all,
        name = "AkitaStage1Prover::fuse_full_sparse_x_y_and_compute_round"
    )]
    pub(super) fn fuse_full_sparse_x_y_and_compute_round(
        &self,
        s_full: &[E],
        r: E,
    ) -> (Vec<E>, EqFactoredUniPoly<E>) {
        debug_assert!(self.use_sparse_x_y_round());
        debug_assert!(self.next_use_sparse_x_y_round_after_current());
        let live_x_cols = self.live_x_cols;
        let y_len = s_full.len() / live_x_cols;
        debug_assert_eq!(y_len % 4, 0);
        let next_y_len = y_len / 2;
        let live_pairs = next_y_len / 2;
        let current_y_half = next_y_len / 2;
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros();
        let block_size = num_first.min(live_pairs);
        let range_pc = &self.range_precomp;
        let full_num_coeffs_q = range_pc.degree_q + 1;
        let num_coeffs_q = full_num_coeffs_q;
        let mut out = vec![E::zero(); live_x_cols * next_y_len];

        #[cfg(feature = "parallel")]
        let q_coeffs = out
            .par_chunks_mut(next_y_len)
            .enumerate()
            .map(|(x, col_out)| {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let col = &s_full[x * y_len..(x + 1) * y_len];
                let j_base = x * current_y_half;
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
                        for (slot, pair_y) in (pair_base..pair_base + 4).enumerate() {
                            let top_y = 2 * pair_y;
                            let top = 4 * pair_y;
                            let s0 = col[top] + r * (col[top + 1] - col[top]);
                            let s1 = col[top + 2] + r * (col[top + 3] - col[top + 2]);
                            col_out[top_y] = s0;
                            col_out[top_y + 1] = s1;
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
                            let pair_y = pair_base + slot;
                            let j_low = (j_base + pair_y) & (num_first - 1);
                            let e_in = e_first[j_low];
                            accumulate_dense_entry_coeffs(
                                &mut inner_accum[..num_coeffs_q],
                                &batch_out[slot][..full_num_coeffs_q],
                                e_in,
                            );
                        }
                    }

                    for pair_y in blk + full_chunks * 4..blk_end {
                        let top_y = 2 * pair_y;
                        let top = 4 * pair_y;
                        let s0 = col[top] + r * (col[top + 1] - col[top]);
                        let s1 = col[top + 2] + r * (col[top + 3] - col[top + 2]);
                        col_out[top_y] = s0;
                        col_out[top_y + 1] = s1;
                        compute_entry_coeffs_from_s(
                            &mut entry_buf,
                            &mut s_pows_buf,
                            range_pc,
                            s0,
                            s1 - s0,
                        );
                        let j_low = (j_base + pair_y) & (num_first - 1);
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
            let mut outer = vec![E::ProductAccum::zero(); num_coeffs_q];
            for (x, col_out) in out.chunks_mut(next_y_len).enumerate() {
                debug_assert!(full_num_coeffs_q <= MAX_AFFINE_COEFFS);
                let col = &s_full[x * y_len..(x + 1) * y_len];
                let j_base = x * current_y_half;
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
                        for (slot, pair_y) in (pair_base..pair_base + 4).enumerate() {
                            let top_y = 2 * pair_y;
                            let top = 4 * pair_y;
                            let s0 = col[top] + r * (col[top + 1] - col[top]);
                            let s1 = col[top + 2] + r * (col[top + 3] - col[top + 2]);
                            col_out[top_y] = s0;
                            col_out[top_y + 1] = s1;
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
                            let pair_y = pair_base + slot;
                            let j_low = (j_base + pair_y) & (num_first - 1);
                            let e_in = e_first[j_low];
                            accumulate_dense_entry_coeffs(
                                &mut inner_accum[..num_coeffs_q],
                                &batch_out[slot][..full_num_coeffs_q],
                                e_in,
                            );
                        }
                    }

                    for pair_y in blk + full_chunks * 4..blk_end {
                        let top_y = 2 * pair_y;
                        let top = 4 * pair_y;
                        let s0 = col[top] + r * (col[top + 1] - col[top]);
                        let s1 = col[top + 2] + r * (col[top + 3] - col[top + 2]);
                        col_out[top_y] = s0;
                        col_out[top_y + 1] = s1;
                        compute_entry_coeffs_from_s(
                            &mut entry_buf,
                            &mut s_pows_buf,
                            range_pc,
                            s0,
                            s1 - s0,
                        );
                        let j_low = (j_base + pair_y) & (num_first - 1);
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

    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::fold_s_full_sparse_x_y")]
    pub(super) fn fold_s_full_sparse_x_y(
        s_full: &[E],
        live_x_cols: usize,
        y_len: usize,
        r: E,
    ) -> Vec<E> {
        debug_assert_eq!(y_len % 2, 0);
        let next_y_len = y_len / 2;
        let mut out = vec![E::zero(); live_x_cols * next_y_len];

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_y_len)
            .enumerate()
            .for_each(|(x, col_out)| {
                let col = &s_full[x * y_len..(x + 1) * y_len];
                for (pair_y, dst) in col_out.iter_mut().enumerate() {
                    let top = 2 * pair_y;
                    let s_0 = col[top];
                    let s_1 = col[top + 1];
                    *dst = s_0 + r * (s_1 - s_0);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (x, col_out) in out.chunks_mut(next_y_len).enumerate() {
            let col = &s_full[x * y_len..(x + 1) * y_len];
            for (pair_y, dst) in col_out.iter_mut().enumerate() {
                let top = 2 * pair_y;
                let s_0 = col[top];
                let s_1 = col[top + 1];
                *dst = s_0 + r * (s_1 - s_0);
            }
        }

        out
    }
}
