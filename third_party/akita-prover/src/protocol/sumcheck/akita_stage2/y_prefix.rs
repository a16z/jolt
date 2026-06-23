use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    #[tracing::instrument(
        skip_all,
        name = "AkitaStage2Prover::compute_round_compact_prefix_y_terms"
    )]
    pub(super) fn compute_round_compact_prefix_y_terms(
        &self,
        w_compact: &[i8],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.in_y_round());
        debug_assert_eq!(w_compact.len(), self.live_x_cols * self.alpha_compact.len());

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_y_half = 1usize << (self.current_y_width() - 1);
        let block_size = num_first.min(current_y_half);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;
        debug_assert_eq!(m_compact.len(), self.current_x_len());

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
                0..self.live_x_cols,
                || ([E::zero(); 2], [E::MulU64Accum::zero(); 6]),
                |(mut virt, mut rel), x| {
                    let column_start = x * alpha_compact.len();
                    let column = &w_compact[column_start..column_start + alpha_compact.len()];
                    let m = m_compact[x];
                    let j_base = x * current_y_half;
                    let mut blk = 0usize;

                    while blk < current_y_half {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base,
                            blk,
                            num_first,
                            first_bits,
                            block_size,
                            current_y_half,
                        );
                        let mut inner_virt = [E::MulU64Accum::zero(); 2];

                        for pair_y in blk..blk_end {
                            let j_low = (j_base + pair_y) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_y;
                            let w0 = column[left] as i32;
                            let w1 = column[left + 1] as i32;
                            let dw = w1 - w0;
                            let w0_i64 = w0 as i64;
                            let dw_i64 = dw as i64;

                            let q0 = w0_i64 * (w0_i64 + 1);
                            if q0 != 0 {
                                inner_virt[0] += e_in.mul_u64_unreduced(q0 as u64);
                            }
                            let q2 = dw_i64 * dw_i64;
                            if q2 != 0 {
                                inner_virt[1] += e_in.mul_u64_unreduced(q2 as u64);
                            }

                            let p0 = alpha_compact[left] * m;
                            let p1 = alpha_compact[left + 1] * m;
                            self.accumulate_fused_relation_trace_signed(
                                &mut rel,
                                w0_i64,
                                dw_i64,
                                column_start + left,
                                column_start + left + 1,
                                p0,
                                p1,
                            );
                        }

                        let reduced_inner: [E; 2] = reduce_compact_virt_skip_linear(inner_virt);
                        let e_out = e_second[j_high];
                        virt[0] += e_out * reduced_inner[0];
                        virt[1] += e_out * reduced_inner[1];
                        blk = blk_end;
                    }

                    (virt, rel)
                },
                |(mut va, mut ra), (vb, rb)| {
                    for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (va, ra)
                }
            );

            (
                NormRoundTerms::SkipLinear(virt_coeffs),
                reduce_compact_rel(rel_accum),
            )
        } else {
            let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
                0..self.live_x_cols,
                || ([E::zero(); 3], [E::MulU64Accum::zero(); 6]),
                |(mut virt, mut rel), x| {
                    let column_start = x * alpha_compact.len();
                    let column = &w_compact[column_start..column_start + alpha_compact.len()];
                    let m = m_compact[x];
                    let j_base = x * current_y_half;
                    let mut blk = 0usize;

                    while blk < current_y_half {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base,
                            blk,
                            num_first,
                            first_bits,
                            block_size,
                            current_y_half,
                        );
                        let mut inner_virt = [E::MulU64Accum::zero(); 4];

                        for pair_y in blk..blk_end {
                            let j_low = (j_base + pair_y) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_y;
                            let w0 = column[left] as i32;
                            let w1 = column[left + 1] as i32;
                            let dw = w1 - w0;
                            let w0_i64 = w0 as i64;
                            let dw_i64 = dw as i64;

                            let q0 = w0_i64 * (w0_i64 + 1);
                            if q0 != 0 {
                                inner_virt[0] += e_in.mul_u64_unreduced(q0 as u64);
                            }
                            let q1 = dw_i64 * (2 * w0_i64 + 1);
                            accum_small_signed::<E>(&mut inner_virt, 1, e_in, q1);
                            let q2 = dw_i64 * dw_i64;
                            if q2 != 0 {
                                inner_virt[3] += e_in.mul_u64_unreduced(q2 as u64);
                            }

                            let p0 = alpha_compact[left] * m;
                            let p1 = alpha_compact[left + 1] * m;
                            self.accumulate_fused_relation_trace_signed(
                                &mut rel,
                                w0_i64,
                                dw_i64,
                                column_start + left,
                                column_start + left + 1,
                                p0,
                                p1,
                            );
                        }

                        let reduced_inner: [E; 3] = reduce_compact_virt(inner_virt);
                        let e_out = e_second[j_high];
                        virt[0] += e_out * reduced_inner[0];
                        virt[1] += e_out * reduced_inner[1];
                        virt[2] += e_out * reduced_inner[2];
                        blk = blk_end;
                    }

                    (virt, rel)
                },
                |(mut va, mut ra), (vb, rb)| {
                    for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (va, ra)
                }
            );

            (
                NormRoundTerms::Full(virt_coeffs),
                reduce_compact_rel(rel_accum),
            )
        }
    }

    #[tracing::instrument(
        skip_all,
        name = "AkitaStage2Prover::compute_round_full_prefix_y_terms"
    )]
    pub(super) fn compute_round_full_prefix_y_terms(
        &self,
        w_full: &[E],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.in_y_round());
        debug_assert_eq!(w_full.len(), self.live_x_cols * self.alpha_compact.len());

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let first_bits = num_first.trailing_zeros() as usize;
        let current_y_half = 1usize << (self.current_y_width() - 1);
        let block_size = num_first.min(current_y_half);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;
        debug_assert_eq!(m_compact.len(), self.current_x_len());

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_coeffs) = cfg_fold_reduce!(
                0..self.live_x_cols,
                || ([E::zero(); 2], [E::zero(); 3]),
                |(mut virt, mut rel), x| {
                    let column_start = x * alpha_compact.len();
                    let column = &w_full[column_start..column_start + alpha_compact.len()];
                    let m = m_compact[x];
                    let j_base = x * current_y_half;
                    let mut blk = 0usize;

                    while blk < current_y_half {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base,
                            blk,
                            num_first,
                            first_bits,
                            block_size,
                            current_y_half,
                        );
                        let mut inner_virt = [E::zero(); 2];

                        for pair_y in blk..blk_end {
                            let j_low = (j_base + pair_y) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_y;
                            let w0 = column[left];
                            let w1 = column[left + 1];
                            let dw = w1 - w0;

                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * dw);

                            let p0 = alpha_compact[left] * m;
                            let p1 = alpha_compact[left + 1] * m;
                            self.accumulate_fused_relation_trace(
                                &mut rel,
                                w0,
                                dw,
                                column_start + left,
                                column_start + left + 1,
                                p0,
                                p1,
                            );
                        }

                        let e_out = e_second[j_high];
                        virt[0] += e_out * inner_virt[0];
                        virt[1] += e_out * inner_virt[1];
                        blk = blk_end;
                    }

                    (virt, rel)
                },
                |(mut va, mut ra), (vb, rb)| {
                    for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (va, ra)
                }
            );
            (NormRoundTerms::SkipLinear(virt_coeffs), rel_coeffs)
        } else {
            let (virt_coeffs, rel_coeffs) = cfg_fold_reduce!(
                0..self.live_x_cols,
                || ([E::zero(); 3], [E::zero(); 3]),
                |(mut virt, mut rel), x| {
                    let column_start = x * alpha_compact.len();
                    let column = &w_full[column_start..column_start + alpha_compact.len()];
                    let m = m_compact[x];
                    let j_base = x * current_y_half;
                    let mut blk = 0usize;

                    while blk < current_y_half {
                        let (j_high, blk_end) = stage2_eq_block(
                            j_base,
                            blk,
                            num_first,
                            first_bits,
                            block_size,
                            current_y_half,
                        );
                        let mut inner_virt = [E::zero(); 3];

                        for pair_y in blk..blk_end {
                            let j_low = (j_base + pair_y) & (num_first - 1);
                            let e_in = e_first[j_low];
                            let left = 2 * pair_y;
                            let w0 = column[left];
                            let w1 = column[left + 1];
                            let dw = w1 - w0;
                            let two_w0_plus_one = w0 + w0 + E::one();

                            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                            inner_virt[1] += e_in * (dw * two_w0_plus_one);
                            inner_virt[2] += e_in * (dw * dw);

                            let p0 = alpha_compact[left] * m;
                            let p1 = alpha_compact[left + 1] * m;
                            self.accumulate_fused_relation_trace(
                                &mut rel,
                                w0,
                                dw,
                                column_start + left,
                                column_start + left + 1,
                                p0,
                                p1,
                            );
                        }

                        let e_out = e_second[j_high];
                        virt[0] += e_out * inner_virt[0];
                        virt[1] += e_out * inner_virt[1];
                        virt[2] += e_out * inner_virt[2];
                        blk = blk_end;
                    }

                    (virt, rel)
                },
                |(mut va, mut ra), (vb, rb)| {
                    for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (va, ra)
                }
            );
            (NormRoundTerms::Full(virt_coeffs), rel_coeffs)
        }
    }

    pub(super) fn fold_full_prefix_y(
        w_full: &[E],
        live_x_cols: usize,
        y_len: usize,
        r: E,
    ) -> Vec<E> {
        debug_assert!(y_len.is_power_of_two());
        debug_assert!(y_len >= 2);
        let next_y_len = y_len >> 1;
        let mut out = vec![E::zero(); live_x_cols * next_y_len];

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_y_len)
            .enumerate()
            .for_each(|(x, column_out)| {
                let column_start = x * y_len;
                let column = &w_full[column_start..column_start + y_len];
                for (pair_y, dst) in column_out.iter_mut().enumerate() {
                    let left = 2 * pair_y;
                    let w0 = column[left];
                    let w1 = column[left + 1];
                    *dst = w0 + r * (w1 - w0);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (x, column_out) in out.chunks_mut(next_y_len).enumerate() {
            let column_start = x * y_len;
            let column = &w_full[column_start..column_start + y_len];
            for (pair_y, dst) in column_out.iter_mut().enumerate() {
                let left = 2 * pair_y;
                let w0 = column[left];
                let w1 = column[left + 1];
                *dst = w0 + r * (w1 - w0);
            }
        }

        out
    }
}
