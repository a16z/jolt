use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    #[tracing::instrument(
        skip_all,
        name = "AkitaStage2Prover::compute_round_compact_dense_terms"
    )]
    pub(super) fn compute_round_compact_dense_terms(
        &self,
        w_compact: &[i8],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        let folding_y_round = self.in_y_round();
        let current_x_width = self.current_x_width();
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
        let current_y_width = self.current_y_width();
        let current_y_mask = (1usize << current_y_width).wrapping_sub(1);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;
        debug_assert_eq!(w_compact.len() / 2, num_first * num_second);

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
                0..num_second,
                || ([E::zero(); 2], [E::MulU64Accum::zero(); 6]),
                |(mut virt, mut rel), j_high| {
                    let mut inner_virt = [E::MulU64Accum::zero(); 2];
                    let base = j_high * num_first;

                    for (j_low, &e_in) in e_first.iter().enumerate() {
                        let j = base + j_low;
                        let w0 = w_compact[2 * j] as i32;
                        let w1 = w_compact[2 * j + 1] as i32;
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

                        let (a0, a1, m0, m1) = if folding_y_round {
                            (
                                alpha_compact[(2 * j) & current_y_mask],
                                alpha_compact[(2 * j + 1) & current_y_mask],
                                m_compact[(2 * j) >> current_y_width],
                                m_compact[(2 * j + 1) >> current_y_width],
                            )
                        } else {
                            (
                                alpha_compact[(2 * j) >> current_x_width],
                                alpha_compact[(2 * j + 1) >> current_x_width],
                                m_compact[(2 * j) & current_x_mask],
                                m_compact[(2 * j + 1) & current_x_mask],
                            )
                        };
                        let p0 = a0 * m0;
                        let p1 = a1 * m1;
                        self.accumulate_fused_relation_trace_signed(
                            &mut rel,
                            w0_i64,
                            dw_i64,
                            2 * j,
                            2 * j + 1,
                            p0,
                            p1,
                        );
                    }

                    let reduced_inner: [E; 2] = reduce_compact_virt_skip_linear(inner_virt);
                    let e_out = e_second[j_high];
                    virt[0] += e_out * reduced_inner[0];
                    virt[1] += e_out * reduced_inner[1];

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
                0..num_second,
                || ([E::zero(); 3], [E::MulU64Accum::zero(); 6]),
                |(mut virt, mut rel), j_high| {
                    let mut inner_virt = [E::MulU64Accum::zero(); 4];
                    let base = j_high * num_first;

                    for (j_low, &e_in) in e_first.iter().enumerate() {
                        let j = base + j_low;
                        let w0 = w_compact[2 * j] as i32;
                        let w1 = w_compact[2 * j + 1] as i32;
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

                        let (a0, a1, m0, m1) = if folding_y_round {
                            (
                                alpha_compact[(2 * j) & current_y_mask],
                                alpha_compact[(2 * j + 1) & current_y_mask],
                                m_compact[(2 * j) >> current_y_width],
                                m_compact[(2 * j + 1) >> current_y_width],
                            )
                        } else {
                            (
                                alpha_compact[(2 * j) >> current_x_width],
                                alpha_compact[(2 * j + 1) >> current_x_width],
                                m_compact[(2 * j) & current_x_mask],
                                m_compact[(2 * j + 1) & current_x_mask],
                            )
                        };
                        let p0 = a0 * m0;
                        let p1 = a1 * m1;
                        self.accumulate_fused_relation_trace_signed(
                            &mut rel,
                            w0_i64,
                            dw_i64,
                            2 * j,
                            2 * j + 1,
                            p0,
                            p1,
                        );
                    }

                    let reduced_inner: [E; 3] = reduce_compact_virt(inner_virt);
                    let e_out = e_second[j_high];
                    virt[0] += e_out * reduced_inner[0];
                    virt[1] += e_out * reduced_inner[1];
                    virt[2] += e_out * reduced_inner[2];

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

    #[tracing::instrument(skip_all, name = "AkitaStage2Prover::compute_round_full_dense_terms")]
    pub(super) fn compute_round_full_dense_terms(
        &self,
        w_full: &[E],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        let folding_y_round = self.in_y_round();
        let current_x_width = self.current_x_width();
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);
        let current_y_width = self.current_y_width();
        let current_y_mask = (1usize << current_y_width).wrapping_sub(1);
        let alpha_compact = &self.alpha_compact;
        let m_compact = &self.m_compact;
        debug_assert_eq!(w_full.len() / 2, num_first * num_second);

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_coeffs) = cfg_fold_reduce!(
                0..num_second,
                || ([E::zero(); 2], [E::zero(); 3]),
                |(mut virt, mut rel), j_high| {
                    let mut inner_virt = [E::zero(); 2];
                    let base = j_high * num_first;

                    for (j_low, &e_in) in e_first.iter().enumerate() {
                        let j = base + j_low;
                        let w0 = w_full[2 * j];
                        let w1 = w_full[2 * j + 1];
                        let dw = w1 - w0;

                        inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                        inner_virt[1] += e_in * (dw * dw);

                        let (a0, a1, m0, m1) = if folding_y_round {
                            (
                                alpha_compact[(2 * j) & current_y_mask],
                                alpha_compact[(2 * j + 1) & current_y_mask],
                                m_compact[(2 * j) >> current_y_width],
                                m_compact[(2 * j + 1) >> current_y_width],
                            )
                        } else {
                            (
                                alpha_compact[(2 * j) >> current_x_width],
                                alpha_compact[(2 * j + 1) >> current_x_width],
                                m_compact[(2 * j) & current_x_mask],
                                m_compact[(2 * j + 1) & current_x_mask],
                            )
                        };
                        let p0 = a0 * m0;
                        let p1 = a1 * m1;
                        self.accumulate_fused_relation_trace(
                            &mut rel,
                            w0,
                            dw,
                            2 * j,
                            2 * j + 1,
                            p0,
                            p1,
                        );
                    }

                    let e_out = e_second[j_high];
                    virt[0] += e_out * inner_virt[0];
                    virt[1] += e_out * inner_virt[1];

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
                0..num_second,
                || ([E::zero(); 3], [E::zero(); 3]),
                |(mut virt, mut rel), j_high| {
                    let mut inner_virt = [E::zero(); 3];
                    let base = j_high * num_first;

                    for (j_low, &e_in) in e_first.iter().enumerate() {
                        let j = base + j_low;
                        let w0 = w_full[2 * j];
                        let w1 = w_full[2 * j + 1];
                        let dw = w1 - w0;
                        let two_w0_plus_one = w0 + w0 + E::one();

                        inner_virt[0] += e_in * (w0 * (w0 + E::one()));
                        inner_virt[1] += e_in * (dw * two_w0_plus_one);
                        inner_virt[2] += e_in * (dw * dw);

                        let (a0, a1, m0, m1) = if folding_y_round {
                            (
                                alpha_compact[(2 * j) & current_y_mask],
                                alpha_compact[(2 * j + 1) & current_y_mask],
                                m_compact[(2 * j) >> current_y_width],
                                m_compact[(2 * j + 1) >> current_y_width],
                            )
                        } else {
                            (
                                alpha_compact[(2 * j) >> current_x_width],
                                alpha_compact[(2 * j + 1) >> current_x_width],
                                m_compact[(2 * j) & current_x_mask],
                                m_compact[(2 * j + 1) & current_x_mask],
                            )
                        };
                        let p0 = a0 * m0;
                        let p1 = a1 * m1;
                        self.accumulate_fused_relation_trace(
                            &mut rel,
                            w0,
                            dw,
                            2 * j,
                            2 * j + 1,
                            p0,
                            p1,
                        );
                    }

                    let e_out = e_second[j_high];
                    virt[0] += e_out * inner_virt[0];
                    virt[1] += e_out * inner_virt[1];
                    virt[2] += e_out * inner_virt[2];

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

    pub(super) fn compute_round_compact_prefix_x_polys(
        &self,
        w_compact: &[i8],
    ) -> (UniPoly<E>, UniPoly<E>) {
        let (virt_q_coeffs, rel_coeffs) = self.compute_round_compact_prefix_x_terms(w_compact);
        self.polys_from_terms(virt_q_coeffs, rel_coeffs)
    }

    #[cfg(all(test, not(feature = "zk")))]
    pub(super) fn compute_round_compact_dense_polys(
        &self,
        w_compact: &[i8],
    ) -> (UniPoly<E>, UniPoly<E>) {
        let (virt_q_coeffs, rel_coeffs) = self.compute_round_compact_dense_terms(w_compact);
        self.polys_from_terms(virt_q_coeffs, rel_coeffs)
    }
}
