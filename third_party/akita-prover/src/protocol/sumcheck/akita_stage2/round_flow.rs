use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    pub(super) fn compute_current_round_poly_from_state(&mut self) -> UniPoly<E> {
        let t_scan = Instant::now();
        let use_two_round_prefix = self.using_two_round_prefix();
        let use_prefix_y_round = !use_two_round_prefix && self.use_prefix_y_round();
        let use_prefix_x_round = !use_two_round_prefix && self.use_prefix_x_round();
        let rounds_completed = self.rounds_completed;
        let poly = if use_two_round_prefix {
            let (virt_poly, rel_poly) = {
                let prefix = self.ensure_two_round_prefix();
                if rounds_completed == 0 {
                    let (virt_poly, rel_poly) = prefix.skip_state.reconstruct_round0_polys();
                    (virt_poly, rel_poly)
                } else {
                    let r0 = prefix
                        .first_challenge
                        .expect("round 1 prefix polynomial requested before ingesting round 0");
                    let (virt_poly, rel_poly) = prefix.skip_state.reconstruct_round1_polys(r0);
                    (virt_poly, rel_poly)
                }
            };
            let combined = self.combine_polys(&virt_poly, &rel_poly);
            self.prev_norm_poly = Some(virt_poly);
            combined
        } else {
            match &self.w_table {
                WTable::Compact(w_compact) => {
                    if use_prefix_y_round {
                        let (virt_q_coeffs, rel_coeffs) =
                            self.compute_round_compact_prefix_y_terms(w_compact);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    } else if use_prefix_x_round {
                        let (virt_poly, rel_poly) =
                            self.compute_round_compact_prefix_x_polys(w_compact);
                        let combined = self.combine_polys(&virt_poly, &rel_poly);
                        self.prev_norm_poly = Some(virt_poly);
                        combined
                    } else {
                        let (virt_q_coeffs, rel_coeffs) =
                            self.compute_round_compact_dense_terms(w_compact);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    }
                }
                WTable::Full(w_full) => {
                    if use_prefix_y_round {
                        let (virt_q_coeffs, rel_coeffs) =
                            self.compute_round_full_prefix_y_terms(w_full);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    } else if use_prefix_x_round {
                        let (virt_q_coeffs, rel_coeffs) =
                            self.compute_round_full_prefix_x_terms(w_full);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    } else {
                        let (virt_q_coeffs, rel_coeffs) =
                            self.compute_round_full_dense_terms(w_full);
                        self.combine_terms(virt_q_coeffs, rel_coeffs)
                    }
                }
            }
        };
        self.scan_time_total += t_scan.elapsed().as_secs_f64();
        poly
    }

    #[inline]
    pub(super) fn build_compact_w_fold_lut(w_compact: &[i8], r: E) -> CompactPairFoldLut<E> {
        let min_w = w_compact
            .iter()
            .copied()
            .map(i32::from)
            .min()
            .unwrap_or(0)
            .min(0);
        let max_w = w_compact
            .iter()
            .copied()
            .map(i32::from)
            .max()
            .unwrap_or(0)
            .max(0);
        CompactPairFoldLut::from_contiguous_range(min_w as i16, max_w as i16, r)
    }

    pub(super) fn fold_compact_to_full(
        w_compact: &[i8],
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        cfg_into_iter!(0..w_compact.len() / 2)
            .map(|j| fold_lut.fold(i16::from(w_compact[2 * j]), i16::from(w_compact[2 * j + 1])))
            .collect()
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps + HasOptimizedFold> SumcheckInstanceProver<E>
    for AkitaStage2Prover<E>
{
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        if let Some(poly) = self.cached_round_poly.take() {
            poly
        } else {
            self.compute_current_round_poly_from_state()
        }
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        let t_fold = Instant::now();
        let _span = tracing::info_span!("AkitaStage2Prover::fold_round").entered();
        if let Some(prev_norm_poly) = self.prev_norm_poly.take() {
            self.prev_norm_claim = prev_norm_poly.evaluate(&r);
        }

        if self.using_two_round_prefix() {
            let rounds_completed = self.rounds_completed;
            self.split_eq.bind(r);
            if rounds_completed == 0 {
                self.ensure_two_round_prefix().first_challenge = Some(r);
            } else {
                let r0 = {
                    let prefix = self.ensure_two_round_prefix();
                    prefix
                        .first_challenge
                        .expect("round 1 ingest requires the round 0 challenge")
                };
                let y_len = self.alpha_compact.len();
                let alpha_round2 = Self::fold_alpha_to_round2(&self.alpha_compact, r0, r);
                if let Some(trace) = self.trace_table.as_mut() {
                    trace.fold_y2(self.live_x_cols, y_len, r0, r);
                }
                // fold_y2 is the two-round handoff; not routed through fold_trace_for_round.
                let mut round2_terms = None;
                self.w_table = match mem::replace(&mut self.w_table, WTable::Full(Vec::new())) {
                    WTable::Compact(w_compact) => {
                        if self.ring_bits() > 2 {
                            let (w_full, virt_terms, rel_coeffs) = self
                                .fuse_compact_to_round2_and_compute_round(
                                    &w_compact,
                                    &alpha_round2,
                                    self.trace_table.as_ref(),
                                    r0,
                                    r,
                                );
                            round2_terms = Some((virt_terms, rel_coeffs));
                            WTable::Full(w_full)
                        } else {
                            WTable::Full(Self::fold_compact_to_round2(
                                &w_compact,
                                self.live_x_cols,
                                y_len,
                                r0,
                                r,
                            ))
                        }
                    }
                    WTable::Full(_) => unreachable!("two-round prefix should hold compact witness"),
                };
                self.alpha_compact = alpha_round2;
                self.two_round_prefix = None;
                self.prefix_r_stage1 = None;
                if let Some((virt_terms, rel_coeffs)) = round2_terms {
                    self.cached_round_poly = Some(self.combine_terms(virt_terms, rel_coeffs));
                }
            }
            self.rounds_completed += 1;
            if self.rounds_completed < self.num_vars {
                if self.cached_round_poly.is_none() {
                    self.cached_round_poly = Some(self.compute_current_round_poly_from_state());
                }
            } else {
                self.cached_round_poly = None;
            }
            drop(_span);
            self.fold_time_total += t_fold.elapsed().as_secs_f64();
            if self.rounds_completed == self.num_vars {
                tracing::debug!(
                    rounds = self.num_vars,
                    scan_s = self.scan_time_total,
                    fold_s = self.fold_time_total,
                    "stage2 sumcheck rounds complete"
                );
            }
            return;
        }

        self.split_eq.bind(r);
        let folding_x_round = !self.in_y_round();
        let use_prefix_x_round = self.use_prefix_x_round();
        let use_prefix_y_round = self.use_prefix_y_round();
        let in_y_round = self.in_y_round();
        let fuse_next_full_prefix_x =
            use_prefix_x_round && self.next_use_prefix_x_round_after_current();
        let y_len = self.alpha_compact.len();
        let live_x_cols = self.live_x_cols;
        let mut fused_full_prefix_x = false;

        self.w_table = match mem::replace(&mut self.w_table, WTable::Full(Vec::new())) {
            WTable::Compact(w_compact) => {
                let fold_lut = Self::build_compact_w_fold_lut(&w_compact, r);
                let w_full = if folding_x_round && use_prefix_x_round {
                    Self::fold_compact_prefix_x(&w_compact, live_x_cols, y_len, &fold_lut)
                } else {
                    Self::fold_compact_to_full(&w_compact, &fold_lut)
                };
                self.fold_trace_for_round(r, folding_x_round);
                WTable::Full(w_full)
            }
            WTable::Full(w_full) => {
                if folding_x_round && use_prefix_x_round {
                    if fuse_next_full_prefix_x {
                        // Fold trace before the fused kernel so relation terms use the same
                        // post-fold table as `compute_round_full_prefix_x_terms`.
                        self.fold_trace_for_round(r, folding_x_round);
                        let (next_w_full, next_m_compact, virt_terms, rel_coeffs) =
                            self.fuse_full_prefix_x_and_compute_round(&w_full, r);
                        self.m_compact = next_m_compact;
                        self.cached_round_poly = Some(self.combine_terms(virt_terms, rel_coeffs));
                        fused_full_prefix_x = true;
                        WTable::Full(next_w_full)
                    } else {
                        let next_w_full = Self::fold_full_prefix_x(&w_full, live_x_cols, y_len, r);
                        self.fold_trace_for_round(r, folding_x_round);
                        WTable::Full(next_w_full)
                    }
                } else if in_y_round && use_prefix_y_round {
                    self.fold_trace_for_round(r, folding_x_round);
                    WTable::Full(Self::fold_full_prefix_y(&w_full, live_x_cols, y_len, r))
                } else {
                    let mut w_full = w_full;
                    fold_evals_in_place(&mut w_full, r);
                    self.fold_trace_for_round(r, folding_x_round);
                    WTable::Full(w_full)
                }
            }
        };

        if folding_x_round {
            if use_prefix_x_round {
                if !fused_full_prefix_x {
                    self.m_compact = Self::fold_m_prefix(&self.m_compact, r);
                }
            } else {
                fold_evals_in_place(&mut self.m_compact, r);
            }
            self.live_x_cols = self.live_x_cols.div_ceil(2);
        } else {
            fold_evals_in_place(&mut self.alpha_compact, r);
        }

        self.rounds_completed += 1;
        if self.rounds_completed < self.num_vars {
            if self.cached_round_poly.is_none() {
                self.cached_round_poly = Some(self.compute_current_round_poly_from_state());
            }
        } else {
            self.cached_round_poly = None;
        }
        drop(_span);
        self.fold_time_total += t_fold.elapsed().as_secs_f64();

        if self.rounds_completed == self.num_vars {
            tracing::debug!(
                rounds = self.num_vars,
                scan_s = self.scan_time_total,
                fold_s = self.fold_time_total,
                "stage2 sumcheck rounds complete"
            );
        }
    }
}
