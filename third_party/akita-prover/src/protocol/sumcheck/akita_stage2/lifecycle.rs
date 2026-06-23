use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    /// Create a fused stage-2 virtual-claim + relation sumcheck prover.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "AkitaStage2Prover::new")]
    pub(crate) fn new(
        batching_coeff: E,
        w_evals_compact: Vec<i8>,
        stage1_point: &[E],
        s_claim: E,
        b: usize,
        alpha_evals_y: Vec<E>,
        m_evals_x: Vec<E>,
        live_x_cols: usize,
        col_bits: usize,
        ring_bits: usize,
        relation_claim: E,
        trace_table: Option<TraceTable<E>>,
        trace_opening_claim: E,
    ) -> Result<Self, AkitaError> {
        let num_vars = col_bits.checked_add(ring_bits).ok_or_else(|| {
            AkitaError::InvalidInput("stage-2 challenge width overflow".to_string())
        })?;
        if live_x_cols == 0 {
            return Err(AkitaError::InvalidInput(
                "live_x_cols must be at least 1".to_string(),
            ));
        }
        let col_bits_u32 = u32::try_from(col_bits)
            .map_err(|_| AkitaError::InvalidInput("stage-2 column width overflow".to_string()))?;
        let ring_bits_u32 = u32::try_from(ring_bits)
            .map_err(|_| AkitaError::InvalidInput("stage-2 ring width overflow".to_string()))?;
        let x_len = 1usize
            .checked_shl(col_bits_u32)
            .ok_or_else(|| AkitaError::InvalidInput("stage-2 column width overflow".to_string()))?;
        if live_x_cols > x_len {
            return Err(AkitaError::InvalidSize {
                expected: x_len,
                actual: live_x_cols,
            });
        }
        let y_len = 1usize
            .checked_shl(ring_bits_u32)
            .ok_or_else(|| AkitaError::InvalidInput("stage-2 ring width overflow".to_string()))?;
        let witness_len = live_x_cols
            .checked_mul(y_len)
            .ok_or_else(|| AkitaError::InvalidInput("stage-2 witness size overflow".to_string()))?;
        if w_evals_compact.len() != witness_len {
            return Err(AkitaError::InvalidSize {
                expected: witness_len,
                actual: w_evals_compact.len(),
            });
        }
        if stage1_point.len() != num_vars {
            return Err(AkitaError::InvalidSize {
                expected: num_vars,
                actual: stage1_point.len(),
            });
        }
        if alpha_evals_y.len() != y_len {
            return Err(AkitaError::InvalidSize {
                expected: y_len,
                actual: alpha_evals_y.len(),
            });
        }
        if m_evals_x.len() != x_len {
            return Err(AkitaError::InvalidSize {
                expected: x_len,
                actual: m_evals_x.len(),
            });
        }
        if let Some(trace) = &trace_table {
            trace.validate_len(witness_len)?;
        }

        let relation_trace_claim = relation_claim + trace_opening_claim;
        let input_claim = batching_coeff * s_claim + relation_trace_claim;

        Ok(Self {
            w_table: WTable::Compact(w_evals_compact),
            b,
            batching_coeff,
            s_claim,
            input_claim,
            split_eq: GruenSplitEq::with_initial_scalar(stage1_point, batching_coeff)?,
            alpha_compact: alpha_evals_y,
            m_compact: m_evals_x,
            trace_table,
            live_x_cols,
            col_bits,
            num_vars,
            relation_trace_claim,
            prev_norm_claim: batching_coeff * s_claim,
            prev_norm_poly: None,
            prefix_r_stage1: can_use_stage2_two_round_prefix(ring_bits, b)
                .then(|| stage1_point.to_vec()),
            two_round_prefix: None,
            cached_round_poly: None,
            scan_time_total: 0.0,
            fold_time_total: 0.0,
            rounds_completed: 0,
        })
    }

    /// Return the fully folded witness evaluation after the final round.
    ///
    /// # Panics
    ///
    /// Panics if called before the witness table has been fully folded to a
    /// single field element.
    pub fn final_w_eval(&self) -> E {
        match &self.w_table {
            WTable::Full(w_full) => {
                assert_eq!(w_full.len(), 1, "w_table not fully folded");
                w_full[0]
            }
            WTable::Compact(_) => panic!("w_table remained compact after final fold"),
        }
    }

    #[inline]
    pub(super) fn ring_bits(&self) -> usize {
        self.num_vars - self.col_bits
    }

    #[inline]
    pub(super) fn y_rounds_completed(&self) -> usize {
        self.rounds_completed.min(self.ring_bits())
    }

    #[inline]
    pub(super) fn x_rounds_completed(&self) -> usize {
        self.rounds_completed.saturating_sub(self.ring_bits())
    }

    #[inline]
    pub(super) fn in_y_round(&self) -> bool {
        self.rounds_completed < self.ring_bits()
    }

    #[inline]
    pub(super) fn current_y_width(&self) -> usize {
        self.ring_bits().saturating_sub(self.y_rounds_completed())
    }

    #[inline]
    pub(super) fn current_x_width(&self) -> usize {
        self.col_bits.saturating_sub(self.x_rounds_completed())
    }

    #[inline]
    pub(super) fn current_x_len(&self) -> usize {
        1usize << self.current_x_width()
    }

    #[inline]
    pub(super) fn use_prefix_y_round(&self) -> bool {
        self.in_y_round() && self.live_x_cols < self.current_x_len()
    }

    #[inline]
    pub(super) fn use_prefix_x_round(&self) -> bool {
        self.rounds_completed >= self.ring_bits()
            && self.x_rounds_completed() < self.col_bits
            && self.live_x_cols < self.current_x_len()
    }

    #[inline]
    pub(super) fn next_use_prefix_x_round_after_current(&self) -> bool {
        self.rounds_completed >= self.ring_bits()
            && self.x_rounds_completed() + 1 < self.col_bits
            && self.live_x_cols.div_ceil(2) < (self.current_x_len() / 2)
    }

    #[inline]
    pub(crate) fn can_use_two_round_prefix(&self) -> bool {
        self.prefix_r_stage1.is_some()
    }

    #[inline]
    pub(super) fn using_two_round_prefix(&self) -> bool {
        self.rounds_completed < 2 && self.can_use_two_round_prefix()
    }

    #[inline]
    pub(super) fn can_skip_norm_linear_coeff(&self) -> bool {
        self.split_eq.can_recover_linear_q_term_from_claim()
    }

    #[inline]
    pub(super) fn norm_poly_from_terms(&self, virt_terms: NormRoundTerms<E>) -> UniPoly<E> {
        match virt_terms {
            NormRoundTerms::Full(virt_q_coeffs) => {
                self.split_eq.gruen_mul(&coeffs_to_poly(virt_q_coeffs))
            }
            NormRoundTerms::SkipLinear([q_constant, q_quadratic]) => self
                .split_eq
                .try_gruen_poly_deg_3(q_constant, q_quadratic, self.prev_norm_claim)
                .expect("split-eq norm claim recovery should succeed"),
        }
    }

    #[inline]
    pub(super) fn polys_from_terms(
        &self,
        virt_terms: NormRoundTerms<E>,
        rel_coeffs: [E; 3],
    ) -> (UniPoly<E>, UniPoly<E>) {
        let virt_poly = self.norm_poly_from_terms(virt_terms);
        let rel_poly = coeffs_to_poly(rel_coeffs);
        (virt_poly, rel_poly)
    }

    #[inline]
    pub(super) fn combine_polys(
        &self,
        virt_poly: &UniPoly<E>,
        relation_poly: &UniPoly<E>,
    ) -> UniPoly<E> {
        let max_len = virt_poly.coeffs.len().max(relation_poly.coeffs.len());
        let mut combined = vec![E::zero(); max_len];
        for (i, c) in virt_poly.coeffs.iter().enumerate() {
            combined[i] += *c;
        }
        for (i, c) in relation_poly.coeffs.iter().enumerate() {
            combined[i] += *c;
        }
        UniPoly::from_coeffs(combined)
    }

    #[inline]
    pub(super) fn combine_terms(
        &mut self,
        virt_terms: NormRoundTerms<E>,
        rel_coeffs: [E; 3],
    ) -> UniPoly<E> {
        let (virt_poly, relation_poly) = self.polys_from_terms(virt_terms, rel_coeffs);
        let combined = self.combine_polys(&virt_poly, &relation_poly);
        self.prev_norm_poly = Some(virt_poly);
        combined
    }

    pub(super) fn ensure_two_round_prefix(&mut self) -> &mut Stage2TwoRoundPrefix<E> {
        if self.two_round_prefix.is_none() {
            let stage1_point = self
                .prefix_r_stage1
                .clone()
                .expect("two-round prefix requested without cached stage-1 challenges");
            let ring_bits = self.num_vars - self.col_bits;
            let w_compact = match &self.w_table {
                WTable::Compact(w_compact) => w_compact,
                WTable::Full(_) => panic!("two-round prefix can only build from compact witness"),
            };
            let proof = build_stage2_bivariate_skip_proof_from_compact(
                w_compact,
                &self.alpha_compact,
                &self.m_compact,
                self.trace_table.as_ref(),
                &stage1_point,
                self.b,
                self.live_x_cols,
                self.col_bits,
                ring_bits,
            )
            .expect("two-round prefix should be available");
            let skip_state = Stage2BivariateSkipState::new(
                &proof,
                &stage1_point,
                self.s_claim,
                self.relation_trace_claim,
                self.batching_coeff,
            )
            .expect("valid bivariate-skip state");
            self.two_round_prefix = Some(Stage2TwoRoundPrefix {
                skip_state,
                first_challenge: None,
            });
        }
        self.two_round_prefix
            .as_mut()
            .expect("two-round prefix should be initialized")
    }
}
