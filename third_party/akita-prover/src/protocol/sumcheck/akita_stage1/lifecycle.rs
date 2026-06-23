use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage1Prover<E> {
    /// Build the stage-1 prover from the compact witness table.
    #[tracing::instrument(skip_all, name = "AkitaStage1Prover::new")]
    pub fn new(
        w_evals_compact: &[i8],
        tau0: &[E],
        b: usize,
        live_x_cols: usize,
        col_bits: usize,
        ring_bits: usize,
    ) -> Result<Self, AkitaError> {
        if b < 2 {
            return Err(AkitaError::InvalidInput("b must be at least 2".to_string()));
        }
        let num_vars = col_bits.checked_add(ring_bits).ok_or_else(|| {
            AkitaError::InvalidInput("stage-1 challenge width overflow".to_string())
        })?;
        let col_bits_u32 = u32::try_from(col_bits)
            .map_err(|_| AkitaError::InvalidInput("stage-1 column width overflow".to_string()))?;
        let x_len = 1usize
            .checked_shl(col_bits_u32)
            .ok_or_else(|| AkitaError::InvalidInput("stage-1 column width overflow".to_string()))?;
        if live_x_cols == 0 || live_x_cols > x_len {
            return Err(AkitaError::InvalidSize {
                expected: x_len,
                actual: live_x_cols,
            });
        }
        let ring_bits_u32 = u32::try_from(ring_bits)
            .map_err(|_| AkitaError::InvalidInput("stage-1 ring width overflow".to_string()))?;
        let y_len = 1usize
            .checked_shl(ring_bits_u32)
            .ok_or_else(|| AkitaError::InvalidInput("stage-1 ring width overflow".to_string()))?;
        let expected = live_x_cols
            .checked_mul(y_len)
            .ok_or_else(|| AkitaError::InvalidInput("stage-1 witness size overflow".to_string()))?;
        if w_evals_compact.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: w_evals_compact.len(),
            });
        }
        if tau0.len() != num_vars {
            return Err(AkitaError::InvalidSize {
                expected: num_vars,
                actual: tau0.len(),
            });
        }
        let s_table = build_compact_s_table(w_evals_compact);

        Ok(Self {
            s_table: STable::Compact(s_table),
            split_eq: GruenSplitEq::new(tau0)?,
            range_precomp: RangeAffineFromSPrecomp::new(b),
            live_x_cols,
            col_bits,
            num_vars,
            b,
            prefix_tau: can_use_stage1_two_round_prefix(ring_bits, b).then(|| tau0.to_vec()),
            two_round_prefix: None,
            cached_round_poly: None,
            prefix_time_total: 0.0,
            dense_time_total: 0.0,
            fold_time_total: 0.0,
            rounds_completed: 0,
        })
    }

    /// Return the fully folded virtual-polynomial claim `S(stage1_point)`.
    ///
    /// # Panics
    ///
    /// Panics if called before the virtual table has been fully folded to a
    /// single field element.
    pub fn final_s_claim(&self) -> E {
        match &self.s_table {
            STable::Full(s_full) => {
                assert_eq!(s_full.len(), 1, "s_table not fully folded");
                s_full[0]
            }
            STable::Compact(_) => panic!("s_table remained compact after final fold"),
        }
    }

    #[inline]
    pub(super) fn ring_bits(&self) -> usize {
        self.num_vars - self.col_bits
    }

    #[inline]
    pub(super) fn in_x_phase(&self) -> bool {
        self.rounds_completed >= self.ring_bits()
    }

    #[inline]
    pub(super) fn current_x_width(&self) -> usize {
        debug_assert!(self.in_x_phase());
        self.num_vars.saturating_sub(self.rounds_completed)
    }

    #[inline]
    pub(super) fn current_x_len(&self) -> usize {
        1usize << self.current_x_width()
    }

    #[inline]
    pub(super) fn use_prefix_x_round(&self) -> bool {
        self.in_x_phase() && self.live_x_cols < self.current_x_len()
    }

    #[inline]
    pub(super) fn next_use_prefix_x_round_after_current(&self) -> bool {
        self.in_x_phase()
            && self.rounds_completed + 1 < self.num_vars
            && self.live_x_cols.div_ceil(2) < (self.current_x_len() / 2)
    }

    #[inline]
    pub(super) fn next_use_sparse_x_y_round_after_current(&self) -> bool {
        !self.in_x_phase() && self.rounds_completed + 1 < self.ring_bits()
    }

    #[inline]
    pub(crate) fn can_use_two_round_prefix(&self) -> bool {
        self.prefix_tau.is_some()
    }

    #[inline]
    pub(super) fn using_two_round_prefix(&self) -> bool {
        self.rounds_completed < 2 && self.can_use_two_round_prefix()
    }

    #[inline]
    pub(super) fn compact_s_values(b: usize) -> Vec<i16> {
        let half = (b / 2) as i16;
        (0..half).map(|k| k * (k + 1)).collect()
    }

    #[inline]
    pub(super) fn build_compact_s_fold_lut(b: usize, r: E) -> CompactPairFoldLut<E> {
        let valid_s = Self::compact_s_values(b);
        CompactPairFoldLut::from_allowed_values(&valid_s, r)
    }

    pub(super) fn ensure_two_round_prefix(&mut self) -> &mut Stage1TwoRoundPrefix<E> {
        if self.two_round_prefix.is_none() {
            let tau0 = self
                .prefix_tau
                .clone()
                .expect("two-round prefix requested without cached tau");
            let ring_bits = self.num_vars - self.col_bits;
            let s_compact = match &self.s_table {
                STable::Compact(s_compact) => s_compact,
                STable::Full(_) => panic!("two-round prefix can only build from compact table"),
            };
            let proof = build_stage1_bivariate_skip_proof_from_s_compact(
                s_compact,
                &tau0,
                self.b,
                self.live_x_cols,
                self.col_bits,
                ring_bits,
            )
            .expect("two-round prefix should be available");
            let skip_state = Stage1BivariateSkipState::new(&proof, &tau0, self.b)
                .expect("valid bivariate-skip state");
            self.two_round_prefix = Some(Stage1TwoRoundPrefix {
                skip_state,
                first_challenge: None,
            });
        }
        self.two_round_prefix
            .as_mut()
            .expect("two-round prefix should be initialized")
    }
}
