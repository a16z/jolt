use super::*;

impl<F: JoltField> OptimizedInstructionReadRafKernel<F> {
    fn init_cycle_rounds(&mut self) -> Result<(), WitnessError> {
        let gamma_sqr = self.gamma * self.gamma;
        let empty_bits = LookupBits::new(0, 0);
        let table_values: Vec<F> = LookupTableKind::<RISCV_XLEN>::iter()
            .map(|table| {
                let suffix_evals: Vec<SuffixEval<F>> = table
                    .suffixes()
                    .iter()
                    .map(|suffix| SuffixEval::from(F::from_u64(suffix.suffix_mle(empty_bits))))
                    .collect();
                table.combine(&self.prefix_checkpoints, &suffix_evals)
            })
            .collect();
        let raf_interleaved =
            self.gamma * self.raf_left.checkpoint + gamma_sqr * self.raf_right.checkpoint;
        // The identity branch is selected by `raf_flag`, so folding
        // γ³·U(r_address) in here applies the mask without a separate
        // cycle-indexed polynomial.
        let mut raf_identity = gamma_sqr * self.raf_identity.checkpoint;
        if CANONICAL_INSTRUCTION_ADDRESS {
            raf_identity += gamma_sqr * self.gamma * self.raf_upper_all_ones.checkpoint;
        }

        let combined_table: Vec<F> = (0..=u8::MAX)
            .map(|packed| {
                let table_value = usize::from(packed & 0x7f)
                    .checked_sub(1)
                    .and_then(|table| table_values.get(table))
                    .map_or_else(F::zero, |value| *value);
                let raf_value = if packed & 0x80 == 0 {
                    raf_interleaved
                } else {
                    raf_identity
                };
                table_value + raf_value
            })
            .collect();

        // `ra_i = Π_{phases p of i} v_p[chunk_p]`: phase pairs tensor into
        // 2^16-entry columns over 16-bit lookup-index chunks, so 16-bit
        // virtual chunks gather without multiplying; wider ones factor into
        // several columns.
        let phases_per_ra = self.phases() / self.dimensions.num_virtual_ra_polys();
        let phases_per_column = phases_per_ra.min(2);
        let v_tables = std::mem::take(&mut self.v_tables);
        let column_tables = map_indices(self.phases() / phases_per_column, |column| {
            let mut table = vec![F::one()];
            for v in &v_tables[column * phases_per_column..(column + 1) * phases_per_column] {
                table = table
                    .iter()
                    .flat_map(|high| v.iter().map(move |low| *high * *low))
                    .collect();
            }
            table
        });
        let columns = column_tables.len();
        let chunk_bits = phases_per_column * CHUNK_LEN;
        let factors = phases_per_ra / phases_per_column;
        // Past this handoff the cycle tables own the rows.
        let released = ScanRows::Collected {
            rows: Arc::new(Vec::new().into()),
            blocks: Vec::new(),
        };
        let ra = match std::mem::replace(&mut self.rows, released) {
            ScanRows::Collected { rows, .. } => {
                // Snap the packed output-claim facts first: past this handoff
                // the final flag walk reads one byte per cycle, not the 40 B
                // row.
                let packed = rows.representatives();
                self.claim_columns = Arc::new(ConstantSuffix::new(
                    map_indices(packed.len(), |j| {
                        claim_byte(packed[j].table_index(), packed[j].raf_flag())
                    }),
                    rows.len(),
                ));
                CycleRa::Lazy(LazyFoldedRa::factored(
                    column_tables,
                    factors,
                    LAZY_MAX_WIDTH,
                    LookupIndexChunks::new(rows, columns, chunk_bits),
                ))
            }
            ScanRows::Resident { lookup_indices, .. } => {
                // The streamed rounds read the combined value by pair row, so
                // the family's suffix starts no earlier than the claim
                // column's.
                let ra = StreamedFoldedRa::factored(
                    column_tables,
                    factors,
                    LAZY_MAX_WIDTH,
                    LookupIndexDigits::new(columns, chunk_bits),
                    lookup_indices,
                    self.dimensions.log_t(),
                    self.claim_columns.prefix_len(),
                )?;
                // A one-cycle domain has no bind to materialize at.
                match ra.single_cycle()? {
                    Some(dense) => CycleRa::Lazy(dense),
                    None => CycleRa::Streamed(ra),
                }
            }
        };

        self.cycle = Some(CycleState {
            gruen: GruenSplitEqPolynomial::new(&self.r_reduction, BindingOrder::LowToHigh),
            combined_val: LazyFoldedRa::factored(
                vec![combined_table],
                1,
                LAZY_MAX_WIDTH,
                ClaimBytes(Arc::clone(&self.claim_columns)),
            ),
            ra,
        });

        // The address-phase state is dead past this point; the collected rows
        // live on in the lazy RA source until it materializes.
        self.u_evals = Vec::new();
        self.prefix_tables = Vec::new();
        self.suffix_tables = Vec::new();
        self.range = Vec::new();
        Ok(())
    }

    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        let round = self.progress.bound();
        if self.progress.bound() < self.address_bits() {
            let bind_dense = |table: &mut Polynomial<F>| {
                table.bind_with_order(challenge, BindingOrder::HighToLow);
            };
            #[cfg(feature = "parallel")]
            let ((), ()) = rayon::join(
                || self.prefix_tables.par_iter_mut().for_each(bind_dense),
                || {
                    self.suffix_tables.par_iter_mut().for_each(|(_, suffixes)| {
                        suffixes.iter_mut().for_each(bind_dense);
                    });
                },
            );
            #[cfg(not(feature = "parallel"))]
            {
                self.prefix_tables.iter_mut().for_each(bind_dense);
                self.suffix_tables
                    .iter_mut()
                    .for_each(|(_, suffixes)| suffixes.iter_mut().for_each(bind_dense));
            }
            self.raf_left.bind(challenge);
            self.raf_right.bind(challenge);
            self.raf_identity.bind(challenge);
            if CANONICAL_INSTRUCTION_ADDRESS {
                self.raf_upper_all_ones.bind(challenge);
            }
            self.phase_challenges.push(challenge);

            if self.phase_challenges.len() == CHUNK_LEN {
                let phase = self.progress.bound() / CHUNK_LEN;
                self.v_tables.push(eq_table(&self.phase_challenges));
                for (&index, table) in self.prefix_indices.iter().zip(&self.prefix_tables) {
                    self.prefix_checkpoints[index] = PrefixEval::from(table.evals()[0]);
                }
                self.raf_left.checkpoint = self.raf_left.prefix.evals()[0];
                self.raf_right.checkpoint = self.raf_right.prefix.evals()[0];
                self.raf_identity.checkpoint = self.raf_identity.prefix.evals()[0];
                if CANONICAL_INSTRUCTION_ADDRESS {
                    self.raf_upper_all_ones.checkpoint = self.raf_upper_all_ones.prefix.evals()[0];
                }

                if phase + 1 < self.phases() {
                    self.init_phase(phase + 1)
                } else {
                    self.init_cycle_rounds()
                }
                .map_err(rows_unavailable)?;
            }
        } else {
            let cycle = self
                .cycle
                .as_mut()
                .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
            cycle
                .gruen
                .round_mut()
                .ok_or_else(|| eq_round_error(round))?
                .bind(challenge);
            cycle.combined_val.bind(challenge);
            match &mut cycle.ra {
                CycleRa::Lazy(ra) => ra.bind(challenge),
                CycleRa::Streamed(ra) => {
                    if let Some(dense) = ra.bind(challenge).map_err(rows_unavailable)? {
                        cycle.ra = CycleRa::Lazy(dense);
                    }
                }
            }
            self.cycle_challenges.push(challenge);
        }
        self.progress.advance();
        Ok(())
    }
}

impl<F: JoltField> ProveRounds<F> for OptimizedInstructionReadRafKernel<F> {
    fn num_rounds(&self) -> usize {
        self.dimensions.sumcheck_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        if self.progress.bound() < self.address_bits() {
            Ok(self.address_message(previous_claim))
        } else {
            self.cycle_message(round, previous_claim)
        }
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}
