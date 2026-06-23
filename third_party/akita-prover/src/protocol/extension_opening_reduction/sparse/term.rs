use super::*;

/// One term in an extension-opening reduction sumcheck.
///
/// A single dense term is the degenerate `1`-term case; the prover treats the
/// dense and batched paths uniformly.
#[derive(Debug, Clone)]
pub struct ExtensionOpeningReductionTerm<E: FieldCore> {
    pub(in crate::protocol::extension_opening_reduction) tables: ExtensionOpeningTables<E>,
    pub(in crate::protocol::extension_opening_reduction) coeff: E,
    /// `coeff`-scaled `(constant, quadratic)` for the next round, pre-computed
    /// by the fused fold in [`Self::ingest_challenge`] for the dense path.
    pub(in crate::protocol::extension_opening_reduction) cached_accumulate: Option<(E, E)>,
}

impl<E: FieldCore> ExtensionOpeningReductionTerm<E> {
    /// Construct one term `coeff * sum_x witness(x) * factor(x)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the witness/factor tables are malformed.
    pub fn new(witness_evals: Vec<E>, factor_evals: Vec<E>, coeff: E) -> Result<Self, AkitaError> {
        validate_reduction_tables(&witness_evals, &factor_evals)?;
        Ok(Self {
            tables: ExtensionOpeningTables::Dense {
                witness: witness_evals,
                factor: factor_evals,
            },
            coeff,
            cached_accumulate: None,
        })
    }

    /// Construct one sparse-witness term `coeff * sum_x witness(x) * factor(x)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the sparse witness and factor table shapes differ.
    pub fn new_sparse(
        witness_evals: SparseExtensionOpeningWitness<E>,
        factor_evals: Vec<E>,
        coeff: E,
    ) -> Result<Self, AkitaError> {
        if witness_evals.table_len() != factor_evals.len() {
            return Err(AkitaError::InvalidSize {
                expected: witness_evals.table_len(),
                actual: factor_evals.len(),
            });
        }
        Ok(Self {
            tables: ExtensionOpeningTables::Sparse {
                witness: witness_evals,
                factor: SparseFactor::Dense(factor_evals),
            },
            coeff,
            cached_accumulate: None,
        })
    }

    /// Construct one sparse-witness term with a lazy transparent tensor factor.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor factor shape and sparse witness domain
    /// differ, or if the tensor opening parameters are malformed.
    pub fn new_sparse_tensor_factor<F>(
        witness_evals: SparseExtensionOpeningWitness<E>,
        tail_point: Vec<E>,
        eta: Vec<E>,
        coeff: E,
        materialize_at: usize,
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        let factor = TensorEqualityFactor::new::<F>(tail_point, eta, materialize_at)?;
        if witness_evals.table_len() != factor.len() {
            return Err(AkitaError::InvalidSize {
                expected: witness_evals.table_len(),
                actual: factor.len(),
            });
        }
        let factor = if factor.is_ready_to_materialize() {
            SparseFactor::Dense(factor.materialize_dense())
        } else {
            SparseFactor::Tensor(factor)
        };
        Ok(Self {
            tables: ExtensionOpeningTables::Sparse {
                witness: witness_evals,
                factor,
            },
            coeff,
            cached_accumulate: None,
        })
    }

    /// Batching coefficient multiplying this term.
    pub fn coeff(&self) -> E {
        self.coeff
    }

    /// Return final folded witness/factor evaluations after all challenges.
    pub fn final_witness_and_factor_evals(&self) -> Option<(E, E)> {
        self.tables.final_witness_and_factor_evals()
    }
}

impl<E: FieldCore + HasUnreducedOps + HasOptimizedFold> ExtensionOpeningReductionTerm<E> {
    /// Add this term's `coeff`-scaled `(constant, quadratic)` round
    /// contribution into the shared accumulators.
    ///
    /// Consumes the cache filled by the previous round's fused fold when
    /// present; otherwise accumulates directly from the current tables (the
    /// first round, and every round of the sparse/tensor paths).
    pub(in crate::protocol::extension_opening_reduction) fn accumulate_into(
        &mut self,
        constant: &mut E,
        quadratic: &mut E,
    ) {
        match self.cached_accumulate.take() {
            Some((cached_constant, cached_quadratic)) => {
                *constant += cached_constant;
                *quadratic += cached_quadratic;
            }
            None => {
                self.tables
                    .accumulate_round(self.coeff, constant, quadratic);
            }
        }
    }

    /// Fold this term's tables by one sumcheck challenge.
    ///
    /// Two shapes fold and pre-compute the next round's `(constant, quadratic)`
    /// in a single pass, caching the `coeff`-scaled result:
    /// - a dense witness/factor with at least four entries, and
    /// - a sparse witness still inside its merge-free plateau, with at least two
    ///   merge-free rounds left so the look-ahead accumulation is also merge-free.
    ///
    /// Every other shape folds in place and clears the cache.
    pub(in crate::protocol::extension_opening_reduction) fn ingest_challenge(
        &mut self,
        r_round: E,
    ) {
        if self.tables.len() <= 1 {
            return;
        }
        let fused = match &mut self.tables {
            ExtensionOpeningTables::Dense { witness, factor } if witness.len() >= 4 => {
                Some(fused_fold_and_accumulate(witness, factor, r_round))
            }
            ExtensionOpeningTables::Sparse { witness, factor }
                if witness.merge_free_rounds_left >= 2 =>
            {
                Some(fused_fold_and_accumulate_sparse(witness, factor, r_round))
            }
            _ => None,
        };
        match fused {
            Some((constant, quadratic)) => {
                self.cached_accumulate = Some((self.coeff * constant, self.coeff * quadratic));
            }
            None => {
                self.tables.fold_in_place(r_round);
                self.cached_accumulate = None;
            }
        }
    }
}
