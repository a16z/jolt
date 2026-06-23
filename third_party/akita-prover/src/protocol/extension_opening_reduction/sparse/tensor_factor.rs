use super::*;

#[derive(Debug, Clone)]
struct TensorFactorTransition<E: FieldCore> {
    zero: Vec<Vec<E>>,
    one: Vec<Vec<E>>,
}

/// Lazy transparent tensor factor for sparse extension-opening terms.
///
/// This stores the exact multilinear folding state for
/// `A_eta(w) = sum_u eq(u, eta) * coord_u(eq(r_tail, w))` without relying on
/// `coord_u` being extension-linear. Once the sparse low block has been folded,
/// it materializes into the ordinary dense factor table and rejoins the shared
/// reduction path.
#[derive(Debug, Clone)]
pub(in crate::protocol::extension_opening_reduction) struct TensorEqualityFactor<E: FieldCore> {
    table_vars: usize,
    round: usize,
    materialize_at: usize,
    prefix_state: Vec<E>,
    transitions: Vec<TensorFactorTransition<E>>,
    suffix_tables: Vec<Vec<E>>,
    low_states: Vec<Vec<E>>,
}

impl<E: FieldCore> TensorEqualityFactor<E> {
    pub(super) fn new<F>(
        tail_point: Vec<E>,
        eta: Vec<E>,
        materialize_at: usize,
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        let (split_bits, width) = tensor_opening_split::<F, E>()?;
        if eta.len() != split_bits {
            return Err(AkitaError::InvalidSize {
                expected: split_bits,
                actual: eta.len(),
            });
        }
        if materialize_at > tail_point.len() {
            return Err(AkitaError::InvalidSize {
                expected: tail_point.len(),
                actual: materialize_at,
            });
        }
        checked_table_len(tail_point.len())?;
        checked_table_len(tail_point.len() - materialize_at)?;

        let eta_weights = EqPolynomial::evals(&eta)?;
        let basis = (0..width)
            .map(|idx| {
                let mut coords = vec![F::zero(); width];
                coords[idx] = F::one();
                E::from_base_slice(&coords)
            })
            .collect::<Vec<_>>();
        let one_coords = E::one().to_base_vec();
        if one_coords.len() != width {
            return Err(AkitaError::InvalidSize {
                expected: width,
                actual: one_coords.len(),
            });
        }
        let prefix_state = one_coords.into_iter().map(E::lift_base).collect::<Vec<_>>();

        let transitions = tail_point[..materialize_at]
            .iter()
            .copied()
            .map(|tail| Self::transition::<F>(&basis, tail, width))
            .collect::<Result<Vec<_>, _>>()?;
        let suffix_eq = EqPolynomial::evals(&tail_point[materialize_at..])?;
        let suffix_tables = basis
            .iter()
            .map(|&basis_elem| {
                suffix_eq
                    .iter()
                    .copied()
                    .map(|suffix| {
                        project_tensor_factor_value::<F, E>(
                            basis_elem * suffix,
                            &eta_weights,
                            width,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut factor = Self {
            table_vars: tail_point.len(),
            round: 0,
            materialize_at,
            prefix_state,
            transitions,
            suffix_tables,
            low_states: Vec::new(),
        };
        factor.rebuild_low_states();
        Ok(factor)
    }

    fn transition<F>(
        basis: &[E],
        tail: E,
        width: usize,
    ) -> Result<TensorFactorTransition<E>, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        let tail_zero = E::one() - tail;
        let tail_one = tail;
        let mut zero = vec![vec![E::zero(); width]; width];
        let mut one = vec![vec![E::zero(); width]; width];
        for (src_idx, &basis_elem) in basis.iter().enumerate() {
            let zero_coords = (basis_elem * tail_zero).to_base_vec();
            let one_coords = (basis_elem * tail_one).to_base_vec();
            if zero_coords.len() != width || one_coords.len() != width {
                return Err(AkitaError::InvalidSize {
                    expected: width,
                    actual: zero_coords.len().max(one_coords.len()),
                });
            }
            for dst_idx in 0..width {
                zero[src_idx][dst_idx] = E::lift_base(zero_coords[dst_idx]);
                one[src_idx][dst_idx] = E::lift_base(one_coords[dst_idx]);
            }
        }
        Ok(TensorFactorTransition { zero, one })
    }

    pub(super) fn len(&self) -> usize {
        1usize << (self.table_vars - self.round)
    }

    pub(super) fn is_ready_to_materialize(&self) -> bool {
        self.round >= self.materialize_at
    }

    fn apply_transition(
        state: &[E],
        transition: &TensorFactorTransition<E>,
        challenge: E,
    ) -> Vec<E> {
        let width = state.len();
        let one_minus = E::one() - challenge;
        let mut next = vec![E::zero(); width];
        for (src_idx, &src) in state.iter().enumerate() {
            if src == E::zero() {
                continue;
            }
            for (dst_idx, dst) in next.iter_mut().enumerate() {
                let step = transition.zero[src_idx][dst_idx] * one_minus
                    + transition.one[src_idx][dst_idx] * challenge;
                *dst += src * step;
            }
        }
        next
    }

    fn apply_boolean_transition(
        state: &[E],
        transition: &TensorFactorTransition<E>,
        bit: usize,
    ) -> Vec<E> {
        let width = state.len();
        let matrix = if bit == 0 {
            &transition.zero
        } else {
            &transition.one
        };
        let mut next = vec![E::zero(); width];
        for (src_idx, &src) in state.iter().enumerate() {
            if src == E::zero() {
                continue;
            }
            for (dst_idx, dst) in next.iter_mut().enumerate() {
                *dst += src * matrix[src_idx][dst_idx];
            }
        }
        next
    }

    fn rebuild_low_states(&mut self) {
        let low_bits = self.materialize_at.saturating_sub(self.round);
        if low_bits == 0 {
            self.low_states.clear();
            return;
        }
        let count = 1usize << low_bits;
        let mut low_states = Vec::with_capacity(count);
        for low in 0..count {
            let mut state = self.prefix_state.clone();
            for bit_idx in 0..low_bits {
                let bit = (low >> bit_idx) & 1;
                state = Self::apply_boolean_transition(
                    &state,
                    &self.transitions[self.round + bit_idx],
                    bit,
                );
            }
            low_states.push(state);
        }
        self.low_states = low_states;
    }

    fn eval_state_at_suffix(&self, state: &[E], suffix_index: usize) -> E {
        self.suffix_tables
            .iter()
            .zip(state.iter().copied())
            .fold(E::zero(), |acc, (table, coeff)| {
                acc + coeff * table[suffix_index]
            })
    }

    pub(super) fn factor_at_index(&self, index: usize) -> E {
        let low_bits = self.materialize_at.saturating_sub(self.round);
        if low_bits == 0 {
            return self.eval_state_at_suffix(&self.prefix_state, index);
        }
        let low_mask = (1usize << low_bits) - 1;
        let low = index & low_mask;
        let suffix_index = index >> low_bits;
        self.eval_state_at_suffix(&self.low_states[low], suffix_index)
    }

    pub(super) fn fold_in_place(&mut self, r_round: E) {
        if self.len() <= 1 {
            return;
        }
        debug_assert!(self.round < self.materialize_at);
        self.prefix_state =
            Self::apply_transition(&self.prefix_state, &self.transitions[self.round], r_round);
        self.round += 1;
        self.rebuild_low_states();
    }

    pub(super) fn materialize_dense(&self) -> Vec<E> {
        debug_assert!(self.is_ready_to_materialize());
        let suffix_len = self.suffix_tables.first().map(Vec::len).unwrap_or(0);
        let _span = tracing::debug_span!(
            "TensorEqualityFactor::materialize_dense",
            suffix_len,
            width = self.prefix_state.len()
        )
        .entered();
        #[cfg(feature = "parallel")]
        {
            (0..suffix_len)
                .into_par_iter()
                .map(|idx| self.eval_state_at_suffix(&self.prefix_state, idx))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..suffix_len)
                .map(|idx| self.eval_state_at_suffix(&self.prefix_state, idx))
                .collect()
        }
    }
}

impl<E: FieldCore + HasUnreducedOps> TensorEqualityFactor<E> {
    /// Factor inner product `sum_i state[i] * suffix_tables[i][suffix_index]`,
    /// reducing once at the end when the field's product accumulator is exact
    /// w.r.t. `Mul`, and otherwise falling back to the per-term
    /// [`Self::eval_state_at_suffix`].
    ///
    /// On the exact path (e.g. the fp32 `FpExt4<Fp32>` campaign field)
    /// each product is widened into `E::ProductAccum` and the
    /// `state.len() == E::EXT_DEGREE` terms are summed before a single
    /// `reduce_product_accum`. The per-coefficient reduction is additive over
    /// the accumulator and the wide sum cannot overflow (`EXT_DEGREE` is a small
    /// power of two — 4 here — far below the accumulator's >= 2^63 headroom), so
    /// the result is byte-identical to `eval_state_at_suffix`.
    ///
    /// Fields whose wide accumulator is lossy versus `Mul` leave
    /// `DELAYED_PRODUCT_SUM_IS_EXACT` at `false` and take the per-term path, so
    /// the emitted factor, and the proof, stay unchanged. `FpExt2<Fp64>` opts into
    /// the exact path only because its accumulator keeps the carry above bit
    /// 128 explicitly.
    ///
    /// The two factor values `a0` (state `low_zero`) and `a1` (state `low_one`)
    /// always share the same `suffix_index`, so both inner products read the
    /// same `suffix_tables[j][suffix_index]` column. They are fused into one
    /// pass over `j` that loads each column entry once and feeds it into both
    /// delayed accumulators, halving the column loads and tightening the loop
    /// without changing the accumulation order, so the result is byte-identical
    /// to two independent evaluations.
    pub(super) fn factor_pair(&self, pair: usize) -> (E, E) {
        let low_bits = self.materialize_at - self.round;
        debug_assert!(low_bits > 0);
        let rest_low_bits = low_bits - 1;
        let low_mask = (1usize << rest_low_bits).saturating_sub(1);
        let low_rest = pair & low_mask;
        let suffix_index = pair >> rest_low_bits;
        let low_zero = low_rest << 1;
        let low_one = low_zero | 1;
        let state_zero = &self.low_states[low_zero];
        let state_one = &self.low_states[low_one];

        if !E::DELAYED_PRODUCT_SUM_IS_EXACT {
            return (
                self.eval_state_at_suffix(state_zero, suffix_index),
                self.eval_state_at_suffix(state_one, suffix_index),
            );
        }

        let mut accum_zero = E::ProductAccum::zero();
        let mut accum_one = E::ProductAccum::zero();
        for ((table, &coeff_zero), &coeff_one) in self
            .suffix_tables
            .iter()
            .zip(state_zero.iter())
            .zip(state_one.iter())
        {
            let column = table[suffix_index];
            accum_zero += coeff_zero.mul_to_product_accum(column);
            accum_one += coeff_one.mul_to_product_accum(column);
        }
        (
            E::reduce_product_accum(accum_zero),
            E::reduce_product_accum(accum_one),
        )
    }
}

/// Transparent factor for a sparse-witness term.
///
/// The lazy [`TensorEqualityFactor`] is only ever paired with a sparse witness,
/// so it lives inside the sparse case rather than as a standalone factor. This
/// is what makes the `(dense witness, tensor factor)` combination unrepresentable.
#[derive(Debug, Clone)]
pub(in crate::protocol::extension_opening_reduction) enum SparseFactor<E: FieldCore> {
    Dense(Vec<E>),
    Tensor(TensorEqualityFactor<E>),
}
