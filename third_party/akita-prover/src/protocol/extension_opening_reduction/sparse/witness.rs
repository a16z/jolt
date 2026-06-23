use super::*;

/// Sparse transformed-witness evaluations for extension-opening reduction.
#[derive(Debug, Clone)]
pub struct SparseExtensionOpeningWitness<E: FieldCore> {
    pub(super) table_len: usize,
    pub(super) entries: Vec<(usize, E)>,
    /// Number of upcoming folds guaranteed to leave at most one entry per pair
    /// (no merges). While positive, the merge-free fast path is exact: the round
    /// message has a closed form and the witness folds in place without
    /// reallocating. Derived once at construction from the entry spacing; see
    /// [`Self::leading_merge_free_rounds`].
    pub(super) merge_free_rounds_left: usize,
}

#[cfg(feature = "parallel")]
const SPARSE_PARALLEL_ENTRY_THRESHOLD: usize = 1 << 14;
#[cfg(feature = "parallel")]
const SPARSE_PARALLEL_CHUNKS_PER_THREAD: usize = 4;

impl<E: FieldCore> SparseExtensionOpeningWitness<E> {
    /// Construct a sparse witness table from `(index, value)` entries.
    ///
    /// Duplicate indices are combined, and zero entries are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if `table_len` is not a nonzero power of two or if an
    /// entry index is out of range.
    pub fn new(table_len: usize, mut entries: Vec<(usize, E)>) -> Result<Self, AkitaError> {
        let _span = tracing::debug_span!(
            "SparseExtensionOpeningWitness::new",
            table_len,
            entries_len = entries.len()
        )
        .entered();
        entries.sort_unstable_by_key(|(idx, _)| *idx);
        Self::from_sorted_entries(table_len, entries)
    }

    /// Construct a sparse witness table from entries already sorted by index.
    ///
    /// Duplicate indices are combined, and zero entries are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if `table_len` is not a nonzero power of two, if an
    /// entry index is out of range, or if entries are not sorted by index.
    pub fn from_sorted_entries(
        table_len: usize,
        entries: Vec<(usize, E)>,
    ) -> Result<Self, AkitaError> {
        let _span = tracing::debug_span!(
            "SparseExtensionOpeningWitness::from_sorted_entries",
            table_len,
            entries_len = entries.len()
        )
        .entered();
        if table_len == 0 || !table_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "sparse extension-opening witness length must be a nonzero power of two"
                    .to_string(),
            ));
        }
        let mut combined: Vec<(usize, E)> = Vec::with_capacity(entries.len());
        let mut previous_idx = None;
        for (idx, value) in entries {
            if idx >= table_len {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness index out of range".to_string(),
                ));
            }
            if previous_idx.is_some_and(|previous| idx < previous) {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness sorted constructor received unsorted entries"
                        .to_string(),
                ));
            }
            previous_idx = Some(idx);
            if value == E::zero() {
                continue;
            }
            if let Some((last_idx, last_value)) = combined.last_mut() {
                if *last_idx == idx {
                    *last_value += value;
                    if *last_value == E::zero() {
                        combined.pop();
                    }
                    continue;
                }
            }
            combined.push((idx, value));
        }
        let merge_free_rounds_left = Self::leading_merge_free_rounds(table_len, &combined);
        Ok(Self {
            table_len,
            entries: combined,
            merge_free_rounds_left,
        })
    }

    /// Construct a sparse witness table from entries already normalized as
    /// strictly sorted, unique, nonzero `(index, value)` pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if `table_len` is not a nonzero power of two, if an
    /// entry index is out of range, if an entry is zero, or if entries are not
    /// strictly sorted by index.
    pub fn from_sorted_unique_entries(
        table_len: usize,
        entries: Vec<(usize, E)>,
    ) -> Result<Self, AkitaError> {
        let _span = tracing::debug_span!(
            "SparseExtensionOpeningWitness::from_sorted_unique_entries",
            table_len,
            entries_len = entries.len()
        )
        .entered();
        if table_len == 0 || !table_len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "sparse extension-opening witness length must be a nonzero power of two"
                    .to_string(),
            ));
        }
        let mut previous_idx = None;
        for &(idx, value) in &entries {
            if idx >= table_len {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness index out of range".to_string(),
                ));
            }
            if previous_idx.is_some_and(|previous| idx <= previous) {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness unique constructor received duplicate or unsorted entries"
                        .to_string(),
                ));
            }
            if value == E::zero() {
                return Err(AkitaError::InvalidInput(
                    "sparse extension-opening witness unique constructor received a zero entry"
                        .to_string(),
                ));
            }
            previous_idx = Some(idx);
        }
        let merge_free_rounds_left = Self::leading_merge_free_rounds(table_len, &entries);
        Ok(Self {
            table_len,
            entries,
            merge_free_rounds_left,
        })
    }

    /// Number of leading folds guaranteed to be merge-free (every pair keeps at
    /// most one entry).
    ///
    /// Two adjacent entries `tᵢ < tⱼ` first land in the same folded index at fold
    /// `bit_length(tᵢ ⊕ tⱼ)`, so the minimum over neighbors is the first merging
    /// fold and the guaranteed merge-free run is that minus one. With fewer than
    /// two entries nothing ever merges, so the whole reduction stays merge-free.
    pub(super) fn leading_merge_free_rounds(table_len: usize, entries: &[(usize, E)]) -> usize {
        let total = table_len.trailing_zeros() as usize;
        if entries.len() < 2 {
            return total;
        }
        let mut first_merge = usize::BITS;
        for window in entries.windows(2) {
            let diff = window[0].0 ^ window[1].0;
            let bit_length = usize::BITS - diff.leading_zeros();
            if bit_length < first_merge {
                first_merge = bit_length;
            }
        }
        (first_merge as usize).saturating_sub(1).min(total)
    }

    /// Dense table length represented by this sparse witness.
    pub fn table_len(&self) -> usize {
        self.table_len
    }

    /// Nonzero sparse entries, sorted by table index.
    pub fn entries(&self) -> &[(usize, E)] {
        &self.entries
    }

    /// Combine sparse witnesses over the same table domain.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no terms or if the sparse witnesses have
    /// different table lengths.
    pub fn linear_combination<'a, I>(terms: I) -> Result<Self, AkitaError>
    where
        I: IntoIterator<Item = (E, &'a Self)>,
        E: 'a,
    {
        let _span =
            tracing::debug_span!("SparseExtensionOpeningWitness::linear_combination").entered();
        let mut table_len = None;
        let mut entries = Vec::new();
        {
            let _span = tracing::debug_span!("sparse_extension_witness_lc_collect").entered();
            for (coeff, witness) in terms {
                match table_len {
                    Some(len) if len != witness.table_len() => {
                        return Err(AkitaError::InvalidSize {
                            expected: len,
                            actual: witness.table_len(),
                        });
                    }
                    None => table_len = Some(witness.table_len()),
                    Some(_) => {}
                }
                entries.extend(
                    witness
                        .entries()
                        .iter()
                        .map(|&(idx, value)| (idx, value * coeff)),
                );
            }
        }
        let table_len = table_len.ok_or_else(|| {
            AkitaError::InvalidInput(
                "sparse extension-opening witness combination requires at least one term"
                    .to_string(),
            )
        })?;
        let _span = tracing::debug_span!(
            "sparse_extension_witness_lc_normalize",
            table_len,
            entries_len = entries.len()
        )
        .entered();
        Self::new(table_len, entries)
    }

    pub(super) fn claim_with_factor(&self, factor_evals: &[E]) -> Result<E, AkitaError> {
        if factor_evals.len() != self.table_len {
            return Err(AkitaError::InvalidSize {
                expected: self.table_len,
                actual: factor_evals.len(),
            });
        }
        Ok(self.entries.iter().fold(E::zero(), |acc, &(idx, value)| {
            acc + value * factor_evals[idx]
        }))
    }

    pub(super) fn claim_with_factor_fn<P>(&self, factor_at: P) -> E
    where
        P: Fn(usize) -> E,
    {
        self.entries
            .iter()
            .fold(E::zero(), |acc, &(idx, value)| acc + value * factor_at(idx))
    }

    pub(super) fn final_eval(&self) -> Option<E> {
        if self.table_len != 1 {
            return None;
        }
        Some(
            self.entries
                .first()
                .map(|(_, value)| *value)
                .unwrap_or(E::zero()),
        )
    }

    pub(super) fn fold_entries(entries: &[(usize, E)], r_round: E) -> Vec<(usize, E)> {
        let one_minus = E::one() - r_round;
        let mut folded = Vec::with_capacity(entries.len());
        let mut i = 0;
        while i < entries.len() {
            let pair = entries[i].0 / 2;
            let mut value = E::zero();
            while i < entries.len() && entries[i].0 / 2 == pair {
                let (idx, entry_value) = entries[i];
                value += if idx & 1 == 0 {
                    entry_value * one_minus
                } else {
                    entry_value * r_round
                };
                i += 1;
            }
            if value != E::zero() {
                folded.push((pair, value));
            }
        }
        folded
    }

    #[cfg(feature = "parallel")]
    pub(super) fn parallel_chunk_size(len: usize) -> usize {
        let target_chunks = rayon::current_num_threads() * SPARSE_PARALLEL_CHUNKS_PER_THREAD;
        len.div_ceil(target_chunks)
            .max(SPARSE_PARALLEL_ENTRY_THRESHOLD)
    }

    #[cfg(feature = "parallel")]
    pub(super) fn pair_aligned_ranges(&self) -> Vec<(usize, usize)> {
        let len = self.entries.len();
        let chunk_size = Self::parallel_chunk_size(len);
        let mut ranges = Vec::with_capacity(len.div_ceil(chunk_size));
        let mut start = 0;
        while start < len {
            let mut end = (start + chunk_size).min(len);
            if end < len {
                let split_pair = self.entries[end].0 / 2;
                while end < len && self.entries[end].0 / 2 == split_pair {
                    end += 1;
                }
            }
            ranges.push((start, end));
            start = end;
        }
        ranges
    }
}

impl<E: FieldCore + HasUnreducedOps> SparseExtensionOpeningWitness<E> {
    pub(super) fn accumulate_entries_with_factor<P>(
        entries: &[(usize, E)],
        coeff: E,
        merge_free: bool,
        factor_pair: &P,
    ) -> (E, E)
    where
        P: Fn(usize) -> (E, E) + Sync,
    {
        // Honor `DELAYED_PRODUCT_SUM_IS_EXACT`: only sum wide products and reduce
        // once for fields whose accumulator is proven exact; otherwise reduce per
        // term so the coefficients stay byte-identical to `Mul`, matching the
        // dense round and `TensorEqualityFactor::factor_pair`.
        //
        // `merge_free` selects the closed-form pass valid while every pair has at
        // most one entry; it accumulates the identical products in the identical
        // order, so both paths agree bit-for-bit.
        let (constant, quadratic) = match (E::DELAYED_PRODUCT_SUM_IS_EXACT, merge_free) {
            (true, false) => Self::accumulate_entries_with_factor_using::<DelayedDeg2<E>, P>(
                entries,
                factor_pair,
            ),
            (false, false) => {
                Self::accumulate_entries_with_factor_using::<DirectDeg2<E>, P>(entries, factor_pair)
            }
            (true, true) => {
                Self::accumulate_entries_merge_free_using::<DelayedDeg2<E>, P>(entries, factor_pair)
            }
            (false, true) => {
                Self::accumulate_entries_merge_free_using::<DirectDeg2<E>, P>(entries, factor_pair)
            }
        };
        (coeff * constant, coeff * quadratic)
    }

    pub(super) fn accumulate_entries_with_factor_using<A, P>(
        entries: &[(usize, E)],
        factor_pair: &P,
    ) -> (E, E)
    where
        A: Deg2RoundAccum<E>,
        P: Fn(usize) -> (E, E) + Sync,
    {
        let mut acc = A::zero();
        let mut i = 0;
        while i < entries.len() {
            let pair = entries[i].0 / 2;
            let mut w0 = E::zero();
            let mut w1 = E::zero();
            while i < entries.len() && entries[i].0 / 2 == pair {
                let (idx, value) = entries[i];
                if idx & 1 == 0 {
                    w0 += value;
                } else {
                    w1 += value;
                }
                i += 1;
            }

            let (a0, a1) = factor_pair(pair);
            let da = a1 - a0;
            if w0 == E::zero() {
                acc.add_quadratic_product(w1, da);
            } else {
                acc.add_constant_product(w0, a0);
                acc.add_quadratic_product(w1 - w0, da);
            }
        }

        acc.finish()
    }

    /// Closed-form merge-free specialization of
    /// [`Self::accumulate_entries_with_factor_using`].
    ///
    /// Valid only while every pair holds at most one entry (the leading
    /// `merge_free_rounds_left` rounds). It is the grouped loop with the inner
    /// pair-grouping `while` removed: each pair contributes exactly its single
    /// child, placed at `w0` or `w1` by parity, so the products, their order, and
    /// the `Deg2RoundAccum` calls are byte-identical to the general path.
    pub(super) fn accumulate_entries_merge_free_using<A, P>(
        entries: &[(usize, E)],
        factor_pair: &P,
    ) -> (E, E)
    where
        A: Deg2RoundAccum<E>,
        P: Fn(usize) -> (E, E) + Sync,
    {
        let mut acc = A::zero();
        for &(idx, value) in entries {
            let (a0, a1) = factor_pair(idx >> 1);
            let da = a1 - a0;
            if idx & 1 == 0 {
                // even child: w0 = value, w1 = 0.
                acc.add_constant_product(value, a0);
                acc.add_quadratic_product(E::zero() - value, da);
            } else {
                // odd child: w0 = 0, w1 = value.
                acc.add_quadratic_product(value, da);
            }
        }
        acc.finish()
    }

    pub(super) fn accumulate_entries(
        entries: &[(usize, E)],
        factor_evals: &[E],
        coeff: E,
        merge_free: bool,
    ) -> (E, E) {
        Self::accumulate_entries_with_factor(entries, coeff, merge_free, &|pair| {
            (factor_evals[2 * pair], factor_evals[2 * pair + 1])
        })
    }

    pub(super) fn accumulate_round(
        &self,
        factor_evals: &[E],
        coeff: E,
        constant: &mut E,
        quadratic: &mut E,
    ) {
        let _span = tracing::trace_span!(
            "SparseExtensionOpeningWitness::accumulate_round",
            table_len = self.table_len,
            entries_len = self.entries.len()
        )
        .entered();
        let merge_free = self.merge_free_rounds_left > 0;
        #[cfg(feature = "parallel")]
        let (round_constant, round_quadratic) =
            if self.entries.len() >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
                if merge_free {
                    let chunk_size = Self::parallel_chunk_size(self.entries.len());
                    self.entries
                        .par_chunks(chunk_size)
                        .map(|entries| Self::accumulate_entries(entries, factor_evals, coeff, true))
                        .reduce(
                            || (E::zero(), E::zero()),
                            |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                        )
                } else {
                    self.pair_aligned_ranges()
                        .into_par_iter()
                        .map(|(start, end)| {
                            Self::accumulate_entries(
                                &self.entries[start..end],
                                factor_evals,
                                coeff,
                                false,
                            )
                        })
                        .reduce(
                            || (E::zero(), E::zero()),
                            |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                        )
                }
            } else {
                Self::accumulate_entries(&self.entries, factor_evals, coeff, merge_free)
            };
        #[cfg(not(feature = "parallel"))]
        let (round_constant, round_quadratic) =
            Self::accumulate_entries(&self.entries, factor_evals, coeff, merge_free);
        *constant += round_constant;
        *quadratic += round_quadratic;
    }

    pub(super) fn accumulate_round_with_factor<P>(
        &self,
        coeff: E,
        constant: &mut E,
        quadratic: &mut E,
        factor_pair: P,
    ) where
        P: Fn(usize) -> (E, E) + Sync,
    {
        let _span = tracing::trace_span!(
            "SparseExtensionOpeningWitness::accumulate_round_with_factor",
            table_len = self.table_len,
            entries_len = self.entries.len()
        )
        .entered();
        let merge_free = self.merge_free_rounds_left > 0;
        #[cfg(feature = "parallel")]
        let (round_constant, round_quadratic) =
            if self.entries.len() >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
                if merge_free {
                    let chunk_size = Self::parallel_chunk_size(self.entries.len());
                    self.entries
                        .par_chunks(chunk_size)
                        .map(|entries| {
                            Self::accumulate_entries_with_factor(entries, coeff, true, &factor_pair)
                        })
                        .reduce(
                            || (E::zero(), E::zero()),
                            |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                        )
                } else {
                    self.pair_aligned_ranges()
                        .into_par_iter()
                        .map(|(start, end)| {
                            Self::accumulate_entries_with_factor(
                                &self.entries[start..end],
                                coeff,
                                false,
                                &factor_pair,
                            )
                        })
                        .reduce(
                            || (E::zero(), E::zero()),
                            |lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1),
                        )
                }
            } else {
                Self::accumulate_entries_with_factor(&self.entries, coeff, merge_free, &factor_pair)
            };
        #[cfg(not(feature = "parallel"))]
        let (round_constant, round_quadratic) =
            Self::accumulate_entries_with_factor(&self.entries, coeff, merge_free, &factor_pair);
        *constant += round_constant;
        *quadratic += round_quadratic;
    }

    /// Fold the witness by one merge-free round AND accumulate the *next*
    /// round's `(constant, quadratic)` in a single sweep over the entries.
    ///
    /// The sparse analogue of the dense [`fused_fold_and_accumulate`]: instead of
    /// one pass to accumulate and a second to fold, fold each entry into the next
    /// round in place and immediately add its next-round contribution. The caller
    /// must have already folded the factor to the next round, so
    /// `next_factor_pair` returns the next round's factor children.
    ///
    /// Precondition: at least two merge-free rounds remain when called, so both
    /// this fold and the look-ahead accumulation stay in the merge-free regime
    /// (every pair still holds at most one entry). Returns the *unscaled*
    /// next-round coefficients; the caller applies the term coefficient.
    pub(super) fn fused_fold_accumulate_merge_free<P>(
        &mut self,
        r_round: E,
        next_factor_pair: &P,
    ) -> (E, E)
    where
        P: Fn(usize) -> (E, E) + Sync,
    {
        let (constant, quadratic) = if E::DELAYED_PRODUCT_SUM_IS_EXACT {
            self.fused_fold_accumulate_merge_free_using::<DelayedDeg2<E>, P>(
                r_round,
                next_factor_pair,
            )
        } else {
            self.fused_fold_accumulate_merge_free_using::<DirectDeg2<E>, P>(
                r_round,
                next_factor_pair,
            )
        };
        self.table_len /= 2;
        self.merge_free_rounds_left -= 1;
        (constant, quadratic)
    }

    pub(super) fn fused_fold_accumulate_merge_free_using<A, P>(
        &mut self,
        r_round: E,
        next_factor_pair: &P,
    ) -> (E, E)
    where
        A: Deg2RoundAccum<E>,
        P: Fn(usize) -> (E, E) + Sync,
    {
        let one_minus = E::one() - r_round;
        // One pass per entry: fold it into the next round in place, then add its
        // next-round merge-free contribution (each pair still holds one entry, so
        // the closed form of `accumulate_entries_merge_free_using` applies to the
        // just-folded entry).
        let fold_accumulate = |chunk: &mut [(usize, E)]| {
            let mut acc = A::zero();
            for entry in chunk {
                let (idx, value) = *entry;
                let folded = if idx & 1 == 0 {
                    value * one_minus
                } else {
                    value * r_round
                };
                let folded_idx = idx >> 1;
                *entry = (folded_idx, folded);
                let (a0, a1) = next_factor_pair(folded_idx >> 1);
                let da = a1 - a0;
                if folded_idx & 1 == 0 {
                    acc.add_constant_product(folded, a0);
                    acc.add_quadratic_product(E::zero() - folded, da);
                } else {
                    acc.add_quadratic_product(folded, da);
                }
            }
            acc
        };

        #[cfg(feature = "parallel")]
        {
            if self.entries.len() >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
                let chunk_size = Self::parallel_chunk_size(self.entries.len());
                return self
                    .entries
                    .par_chunks_mut(chunk_size)
                    .map(fold_accumulate)
                    .reduce(A::zero, A::merge)
                    .finish();
            }
        }
        fold_accumulate(self.entries.as_mut_slice()).finish()
    }
}

impl<E: FieldCore> SparseExtensionOpeningWitness<E> {
    pub(super) fn fold_in_place(&mut self, r_round: E) {
        let _span = tracing::trace_span!(
            "SparseExtensionOpeningWitness::fold_in_place",
            table_len = self.table_len,
            entries_len = self.entries.len()
        )
        .entered();
        if self.table_len <= 1 {
            return;
        }
        // Merge-free regime: no pair merges this fold, so each entry just drops
        // its low tail bit and scales by the matching challenge weight. Fold in
        // place — no reallocation, no dedup, no pair-range scan.
        if self.merge_free_rounds_left > 0 {
            self.fold_in_place_merge_free(r_round);
            self.table_len /= 2;
            self.merge_free_rounds_left -= 1;
            return;
        }
        #[cfg(feature = "parallel")]
        let folded = if self.entries.len() >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
            let chunks = self
                .pair_aligned_ranges()
                .into_par_iter()
                .map(|(start, end)| Self::fold_entries(&self.entries[start..end], r_round))
                .collect::<Vec<_>>();
            let len = chunks.iter().map(Vec::len).sum();
            let mut folded = Vec::with_capacity(len);
            for chunk in chunks {
                folded.extend(chunk);
            }
            folded
        } else {
            Self::fold_entries(&self.entries, r_round)
        };
        #[cfg(not(feature = "parallel"))]
        let folded = Self::fold_entries(&self.entries, r_round);
        self.table_len /= 2;
        self.entries = folded;
    }

    /// Alloc-free in-place fold for the merge-free regime.
    ///
    /// No pair has two entries, so folding never combines values: each entry
    /// `(idx, value)` becomes `(idx >> 1, value · weight)` with `weight` the
    /// even/odd challenge factor. Byte-identical to [`Self::fold_entries`] when
    /// every occupied pair holds one entry, and trivially parallel (no cross-entry
    /// dependency, no merges).
    pub(super) fn fold_in_place_merge_free(&mut self, r_round: E) {
        let one_minus = E::one() - r_round;
        let fold_one = |entry: &mut (usize, E)| {
            let (idx, value) = *entry;
            let folded = if idx & 1 == 0 {
                value * one_minus
            } else {
                value * r_round
            };
            *entry = (idx >> 1, folded);
        };
        #[cfg(feature = "parallel")]
        {
            let len = self.entries.len();
            if len >= SPARSE_PARALLEL_ENTRY_THRESHOLD {
                let chunk_size = Self::parallel_chunk_size(len);
                self.entries
                    .par_chunks_mut(chunk_size)
                    .for_each(|chunk| chunk.iter_mut().for_each(fold_one));
                return;
            }
        }
        self.entries.iter_mut().for_each(fold_one);
    }
}
