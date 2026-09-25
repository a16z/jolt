//! The sequential cycle walk driving the atomic extractors, and the
//! trace-backed implementation of the streaming pass.

use super::*;
use crate::consumer::ChunkVisitor;
use crate::witnesses::{Extract, ExtractIndexed, RaChunkSelector, ToField, WitnessEnv};
use jolt_riscv::JoltTraceRow as TraceRow;
#[cfg(feature = "parallel")]
use jolt_utils::par_collect_windows;
use std::ops::Range;

use crate::{BundleSource, RandomAccessRows, RowSource, WitnessBundle};
use std::sync::Arc;

impl<T: TraceSource> TraceBackend<T> {
    /// Materializes one cycle-domain witness column by walking the trace
    /// once; all per-witness logic lives on `W`.
    pub(crate) fn materialize_cycle<F: JoltField, W: Extract + ToField>(
        &self,
    ) -> Result<Vec<F>, WitnessError> {
        self.walk_cycles(|row, next, env| W::extract(row, next, env).map(ToField::to_field))
    }

    /// [`Self::materialize_cycle`] for indexed witness families; `index`
    /// selects the family member.
    pub(crate) fn materialize_cycle_indexed<
        F: JoltField,
        W: ExtractIndexed<I> + ToField,
        I: Copy + Send + Sync,
    >(
        &self,
        index: I,
    ) -> Result<Vec<F>, WitnessError> {
        self.walk_cycles(|row, next, env| {
            W::extract_indexed(index, row, next, env).map(ToField::to_field)
        })
    }

    /// Materializes one member of a one-hot RA decomposition as the flat
    /// address-major `(K x T)` grid, `K = 2^chunk_bits`: one cycle walk
    /// collecting the per-cycle hot addresses (`None` is a cold cycle),
    /// then a scatter of ones.
    ///
    /// The walk's padding (default no-op) rows coincide with the one-hot
    /// conventions by construction: a no-op's lookup index is 0 and `get_pc`
    /// short-circuits no-ops to slot 0, so instruction/bytecode grids pad to
    /// the address-0 chunk and RAM grids to cold cycles.
    pub(crate) fn materialize_one_hot<F, W>(
        &self,
        index: usize,
        chunks: usize,
        chunk_bits: usize,
    ) -> Result<Vec<F>, WitnessError>
    where
        F: JoltField,
        W: ExtractIndexed<RaChunkSelector> + Into<Option<usize>>,
    {
        let selector = RaChunkSelector::new(index, chunks, chunk_bits)?;
        let cycles = checked_pow2(self.config.log_t)?;
        let len = checked_dense_grid_len::<F>(checked_pow2(chunk_bits)?, cycles)?;
        let hot_addresses: Vec<Option<usize>> = self.walk_cycles(|row, next, env| {
            W::extract_indexed(selector, row, next, env).map(W::into)
        })?;
        // The selector's mask bounds every hot address below `2^chunk_bits`.
        let mut values = jolt_utils::unsafe_allocate_zero_vec(len);
        for (cycle, address) in hot_addresses.into_iter().enumerate() {
            if let Some(address) = address {
                values[address * cycles + cycle] = F::one();
            }
        }
        Ok(values)
    }

    /// Materializes one `BalancedIncDigit`/`BalancedIncCarry` column of the
    /// packed (lattice) witness as the flat address-major `(K x T)` grid,
    /// `K = 2^committed_chunk_bits`. Every cycle is hot: padding rows encode
    /// the zero delta in row zero of every digit and the carry.
    pub(crate) fn materialize_balanced_inc_one_hot<F: JoltField>(
        &self,
        column: crate::witnesses::BalancedIncColumn,
    ) -> Result<Vec<F>, WitnessError> {
        let chunk_bits = self.config.one_hot.committed_chunk_bits();
        let cycles = checked_pow2(self.config.log_t)?;
        let len = checked_dense_grid_len::<F>(checked_pow2(chunk_bits)?, cycles)?;
        let selected_rows: Vec<usize> = self.walk_cycles(|row, next, env| {
            crate::witnesses::BalancedIncRow::extract_indexed(column, row, next, env)
                .map(|selected| selected.0)
        })?;
        let mut values = vec![F::zero(); len];
        for (cycle, selected_row) in selected_rows.into_iter().enumerate() {
            if selected_row >> chunk_bits != 0 {
                return Err(WitnessError::InvalidWitnessData {
                    label: JOLT_VM_LABEL,
                    reason: format!(
                        "balanced-inc row {selected_row} outside the 2^{chunk_bits} row domain"
                    ),
                });
            }
            values[selected_row * cycles + cycle] = F::one();
        }
        Ok(values)
    }

    /// One pass over `2^log_t` cycles with the one-row lookahead window;
    /// rows beyond the trace are padding (default) rows.
    ///
    /// Extraction is pure per cycle window and parallel when enabled.
    fn walk_cycles<V: Copy + Send>(
        &self,
        value: impl Fn(&TraceRow, Option<&TraceRow>, &WitnessEnv<'_>) -> Result<V, WitnessError>
            + Send
            + Sync,
    ) -> Result<Vec<V>, WitnessError> {
        let rows = checked_pow2(self.config.log_t)?;
        let env = WitnessEnv::new(&self.preprocessing);
        let physical = self.trace.trace.as_slice();
        let padding = TraceRow::default();
        let window = |index: usize| {
            let current = physical.get(index).unwrap_or(&padding);
            let next = (index + 1 < rows).then(|| physical.get(index + 1).unwrap_or(&padding));
            value(current, next, &env)
        };
        #[cfg(feature = "parallel")]
        return par_collect_windows(rows, window);
        #[cfg(not(feature = "parallel"))]
        return (0..rows).map(window).collect();
    }
}

impl<T: TraceSource> TraceBackend<T> {
    fn bundle_rows(&self) -> Result<RandomAccessRows, WitnessError> {
        #[cfg(feature = "field-inline")]
        if let Some(field_inline) = &self.field_inline {
            return Ok(field_inline.rows_source().clone());
        }
        RandomAccessRows::new(
            Arc::clone(&self.trace.trace),
            checked_pow2(self.config.log_t)?,
            Arc::clone(&self.preprocessing),
        )
    }
}

impl<T: TraceSource> RowSource for TraceBackend<T> {
    fn random_access(&self) -> Option<RandomAccessRows> {
        self.bundle_rows().ok()
    }

    fn visit_chunks(
        &self,
        range: Range<usize>,
        chunk_size: usize,
        visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError> {
        self.bundle_rows()?.visit_chunks(range, chunk_size, visitor)
    }
}

impl<T: TraceSource> BundleSource for TraceBackend<T> {
    fn bundles<B: WitnessBundle + Clone + Send + Sync>(&self) -> Result<Vec<B>, WitnessError> {
        crate::collect_bundles(self, checked_pow2(self.config.log_t)?)
    }
}
