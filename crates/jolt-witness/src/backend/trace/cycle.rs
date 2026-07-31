//! The sequential cycle walk driving the atomic extractors, and the
//! trace-backed implementation of the streaming pass.

use super::*;
use crate::consumer::ChunkVisitor;
use crate::witnesses::{Extract, ExtractIndexed, RaChunkSelector, ToField, WitnessEnv};
use std::ops::Range;

use crate::{BundleSource, RowSource, WitnessBundle};

impl<T: TraceSource + Clone> TraceBackend<T> {
    /// Materializes one cycle-domain witness column by walking the trace
    /// once; all per-witness logic lives on `W`.
    pub(crate) fn materialize_cycle<F: Field, W: Extract + ToField>(
        &self,
    ) -> Result<Vec<F>, WitnessError> {
        self.walk_cycles(|row, next, env| W::extract(row, next, env).map(ToField::to_field))
    }

    /// [`Self::materialize_cycle`] for indexed witness families; `index`
    /// selects the family member.
    pub(crate) fn materialize_cycle_indexed<
        F: Field,
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
        F: Field,
        W: ExtractIndexed<RaChunkSelector> + Into<Option<usize>>,
    {
        let selector = RaChunkSelector::new(index, chunks, chunk_bits)?;
        let cycles = checked_pow2(self.config.log_t)?;
        let log_rows = chunk_bits.checked_add(self.config.log_t).ok_or_else(|| {
            WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: "one-hot rows overflow".to_owned(),
            }
        })?;
        let hot_addresses: Vec<Option<usize>> = self.walk_cycles(|row, next, env| {
            W::extract_indexed(selector, row, next, env).map(W::into)
        })?;
        // The selector's mask bounds every hot address below `2^chunk_bits`.
        let mut values = jolt_utils::unsafe_allocate_zero_vec(checked_pow2(log_rows)?);
        for (cycle, address) in hot_addresses.into_iter().enumerate() {
            if let Some(address) = address {
                values[address * cycles + cycle] = F::one();
            }
        }
        Ok(values)
    }

    /// Materializes one `UnsignedIncChunk`/`UnsignedIncMsb` column of the
    /// packed (lattice) witness as the flat address-major `(K x T)` grid,
    /// `K = 2^committed_chunk_bits`. Every cycle is hot: padding rows encode
    /// the zero delta as lane 0 of each chunk and lane 1 of the msb.
    pub(crate) fn materialize_unsigned_inc_one_hot<F: Field>(
        &self,
        lane: crate::witnesses::UnsignedIncLane,
    ) -> Result<Vec<F>, WitnessError> {
        let chunk_bits = self.config.one_hot.committed_chunk_bits();
        let cycles = checked_pow2(self.config.log_t)?;
        let hot_addresses: Vec<usize> = self.walk_cycles(|row, next, env| {
            crate::witnesses::UnsignedIncHot::extract_indexed(lane, row, next, env).map(|hot| hot.0)
        })?;
        let mut values = vec![F::zero(); checked_pow2(self.one_hot_log_rows()?)?];
        for (cycle, address) in hot_addresses.into_iter().enumerate() {
            if address >> chunk_bits != 0 {
                return Err(WitnessError::InvalidWitnessData {
                    label: JOLT_VM_LABEL,
                    reason: format!(
                        "unsigned-inc hot lane {address} outside the 2^{chunk_bits} lane domain"
                    ),
                });
            }
            values[address * cycles + cycle] = F::one();
        }
        Ok(values)
    }

    /// One pass over `2^log_t` cycles with the one-row lookahead window;
    /// rows beyond the trace are padding (default) rows.
    ///
    /// A slice-backed trace ([`TraceSource::rows`]) takes an index-parallel
    /// path — extraction is pure per cycle window, so the walk order is
    /// unobservable. The sequential walk remains the fallback (and the only
    /// public contract) for re-emulating sources. `V: Copy` keeps the
    /// parallel collector's leak-free-on-error invariant compiler-checked.
    fn walk_cycles<V: Copy + Send>(
        &self,
        value: impl Fn(&TraceRow, Option<&TraceRow>, &WitnessEnv<'_>) -> Result<V, WitnessError>
            + Send
            + Sync,
    ) -> Result<Vec<V>, WitnessError> {
        let rows = checked_pow2(self.config.log_t)?;
        let env = WitnessEnv {
            preprocessing: &self.preprocessing,
        };
        if let Some(physical) = self.trace.trace.rows() {
            let padding = TraceRow::default();
            let window = |index: usize| {
                let current = physical.get(index).unwrap_or(&padding);
                let next = (index + 1 < rows).then(|| physical.get(index + 1).unwrap_or(&padding));
                value(current, next, &env)
            };
            #[cfg(feature = "parallel")]
            return jolt_utils::par_collect_windows(rows, window);
            #[cfg(not(feature = "parallel"))]
            return (0..rows).map(window).collect();
        }
        let mut values = Vec::with_capacity(rows);
        let mut trace = self.trace.trace.clone();
        let mut current = trace.next_row().unwrap_or_default();
        for index in 0..rows {
            let next = (index + 1 < rows).then(|| trace.next_row().unwrap_or_default());
            values.push(value(&current, next.as_ref(), &env)?);
            if let Some(row) = next {
                current = row;
            }
        }
        Ok(values)
    }
}

impl<T: TraceSource + Clone> RowSource for TraceBackend<T> {
    fn rows(&self) -> Option<&[TraceRow]> {
        self.trace.trace.rows()
    }

    fn visit_chunks(
        &self,
        range: Range<usize>,
        chunk_size: usize,
        visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError> {
        let total = checked_pow2(self.config.log_t)?;
        if range.start > range.end || range.end > total {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "cycle range [{}, {}) exceeds the domain of {total} cycles",
                    range.start, range.end
                ),
            });
        }
        let env = WitnessEnv {
            preprocessing: &self.preprocessing,
        };
        // Slice-backed traces visit borrowed subslices directly — no per-row
        // copies; only a buffer overlapping the padding tail materializes.
        if let Some(physical) = self.trace.trace.rows() {
            let padding = TraceRow::default();
            let mut position = range.start;
            while position < range.end {
                let chunk_end = (position + chunk_size).min(range.end);
                let next_after: Option<&TraceRow> =
                    (chunk_end < total).then(|| physical.get(chunk_end).unwrap_or(&padding));
                if chunk_end <= physical.len() {
                    visitor(&physical[position..chunk_end], next_after, &env)?;
                } else {
                    let mut rows = Vec::with_capacity(chunk_end - position);
                    rows.extend_from_slice(&physical[position.min(physical.len())..]);
                    rows.resize(chunk_end - position, TraceRow::default());
                    visitor(&rows, next_after, &env)?;
                }
                position = chunk_end;
            }
            return Ok(());
        }
        let mut trace = self.trace.trace.clone();
        for _ in 0..range.start {
            let _ = trace.next_row();
        }
        // Rows beyond the physical trace are padding (default) rows; the
        // lookahead row after each buffer doubles as the first row of the
        // next one.
        let mut position = range.start;
        let mut carried: Option<TraceRow> = None;
        while position < range.end {
            let chunk_end = (position + chunk_size).min(range.end);
            let mut rows = Vec::with_capacity(chunk_end - position);
            if let Some(row) = carried.take() {
                rows.push(row);
            }
            while position + rows.len() < chunk_end {
                rows.push(trace.next_row().unwrap_or_default());
            }
            position = chunk_end;
            // The lookahead row doubles as the first row of the next buffer.
            carried = (position < total).then(|| trace.next_row().unwrap_or_default());
            visitor(&rows, carried.as_ref(), &env)?;
        }
        Ok(())
    }
}

impl<T: TraceSource + Clone> BundleSource for TraceBackend<T> {
    fn bundles<B: WitnessBundle + Clone + Send + Sync>(&self) -> Result<Vec<B>, WitnessError> {
        crate::collect_bundles(self, checked_pow2(self.config.log_t)?)
    }
}
