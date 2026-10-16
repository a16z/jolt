use super::*;
use crate::consumer::ChunkVisitor;
use crate::witnesses::{
    BalancedIncColumn, BalancedIncRow, Extract, ExtractIndexed, RaChunkSelector, ToField,
    WitnessEnv,
};
use jolt_riscv::JoltTraceRow as TraceRow;
#[cfg(feature = "parallel")]
use jolt_utils::FirstErrorLatch;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::ops::Range;

use crate::{BundleSource, RowSource, WitnessBundle};

impl TraceBackend {
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
    /// address-major `(K x T)` grid, `K = 2^chunk_bits`: one sequential visit
    /// writes a one at each cycle's hot address (`None` is a cold cycle).
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
        let rows = self.trace_rows()?;
        let cycles = rows.cycles();
        let len = checked_dense_grid_len::<F>(checked_pow2(chunk_bits)?, cycles)?;
        // The selector's mask bounds every hot address below `2^chunk_bits`.
        let mut values = jolt_utils::unsafe_allocate_zero_vec(len);
        let mut cycle = 0;
        rows.visit(0..cycles, WALK_ROWS, &mut |chunk, next, env| {
            for (index, row) in chunk.iter().enumerate() {
                let hot: Option<usize> =
                    W::extract_indexed(selector, row, chunk.get(index + 1).or(next), env)?.into();
                if let Some(address) = hot {
                    values[address * cycles + cycle] = F::one();
                }
                cycle += 1;
            }
            Ok(())
        })?;
        Ok(values)
    }

    /// Materializes one `BalancedIncDigit`/`BalancedIncCarry` column of the
    /// packed (lattice) witness as the flat address-major `(K x T)` grid,
    /// `K = 2^committed_chunk_bits`. Every cycle is hot: padding rows encode
    /// the zero delta in row zero of every digit and the carry.
    pub(crate) fn materialize_balanced_inc_one_hot<F: JoltField>(
        &self,
        column: BalancedIncColumn,
    ) -> Result<Vec<F>, WitnessError> {
        let chunk_bits = self.config.one_hot.committed_chunk_bits();
        let rows = self.trace_rows()?;
        let cycles = rows.cycles();
        let len = checked_dense_grid_len::<F>(checked_pow2(chunk_bits)?, cycles)?;
        let mut values = vec![F::zero(); len];
        let mut cycle = 0;
        rows.visit(0..cycles, WALK_ROWS, &mut |chunk, next, env| {
            for (index, row) in chunk.iter().enumerate() {
                let selected_row = BalancedIncRow::extract_indexed(
                    column,
                    row,
                    chunk.get(index + 1).or(next),
                    env,
                )?
                .0;
                if selected_row >> chunk_bits != 0 {
                    return Err(WitnessError::InvalidWitnessData {
                        label: JOLT_VM_LABEL,
                        reason: format!(
                            "balanced-inc row {selected_row} outside the 2^{chunk_bits} row domain"
                        ),
                    });
                }
                values[selected_row * cycles + cycle] = F::one();
                cycle += 1;
            }
            Ok(())
        })?;
        Ok(values)
    }

    /// Fills the cycle-domain column `value(row, lookahead)`. Parallel tasks
    /// of `WALK_ROWS` rows each write their own slice of the output through
    /// one serial visit. The lowest-index failure wins.
    fn walk_cycles<F: JoltField>(
        &self,
        value: impl Fn(&TraceRow, Option<&TraceRow>, &WitnessEnv<'_>) -> Result<F, WitnessError> + Sync,
    ) -> Result<Vec<F>, WitnessError> {
        let rows = self.trace_rows()?;
        let mut values = jolt_utils::unsafe_allocate_zero_vec(rows.cycles());
        let fill = |first: usize, values: &mut [F]| {
            let mut rest = values;
            rows.visit(
                first..first + rest.len(),
                WALK_ROWS,
                &mut |chunk, next, env| {
                    let (head, tail) = std::mem::take(&mut rest).split_at_mut(chunk.len());
                    rest = tail;
                    for (index, (row, slot)) in chunk.iter().zip(head).enumerate() {
                        *slot = value(row, chunk.get(index + 1).or(next), env)?;
                    }
                    Ok(())
                },
            )
        };
        #[cfg(feature = "parallel")]
        {
            let error = FirstErrorLatch::new();
            values
                .par_chunks_mut(WALK_ROWS)
                .enumerate()
                .for_each(|(index, values)| {
                    if let Err(failure) = fill(index * WALK_ROWS, values) {
                        error.record(index, failure);
                    }
                });
            if let Some(failure) = error.take() {
                return Err(failure);
            }
        }
        #[cfg(not(feature = "parallel"))]
        fill(0, &mut values)?;
        Ok(values)
    }
}

impl RowSource for TraceBackend {
    fn rows(&self) -> Option<TraceRows> {
        self.trace_rows().ok()
    }

    fn visit_chunks(
        &self,
        range: Range<usize>,
        chunk_size: usize,
        visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError> {
        self.trace_rows()?.visit(range, chunk_size, visitor)
    }
}

impl BundleSource for TraceBackend {
    fn bundles<B: WitnessBundle + Clone + Send + Sync>(&self) -> Result<Vec<B>, WitnessError> {
        crate::collect_bundles(self, checked_pow2(self.config.log_t)?)
    }
}
