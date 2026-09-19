//! The optimized joint-opening kernel: lazy grid embeddings for the stage-8
//! batch opening.
//!
//! The reference slot materializes every opened polynomial dense over the
//! full `2^total_vars` commitment domain — `O(polys · 2^total_vars)` field
//! elements retained across the whole PCS opening (the stage-8 memory wall).
//! The batch opening only ever drives the returned polynomials through
//! [`MultilinearPoly::fold_rows`] (Dory's one vector-matrix product per
//! opening; `RlcSource` distributes it per constituent) and — in hiding
//! mode — [`MultilinearPoly::evaluate`], so nothing needs the dense table.
//! This kernel returns lazy views that answer both from compact per-cycle
//! trace columns:
//!
//! - **Sparse one-hot VMP**:
//!   a one-hot polynomial's fold is `result[col(idx)] += left[row(idx)]` at
//!   the single hot grid index per cycle — `O(T)` group-free additions
//!   instead of an `O(K · T)` dense walk over a materialized grid.
//! - **Streaming trace columns**: the committed values are re-derived from one typed
//!   witness pass (`CommittedColumnsWitness`, the same bundle the commit
//!   kernel consumed) into packed per-cycle columns — `O(T)` small scalars
//!   shared by every trace polynomial via [`Arc`], never `K × T` oracle
//!   tables.
//! - **Strided dense scatter**: a dense trace column contributes
//!   `result[col] += left[row] · value` at `index = t · t_stride`, covering
//!   both coefficient orders with one placement formula.
//! - **Precommitted block contribution**: advice / committed-program tables
//!   fold from their own balanced `(2^{ν_p} × 2^{σ_p})` matrix into the
//!   grid's top-left block, `O(len)` work and space.
//!
//! Commit-time hints (row commitments) are combined homomorphically by the
//! batch opener (`combine_hints`), so no re-commit touches these views. The
//! placement formulas are exactly the reference embeddings'
//! (`reference::opening`); the in-module tests pin dense equality against
//! the reference slot on a real synthetic trace.

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::committed_openings::final_opening_id;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, TracePolynomialOrder};
use jolt_field::JoltField;
use jolt_poly::{MultilinearPoly, TensorEqTable};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_witness::witnesses::{BytecodePc, LookupIndex, RamInc, RdInc, RemappedRamAddress};
use jolt_witness::{stream_witnesses, JoltWitnessPlane, RandomAccessRows, StreamConsumer};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::commitment::{CommitmentGrid, CommittedColumnsWitness};
use crate::opening::JointOpeningPolynomials;
use crate::reference::commitment::{column_kinds, ColumnKind};
use crate::reference::views::dense_view;
use crate::{KernelError, OptimizedBackend, ProofSession};

/// Column sentinel for cycles with no hot address (no bytecode row, no
/// remappable RAM access).
const COLD: u64 = u64::MAX;

/// The row-window size of the column-collection pass.
const COLLECT_CHUNK: usize = 1 << 12;

/// Minimum per-range work of the parallel scatter/sum drivers; below it the
/// range split costs more than the loop.
#[cfg(feature = "parallel")]
const MIN_RANGE: usize = 1 << 12;

impl<F: JoltField> JointOpeningPolynomials<F> for OptimizedBackend {
    #[tracing::instrument(
        skip_all,
        name = "OptimizedJointOpening::prepare",
        fields(polynomials = polynomials.len(), total_vars = grid.total_vars)
    )]
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        polynomials: &[JoltCommittedPolynomial],
        mut precommitted_tables: BTreeMap<JoltCommittedPolynomial, Vec<F>>,
        grid: CommitmentGrid,
    ) -> Result<Vec<Box<dyn MultilinearPoly<F>>>, KernelError<F>> {
        if grid.total_vars < grid.log_t + grid.log_k_chunk {
            return Err(KernelError::InvalidGeometry {
                reason: format!(
                    "grid of {} variables cannot hold a (2^{} × 2^{}) one-hot matrix",
                    grid.total_vars, grid.log_k_chunk, grid.log_t
                ),
            });
        }

        // Chunk selectors for the trace-derived subset — the same family-size
        // counting and selector math as the commit kernel, so the opened
        // values are the committed values by construction.
        let trace_ids: Vec<JoltCommittedPolynomial> = polynomials
            .iter()
            .copied()
            .filter(|id| !is_block_embedded(*id))
            .collect();
        let kinds = column_kinds(&trace_ids, grid)?;
        let kind_by_id: BTreeMap<JoltCommittedPolynomial, ColumnKind> =
            trace_ids.into_iter().zip(kinds).collect();

        // One typed trace pass shared by every trace polynomial.
        let columns = if kind_by_id.is_empty() {
            None
        } else {
            Some(Arc::new(OpeningColumns::collect(witness, grid.log_t)?))
        };
        let placement = TracePlacement::new(grid);

        polynomials
            .iter()
            .map(|&polynomial| {
                if is_block_embedded(polynomial) {
                    let table = match precommitted_tables.remove(&polynomial) {
                        Some(table) => table,
                        None => dense_view(witness, final_opening_id(polynomial))?,
                    };
                    let poly = BlockOpeningPoly::new(table, grid, polynomial)?;
                    Ok(Box::new(poly) as Box<dyn MultilinearPoly<F>>)
                } else {
                    let kind =
                        *kind_by_id
                            .get(&polynomial)
                            .ok_or(KernelError::InvariantViolation {
                                reason: "trace polynomial missing from the resolved column kinds",
                            })?;
                    let columns =
                        Arc::clone(columns.as_ref().ok_or(KernelError::InvariantViolation {
                            reason: "trace columns not collected despite trace polynomials",
                        })?);
                    Ok(Box::new(TraceOpeningPoly::<F> {
                        columns,
                        kind,
                        placement,
                        _field: PhantomData,
                    }) as Box<dyn MultilinearPoly<F>>)
                }
            })
            .collect()
    }
}

/// Whether `polynomial` embeds as its own balanced matrix in the grid's
/// top-left block (advice and committed-program polynomials) rather than by
/// the trace placement.
const fn is_block_embedded(polynomial: JoltCommittedPolynomial) -> bool {
    matches!(
        polynomial,
        JoltCommittedPolynomial::TrustedAdvice
            | JoltCommittedPolynomial::UntrustedAdvice
            | JoltCommittedPolynomial::BytecodeChunk(_)
            | JoltCommittedPolynomial::ProgramImageInit
    )
}

// ---------------------------------------------------------------------------
// Shared per-cycle trace columns
// ---------------------------------------------------------------------------

/// Packed per-cycle facts behind every committed trace column — the
/// [`CommittedColumnsWitness`] bundle stored column-major with `Option`s
/// packed as [`COLD`] sentinels: 64 bytes per cycle, shared by every trace
/// polynomial view.
pub(crate) struct OpeningColumns {
    rd_inc: Vec<i128>,
    ram_inc: Vec<i128>,
    lookup_index: Vec<u128>,
    /// Bytecode slot per cycle; total, so no
    /// bytecode row.
    bytecode_pc: Vec<u64>,
    /// Remapped RAM word address per cycle; [`COLD`] on no-access cycles.
    ram_address: Vec<u64>,
}

impl OpeningColumns {
    fn collect<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<Self, KernelError<F>> {
        let cycles = 1usize << log_t;
        // Slice-backed sources fill the five columns index-parallel — the
        // chunked walk serializes on staging buffers and the consume copy.
        #[cfg(feature = "parallel")]
        if let Some(access) = witness.random_access() {
            if cycles <= access.cycles() {
                return Self::collect_par(&access, cycles);
            }
        }
        let mut consumers = (CollectOpeningColumns {
            columns: Self {
                rd_inc: Vec::with_capacity(cycles),
                ram_inc: Vec::with_capacity(cycles),
                lookup_index: Vec::with_capacity(cycles),
                bytecode_pc: Vec::with_capacity(cycles),
                ram_address: Vec::with_capacity(cycles),
            },
        },);
        stream_witnesses(witness, 0..cycles, COLLECT_CHUNK, &mut consumers)?;
        let columns = consumers.0.columns;
        debug_assert_eq!(
            columns.cycles(),
            cycles,
            "opening columns must cover the full padded cycle domain"
        );
        Ok(columns)
    }

    /// Index-parallel column collection over a slice-backed source: workers
    /// extract straight into the five pre-zeroed columns (values identical
    /// to the streaming pass — extraction is pure per cycle window, and
    /// every slot is written).
    #[cfg(feature = "parallel")]
    fn collect_par<F: JoltField>(
        access: &RandomAccessRows,
        cycles: usize,
    ) -> Result<Self, KernelError<F>> {
        /// The scatter grain: big enough to amortize rayon dispatch, small
        /// enough to load-balance skewed extraction.
        const CHUNK: usize = 1 << 12;
        let mut rd_inc: Vec<i128> = unsafe_allocate_zero_vec(cycles);
        let mut ram_inc: Vec<i128> = unsafe_allocate_zero_vec(cycles);
        let mut lookup_index: Vec<u128> = unsafe_allocate_zero_vec(cycles);
        let mut bytecode_pc: Vec<u64> = unsafe_allocate_zero_vec(cycles);
        let mut ram_address: Vec<u64> = unsafe_allocate_zero_vec(cycles);
        let error = std::sync::Mutex::new(None);
        (
            rd_inc.par_chunks_mut(CHUNK),
            ram_inc.par_chunks_mut(CHUNK),
            lookup_index.par_chunks_mut(CHUNK),
            bytecode_pc.par_chunks_mut(CHUNK),
            ram_address.par_chunks_mut(CHUNK),
        )
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_index, (rd, ram, lookup, pc, address))| {
                let base = chunk_index * CHUNK;
                for offset in 0..rd.len() {
                    match access.window::<CommittedColumnsWitness>(base + offset) {
                        Ok(row) => {
                            debug_assert_ne!(
                                row.ram_address.0,
                                Some(COLD),
                                "a live remapped RAM address collides with the COLD sentinel"
                            );
                            rd[offset] = row.rd_inc.0;
                            ram[offset] = row.ram_inc.0;
                            lookup[offset] = row.lookup_index.0;
                            pc[offset] = row.bytecode_pc.0 as u64;
                            address[offset] = row.ram_address.0.unwrap_or(COLD);
                        }
                        Err(failure) => {
                            if let Ok(mut guard) = error.try_lock() {
                                let _ = guard.get_or_insert(failure);
                            }
                            return;
                        }
                    }
                }
            });
        #[expect(clippy::unwrap_used, reason = "no lock user can panic")]
        if let Some(failure) = error.into_inner().unwrap() {
            return Err(failure.into());
        }
        Ok(Self {
            rd_inc,
            ram_inc,
            lookup_index,
            bytecode_pc,
            ram_address,
        })
    }

    fn cycles(&self) -> usize {
        self.rd_inc.len()
    }

    /// The cycle's fact bundle, reassembled for [`ColumnKind`]'s accessors.
    #[inline]
    fn witness_row(&self, cycle: usize) -> CommittedColumnsWitness {
        let bytecode_pc = self.bytecode_pc[cycle];
        let ram_address = self.ram_address[cycle];
        CommittedColumnsWitness {
            rd_inc: RdInc(self.rd_inc[cycle]),
            ram_inc: RamInc(self.ram_inc[cycle]),
            lookup_index: LookupIndex(self.lookup_index[cycle]),
            bytecode_pc: BytecodePc(bytecode_pc as usize),
            ram_address: RemappedRamAddress((ram_address != COLD).then_some(ram_address)),
        }
    }
}

struct CollectOpeningColumns {
    columns: OpeningColumns,
}

impl StreamConsumer for CollectOpeningColumns {
    type Witness = CommittedColumnsWitness;

    fn consume(&mut self, chunk: &[CommittedColumnsWitness]) {
        let columns = &mut self.columns;
        for row in chunk {
            debug_assert_ne!(
                row.ram_address.0,
                Some(COLD),
                "a live remapped RAM address collides with the COLD sentinel"
            );
            columns.rd_inc.push(row.rd_inc.0);
            columns.ram_inc.push(row.ram_inc.0);
            columns.lookup_index.push(row.lookup_index.0);
            columns.bytecode_pc.push(row.bytecode_pc.0 as u64);
            columns.ram_address.push(row.ram_address.0.unwrap_or(COLD));
        }
    }
}

// ---------------------------------------------------------------------------
// Grid placement
// ---------------------------------------------------------------------------

/// A trace coefficient's grid index: `(k, t) ↦ t · t_stride + k · k_stride`,
/// dense columns at `k = 0`. Covers both proof orders with one formula:
/// cycle-major prefix-embeds the flat address-major `(K × T)` matrix
/// (`t_stride = 1`, `k_stride = 2^log_t`); address-major scatters
/// cycle-block-strided (`t_stride = cycle_stride`, `k_stride =
/// one_hot_stride`) — the reference embeddings' index maps verbatim.
#[derive(Clone, Copy, Debug)]
struct TracePlacement {
    total_vars: usize,
    t_stride: usize,
    k_stride: usize,
}

impl TracePlacement {
    fn new(grid: CommitmentGrid) -> Self {
        match grid.order {
            TracePolynomialOrder::CycleMajor => Self {
                total_vars: grid.total_vars,
                t_stride: 1,
                k_stride: 1usize << grid.log_t,
            },
            TracePolynomialOrder::AddressMajor => Self {
                total_vars: grid.total_vars,
                t_stride: grid.cycle_stride(),
                k_stride: grid.one_hot_stride(),
            },
        }
    }

    #[inline(always)]
    const fn index(self, cycle: usize, address: usize) -> usize {
        cycle * self.t_stride + address * self.k_stride
    }
}

// ---------------------------------------------------------------------------
// Parallel scatter/sum drivers
// ---------------------------------------------------------------------------

/// Fold `total` source slots into a `num_cols`-sized accumulator through
/// `fill`, splitting into per-thread partial accumulators when parallel.
/// Field addition is exact, so the merge order cannot change the values.
fn scatter_fold<F: JoltField>(
    total: usize,
    num_cols: usize,
    fill: impl Fn(Range<usize>, &mut [F]) + Send + Sync,
) -> Vec<F> {
    #[cfg(feature = "parallel")]
    if total > MIN_RANGE {
        let ranges = split_ranges(total);
        return ranges
            .into_par_iter()
            .map(|range| {
                let mut acc: Vec<F> = unsafe_allocate_zero_vec(num_cols);
                fill(range, &mut acc);
                acc
            })
            .reduce(
                || unsafe_allocate_zero_vec(num_cols),
                super::support::merge_evals,
            );
    }
    let mut acc: Vec<F> = unsafe_allocate_zero_vec(num_cols);
    fill(0..total, &mut acc);
    acc
}

/// Sum a per-slot contribution over `total` source slots, in parallel when
/// worthwhile.
fn scatter_sum<F: JoltField>(total: usize, sum: impl Fn(Range<usize>) -> F + Send + Sync) -> F {
    #[cfg(feature = "parallel")]
    if total > MIN_RANGE {
        let ranges = split_ranges(total);
        return ranges
            .into_par_iter()
            .map(sum)
            .reduce(F::zero, |left, right| left + right);
    }
    sum(0..total)
}

#[cfg(feature = "parallel")]
fn split_ranges(total: usize) -> Vec<Range<usize>> {
    let max_ranges = rayon::current_num_threads() * 4;
    let ranges = (total / MIN_RANGE).clamp(1, max_ranges.max(1));
    let chunk = total.div_ceil(ranges);
    (0..total)
        .step_by(chunk)
        .map(|start| start..(start + chunk).min(total))
        .collect()
}

// ---------------------------------------------------------------------------
// Lazy trace polynomial (one-hot and dense-increment columns)
// ---------------------------------------------------------------------------

/// One committed trace polynomial as a lazy view over the shared columns:
/// per cycle one hot grid index (one-hot kinds) or one increment value at
/// address slot zero (dense kinds).
struct TraceOpeningPoly<F: JoltField> {
    columns: Arc<OpeningColumns>,
    kind: ColumnKind,
    placement: TracePlacement,
    _field: PhantomData<F>,
}

impl<F: JoltField> TraceOpeningPoly<F> {
    /// The cycle's grid entry, `None` when the cycle contributes nothing
    /// (cold one-hot cycle, zero increment).
    #[inline]
    fn entry(&self, cycle: usize) -> Option<(usize, F)> {
        let row = self.columns.witness_row(cycle);
        if self.kind.is_one_hot() {
            let address = self.kind.hot_address(&row)?;
            Some((self.placement.index(cycle, address), F::one()))
        } else {
            let increment = self.kind.increment(&row);
            if increment == 0 {
                return None;
            }
            Some((self.placement.index(cycle, 0), F::from_i128(increment)))
        }
    }
}

impl<F: JoltField> MultilinearPoly<F> for TraceOpeningPoly<F> {
    fn num_vars(&self) -> usize {
        self.placement.total_vars
    }

    fn evaluate(&self, point: &[F]) -> F {
        debug_assert_eq!(point.len(), self.placement.total_vars);
        let eq = TensorEqTable::new(point);
        scatter_sum(self.columns.cycles(), |range| {
            let mut acc = F::zero();
            for cycle in range {
                if let Some((index, value)) = self.entry(cycle) {
                    acc += value * eq.evaluate_index(index);
                }
            }
            acc
        })
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F])) {
        let entries = (0..self.columns.cycles())
            .filter_map(|cycle| self.entry(cycle))
            .collect();
        emit_sorted_rows(entries, self.num_vars(), sigma, f);
    }

    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        debug_assert_eq!(
            left.len(),
            1usize << self.num_vars().saturating_sub(sigma),
            "left vector length must equal number of rows"
        );
        let num_cols = 1usize << sigma;
        let mask = num_cols - 1;
        scatter_fold(self.columns.cycles(), num_cols, |range, acc| {
            for cycle in range {
                if let Some((index, value)) = self.entry(cycle) {
                    if self.kind.is_one_hot() {
                        acc[index & mask] += left[index >> sigma];
                    } else {
                        acc[index & mask] += left[index >> sigma] * value;
                    }
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Lazy block-embedded polynomial (advice, committed program)
// ---------------------------------------------------------------------------

/// A precommitted polynomial as a lazy view of its top-left block embedding:
/// its own balanced `(2^{ν_p} × 2^{σ_p})` matrix lands row-aligned in the
/// grid matrix — coefficient `r · 2^{σ_p} + c` at grid index
/// `r · 2^{σ_grid} + c`.
struct BlockOpeningPoly<F: JoltField> {
    table: Vec<F>,
    sigma_table: usize,
    sigma_grid: usize,
    total_vars: usize,
}

impl<F: JoltField> BlockOpeningPoly<F> {
    fn new(
        table: Vec<F>,
        grid: CommitmentGrid,
        polynomial: JoltCommittedPolynomial,
    ) -> Result<Self, KernelError<F>> {
        if !table.len().is_power_of_two() {
            return Err(KernelError::TableSizeMismatch {
                table: format!("{polynomial:?}"),
                expected: table.len().next_power_of_two(),
                got: table.len(),
            });
        }
        if table.len() > 1usize << grid.total_vars {
            return Err(KernelError::TableSizeMismatch {
                table: format!("{polynomial:?}"),
                expected: 1usize << grid.total_vars,
                got: table.len(),
            });
        }
        let table_vars = table.len().ilog2() as usize;
        Ok(Self {
            table,
            sigma_table: table_vars.div_ceil(2),
            sigma_grid: grid.total_vars.div_ceil(2),
            total_vars: grid.total_vars,
        })
    }

    /// Grid index of the table coefficient at flat index `i`.
    #[inline(always)]
    fn index(&self, i: usize) -> usize {
        let column_mask = (1usize << self.sigma_table) - 1;
        ((i >> self.sigma_table) << self.sigma_grid) | (i & column_mask)
    }
}

impl<F: JoltField> MultilinearPoly<F> for BlockOpeningPoly<F> {
    fn num_vars(&self) -> usize {
        self.total_vars
    }

    fn evaluate(&self, point: &[F]) -> F {
        debug_assert_eq!(point.len(), self.total_vars);
        let eq = TensorEqTable::new(point);
        scatter_sum(self.table.len(), |range| {
            let mut acc = F::zero();
            for i in range {
                let value = self.table[i];
                if !value.is_zero() {
                    acc += value * eq.evaluate_index(self.index(i));
                }
            }
            acc
        })
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F])) {
        // Table order is grid-index order (row-aligned block), so the
        // entries arrive pre-sorted.
        let entries = (0..self.table.len())
            .map(|i| (self.index(i), self.table[i]))
            .collect();
        emit_sorted_rows(entries, self.total_vars, sigma, f);
    }

    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        debug_assert_eq!(
            left.len(),
            1usize << self.total_vars.saturating_sub(sigma),
            "left vector length must equal number of rows"
        );
        let num_cols = 1usize << sigma;
        let mask = num_cols - 1;
        scatter_fold(self.table.len(), num_cols, |range, acc| {
            for i in range {
                let value = self.table[i];
                if !value.is_zero() {
                    let index = self.index(i);
                    acc[index & mask] += left[index >> sigma] * value;
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Row emission (off the stage-8 path)
// ---------------------------------------------------------------------------

/// Emit the `(2^{n-σ} × 2^σ)` matrix rows of a sparse entry set. Sorts the
/// entries and cursor-walks them into one reused row buffer — `O(N log N +
/// 2^n)` time, `O(N + 2^σ)` space. The batch opening never calls this
/// (it drives `fold_rows`); it serves the general [`MultilinearPoly`]
/// contract (`to_dense`, tests).
fn emit_sorted_rows<F: JoltField>(
    mut entries: Vec<(usize, F)>,
    num_vars: usize,
    sigma: usize,
    f: &mut dyn FnMut(usize, &[F]),
) {
    entries.sort_unstable_by_key(|&(index, _)| index);
    let num_cols = 1usize << sigma;
    let num_rows = 1usize << num_vars.saturating_sub(sigma);
    let mut row_buffer: Vec<F> = unsafe_allocate_zero_vec(num_cols);
    let mut cursor = 0usize;
    for row in 0..num_rows {
        row_buffer.fill(F::zero());
        let row_base = row << sigma;
        while let Some(&(index, value)) = entries.get(cursor) {
            if index >= row_base + num_cols {
                break;
            }
            row_buffer[index - row_base] = value;
            cursor += 1;
        }
        f(row, &row_buffer);
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module: fail loudly")]
mod tests {
    use jolt_field::Fr;

    use super::*;
    use crate::optimized::testing::{random_scalars, with_ram_fixture, FixtureShape, RamOp};
    use crate::ReferenceBackend;

    const LOG_T: usize = 4;
    const RAM_K: usize = 16;

    fn fixture_ops() -> Vec<RamOp> {
        vec![
            RamOp::Write { word: 2, post: 7 },
            RamOp::Read { word: 2 },
            RamOp::None,
            RamOp::Write { word: 5, post: 11 },
            RamOp::Read { word: 9 },
            RamOp::Write { word: 2, post: 3 },
            RamOp::None,
            RamOp::Read { word: 5 },
        ]
    }

    /// Both slots' outputs over the same order/grid/precommitted tables.
    #[expect(clippy::type_complexity, reason = "test helper pairing both tiers")]
    fn prepare_both(
        witness: &dyn JoltWitnessPlane<Fr>,
        grid: CommitmentGrid,
    ) -> (
        Vec<JoltCommittedPolynomial>,
        Vec<Box<dyn MultilinearPoly<Fr>>>,
        Vec<Box<dyn MultilinearPoly<Fr>>>,
    ) {
        let mut order = witness.committed_order().unwrap();
        order.push(JoltCommittedPolynomial::BytecodeChunk(0));
        order.push(JoltCommittedPolynomial::ProgramImageInit);
        let mut precommitted_tables = BTreeMap::new();
        // Synthetic block-embedded tables: the slot embeds whatever the map
        // carries, so random tables of odd/even variable counts exercise the
        // block placement without a committed-program fixture.
        let _ = precommitted_tables.insert(
            JoltCommittedPolynomial::BytecodeChunk(0),
            random_scalars(1 << 3, 17),
        );
        let _ = precommitted_tables.insert(
            JoltCommittedPolynomial::ProgramImageInit,
            random_scalars(1 << 4, 19),
        );

        let reference = JointOpeningPolynomials::<Fr>::prepare(
            &ReferenceBackend,
            &mut ProofSession::default(),
            witness,
            &order,
            precommitted_tables.clone(),
            grid,
        )
        .unwrap();
        let optimized = JointOpeningPolynomials::<Fr>::prepare(
            &OptimizedBackend,
            &mut ProofSession::default(),
            witness,
            &order,
            precommitted_tables,
            grid,
        )
        .unwrap();
        (order, reference, optimized)
    }

    fn assert_parity(order: TracePolynomialOrder, widen: usize) {
        let shape = FixtureShape {
            log_t: LOG_T,
            ram_k: RAM_K,
        };
        with_ram_fixture(shape, fixture_ops(), |witness| {
            let grid = CommitmentGrid {
                total_vars: LOG_T + 4 + widen,
                log_t: LOG_T,
                log_k_chunk: 4,
                order,
            };
            let (ids, reference, optimized) = prepare_both(witness, grid);
            let point = random_scalars(grid.total_vars, 23);
            let sigmas = [
                grid.total_vars.div_ceil(2),
                grid.total_vars.div_ceil(2) + 1,
                grid.total_vars,
            ];
            for ((id, reference), optimized) in ids.iter().zip(&reference).zip(&optimized) {
                assert_eq!(
                    optimized.num_vars(),
                    reference.num_vars(),
                    "{id:?}: num_vars diverged"
                );
                assert_eq!(
                    optimized.to_dense().as_ref(),
                    reference.to_dense().as_ref(),
                    "{id:?}: dense table diverged"
                );
                for sigma in sigmas {
                    let left = random_scalars(1 << (grid.total_vars - sigma), 29 + sigma as u64);
                    assert_eq!(
                        optimized.fold_rows(&left, sigma),
                        reference.fold_rows(&left, sigma),
                        "{id:?}: fold_rows diverged at sigma {sigma}"
                    );
                }
                assert_eq!(
                    optimized.evaluate(&point),
                    reference.evaluate(&point),
                    "{id:?}: evaluation diverged"
                );
            }
        });
    }

    #[test]
    fn cycle_major_matches_reference() {
        assert_parity(TracePolynomialOrder::CycleMajor, 0);
    }

    #[test]
    fn cycle_major_widened_matches_reference() {
        assert_parity(TracePolynomialOrder::CycleMajor, 2);
    }

    #[test]
    fn cycle_major_odd_widening_matches_reference() {
        assert_parity(TracePolynomialOrder::CycleMajor, 1);
    }

    #[test]
    fn address_major_matches_reference() {
        assert_parity(TracePolynomialOrder::AddressMajor, 0);
    }

    #[test]
    fn address_major_widened_matches_reference() {
        assert_parity(TracePolynomialOrder::AddressMajor, 2);
    }

    #[test]
    fn address_major_odd_widening_matches_reference() {
        assert_parity(TracePolynomialOrder::AddressMajor, 1);
    }
}
