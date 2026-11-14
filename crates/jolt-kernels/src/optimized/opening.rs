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
//! - **Resident trace rows**: [`StreamingJointOpening`], the stage-8 slot of
//!   the `instruction_rows` streaming family, re-extracts that bundle from
//!   the witness's resident rows on every read instead, so the packed
//!   columns never exist beside the trace.
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
use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::committed_openings::final_opening_id;
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_field::JoltField;
use jolt_poly::{MultilinearPoly, TensorEqTable};
#[cfg(feature = "parallel")]
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_witness::witnesses::{BytecodePc, LookupIndex, RamInc, RdInc, RemappedRamAddress};
#[cfg(feature = "parallel")]
use jolt_witness::RandomAccessRows;
use jolt_witness::{stream_witnesses, JoltWitnessPlane, StreamConsumer};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::{BundleAccess, BundleStore};
use crate::commitment::CommitmentGrid;
use crate::commitment::CommittedColumnsWitness;
use crate::opening::{JointOpeningPolynomials, PrecommittedOpeningTables};
use crate::reference::commitment::{column_kinds, ColumnKind};
use crate::reference::views::dense_view;
use crate::trace_column::{emit_sorted_rows, scatter_fold, scatter_sum, TracePlacement};
#[cfg(feature = "field-inline")]
use crate::ReferenceBackend;
use crate::{KernelError, OptimizedBackend, ProofSession};

const COLD: u64 = u64::MAX;

const COLLECT_CHUNK: usize = 1 << 12;

const ROWS: &str = "joint-opening trace rows";

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
        precommitted_tables: PrecommittedOpeningTables<'_, F>,
        grid: CommitmentGrid,
    ) -> Result<Vec<Box<dyn MultilinearPoly<F>>>, KernelError<F>> {
        lazy_views(witness, polynomials, precommitted_tables, grid, || {
            OpeningColumns::collect(witness, grid.log_t)
        })
    }

    #[cfg(feature = "field-inline")]
    fn prepare_field_inline(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        grid: CommitmentGrid,
    ) -> Result<Box<dyn MultilinearPoly<F>>, KernelError<F>> {
        JointOpeningPolynomials::prepare_field_inline(&ReferenceBackend, session, witness, grid)
    }
}

/// The stage-8 joint opening of
/// [`StreamingFamilies::instruction_rows`](super::StreamingFamilies::instruction_rows):
/// the optimized kernel's lazy views, whose trace views read the witness's
/// resident rows instead of sharing packed columns collected from them.
pub struct StreamingJointOpening;

impl<F: JoltField> JointOpeningPolynomials<F> for StreamingJointOpening {
    #[tracing::instrument(
        skip_all,
        name = "StreamingJointOpening::prepare",
        fields(polynomials = polynomials.len(), total_vars = grid.total_vars)
    )]
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        polynomials: &[JoltCommittedPolynomial],
        precommitted_tables: PrecommittedOpeningTables<'_, F>,
        grid: CommitmentGrid,
    ) -> Result<Vec<Box<dyn MultilinearPoly<F>>>, KernelError<F>> {
        lazy_views(witness, polynomials, precommitted_tables, grid, || {
            ResidentRows::new(witness, grid.log_t)
        })
    }

    #[cfg(feature = "field-inline")]
    fn prepare_field_inline(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        grid: CommitmentGrid,
    ) -> Result<Box<dyn MultilinearPoly<F>>, KernelError<F>> {
        JointOpeningPolynomials::prepare_field_inline(&OptimizedBackend, session, witness, grid)
    }
}

/// The lazy views of `polynomials` over `grid`: block-embedded tables from
/// `precommitted_tables` (the witness's dense view when absent), and trace
/// views sharing the rows `rows` resolves, called only when a trace
/// polynomial is opened.
fn lazy_views<F: JoltField, R: OpeningRows>(
    witness: &dyn JoltWitnessPlane<F>,
    polynomials: &[JoltCommittedPolynomial],
    precommitted_tables: PrecommittedOpeningTables<'_, F>,
    grid: CommitmentGrid,
    rows: impl FnOnce() -> Result<R, KernelError<F>>,
) -> Result<Vec<Box<dyn MultilinearPoly<F>>>, KernelError<F>> {
    let mut precommitted_tables = precommitted_tables()?;
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

    let rows = if kind_by_id.is_empty() {
        None
    } else {
        Some(Arc::new(rows()?))
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
                let kind = *kind_by_id
                    .get(&polynomial)
                    .ok_or(KernelError::InvariantViolation {
                        reason: "trace polynomial missing from the resolved column kinds",
                    })?;
                let rows = Arc::clone(rows.as_ref().ok_or(KernelError::InvariantViolation {
                    reason: "trace rows not resolved despite trace polynomials",
                })?);
                Ok(Box::new(TraceOpeningPoly::<F, R> {
                    rows,
                    kind,
                    placement,
                    _field: PhantomData,
                }) as Box<dyn MultilinearPoly<F>>)
            }
        })
        .collect()
}

const fn is_block_embedded(polynomial: JoltCommittedPolynomial) -> bool {
    matches!(
        polynomial,
        JoltCommittedPolynomial::TrustedAdvice
            | JoltCommittedPolynomial::UntrustedAdvice
            | JoltCommittedPolynomial::ProgramBytecode
            | JoltCommittedPolynomial::ProgramImageInit
    )
}

/// The committed values the trace views read: a row per cycle of the opened
/// domain, constant from `prefix_len` on.
trait OpeningRows: Send + Sync + 'static {
    fn cycles(&self) -> usize;

    fn prefix_len(&self) -> usize;

    /// The row of every cycle from `prefix_len` on, if any.
    fn tail(&self) -> Option<CommittedColumnsWitness>;

    fn row(&self, cycle: usize) -> CommittedColumnsWitness;
}

pub(crate) struct OpeningColumns {
    rd_inc: Vec<i128>,
    ram_inc: Vec<i128>,
    lookup_index: Vec<u128>,
    bytecode_pc: Vec<u64>,
    ram_address: Vec<u64>,
    logical_cycles: usize,
    tail: Option<CommittedColumnsWitness>,
}

impl OpeningColumns {
    fn collect<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<Self, KernelError<F>> {
        let cycles = 1usize << log_t;
        let mut retained = cycles;
        let mut tail = None;
        if let Some(access) = witness.random_access() {
            if cycles <= access.cycles() {
                retained = BundleAccess::<CommittedColumnsWitness>::View(&access)
                    .constant_suffix_start(cycles)?;
                if retained < cycles {
                    tail = Some(access.window::<CommittedColumnsWitness>(retained)?);
                }
                #[cfg(feature = "parallel")]
                {
                    let mut columns = Self::collect_par(&access, retained)?;
                    columns.logical_cycles = cycles;
                    columns.tail = tail;
                    return Ok(columns);
                }
            }
        }
        let mut consumers = (CollectOpeningColumns {
            columns: Self {
                rd_inc: Vec::with_capacity(retained),
                ram_inc: Vec::with_capacity(retained),
                lookup_index: Vec::with_capacity(retained),
                bytecode_pc: Vec::with_capacity(retained),
                ram_address: Vec::with_capacity(retained),
                logical_cycles: cycles,
                tail,
            },
        },);
        stream_witnesses(witness, 0..retained, COLLECT_CHUNK, &mut consumers)?;
        let columns = consumers.0.columns;
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
            logical_cycles: cycles,
            tail: None,
        })
    }
}

impl OpeningRows for OpeningColumns {
    fn cycles(&self) -> usize {
        self.logical_cycles
    }

    fn prefix_len(&self) -> usize {
        self.rd_inc.len()
    }

    fn tail(&self) -> Option<CommittedColumnsWitness> {
        self.tail
    }

    #[inline]
    fn row(&self, cycle: usize) -> CommittedColumnsWitness {
        if let Some(tail) = self.tail.filter(|_| cycle >= self.prefix_len()) {
            return tail;
        }
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

/// The witness's resident rows over the opened domain: every read
/// re-extracts its cycle's row.
struct ResidentRows {
    rows: BundleStore<CommittedColumnsWitness>,
    cycles: usize,
    prefix_len: usize,
    tail: Option<CommittedColumnsWitness>,
}

impl ResidentRows {
    /// The rows of the `2^log_t`-cycle domain. Every row before the constant
    /// suffix is extracted once here, so that an extraction error fails the
    /// preparation and the views' reads, which cannot fail, repeat only
    /// extractions that succeeded.
    fn new<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<Self, KernelError<F>> {
        let rows = BundleStore::<CommittedColumnsWitness>::resident(witness, log_t, ROWS)?;
        // The rows span `2^log_t` cycles, so the shift cannot overflow.
        let cycles = 1usize << log_t;
        let access = rows.access();
        let prefix_len = access.constant_suffix_start(cycles)?;
        let tail = (prefix_len < cycles)
            .then(|| access.row(prefix_len))
            .transpose()?;
        let extract = |cycle: usize| access.row(cycle).map(|_| ());
        #[cfg(feature = "parallel")]
        (0..prefix_len).into_par_iter().try_for_each(extract)?;
        #[cfg(not(feature = "parallel"))]
        (0..prefix_len).try_for_each(extract)?;
        Ok(Self {
            rows,
            cycles,
            prefix_len,
            tail,
        })
    }
}

impl OpeningRows for ResidentRows {
    fn cycles(&self) -> usize {
        self.cycles
    }

    fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    fn tail(&self) -> Option<CommittedColumnsWitness> {
        self.tail
    }

    #[expect(
        clippy::expect_used,
        reason = "`new` extracted every row before the suffix, and a row is a pure function of \
                  the immutable trace and its cycle"
    )]
    #[inline]
    fn row(&self, cycle: usize) -> CommittedColumnsWitness {
        match self.tail {
            Some(tail) if cycle >= self.prefix_len => tail,
            _ => self
                .rows
                .access()
                .row(cycle)
                .expect("`ResidentRows::new` extracted every row before the suffix"),
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
// Trace openings
// ---------------------------------------------------------------------------

struct TraceOpeningPoly<F: JoltField, R> {
    rows: Arc<R>,
    kind: ColumnKind,
    placement: TracePlacement,
    _field: PhantomData<F>,
}

impl<F: JoltField, R: OpeningRows> TraceOpeningPoly<F, R> {
    #[inline]
    fn column_value(&self, row: &CommittedColumnsWitness) -> Option<(usize, F)> {
        if self.kind.is_one_hot() {
            self.kind
                .hot_address(row)
                .map(|address| (address, F::one()))
        } else {
            let increment = self.kind.increment(row);
            (increment != 0).then(|| (0, F::from_i128(increment)))
        }
    }

    #[inline]
    fn entry(&self, cycle: usize) -> Option<(usize, F)> {
        self.column_value(&self.rows.row(cycle))
            .map(|(address, value)| (self.placement.index(cycle, address), value))
    }
}

impl<F: JoltField, R: OpeningRows> MultilinearPoly<F> for TraceOpeningPoly<F, R> {
    fn num_vars(&self) -> usize {
        self.placement.total_vars
    }

    fn evaluate(&self, point: &[F]) -> F {
        debug_assert_eq!(point.len(), self.placement.total_vars);
        let eq = TensorEqTable::new(point);
        let explicit = scatter_sum(self.rows.prefix_len(), |range| {
            let mut acc = F::zero();
            for cycle in range {
                if let Some((index, value)) = self.entry(cycle) {
                    acc += value * eq.evaluate_index(index);
                }
            }
            acc
        });
        explicit
            + self
                .rows
                .tail()
                .and_then(|row| self.column_value(&row))
                .map_or(F::zero(), |(address, value)| {
                    value
                        * self.placement.suffix_evaluation(
                            point,
                            self.rows.prefix_len(),
                            self.rows.cycles(),
                            address,
                        )
                })
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F])) {
        let entries = (0..self.rows.cycles())
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
        let mut result = scatter_fold(self.rows.prefix_len(), num_cols, |range, acc| {
            for cycle in range {
                if let Some((index, value)) = self.entry(cycle) {
                    if self.kind.is_one_hot() {
                        acc[index & mask] += left[index >> sigma];
                    } else {
                        acc[index & mask] += left[index >> sigma] * value;
                    }
                }
            }
        });
        if let Some((address, value)) = self.rows.tail().and_then(|row| self.column_value(&row)) {
            self.placement.fold_suffix(
                left,
                sigma,
                self.rows.prefix_len()..self.rows.cycles(),
                address,
                value,
                &mut result,
            );
        }
        result
    }
}

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

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module: fail loudly")]
mod tests {
    use jolt_claims::protocols::jolt::TracePolynomialOrder;
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_witness::{JoltVmWitnessInputs, JoltWitnessOracle, TraceBackend, WitnessError};

    use super::*;
    use crate::optimized::heap_peak::peak_above_start;
    use crate::optimized::streaming::StreamingFamilies;
    use crate::optimized::testing::{
        ram_fixture, random_scalars, with_ram_fixture, FixtureShape, RamOp,
    };
    use crate::{JoltBackend, ReferenceBackend};

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

    /// The reference views, then the optimized and the streaming ones.
    #[expect(clippy::type_complexity, reason = "test helper pairing the tiers")]
    fn prepare_tiers(
        witness: &dyn JoltWitnessPlane<Fr>,
        grid: CommitmentGrid,
        mut order: Vec<JoltCommittedPolynomial>,
    ) -> (
        Vec<JoltCommittedPolynomial>,
        Vec<Box<dyn MultilinearPoly<Fr>>>,
        [Vec<Box<dyn MultilinearPoly<Fr>>>; 2],
    ) {
        order.push(JoltCommittedPolynomial::ProgramBytecode);
        order.push(JoltCommittedPolynomial::ProgramImageInit);
        let mut precommitted_tables = BTreeMap::new();
        let _ = precommitted_tables.insert(
            JoltCommittedPolynomial::ProgramBytecode,
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
            Box::new(|| Ok(precommitted_tables.clone())),
            grid,
        )
        .unwrap();
        let lazy = [
            &OptimizedBackend as &dyn JointOpeningPolynomials<Fr>,
            &StreamingJointOpening,
        ]
        .map(|backend| {
            backend
                .prepare(
                    &mut ProofSession::default(),
                    witness,
                    &order,
                    Box::new(|| Ok(precommitted_tables.clone())),
                    grid,
                )
                .unwrap()
        });
        (order, reference, lazy)
    }

    /// The trace polynomials `witness` commits.
    fn trace_ids(witness: &TraceBackend) -> Vec<JoltCommittedPolynomial> {
        JoltWitnessOracle::<Fr>::committed_order(witness)
            .unwrap()
            .into_iter()
            .filter(|id| !is_block_embedded(*id))
            .collect()
    }

    fn assert_parity(order: TracePolynomialOrder, widen: usize) {
        for prefix in [1, 2, 3, 7, 8, 9, 15, 16] {
            assert_prefix_parity(order, widen, prefix);
        }
    }

    fn assert_prefix_parity(order: TracePolynomialOrder, widen: usize, prefix: usize) {
        let shape = FixtureShape {
            log_t: LOG_T,
            ram_k: RAM_K,
        };
        let ops = fixture_ops().into_iter().cycle().take(prefix).collect();
        with_ram_fixture(shape, ops, |witness| {
            let grid = CommitmentGrid {
                total_vars: LOG_T + 4 + widen,
                log_t: LOG_T,
                log_k_chunk: 4,
                order,
            };
            let ids: Vec<JoltCommittedPolynomial> = witness
                .committed_order()
                .unwrap()
                .into_iter()
                .filter(|id| {
                    !matches!(
                        id,
                        JoltCommittedPolynomial::TrustedAdvice
                            | JoltCommittedPolynomial::UntrustedAdvice
                    )
                })
                .collect();
            let (ids, reference, lazy) = prepare_tiers(witness, grid, ids);
            let point = random_scalars(grid.total_vars, 23);
            for (tier, views) in ["optimized", "streaming"].into_iter().zip(&lazy) {
                for ((id, reference), view) in ids.iter().zip(&reference).zip(views) {
                    assert_eq!(
                        view.num_vars(),
                        reference.num_vars(),
                        "{tier} {id:?}: num_vars diverged"
                    );
                    assert_eq!(
                        view.to_dense().as_ref(),
                        reference.to_dense().as_ref(),
                        "{tier} {id:?}: dense table diverged"
                    );
                    for sigma in 0..=grid.total_vars {
                        let left =
                            random_scalars(1 << (grid.total_vars - sigma), 29 + sigma as u64);
                        assert_eq!(
                            view.fold_rows(&left, sigma),
                            reference.fold_rows(&left, sigma),
                            "{tier} {id:?}: fold_rows diverged at sigma {sigma}"
                        );
                    }
                    assert_eq!(
                        view.evaluate(&point),
                        reference.evaluate(&point),
                        "{tier} {id:?}: evaluation diverged"
                    );
                }
            }
        });
    }

    #[test]
    fn cycle_major_matches_reference() {
        assert_parity(TracePolynomialOrder::CycleMajor, 0);
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
    fn address_major_odd_widening_matches_reference() {
        assert_parity(TracePolynomialOrder::AddressMajor, 1);
    }

    /// `with_streaming` installs the streaming opening. Its views open the
    /// default opening's values, so only its behavior shows which slot a
    /// backend holds: it refuses a witness without resident rows, from which
    /// the default opening collects its columns. The module builds under
    /// Dory only.
    #[test]
    fn with_streaming_installs_the_streaming_joint_opening() {
        let resident = ram_fixture(
            FixtureShape {
                log_t: LOG_T,
                ram_k: RAM_K,
            },
            Vec::new(),
            fixture_ops(),
        );
        // A configured domain shorter than the trace leaves no resident
        // rows; chunk visits still serve its `2^log_t` cycles.
        let mut chunk_only = TraceBackend::new(
            resident.config.clone(),
            JoltVmWitnessInputs::new(
                &resident.program,
                &resident.preprocessing,
                resident.trace.clone(),
            ),
        );
        chunk_only.config.log_t = LOG_T - 1;
        let grid = CommitmentGrid {
            total_vars: LOG_T - 1 + 4,
            log_t: LOG_T - 1,
            log_k_chunk: 4,
            order: TracePolynomialOrder::CycleMajor,
        };
        let ids = trace_ids(&chunk_only);
        let prepare = |backend: JoltBackend<Fr, DoryScheme>| {
            backend
                .joint_opening
                .prepare(
                    &mut ProofSession::default(),
                    &chunk_only,
                    &ids,
                    Box::new(|| Ok(BTreeMap::new())),
                    grid,
                )
                .map(|_| ())
        };
        prepare(JoltBackend::optimized()).unwrap();
        assert!(matches!(
            prepare(JoltBackend::optimized().with_streaming(StreamingFamilies {
                instruction_rows: true,
                ..StreamingFamilies::default()
            })),
            Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
        ));
    }

    /// The streaming views hold no per-cycle column: preparing them stays
    /// far below the 64 bytes per cycle the default opening collects.
    #[test]
    fn streaming_opening_collects_no_columns() {
        let (log_t, ops) = (12, 3000);
        let witness = ram_fixture(
            FixtureShape {
                log_t,
                ram_k: RAM_K,
            },
            Vec::new(),
            fixture_ops().into_iter().cycle().take(ops).collect(),
        );
        let grid = CommitmentGrid {
            total_vars: log_t + 4,
            log_t,
            log_k_chunk: 4,
            order: TracePolynomialOrder::CycleMajor,
        };
        let ids = trace_ids(&witness);
        let prepare = || {
            JointOpeningPolynomials::<Fr>::prepare(
                &StreamingJointOpening,
                &mut ProofSession::default(),
                &witness,
                &ids,
                Box::new(|| Ok(BTreeMap::new())),
                grid,
            )
            .unwrap()
        };
        let (views, held) = peak_above_start(prepare);
        assert_eq!(views.len(), ids.len());
        assert!(
            held < 64 * ops / 8,
            "preparing the streaming views held {held} bytes over {ops} rows"
        );
    }
}
