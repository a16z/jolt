//! Optimized instruction read+RAF checking (stage 5) kernel.
//!
//! Same math and phase structure as the reference kernel (see
//! `reference/instruction_read_raf.rs`): 8-variable prefix–suffix phases over
//! the 128 address rounds, then a plain multilinear product over the `log_T`
//! cycle rounds. Field arithmetic is exact, so every reorganization below
//! emits byte-identical round polynomials and output claims. The ported
//! legacy optimizations:
//!
//! - **Fused, block-bucketed phase scans**: the trace is cut once into
//!   cache-sized blocks whose cycle offsets are bucketed by (lookup table,
//!   RAF flag). Each phase is then one parallel pass: per block, condense
//!   the eq weights in cycle order, then accumulate the RAF and per-table
//!   suffix `Q` sums run by run, gathering only from the cache-resident
//!   block under one table and flag — no table-order gathers across the
//!   trace, and predictable dispatch.
//! - **Deferred-reduction accumulation** (`F::Accumulator`) with primitive
//!   scalar multiplies (`mul_u64`/`mul_u128`, no Montgomery conversion of the
//!   scalar): the scans avoid a full reduction per row; suffixes are
//!   classified once per table (`One` / {0,1}-valued / general) so most rows
//!   add instead of multiply, and every `One` suffix shares one weight-mass
//!   sum per table.
//! - **Allocation-free address messages**: prefix/suffix extension
//!   evaluations go into per-thread scratch reused across the chunk domain
//!   (the reference allocates fresh eval vectors per point), evaluated at
//!   `c ∈ {0,2}` only with `s(1) = previous_claim − s(0)`.
//! - **Gruen split-eq cycle rounds** (`GruenSplitEqPolynomial`): the
//!   `eq(r_reduction, ·)` factor is never materialized or bound as a `T`-sized
//!   table; each round computes `q(t) = Σ_y E_out·E_in·(Val·Πra)(t, y)` with
//!   incrementally-updated factor evaluations and multiplies by the linear eq
//!   factor `ℓ(t)` once.
//! - **Split-eq flag claims**: the output lookup-table/RAF flag sums use the
//!   `E_hi ⊗ E_lo` factorization of `eq(r_cycle, ·)` instead of a `T`-sized
//!   eq table.
//! - **Shared witness rows**: re-emulating sources keep a strong
//!   `SharedInstructionRows` carry in the [`ProofSession`]; slice-backed
//!   sources keep only `SharedInstructionRowsWeak`, sharing inside one stage
//!   and rebuilding index-parallel later so `40 B × T` rows do not survive
//!   through the prover's peak window.

use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::instruction::{
    InstructionReadRafDimensions, CANONICAL_INSTRUCTION_ADDRESS,
};
use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafOutputClaims;
use jolt_field::{Accumulator, JoltField};
use jolt_lookup_tables::tables::prefixes::{PrefixEval, ALL_PREFIXES};
use jolt_lookup_tables::tables::suffixes::{SuffixEval, Suffixes};
use jolt_lookup_tables::{LookupBits, LookupTableKind, XLEN as RISCV_XLEN};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, TensorEqTable, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::SumcheckInputClaims;
use jolt_verifier::stages::stage5::InstructionReadRaf;
#[cfg(feature = "akita")]
use jolt_witness::witnesses::{BalancedIncColumn, FusedInc};
use jolt_witness::witnesses::{
    BytecodePc, InstructionRafFlag, LookupIndex, RemappedRamAddress, TableIndex,
};
use jolt_witness::{stream_witnesses, JoltWitnessPlane, StreamConsumer, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::{
    accumulate_product_grid, collect_par_map, map_indices, map_reduce_chunks, scan_chunk_size,
    RoundProgress,
};
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Address variables bound per phase — identical to the reference kernel (and
/// to the legacy prover below its 2^24-cycle threshold).
const CHUNK_LEN: usize = 8;
const CHUNK_SIZE: usize = 1 << CHUNK_LEN;

const _: () = assert!(
    LookupTableKind::<RISCV_XLEN>::COUNT < u8::MAX as usize,
    "InstructionCycleRow packs lookup table indices as u8"
);

/// One packed per-cycle row: the stage-5 facts plus the bytecode/RAM and
/// packed fused-inc sources used by later one-hot kernels. The lookup index
/// is split into native limbs and the PC/table/flags share one word, keeping
/// the retained row at 40 bytes in Akita mode.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct InstructionCycleRow {
    lookup_index_lo: u64,
    lookup_index_hi: u64,
    ram_address_plus_one: u64,
    #[cfg(feature = "akita")]
    fused_inc_magnitude: u64,
    packed_pc_and_flags: u64,
}

const PACKED_PC_BITS: u32 = 56;
const PACKED_TABLE_BITS: u32 = 6;
const PACKED_PC_MASK: u64 = (1 << PACKED_PC_BITS) - 1;
const PACKED_TABLE_MASK: u64 = (1 << PACKED_TABLE_BITS) - 1;
const PACKED_TABLE_SHIFT: u32 = PACKED_PC_BITS;
const PACKED_RAF_SHIFT: u32 = PACKED_TABLE_SHIFT + PACKED_TABLE_BITS;
#[cfg(feature = "akita")]
const PACKED_INC_SIGN_SHIFT: u32 = PACKED_RAF_SHIFT + 1;

const _: () = assert!(LookupTableKind::<RISCV_XLEN>::COUNT < 1 << PACKED_TABLE_BITS);

impl InstructionCycleRow {
    pub(crate) fn new(
        lookup_index: u128,
        table_index: Option<usize>,
        raf_flag: bool,
        bytecode_pc: usize,
        remapped_ram_address: Option<u64>,
        #[cfg(feature = "akita")] fused_inc: FusedInc,
    ) -> Self {
        debug_assert!(table_index.is_none_or(|index| index < u8::MAX as usize));
        #[cfg(feature = "akita")]
        debug_assert!(fused_inc.0.unsigned_abs() <= u64::MAX as u128);
        let pc = bytecode_pc as u64;
        assert!(pc <= PACKED_PC_MASK, "bytecode PC exceeds packed row");
        let table_plus_one = table_index.map_or(0, |index| index as u64 + 1);
        let packed_pc_and_flags =
            pc | (table_plus_one << PACKED_TABLE_SHIFT) | (u64::from(raf_flag) << PACKED_RAF_SHIFT);
        #[cfg(feature = "akita")]
        let packed_pc_and_flags =
            packed_pc_and_flags | (u64::from(fused_inc.0 < 0) << PACKED_INC_SIGN_SHIFT);
        Self {
            lookup_index_lo: lookup_index as u64,
            lookup_index_hi: (lookup_index >> 64) as u64,
            ram_address_plus_one: remapped_ram_address.map_or(0, |address| address + 1),
            #[cfg(feature = "akita")]
            fused_inc_magnitude: fused_inc.0.unsigned_abs() as u64,
            packed_pc_and_flags,
        }
    }

    #[inline(always)]
    pub(crate) fn lookup_index(&self) -> u128 {
        u128::from(self.lookup_index_lo) | (u128::from(self.lookup_index_hi) << 64)
    }

    #[inline]
    pub(crate) fn table_index(&self) -> Option<usize> {
        let table_plus_one =
            ((self.packed_pc_and_flags >> PACKED_TABLE_SHIFT) & PACKED_TABLE_MASK) as usize;
        table_plus_one.checked_sub(1)
    }

    #[inline]
    pub(crate) fn bytecode_pc(&self) -> usize {
        (self.packed_pc_and_flags & PACKED_PC_MASK) as usize
    }

    #[inline]
    pub(crate) fn remapped_ram_address(&self) -> Option<u64> {
        self.ram_address_plus_one.checked_sub(1)
    }

    #[inline]
    pub(crate) fn raf_flag(&self) -> bool {
        self.packed_pc_and_flags & (1 << PACKED_RAF_SHIFT) != 0
    }

    #[cfg(feature = "akita")]
    #[inline]
    pub(crate) fn fused_inc_row(&self, column: BalancedIncColumn) -> usize {
        let magnitude = i128::from(self.fused_inc_magnitude);
        let value = if self.packed_pc_and_flags & (1 << PACKED_INC_SIGN_SHIFT) != 0 {
            -magnitude
        } else {
            magnitude
        };
        FusedInc(value).selected_row(column)
    }

    #[cfg(feature = "akita")]
    #[inline]
    pub(crate) fn fused_inc<F: JoltField>(&self) -> F {
        let magnitude = F::from_u64(self.fused_inc_magnitude);
        if self.packed_pc_and_flags & (1 << PACKED_INC_SIGN_SHIFT) != 0 {
            -magnitude
        } else {
            magnitude
        }
    }
}

#[cfg(feature = "akita")]
const _: () = assert!(std::mem::size_of::<InstructionCycleRow>() == 40);
#[cfg(not(feature = "akita"))]
const _: () = assert!(std::mem::size_of::<InstructionCycleRow>() == 32);

/// The bundle row the packing pass extracts; never materialized beyond one
/// streaming chunk.
#[derive(Clone, Copy, Debug, WitnessBundle)]
struct WideInstructionRow {
    lookup_index: LookupIndex,
    table_index: TableIndex,
    raf_flag: InstructionRafFlag,
    bytecode_pc: BytecodePc,
    remapped_ram_address: RemappedRamAddress,
    #[cfg(feature = "akita")]
    fused_inc: FusedInc,
}

struct PackRows {
    rows: Vec<InstructionCycleRow>,
}

impl StreamConsumer for PackRows {
    type Witness = WideInstructionRow;

    fn consume(&mut self, chunk: &[WideInstructionRow]) {
        self.rows.extend(chunk.iter().map(|row| {
            InstructionCycleRow::new(
                row.lookup_index.0,
                row.table_index.0,
                row.raf_flag.0,
                row.bytecode_pc.0,
                row.remapped_ram_address.0,
                #[cfg(feature = "akita")]
                row.fused_inc,
            )
        }));
    }
}

impl InstructionCycleRow {
    /// One streaming bundle pass over the cycle domain, packed row by row (the
    /// wide bundle row exists only per chunk).
    pub(crate) fn collect<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<Vec<Self>, KernelError<F>> {
        // Slice-backed sources pack index-parallel (the wide bundle row still
        // never exists beyond a register); re-emulating sources stream.
        if let Some(access) = witness.random_access() {
            if cycles <= access.cycles() {
                let rows = collect_par_map(&access, cycles, |row: WideInstructionRow| {
                    Self::new(
                        row.lookup_index.0,
                        row.table_index.0,
                        row.raf_flag.0,
                        row.bytecode_pc.0,
                        row.remapped_ram_address.0,
                        #[cfg(feature = "akita")]
                        row.fused_inc,
                    )
                })?;
                return Ok(rows);
            }
        }
        let mut consumers = (PackRows {
            rows: Vec::with_capacity(cycles),
        },);
        stream_witnesses(witness, 0..cycles, 1 << 12, &mut consumers)?;
        Ok(consumers.0.rows)
    }
}

/// The collected stage-5 rows, parked in the [`ProofSession`] for the
/// stage-6b instruction RA virtualization kernel (its committed one-hot
/// chunks are chunks of the same per-cycle lookup index) and the
/// stage-6a/6b booleanity kernels (all three one-hot chunk families).
///
/// Non-final consumers reclaim with `take`, clone the [`Arc`], and park the
/// carry back for the later stages.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SharedInstructionRows(pub(crate) Arc<Vec<InstructionCycleRow>>);

/// The slice-backed counterpart of [`SharedInstructionRows`]: a weak handle,
/// so same-stage co-consumers share one collection but the 40 B × T rows
/// never outlive their stage — later stages re-derive them index-parallel
/// instead of carrying them across the prover's peak window.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SharedInstructionRowsWeak(pub(crate) std::sync::Weak<Vec<InstructionCycleRow>>);

impl InstructionCycleRow {
    /// Reclaim the parked stage-5 rows (the length guard makes a stale carry
    /// impossible to consume) or collect them fresh, and park the carry back
    /// for later consumers.
    pub(crate) fn shared<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<Arc<Vec<Self>>, KernelError<F>> {
        // A parked strong carry is always honored (re-emulating sources, and
        // tests that inject rows a witness would not produce).
        let carried = match session.take::<SharedInstructionRows>() {
            Some(SharedInstructionRows(rows)) if rows.len() == cycles => Some(rows),
            _ => None,
        };
        if witness.random_access().is_some() {
            // Slice-backed: consumers share within a stage through a weak
            // handle; once the stage's kernels drop, the rows free, and later
            // stages re-derive them index-parallel.
            let upgraded = || {
                session
                    .state::<SharedInstructionRowsWeak>()
                    .and_then(|weak| weak.0.upgrade())
                    .filter(|rows| rows.len() == cycles)
            };
            let rows = match carried.or_else(upgraded) {
                Some(rows) => rows,
                None => Arc::new(Self::collect(witness, cycles)?),
            };
            session.park(SharedInstructionRowsWeak(Arc::downgrade(&rows)));
            return Ok(rows);
        }
        let rows = match carried {
            Some(rows) => rows,
            None => Arc::new(Self::collect(witness, cycles)?),
        };
        session.park(SharedInstructionRows(Arc::clone(&rows)));
        Ok(rows)
    }
}

/// Optimized [`PrepareKernel`] implementor for the `instruction_read_raf`
/// slot.
pub struct OptimizedInstructionReadRaf;

impl<F: JoltField> PrepareKernel<F, InstructionReadRaf<F>> for OptimizedInstructionReadRaf {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionReadRaf<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionReadRaf<F>>>, KernelError<F>> {
        let dimensions = inputs.relation.dimensions();
        let rows: Arc<Vec<InstructionCycleRow>> = Arc::new(InstructionCycleRow::collect(
            witness,
            1 << dimensions.log_t(),
        )?);
        if witness.random_access().is_some() {
            session.park(SharedInstructionRowsWeak(Arc::downgrade(&rows)));
        } else {
            session.park(SharedInstructionRows(Arc::clone(&rows)));
        }
        Ok(Box::new(OptimizedInstructionReadRafKernel::new(
            dimensions,
            &inputs.points.lookup_output,
            rows,
            inputs.challenges.gamma,
        )?))
    }
}

/// One RAF prefix–suffix decomposition — same shape and binding as the
/// reference kernel's.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct RafDecomposition<F: JoltField> {
    prefix: Polynomial<F>,
    q_shift: Polynomial<F>,
    q_value: Polynomial<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    checkpoint: F,
}

impl<F: JoltField> RafDecomposition<F> {
    fn empty() -> Self {
        Self {
            prefix: Polynomial::new(vec![F::zero()]),
            q_shift: Polynomial::new(vec![F::zero()]),
            q_value: Polynomial::new(vec![F::zero()]),
            checkpoint: F::zero(),
        }
    }

    /// WARNING: the canonical-address decomposition is an AND over address
    /// bits, so its bound-prefix accumulator is a *product* and its empty
    /// value is one (see the reference kernel).
    fn empty_product() -> Self {
        Self {
            checkpoint: F::one(),
            ..Self::empty()
        }
    }

    /// `(eval at c = 0, eval at c = 2)` of `prefix · q_shift + q_value` at
    /// chunk-domain index `b`.
    #[inline]
    fn message_evals(&self, b: usize, half: usize) -> (F, F) {
        let (p0, p2) = extension_pair(self.prefix.evals(), b, half);
        let (s0, s2) = extension_pair(self.q_shift.evals(), b, half);
        let (v0, v2) = extension_pair(self.q_value.evals(), b, half);
        (p0 * s0 + v0, p2 * s2 + v2)
    }

    fn bind(&mut self, challenge: F) {
        self.prefix
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.q_shift
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.q_value
            .bind_with_order(challenge, BindingOrder::HighToLow);
    }
}

/// Linear extension of a dense table's top variable at `c = 0` and `c = 2`:
/// `(evals[b], 2·evals[b + half] − evals[b])`.
#[inline]
fn extension_pair<F: JoltField>(evals: &[F], b: usize, half: usize) -> (F, F) {
    let lo = evals[b];
    let hi = evals[b + half];
    (lo, hi + hi - lo)
}

/// Cycle-round state: the Gruen-split eq factor plus the cycle tables.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct CycleState<F: JoltField> {
    gruen: GruenSplitEqPolynomial<F>,
    tables: CycleTables<F>,
    /// Reused low-to-high binding buffer (swapped through every bind).
    bind_scratch: Vec<F>,
}

/// The cycle tables' lifecycle. The address/cycle handoff leaves them
/// *pending*: the first cycle round's message evaluates the bases on the
/// fly (a packed-byte lookup for the combined value, `v_table` products
/// for the ra decomposition), and the first cycle bind materializes the
/// half-domain tables directly under that challenge — the full-T dense
/// tables ((1 + ra_count) × 32 B × T, the stage-5 peak allocation) never
/// exist. Values are identical to materialize-then-bind: the bases are the
/// same, and `lo + r·(hi − lo)` is the binding formula either way.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum CycleTables<F: JoltField> {
    Pending(PendingCycleTables<F>),
    Dense {
        combined_val: Polynomial<F>,
        ra: Vec<Polynomial<F>>,
    },
}

/// Everything the pending-base evaluations need beyond the kernel's own
/// rows / claim columns / phase eq tables.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct PendingCycleTables<F: JoltField> {
    /// Per-table combined value at the bound address point.
    table_values: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    raf_interleaved: F,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    raf_identity: F,
}

/// Per-thread RAF scan accumulators over one phase's chunk domain, in
/// deferred-reduction form.
struct RafScan<F: JoltField> {
    shift_half: Vec<F::Accumulator>,
    left: Vec<F::Accumulator>,
    right: Vec<F::Accumulator>,
    shift_full: Vec<F::Accumulator>,
    identity: Vec<F::Accumulator>,
    upper_all_ones: Vec<F::Accumulator>,
}

/// The reduced (field-element) form of one thread's [`RafScan`].
struct RafSums<F> {
    shift_half: Vec<F>,
    left: Vec<F>,
    right: Vec<F>,
    shift_full: Vec<F>,
    identity: Vec<F>,
    upper_all_ones: Vec<F>,
}

impl<F: JoltField> RafScan<F> {
    fn new() -> Self {
        Self {
            shift_half: vec![F::Accumulator::default(); CHUNK_SIZE],
            left: vec![F::Accumulator::default(); CHUNK_SIZE],
            right: vec![F::Accumulator::default(); CHUNK_SIZE],
            shift_full: vec![F::Accumulator::default(); CHUNK_SIZE],
            identity: vec![F::Accumulator::default(); CHUNK_SIZE],
            upper_all_ones: vec![F::Accumulator::default(); CHUNK_SIZE],
        }
    }

    fn reduce(self) -> RafSums<F> {
        let reduce = |accumulators: Vec<F::Accumulator>| -> Vec<F> {
            accumulators
                .into_iter()
                .map(|accumulator| accumulator.reduce())
                .collect()
        };
        RafSums {
            shift_half: reduce(self.shift_half),
            left: reduce(self.left),
            right: reduce(self.right),
            shift_full: reduce(self.shift_full),
            identity: reduce(self.identity),
            upper_all_ones: reduce(self.upper_all_ones),
        }
    }
}

impl<F: JoltField> RafSums<F> {
    fn zero() -> Self {
        Self {
            shift_half: vec![F::zero(); CHUNK_SIZE],
            left: vec![F::zero(); CHUNK_SIZE],
            right: vec![F::zero(); CHUNK_SIZE],
            shift_full: vec![F::zero(); CHUNK_SIZE],
            identity: vec![F::zero(); CHUNK_SIZE],
            upper_all_ones: vec![F::zero(); CHUNK_SIZE],
        }
    }

    fn merge(mut self, other: Self) -> Self {
        let pairs = [
            (&mut self.shift_half, &other.shift_half),
            (&mut self.left, &other.left),
            (&mut self.right, &other.right),
            (&mut self.shift_full, &other.shift_full),
            (&mut self.identity, &other.identity),
            (&mut self.upper_all_ones, &other.upper_all_ones),
        ];
        for (into, from) in pairs {
            for (a, b) in into.iter_mut().zip(from) {
                *a += *b;
            }
        }
        self
    }
}

/// Scan classes: `2 · (table + 1) + raf_flag`, with `2 · 0 + raf_flag` for
/// no-table cycles.
const SCAN_CLASSES: usize = 2 * (LookupTableKind::<RISCV_XLEN>::COUNT + 1);

/// Cycles per scan block: a block's rows and weights stay cache-resident
/// while its class runs gather from them.
const SCAN_BLOCK: usize = 1 << 10;

const _: () = assert!(SCAN_BLOCK <= u16::MAX as usize, "block offsets are u16");

/// One block of `SCAN_BLOCK` consecutive cycles (fewer for a trace shorter
/// than a block), bucketed by scan class.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct ScanBlock {
    /// `starts[class]..starts[class + 1]` indexes the class's `offsets`.
    starts: [u16; SCAN_CLASSES + 1],
    /// The block's cycle offsets, stably sorted by scan class.
    offsets: [u16; SCAN_BLOCK],
}

impl ScanBlock {
    /// Buckets one block of rows whose table indices are already checked.
    fn new(rows: &[InstructionCycleRow]) -> Self {
        let mut starts = [0u16; SCAN_CLASSES + 1];
        for row in rows {
            starts[Self::class(row) + 1] += 1;
        }
        for class in 0..SCAN_CLASSES {
            starts[class + 1] += starts[class];
        }
        let mut next = starts;
        let mut offsets = [0u16; SCAN_BLOCK];
        for (offset, row) in rows.iter().enumerate() {
            let position = &mut next[Self::class(row)];
            offsets[usize::from(*position)] = offset as u16;
            *position += 1;
        }
        Self { starts, offsets }
    }

    fn class(row: &InstructionCycleRow) -> usize {
        2 * row.table_index().map_or(0, |table| table + 1) + usize::from(row.raf_flag())
    }
}

/// Scan tasks per pool thread, so work stealing balances the phase scan
/// across heterogeneous cores.
const SCAN_TASKS_PER_THREAD: usize = 8;

/// One phase's fused-scan parameters.
struct PhaseScan<'a, F> {
    /// The previous phase's chunk shift and bound-challenge eq table, folded
    /// into the weights before they are read (every phase but the first).
    condense: Option<(usize, &'a [F])>,
    suffix_len: usize,
    suffix_mask: u128,
    upper_suffix_bits: usize,
    /// Every lookup table, by `LookupTableKind::index()`.
    tables: Vec<LookupTableKind<RISCV_XLEN>>,
}

/// One phase's scan sums: the RAF sums, plus each table's suffix sums
/// (`table.suffixes()`-major over the chunk domain) by
/// `LookupTableKind::index()`, empty for tables no cycle selects.
struct PhaseSums<F> {
    raf: RafSums<F>,
    suffixes: Vec<Vec<F>>,
}

impl<F: JoltField> PhaseSums<F> {
    fn zero() -> Self {
        Self {
            raf: RafSums::zero(),
            suffixes: vec![Vec::new(); LookupTableKind::<RISCV_XLEN>::COUNT],
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.raf = self.raf.merge(other.raf);
        for (sums, other) in self.suffixes.iter_mut().zip(other.suffixes) {
            if sums.is_empty() {
                *sums = other;
            } else {
                for (a, b) in sums.iter_mut().zip(&other) {
                    *a += *b;
                }
            }
        }
        self
    }
}

/// One scan task's suffix accumulators for one lookup table: its `One`
/// sums (the table's weight mass per chunk, shared by every `One` suffix)
/// when it has a `One` suffix, then one row per other suffix.
struct TableScan<F: JoltField> {
    table: LookupTableKind<RISCV_XLEN>,
    has_one: bool,
    /// Non-`One` suffixes in `table.suffixes()` order, with their
    /// {0,1}-valued classification.
    row_suffixes: Vec<(Suffixes, bool)>,
    accumulators: Vec<F::Accumulator>,
}

impl<F: JoltField> TableScan<F> {
    fn new(table: LookupTableKind<RISCV_XLEN>) -> Self {
        let suffixes = table.suffixes();
        let has_one = suffixes
            .iter()
            .any(|suffix| matches!(suffix, Suffixes::One));
        let row_suffixes: Vec<(Suffixes, bool)> = suffixes
            .iter()
            .filter(|suffix| !matches!(suffix, Suffixes::One))
            .map(|&suffix| (suffix, suffix.is_01_valued()))
            .collect();
        let rows = usize::from(has_one) + row_suffixes.len();
        Self {
            table,
            has_one,
            row_suffixes,
            accumulators: vec![F::Accumulator::default(); rows * CHUNK_SIZE],
        }
    }

    /// `One` adds the weight, {0,1}-valued suffixes add it conditionally,
    /// general ones use the primitive-scalar multiply.
    #[inline]
    fn accumulate(&mut self, chunk: usize, suffix_bits: LookupBits, u: F) {
        let (one, rows) = self
            .accumulators
            .split_at_mut(usize::from(self.has_one) * CHUNK_SIZE);
        if self.has_one {
            one[chunk].add(u);
        }
        for (&(suffix, zero_one_valued), row) in self
            .row_suffixes
            .iter()
            .zip(rows.chunks_exact_mut(CHUNK_SIZE))
        {
            if zero_one_valued {
                if suffix.suffix_mle(suffix_bits) == 1 {
                    row[chunk].add(u);
                }
            } else {
                let value = suffix.suffix_mle(suffix_bits);
                if value != 0 {
                    row[chunk].fmadd_u64(u, value);
                }
            }
        }
    }

    /// The table's reduced suffix sums, in `table.suffixes()` order.
    fn reduce(self) -> Vec<F> {
        let mut rows = self
            .accumulators
            .chunks_exact(CHUNK_SIZE)
            .map(|row| row.iter().map(|accumulator| accumulator.reduce()));
        let one: Vec<F> = if self.has_one {
            rows.next().into_iter().flatten().collect()
        } else {
            Vec::new()
        };
        let mut sums = Vec::with_capacity(self.table.suffixes().len() * CHUNK_SIZE);
        for suffix in self.table.suffixes() {
            if matches!(suffix, Suffixes::One) {
                sums.extend_from_slice(&one);
            } else {
                sums.extend(rows.next().into_iter().flatten());
            }
        }
        sums
    }
}

impl<F: JoltField> PhaseScan<'_, F> {
    /// Scans the trace in parallel tasks of consecutive blocks, condensing
    /// `u_evals` in place.
    fn run(
        &self,
        blocks: &[ScanBlock],
        rows: &[InstructionCycleRow],
        u_evals: &mut [F],
    ) -> PhaseSums<F> {
        let task_blocks = scan_chunk_size(rows.len()).div_ceil(SCAN_TASKS_PER_THREAD * SCAN_BLOCK);
        let task_rows = task_blocks * SCAN_BLOCK;
        #[cfg(feature = "parallel")]
        {
            blocks
                .par_chunks(task_blocks)
                .zip(rows.par_chunks(task_rows))
                .zip(u_evals.par_chunks_mut(task_rows))
                .map(|((blocks, rows), u_evals)| self.scan(blocks, rows, u_evals))
                .reduce(PhaseSums::zero, PhaseSums::merge)
        }
        #[cfg(not(feature = "parallel"))]
        {
            blocks
                .chunks(task_blocks)
                .zip(rows.chunks(task_rows))
                .zip(u_evals.chunks_mut(task_rows))
                .map(|((blocks, rows), u_evals)| self.scan(blocks, rows, u_evals))
                .fold(PhaseSums::zero(), PhaseSums::merge)
        }
    }

    /// One task, a block at a time: condense the block's weights in cycle
    /// order (streaming the block into cache), then accumulate each class
    /// run — gathered from the cache-resident block under one table and
    /// one RAF flag, so the dispatch stays predictable.
    fn scan(
        &self,
        blocks: &[ScanBlock],
        rows: &[InstructionCycleRow],
        u_evals: &mut [F],
    ) -> PhaseSums<F> {
        let mut raf = RafScan::<F>::new();
        let mut table_scans: Vec<Option<TableScan<F>>> = (0..LookupTableKind::<RISCV_XLEN>::COUNT)
            .map(|_| None)
            .collect();
        for ((block, rows), u_evals) in blocks
            .iter()
            .zip(rows.chunks(SCAN_BLOCK))
            .zip(u_evals.chunks_mut(SCAN_BLOCK))
        {
            if let Some((shift, v_prev)) = self.condense {
                for (row, u) in rows.iter().zip(u_evals.iter_mut()) {
                    *u *= v_prev[((row.lookup_index() >> shift) as usize) & (CHUNK_SIZE - 1)];
                }
            }
            for (class, range) in block.starts.windows(2).enumerate() {
                let offsets = &block.offsets[usize::from(range[0])..usize::from(range[1])];
                if offsets.is_empty() {
                    continue;
                }
                let table_scan = (class / 2).checked_sub(1).map(|table| {
                    table_scans[table].get_or_insert_with(|| TableScan::new(self.tables[table]))
                });
                self.accumulate(class % 2 == 1, offsets, rows, u_evals, &mut raf, table_scan);
            }
        }
        PhaseSums {
            raf: raf.reduce(),
            suffixes: table_scans
                .into_iter()
                .map(|scan| scan.map_or_else(Vec::new, TableScan::reduce))
                .collect(),
        }
    }

    /// One class run: the RAF sums (deferred-reduction, primitive-scalar
    /// multiplies) and, for a table's run, its suffix sums.
    #[inline]
    fn accumulate(
        &self,
        raf_flag: bool,
        offsets: &[u16],
        rows: &[InstructionCycleRow],
        u_evals: &[F],
        raf: &mut RafScan<F>,
        mut table_scan: Option<&mut TableScan<F>>,
    ) {
        let &Self {
            suffix_len,
            suffix_mask,
            upper_suffix_bits,
            ..
        } = self;
        for &offset in offsets {
            let lookup_index = rows[usize::from(offset)].lookup_index();
            let u = u_evals[usize::from(offset)];
            let chunk = ((lookup_index >> suffix_len) as usize) & (CHUNK_SIZE - 1);
            let suffix_bits = lookup_index & suffix_mask;
            if CANONICAL_INSTRUCTION_ADDRESS
                && raf_flag
                && (upper_suffix_bits == 0
                    || (suffix_bits >> (suffix_len - upper_suffix_bits))
                        == (1u128 << upper_suffix_bits) - 1)
            {
                raf.upper_all_ones[chunk].add(u);
            }
            if !raf_flag {
                raf.shift_half[chunk].add(u);
                let (left, right) = LookupBits::new(suffix_bits, suffix_len).uninterleave();
                let left = u64::from(left);
                if left != 0 {
                    raf.left[chunk].fmadd_u64(u, left);
                }
                let right = u64::from(right);
                if right != 0 {
                    raf.right[chunk].fmadd_u64(u, right);
                }
            } else {
                raf.shift_full[chunk].add(u);
                if suffix_bits != 0 {
                    raf.identity[chunk].fmadd_u128(u, suffix_bits);
                }
            }
            if let Some(table_scan) = table_scan.as_deref_mut() {
                table_scan.accumulate(chunk, LookupBits::new(suffix_bits, suffix_len), u);
            }
        }
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct OptimizedInstructionReadRafKernel<F: JoltField> {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    dimensions: InstructionReadRafDimensions,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gamma: F,
    r_reduction: Vec<F>,
    rows: Arc<Vec<InstructionCycleRow>>,
    /// The trace in blocks of `SCAN_BLOCK` cycles, each bucketed by
    /// (lookup table, RAF flag).
    buckets: Vec<ScanBlock>,
    /// Condensed per-cycle eq weights (see the reference kernel).
    u_evals: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    prefix_checkpoints: Vec<PrefixEval<F>>,
    /// `ALL_PREFIXES` indices referenced by tables some cycle selects.
    prefix_indices: Vec<usize>,
    /// Materialized prefix chunk polynomials for the current phase, in
    /// `prefix_indices` order.
    prefix_tables: Vec<Polynomial<F>>,
    /// Per present table: enum value + suffix `Q` polynomials in
    /// `table.suffixes()` order.
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_keyed_polys))]
    suffix_tables: Vec<(LookupTableKind<RISCV_XLEN>, Vec<Polynomial<F>>)>,
    raf_left: RafDecomposition<F>,
    raf_right: RafDecomposition<F>,
    raf_identity: RafDecomposition<F>,
    raf_upper_all_ones: RafDecomposition<F>,
    /// Completed phases' bound-challenge eq tables.
    v_tables: Vec<Vec<F>>,
    phase_challenges: Vec<F>,
    cycle_challenges: Vec<F>,
    cycle: Option<CycleState<F>>,
    /// Packed per-cycle output-claim facts (bits 0..=6: `table_index + 1`,
    /// 0 for none; bit 7: the RAF flag), snapped at the address/cycle
    /// handoff so the full 40 B rows can free — the final flag walk needs
    /// only this byte per cycle.
    claim_columns: Vec<u8>,
    progress: RoundProgress,
}

impl<F: JoltField> OptimizedInstructionReadRafKernel<F> {
    pub(crate) fn new(
        dimensions: InstructionReadRafDimensions,
        r_reduction: &[F],
        rows: Arc<Vec<InstructionCycleRow>>,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        let address_bits = dimensions.instruction_address_bits();
        let log_t = dimensions.log_t();
        if address_bits != 2 * RISCV_XLEN {
            return Err(KernelError::Unsupported {
                reason: "instruction read-RAF supports only the 2·XLEN interleaved-operand \
                         address width",
            });
        }
        let ra_count = dimensions.num_virtual_ra_polys();
        if !address_bits.is_multiple_of(ra_count)
            || !(address_bits / ra_count).is_multiple_of(CHUNK_LEN)
        {
            return Err(KernelError::Unsupported {
                reason: "virtual RA chunk width must be a multiple of the phase width",
            });
        }
        if rows.len() != 1 << log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "stage-5 instruction rows".to_owned(),
                expected: 1 << log_t,
                got: rows.len(),
            });
        }
        if r_reduction.len() != log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "instruction claim-reduction point".to_owned(),
                expected: log_t,
                got: r_reduction.len(),
            });
        }

        let num_tables = LookupTableKind::<RISCV_XLEN>::COUNT;
        let present_tables = map_reduce_chunks(
            rows.len(),
            scan_chunk_size(rows.len()),
            |range| {
                let mut present = 0u64;
                for row in &rows[range] {
                    if let Some(table) = row.table_index() {
                        if table >= num_tables {
                            return Err(KernelError::InvariantViolation {
                                reason: "stage-5 row selects an unknown lookup table",
                            });
                        }
                        present |= 1 << table;
                    }
                }
                Ok(present)
            },
            |a, b| Ok(a? | b?),
            || Ok(0),
        )?;
        let mut present_prefixes = vec![false; ALL_PREFIXES.len()];
        for table in LookupTableKind::<RISCV_XLEN>::iter()
            .filter(|table| present_tables & (1 << table.index()) != 0)
        {
            for prefix in table.prefixes() {
                present_prefixes[*prefix as usize] = true;
            }
        }
        let prefix_indices = present_prefixes
            .into_iter()
            .enumerate()
            .filter_map(|(index, present)| present.then_some(index))
            .collect();

        let buckets = map_indices(rows.len().div_ceil(SCAN_BLOCK), |block| {
            ScanBlock::new(&rows[block * SCAN_BLOCK..((block + 1) * SCAN_BLOCK).min(rows.len())])
        });
        let mut kernel = Self {
            dimensions,
            gamma,
            r_reduction: r_reduction.to_vec(),
            rows,
            buckets,
            u_evals: eq_table(r_reduction),
            prefix_checkpoints: ALL_PREFIXES
                .iter()
                .map(|prefix| prefix.default_checkpoint::<F>())
                .collect(),
            prefix_indices,
            prefix_tables: Vec::new(),
            suffix_tables: Vec::new(),
            raf_left: RafDecomposition::empty(),
            raf_right: RafDecomposition::empty(),
            raf_identity: RafDecomposition::empty(),
            raf_upper_all_ones: RafDecomposition::empty_product(),
            v_tables: Vec::new(),
            phase_challenges: Vec::new(),
            cycle_challenges: Vec::new(),
            cycle: None,
            claim_columns: Vec::new(),
            progress: RoundProgress::new(dimensions.sumcheck_rounds()),
        };
        kernel.init_phase(0);
        Ok(kernel)
    }

    fn address_bits(&self) -> usize {
        self.dimensions.instruction_address_bits()
    }

    fn phases(&self) -> usize {
        self.address_bits() / CHUNK_LEN
    }

    /// Bits below (and excluding) phase `p`'s chunk.
    fn suffix_len(&self, phase: usize) -> usize {
        self.address_bits() - (phase + 1) * CHUNK_LEN
    }

    fn init_phase(&mut self, phase: usize) {
        let suffix_len = self.suffix_len(phase);
        let previous = phase
            .checked_sub(1)
            .map(|previous| (previous, self.suffix_len(previous)));
        let scan = PhaseScan {
            condense: previous.map(|(previous, shift)| (shift, self.v_tables[previous].as_slice())),
            suffix_len,
            suffix_mask: if suffix_len == 128 {
                u128::MAX
            } else {
                (1u128 << suffix_len) - 1
            },
            upper_suffix_bits: suffix_len.saturating_sub(self.address_bits() / 2),
            tables: LookupTableKind::<RISCV_XLEN>::iter().collect(),
        };
        let PhaseSums { raf, mut suffixes } =
            scan.run(&self.buckets, &self.rows, &mut self.u_evals);

        let q_shift_half: Vec<F> = raf
            .shift_half
            .iter()
            .map(|value| value.mul_pow_2(suffix_len / 2))
            .collect();
        let q_shift_full: Vec<F> = raf
            .shift_full
            .iter()
            .map(|value| value.mul_pow_2(suffix_len))
            .collect();

        // RAF prefix chunk polynomials from the checkpoints — identical
        // construction to the reference kernel.
        let identity_prefix: Vec<F> = (0..CHUNK_SIZE)
            .map(|x| self.raf_identity.checkpoint.mul_pow_2(CHUNK_LEN) + F::from_u64(x as u64))
            .collect();
        let (left_prefix, right_prefix): (Vec<F>, Vec<F>) = (0..CHUNK_SIZE)
            .map(|x| {
                let (left, right) = LookupBits::new(x as u128, CHUNK_LEN).uninterleave();
                (
                    self.raf_left.checkpoint.mul_pow_2(CHUNK_LEN / 2)
                        + F::from_u64(u64::from(left)),
                    self.raf_right.checkpoint.mul_pow_2(CHUNK_LEN / 2)
                        + F::from_u64(u64::from(right)),
                )
            })
            .unzip();
        self.raf_left.prefix = Polynomial::new(left_prefix);
        self.raf_left.q_shift = Polynomial::new(q_shift_half.clone());
        self.raf_left.q_value = Polynomial::new(raf.left);
        self.raf_right.prefix = Polynomial::new(right_prefix);
        self.raf_right.q_shift = Polynomial::new(q_shift_half);
        self.raf_right.q_value = Polynomial::new(raf.right);
        self.raf_identity.prefix = Polynomial::new(identity_prefix);
        self.raf_identity.q_shift = Polynomial::new(q_shift_full);
        self.raf_identity.q_value = Polynomial::new(raf.identity);

        if CANONICAL_INSTRUCTION_ADDRESS {
            let chunk_upper_bits = (self.address_bits() / 2)
                .saturating_sub(phase * CHUNK_LEN)
                .min(CHUNK_LEN);
            let checkpoint = self.raf_upper_all_ones.checkpoint;
            let upper_prefix: Vec<F> = (0..CHUNK_SIZE)
                .map(|x| {
                    if chunk_upper_bits == 0
                        || (x >> (CHUNK_LEN - chunk_upper_bits)) == (1 << chunk_upper_bits) - 1
                    {
                        checkpoint
                    } else {
                        F::zero()
                    }
                })
                .collect();
            self.raf_upper_all_ones.prefix = Polynomial::new(upper_prefix);
            self.raf_upper_all_ones.q_shift = Polynomial::new(raf.upper_all_ones);
            self.raf_upper_all_ones.q_value = Polynomial::new(vec![F::zero(); CHUNK_SIZE]);
        }

        self.suffix_tables = LookupTableKind::<RISCV_XLEN>::iter()
            .filter_map(|table| {
                let sums = std::mem::take(&mut suffixes[table.index()]);
                (!sums.is_empty()).then(|| {
                    let polynomials = sums
                        .chunks_exact(CHUNK_SIZE)
                        .map(|coefficients| Polynomial::new(coefficients.to_vec()))
                        .collect();
                    (table, polynomials)
                })
            })
            .collect();

        // Table-prefix chunk polynomials from the checkpoints, one prefix per
        // parallel task.
        let checkpoints = self.prefix_checkpoints.as_slice();
        let prefix_indices = self.prefix_indices.as_slice();
        self.prefix_tables = map_indices(prefix_indices.len(), |position| {
            let index = prefix_indices[position];
            let prefix = &ALL_PREFIXES[index];
            Polynomial::new(
                (0..CHUNK_SIZE)
                    .map(|x| {
                        prefix
                            .evaluate::<F>(
                                checkpoints,
                                LookupBits::new(x as u128, CHUNK_LEN),
                                suffix_len,
                            )
                            .value()
                    })
                    .collect(),
            )
        });

        self.phase_challenges.clear();
    }

    /// The address-round quadratic, evaluated at `c ∈ {0, 2}` with
    /// `s(1) = previous_claim − s(0)` (the engine-checked hint), emitted
    /// through the same `from_evals` constructor as the reference.
    fn address_message(&self, previous_claim: F) -> UnivariatePoly<F> {
        let half = self.raf_left.prefix.evals().len() / 2;
        // Partial sums: [read, left, right, identity, upper] × {c=0, c=2}.
        let sums = map_reduce_chunks(
            half,
            (half / 8).max(8),
            |range| {
                let mut sums = [F::zero(); 10];
                // Per-thread scratch: full prefix eval rows (indexed by the
                // `Prefixes` discriminant, as `combine` expects) plus suffix
                // eval rows reused across tables.
                let mut p0 = vec![PrefixEval::from(F::zero()); self.prefix_checkpoints.len()];
                let mut p2 = vec![PrefixEval::from(F::zero()); self.prefix_checkpoints.len()];
                let mut s0: Vec<SuffixEval<F>> = Vec::new();
                let mut s2: Vec<SuffixEval<F>> = Vec::new();
                for b in range {
                    for (&index, table) in self.prefix_indices.iter().zip(&self.prefix_tables) {
                        let (lo, ext) = extension_pair(table.evals(), b, half);
                        p0[index] = PrefixEval::from(lo);
                        p2[index] = PrefixEval::from(ext);
                    }
                    for (table, suffixes) in &self.suffix_tables {
                        s0.clear();
                        s2.clear();
                        for q in suffixes {
                            let (lo, ext) = extension_pair(q.evals(), b, half);
                            s0.push(SuffixEval::from(lo));
                            s2.push(SuffixEval::from(ext));
                        }
                        sums[0] += table.combine(&p0, &s0);
                        sums[1] += table.combine(&p2, &s2);
                    }
                    let (left0, left2) = self.raf_left.message_evals(b, half);
                    let (right0, right2) = self.raf_right.message_evals(b, half);
                    let (id0, id2) = self.raf_identity.message_evals(b, half);
                    sums[2] += left0;
                    sums[3] += left2;
                    sums[4] += right0;
                    sums[5] += right2;
                    sums[6] += id0;
                    sums[7] += id2;
                    if CANONICAL_INSTRUCTION_ADDRESS {
                        let (upper0, upper2) = self.raf_upper_all_ones.message_evals(b, half);
                        sums[8] += upper0;
                        sums[9] += upper2;
                    }
                }
                sums
            },
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(&b) {
                    *a += *b;
                }
                a
            },
            || [F::zero(); 10],
        );

        let gamma_sqr = self.gamma * self.gamma;
        let mut eval_0 = sums[0] + self.gamma * sums[2] + gamma_sqr * (sums[4] + sums[6]);
        let mut eval_2 = sums[1] + self.gamma * sums[3] + gamma_sqr * (sums[5] + sums[7]);
        if CANONICAL_INSTRUCTION_ADDRESS {
            eval_0 += gamma_sqr * self.gamma * sums[8];
            eval_2 += gamma_sqr * self.gamma * sums[9];
        }
        let eval_1 = previous_claim - eval_0;
        UnivariatePoly::from_evals(&[eval_0, eval_1, eval_2])
    }

    /// The cycle-round polynomial via the Gruen factorization: the true
    /// degree-`(ra_count + 2)` polynomial is `s(t) = ℓ(t) · q(t)` with `ℓ`
    /// the current linear eq factor and `q(t) = Σ_y E(y) · (Val · Π ra)(t,
    /// y)`. `q` is evaluated on the grid `[1, …, F−1, ∞]` (`F = 1 +
    /// ra_count` linear factors): `e_in` folds into the `Val` pair so the
    /// per-point products accumulate unreduced across the whole inner block
    /// with no per-row reductions (legacy `eval_linear_prod_accumulate`).
    /// `q(0)` is recovered from `s(0) + s(1) = previous_claim` and the
    /// unique degree-`(F+1)` coefficient vector recomposed — byte-identical
    /// to explicit-point interpolation.
    fn cycle_message(
        &self,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        let factors = 1 + self.dimensions.num_virtual_ra_polys();

        struct Scratch<F: JoltField> {
            /// Cross-row lanes for `q(1), …, q(F−1), q(∞)` — `e_in` rides in
            /// the `Val` factor, so these stay unreduced across the block.
            lanes: Vec<F::Accumulator>,
            evals: Vec<F>,
            steps: Vec<F>,
        }

        let block_lanes = cycle.gruen.par_fold_out_in(
            || Scratch {
                lanes: vec![F::Accumulator::default(); factors],
                evals: vec![F::zero(); factors],
                steps: vec![F::zero(); factors],
            },
            |scratch, row, _x_in, e_in| {
                match &cycle.tables {
                    CycleTables::Dense { combined_val, ra } => {
                        {
                            let val = combined_val.evals();
                            let lo = e_in * val[2 * row];
                            let hi = e_in * val[2 * row + 1];
                            scratch.evals[0] = hi;
                            scratch.steps[0] = hi - lo;
                        }
                        for ((ra, eval), step) in ra
                            .iter()
                            .zip(scratch.evals[1..].iter_mut())
                            .zip(scratch.steps[1..].iter_mut())
                        {
                            let table = ra.evals();
                            let lo = table[2 * row];
                            let hi = table[2 * row + 1];
                            *eval = hi;
                            *step = hi - lo;
                        }
                    }
                    CycleTables::Pending(pending) => {
                        // First cycle round: same pair math over the bases —
                        // the values a dense materialization would hold.
                        {
                            let lo = e_in * self.pending_combined_base(pending, 2 * row);
                            let hi = e_in * self.pending_combined_base(pending, 2 * row + 1);
                            scratch.evals[0] = hi;
                            scratch.steps[0] = hi - lo;
                        }
                        let ra_count = scratch.evals.len() - 1;
                        for i in 0..ra_count {
                            let lo = self.pending_ra_base(i, 2 * row);
                            let hi = self.pending_ra_base(i, 2 * row + 1);
                            scratch.evals[1 + i] = hi;
                            scratch.steps[1 + i] = hi - lo;
                        }
                    }
                }
                accumulate_product_grid(&mut scratch.evals, &scratch.steps, &mut scratch.lanes);
            },
            |_x_out, e_out, scratch| {
                let mut out = vec![F::Accumulator::default(); factors];
                for (out, lane) in out.iter_mut().zip(scratch.lanes) {
                    out.fmadd(e_out, lane.reduce());
                }
                out
            },
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(b) {
                    a.merge(b);
                }
                a
            },
        );
        let q_evals: Vec<F> = block_lanes.into_iter().map(|lane| lane.reduce()).collect();
        Ok(cycle.gruen.gruen_poly_from_evals(&q_evals, previous_claim))
    }

    /// Handoff at the address/cycle boundary — same collapse as the
    /// reference, with parallel materialization and a Gruen-split eq factor
    /// instead of a dense `T`-sized eq table.
    fn init_cycle_rounds(&mut self) {
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

        // Snap the packed output-claim facts first: past this handoff the
        // final flag walk reads one byte per cycle, not the 40 B row.
        let rows = self.rows.as_slice();
        const {
            assert!(
                LookupTableKind::<RISCV_XLEN>::COUNT < 0x7f,
                "table indices must fit the packed claim byte"
            );
        }
        self.claim_columns = map_indices(rows.len(), |j| {
            let row = &rows[j];
            let table = row.table_index().map_or(0, |index| index as u8 + 1);
            table | (u8::from(row.raf_flag()) << 7)
        });

        // The tables stay pending: the first cycle message evaluates these
        // bases per row, and the first cycle bind materializes half-domain
        // tables directly (rows and the phase eq tables stay alive until
        // then).
        self.cycle = Some(CycleState {
            gruen: GruenSplitEqPolynomial::new(&self.r_reduction, BindingOrder::LowToHigh),
            tables: CycleTables::Pending(PendingCycleTables {
                table_values,
                raf_interleaved,
                raf_identity,
            }),
            bind_scratch: Vec::new(),
        });

        // The address-phase state is dead past this point — except the
        // bound-challenge eq tables, which the pending ra bases read until
        // the first cycle bind materializes the dense tables.
        self.u_evals = Vec::new();
        self.prefix_tables = Vec::new();
        self.suffix_tables = Vec::new();
        self.buckets = Vec::new();
    }

    /// The pending combined-value base at cycle `j` (packed-byte lookup).
    #[inline]
    fn pending_combined_base(&self, pending: &PendingCycleTables<F>, j: usize) -> F {
        let packed = self.claim_columns[j];
        let table_value = match packed & 0x7f {
            0 => F::zero(),
            table => pending.table_values[usize::from(table) - 1],
        };
        let raf_value = if packed & 0x80 == 0 {
            pending.raf_interleaved
        } else {
            pending.raf_identity
        };
        table_value + raf_value
    }

    /// The pending `ra_i` base at cycle `j` (the phase eq-table product).
    #[inline]
    fn pending_ra_base(&self, i: usize, j: usize) -> F {
        let ra_count = self.dimensions.num_virtual_ra_polys();
        let phases_per_ra = self.phases() / ra_count;
        let address_bits = self.address_bits();
        let index = self.rows[j].lookup_index();
        let mut phase = i * phases_per_ra;
        let mut shift = address_bits - (phase + 1) * CHUNK_LEN;
        let mut product = self.v_tables[phase][((index >> shift) as usize) & (CHUNK_SIZE - 1)];
        for _ in 1..phases_per_ra {
            phase += 1;
            shift -= CHUNK_LEN;
            product *= self.v_tables[phase][((index >> shift) as usize) & (CHUNK_SIZE - 1)];
        }
        product
    }

    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
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
                    self.init_phase(phase + 1);
                } else {
                    self.init_cycle_rounds();
                }
            }
        } else {
            let pending = {
                let cycle = self
                    .cycle
                    .as_mut()
                    .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
                cycle.gruen.bind(challenge);
                match &mut cycle.tables {
                    CycleTables::Pending(pending) => Some(core::mem::replace(
                        pending,
                        PendingCycleTables {
                            table_values: Vec::new(),
                            raf_interleaved: F::zero(),
                            raf_identity: F::zero(),
                        },
                    )),
                    CycleTables::Dense { combined_val, ra } => {
                        combined_val
                            .bind_low_to_high_reusing_scratch(challenge, &mut cycle.bind_scratch);
                        for ra in ra {
                            ra.bind_low_to_high_reusing_scratch(challenge, &mut cycle.bind_scratch);
                        }
                        None
                    }
                }
            };
            if let Some(pending) = pending {
                // First cycle bind: materialize the half-domain tables
                // straight from the bases under this challenge — the same
                // values a full-T materialization would bind to, without
                // the full-T tables ever existing.
                let half = self.claim_columns.len() / 2;
                let combined_val: Vec<F> = map_indices(half, |position| {
                    let lo = self.pending_combined_base(&pending, 2 * position);
                    let hi = self.pending_combined_base(&pending, 2 * position + 1);
                    lo + challenge * (hi - lo)
                });
                let ra_count = self.dimensions.num_virtual_ra_polys();
                let ra: Vec<Polynomial<F>> = (0..ra_count)
                    .map(|i| {
                        Polynomial::new(map_indices(half, |position| {
                            let lo = self.pending_ra_base(i, 2 * position);
                            let hi = self.pending_ra_base(i, 2 * position + 1);
                            lo + challenge * (hi - lo)
                        }))
                    })
                    .collect();
                // The rows' and phase eq tables' last read is behind us.
                self.rows = Arc::new(Vec::new());
                self.v_tables = Vec::new();
                if let Some(cycle) = self.cycle.as_mut() {
                    cycle.tables = CycleTables::Dense {
                        combined_val: Polynomial::new(combined_val),
                        ra,
                    };
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

impl<F: JoltField> SumcheckKernel<F> for OptimizedInstructionReadRafKernel<F> {
    type Relation = InstructionReadRaf<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<InstructionReadRafOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "cycle tables absent after full binding",
            })?;

        // Flag claims at the normalized (big-endian) cycle point via the
        // split-eq factorization `eq(r_cycle, j) = E_hi[j_hi] · E_lo[j_lo]`:
        // per-table masses accumulate over the low half and scale by `E_hi`
        // once per block (exact by distributivity).
        let r_cycle: Vec<F> = self.cycle_challenges.iter().rev().copied().collect();
        let eq_cycle = TensorEqTable::<F>::new(&r_cycle);
        let num_tables = LookupTableKind::<RISCV_XLEN>::COUNT;
        let claim_columns = self.claim_columns.as_slice();
        let (lookup_table_flags, instruction_raf_flag) = eq_cycle.par_fold_out_in(
            || vec![F::Accumulator::default(); num_tables + 1],
            |accumulators, row_index, _x_in, e_in| {
                let packed = claim_columns[row_index];
                if packed & 0x7f != 0 {
                    accumulators[usize::from(packed & 0x7f) - 1].add(e_in);
                }
                if packed & 0x80 != 0 {
                    accumulators[num_tables].add(e_in);
                }
            },
            |_x_out, e_out, accumulators| {
                let mut values: Vec<F> = accumulators
                    .into_iter()
                    .map(|accumulator| e_out * accumulator.reduce())
                    .collect();
                let raf = values.pop().unwrap_or_else(F::zero);
                (values, raf)
            },
            |(mut flags, raf_a), (other, raf_b)| {
                for (a, b) in flags.iter_mut().zip(&other) {
                    *a += *b;
                }
                (flags, raf_a + raf_b)
            },
        );

        let CycleTables::Dense { ra, .. } = &cycle.tables else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "cycle tables still pending after full binding",
            });
        };
        Ok(InstructionReadRafOutputClaims {
            lookup_table_flags,
            instruction_ra: ra.iter().map(|ra| ra.evals()[0]).collect(),
            instruction_raf_flag,
        })
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::geometry::instruction::{
        InstructionReadRafDimensions, CANONICAL_INSTRUCTION_ADDRESS,
    };
    use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafInputClaims;
    use jolt_field::{Fr, Ring};
    use jolt_lookup_tables::{LookupBits, LookupTableKind, XLEN as RISCV_XLEN};
    use jolt_sumcheck::ProveRounds;
    #[cfg(feature = "akita")]
    use jolt_witness::witnesses::FusedInc;
    use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};

    use crate::reference::instruction_read_raf::{
        InstructionReadRafKernel, InstructionReadRafWitness,
    };
    use crate::reference::views::eq_table;
    use crate::SumcheckKernel;

    use super::{InstructionCycleRow, OptimizedInstructionReadRafKernel};

    /// Packs reference-typed fixture rows into the optimized kernel's shared
    /// row form (the stage-5 kernel reads no PC/RAM columns).
    fn pack(rows: &[InstructionReadRafWitness]) -> Vec<InstructionCycleRow> {
        rows.iter()
            .map(|row| {
                InstructionCycleRow::new(
                    row.lookup_index.0,
                    row.table_index.0,
                    row.raf_flag.0,
                    0,
                    None,
                    #[cfg(feature = "akita")]
                    FusedInc::default(),
                )
            })
            .collect()
    }

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// Deterministic non-Boolean challenge stream.
    fn challenge(round: usize) -> Fr {
        fr(0x9E37_79B9_7F4A_7C15 ^ (round as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ 0x11)
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Synthetic rows exercising every branch: a handful of present tables,
    /// no-table rows, both RAF branches, and edge indices (0, all-ones,
    /// all-ones upper half — the canonical-address path).
    fn fixture_rows(log_t: usize, seed: u64) -> Vec<InstructionReadRafWitness> {
        let tables = [
            LookupTableKind::<RISCV_XLEN>::And(Default::default()).index(),
            LookupTableKind::<RISCV_XLEN>::Andn(Default::default()).index(),
            LookupTableKind::<RISCV_XLEN>::Or(Default::default()).index(),
            LookupTableKind::<RISCV_XLEN>::Xor(Default::default()).index(),
            LookupTableKind::<RISCV_XLEN>::VirtualXORROTW7(Default::default()).index(),
        ];
        let mut state = seed;
        (0..1usize << log_t)
            .map(|j| {
                let lookup_index = match j {
                    0 => 0u128,
                    1 => u128::MAX,
                    2 => ((u64::MAX as u128) << 64) | splitmix(&mut state) as u128,
                    _ => ((splitmix(&mut state) as u128) << 64) | splitmix(&mut state) as u128,
                };
                let table_index = if j % 7 == 3 {
                    None
                } else {
                    Some(tables[j % tables.len()])
                };
                InstructionReadRafWitness {
                    lookup_index: LookupIndex(lookup_index),
                    table_index: TableIndex(table_index),
                    raf_flag: InstructionRafFlag(j % 3 == 0),
                }
            })
            .collect()
    }

    #[test]
    fn packed_instruction_row_roundtrips() {
        let lookup_index = u128::MAX - 17;
        let table = LookupTableKind::<RISCV_XLEN>::COUNT - 1;
        let row = InstructionCycleRow::new(
            lookup_index,
            Some(table),
            true,
            u32::MAX as usize,
            Some(u64::MAX - 1),
            #[cfg(feature = "akita")]
            FusedInc(-123),
        );
        assert_eq!(row.lookup_index(), lookup_index);
        assert_eq!(row.table_index(), Some(table));
        assert_eq!(row.bytecode_pc(), u32::MAX as usize);
        assert_eq!(row.remapped_ram_address(), Some(u64::MAX - 1));
        assert!(row.raf_flag());
        #[cfg(feature = "akita")]
        assert_eq!(row.fused_inc::<Fr>(), -Fr::from_u64(123));
    }

    /// The sumcheck input claim from first principles:
    /// `Σ_j eq(r_reduction, j) · (Val_j(k_j) + γ·RafVal_j(k_j))` with the
    /// point-mass `ra` collapsed at each cycle's lookup index. Pins both
    /// kernels to the protocol, not merely to each other (each kernel's own
    /// `s(0) + s(1) = claim` self-check would reject a drifted round 0).
    fn input_claim(rows: &[InstructionReadRafWitness], r_reduction: &[Fr], gamma: Fr) -> Fr {
        let tables: Vec<LookupTableKind<RISCV_XLEN>> = LookupTableKind::iter().collect();
        let gamma_sqr = gamma * gamma;
        let address_bits = 2 * RISCV_XLEN;
        eq_table(r_reduction)
            .iter()
            .zip(rows)
            .map(|(&u, row)| {
                let k = row.lookup_index.0;
                let value = row
                    .table_index
                    .0
                    .map_or_else(|| fr(0), |index| fr(tables[index].materialize_entry(k)));
                let raf = if !row.raf_flag.0 {
                    let (left, right) = LookupBits::new(k, address_bits).uninterleave();
                    gamma * fr(u64::from(left)) + gamma_sqr * fr(u64::from(right))
                } else {
                    let mut raf = gamma_sqr * (fr(k as u64) + fr((k >> 64) as u64).mul_pow_2(64));
                    if CANONICAL_INSTRUCTION_ADDRESS
                        && (k >> (address_bits / 2)) == (1u128 << (address_bits / 2)) - 1
                    {
                        raf += gamma_sqr * gamma;
                    }
                    raf
                };
                u * (value + raf)
            })
            .sum()
    }

    /// Runs reference and optimized kernels through the full round loop with
    /// identical challenges, asserting byte-equal round polynomials (equal
    /// canonical coefficient vectors of equal length) every round and equal
    /// output claims.
    fn assert_parity(log_t: usize, num_virtual_ra_polys: usize, seed: u64) {
        let dimensions = InstructionReadRafDimensions::new(
            log_t,
            2 * RISCV_XLEN,
            NonZeroUsize::new(num_virtual_ra_polys).unwrap(),
        );
        let rows = fixture_rows(log_t, seed);
        let r_reduction: Vec<Fr> = (0..log_t).map(|i| fr(1000 + 37 * i as u64)).collect();
        let gamma = fr(0xACE1_57EF);

        let mut reference =
            InstructionReadRafKernel::new(dimensions, &r_reduction, rows.clone(), gamma).unwrap();
        let mut optimized = OptimizedInstructionReadRafKernel::new(
            dimensions,
            &r_reduction,
            Arc::new(pack(&rows)),
            gamma,
        )
        .unwrap();

        let rounds = reference.num_rounds();
        assert_eq!(rounds, optimized.num_rounds());
        let mut claim = input_claim(&rows, &r_reduction, gamma);
        for round in 0..rounds {
            let bind = round.checked_sub(1).map(challenge);
            let reference_poly = reference.prove_round(bind, round, claim).unwrap();
            let optimized_poly = optimized.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                reference_poly.coefficients(),
                optimized_poly.coefficients(),
                "round {round} polynomial mismatch (log_t={log_t}, ra={num_virtual_ra_polys})"
            );
            claim = reference_poly.evaluate(challenge(round));
        }
        reference.finish_rounds(challenge(rounds - 1)).unwrap();
        optimized.finish_rounds(challenge(rounds - 1)).unwrap();

        let inputs = InstructionReadRafInputClaims {
            lookup_output: fr(0),
            left_lookup_operand: fr(0),
            right_lookup_operand: fr(0),
        };
        let reference_outputs = reference.output_claims(&inputs).unwrap();
        let optimized_outputs = optimized.output_claims(&inputs).unwrap();
        assert_eq!(
            reference_outputs.lookup_table_flags,
            optimized_outputs.lookup_table_flags
        );
        assert_eq!(
            reference_outputs.instruction_ra,
            optimized_outputs.instruction_ra
        );
        assert_eq!(
            reference_outputs.instruction_raf_flag,
            optimized_outputs.instruction_raf_flag
        );
    }

    #[test]
    fn parity_default_geometry() {
        assert_parity(4, 8, 12345);
    }

    #[test]
    fn parity_wide_virtual_chunks_and_odd_log_t() {
        assert_parity(3, 4, 67890);
    }

    /// Enough cycles for several scan tasks, so the phase sums merge tasks
    /// that accumulated the same tables.
    #[test]
    fn parity_multi_task_scan() {
        assert_parity(11, 8, 24680);
    }

    /// All-RAF rows: the identity path (and the canonical-address guard) is
    /// the entire summand; interleaved-operand accumulators stay empty.
    #[test]
    fn parity_all_raf_rows() {
        let log_t = 3;
        let dimensions =
            InstructionReadRafDimensions::new(log_t, 2 * RISCV_XLEN, NonZeroUsize::new(8).unwrap());
        let rows: Vec<InstructionReadRafWitness> = fixture_rows(log_t, 555)
            .into_iter()
            .map(|mut row| {
                row.raf_flag = InstructionRafFlag(true);
                row
            })
            .collect();
        let r_reduction: Vec<Fr> = (0..log_t).map(|i| fr(2000 + 11 * i as u64)).collect();
        let gamma = fr(0xBEEF);

        let mut reference =
            InstructionReadRafKernel::new(dimensions, &r_reduction, rows.clone(), gamma).unwrap();
        let mut optimized = OptimizedInstructionReadRafKernel::new(
            dimensions,
            &r_reduction,
            Arc::new(pack(&rows)),
            gamma,
        )
        .unwrap();
        let mut claim = input_claim(&rows, &r_reduction, gamma);
        for round in 0..reference.num_rounds() {
            let bind = round.checked_sub(1).map(challenge);
            let reference_poly = reference.prove_round(bind, round, claim).unwrap();
            let optimized_poly = optimized.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                reference_poly.coefficients(),
                optimized_poly.coefficients(),
                "round {round}"
            );
            claim = reference_poly.evaluate(challenge(round));
        }
    }
}
