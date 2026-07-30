//! Optimized instruction read+RAF checking (stage 5) kernel.
//!
//! Same math and phase structure as the reference kernel (see
//! `reference/instruction_read_raf.rs`): 8-variable prefix–suffix phases over
//! the 128 address rounds, then a plain multilinear product over the `log_T`
//! cycle rounds. Field arithmetic is exact, so every reorganization below
//! emits byte-identical round polynomials and output claims. The ported
//! legacy optimizations:
//!
//! - **Parallel phase scans** (rayon fold/reduce): the per-phase RAF `Q`
//!   accumulation is one fused trace scan, the per-table suffix `Q`
//!   accumulation runs tables × bucket-chunks in parallel.
//! - **Deferred-reduction accumulation** (`F::Accumulator`) with primitive
//!   scalar multiplies (`mul_u64`/`mul_u128`, no Montgomery conversion of the
//!   scalar): the scans avoid a full reduction per row; suffixes are
//!   classified once per table (`One` / {0,1}-valued / general) so most rows
//!   add instead of multiply.
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
//! - **Shared witness rows**: the collected per-cycle rows are parked in the
//!   [`ProofSession`] (keyed by [`SharedInstructionRows`]) so the stage-6b
//!   instruction RA virtualization kernel and the stage-6a/6b booleanity
//!   kernels reuse them instead of re-scanning the trace — the packed row
//!   carries the mapped PC and remapped RAM address alongside the stage-5
//!   facts at no size cost.

use std::ops::Range;
use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::instruction::{
    InstructionReadRafDimensions, CANONICAL_INSTRUCTION_ADDRESS,
};
use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafOutputClaims;
use jolt_field::{AdditiveAccumulator, Field, RingAccumulator};
use jolt_lookup_tables::tables::prefixes::{PrefixEval, ALL_PREFIXES};
use jolt_lookup_tables::tables::suffixes::{SuffixEval, Suffixes};
use jolt_lookup_tables::{LookupBits, LookupTableKind, XLEN as RISCV_XLEN};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, TensorEqTable, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::SumcheckInputClaims;
use jolt_verifier::stages::stage5::InstructionReadRaf;
use jolt_witness::witnesses::{
    InstructionRafFlag, LookupIndex, MappedPc, RemappedRamAddress, TableIndex,
};
use jolt_witness::{stream_witnesses, JoltWitnessPlane, StreamConsumer, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::accumulate_product;
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

/// One packed per-cycle row: the stage-5 facts (lookup index, lookup table,
/// RAF flag) plus the bytecode/RAM one-hot chunk sources the stage-6a/6b
/// consumers gather from. Sentinel packing (`0` = cold) keeps the row at 48
/// bytes — the same as a stage-5-only bundle row — so sharing the extra
/// columns across stages costs no memory. `repr(C)` so the Metal tier can
/// view a row slice as flat `u32` words (12 per row, layout pinned below).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct InstructionCycleRow {
    pub(crate) lookup_index: u128,
    pc_plus_one: u64,
    ram_address_plus_one: u64,
    /// `0` = no lookup table (the `COUNT < u8::MAX` assert keeps `+ 1` in
    /// range).
    table_index_plus_one: u8,
    pub(crate) raf_flag: bool,
}

// The device-view contract: 48 bytes per row, lookup index limbs first, the
// table/flag bytes at fixed offsets behind the PC/RAM columns.
const _: () = assert!(size_of::<InstructionCycleRow>() == 48);
const _: () = assert!(std::mem::offset_of!(InstructionCycleRow, lookup_index) == 0);
const _: () = assert!(std::mem::offset_of!(InstructionCycleRow, pc_plus_one) == 16);
const _: () = assert!(std::mem::offset_of!(InstructionCycleRow, ram_address_plus_one) == 24);
const _: () = assert!(std::mem::offset_of!(InstructionCycleRow, table_index_plus_one) == 32);
const _: () = assert!(std::mem::offset_of!(InstructionCycleRow, raf_flag) == 33);

impl InstructionCycleRow {
    pub(crate) fn new(
        lookup_index: u128,
        table_index: Option<usize>,
        raf_flag: bool,
        mapped_pc: Option<usize>,
        remapped_ram_address: Option<u64>,
    ) -> Self {
        debug_assert!(table_index.is_none_or(|index| index < u8::MAX as usize));
        Self {
            lookup_index,
            pc_plus_one: mapped_pc.map_or(0, |pc| pc as u64 + 1),
            ram_address_plus_one: remapped_ram_address.map_or(0, |address| address + 1),
            table_index_plus_one: table_index.map_or(0, |index| index as u8 + 1),
            raf_flag,
        }
    }

    #[inline]
    pub(crate) fn table_index(&self) -> Option<usize> {
        (self.table_index_plus_one as usize).checked_sub(1)
    }

    #[inline]
    pub(crate) fn mapped_pc(&self) -> Option<usize> {
        self.pc_plus_one.checked_sub(1).map(|pc| pc as usize)
    }

    #[inline]
    pub(crate) fn remapped_ram_address(&self) -> Option<u64> {
        self.ram_address_plus_one.checked_sub(1)
    }
}

/// The bundle row the packing pass extracts; never materialized beyond one
/// streaming chunk.
#[derive(Clone, Copy, Debug, WitnessBundle)]
struct WideInstructionRow {
    lookup_index: LookupIndex,
    table_index: TableIndex,
    raf_flag: InstructionRafFlag,
    mapped_pc: MappedPc,
    remapped_ram_address: RemappedRamAddress,
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
                row.mapped_pc.0,
                row.remapped_ram_address.0,
            )
        }));
    }
}

/// One streaming bundle pass over the cycle domain, packed row by row (the
/// wide bundle row exists only per chunk).
pub(crate) fn collect_instruction_cycle_rows<F: Field>(
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
) -> Result<Vec<InstructionCycleRow>, KernelError<F>> {
    let mut consumers = (PackRows {
        rows: Vec::with_capacity(cycles),
    },);
    stream_witnesses(witness, 0..cycles, 1 << 12, &mut consumers)?;
    Ok(consumers.0.rows)
}

/// The collected stage-5 rows, parked in the [`ProofSession`] for the
/// stage-6b instruction RA virtualization kernel (its committed one-hot
/// chunks are chunks of the same per-cycle lookup index) and the
/// stage-6a/6b booleanity kernels (all three one-hot chunk families).
///
/// Non-final consumers reclaim with `take`, clone the [`Arc`], and park the
/// carry back for the later stages.
pub(crate) struct SharedInstructionRows(pub(crate) Arc<Vec<InstructionCycleRow>>);

/// Reclaim the parked stage-5 rows (the length guard makes a stale carry
/// impossible to consume) or collect them fresh, and park the carry back
/// for later consumers.
pub(crate) fn shared_instruction_rows<F: Field>(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
) -> Result<Arc<Vec<InstructionCycleRow>>, KernelError<F>> {
    let rows = match session.take::<SharedInstructionRows>() {
        Some(SharedInstructionRows(rows)) if rows.len() == cycles => rows,
        _ => Arc::new(collect_instruction_cycle_rows(witness, cycles)?),
    };
    session.park(SharedInstructionRows(Arc::clone(&rows)));
    Ok(rows)
}

/// Optimized [`PrepareKernel`] implementor for the `instruction_read_raf`
/// slot.
pub struct OptimizedInstructionReadRaf;

impl<F: Field> PrepareKernel<F, InstructionReadRaf<F>> for OptimizedInstructionReadRaf {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionReadRaf<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionReadRaf<F>>>, KernelError<F>> {
        let dimensions = inputs.relation.dimensions();
        // Reclaims the rows the trace record's walk co-produced (parked at
        // stage 1); collects fresh only when no record was built.
        let rows = shared_instruction_rows(session, witness, 1 << dimensions.log_t())?;
        Ok(Box::new(OptimizedInstructionReadRafKernel::new(
            dimensions,
            &inputs.points.lookup_output,
            rows,
            inputs.challenges.gamma,
        )?))
    }
}

// --- parallel shims -------------------------------------------------------
//
// The kernel's custom scans need chunked map-reduce and indexed maps; the
// serial fallbacks compute the same field values (sums and products of the
// same terms), so parity is unaffected by the feature.

/// `merge`-fold of `map` over index chunks of at most `chunk_size`.
fn map_reduce_chunks<R: Send>(
    len: usize,
    chunk_size: usize,
    map: impl Fn(Range<usize>) -> R + Send + Sync,
    merge: impl Fn(R, R) -> R + Send + Sync,
    identity: impl Fn() -> R + Send + Sync,
) -> R {
    if len == 0 {
        return identity();
    }
    #[cfg(feature = "parallel")]
    {
        let chunks = len.div_ceil(chunk_size);
        (0..chunks)
            .into_par_iter()
            .map(|c| map(c * chunk_size..((c + 1) * chunk_size).min(len)))
            .reduce(identity, merge)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (merge, identity, chunk_size);
        map(0..len)
    }
}

/// Collect `f(0), …, f(len − 1)`.
fn map_indices<T: Send>(len: usize, f: impl Fn(usize) -> T + Send + Sync) -> Vec<T> {
    #[cfg(feature = "parallel")]
    {
        (0..len).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..len).map(f).collect()
    }
}

/// Indexed in-place update of a slice.
fn for_each_index_mut<T: Send>(items: &mut [T], f: impl Fn(usize, &mut T) + Send + Sync) {
    #[cfg(feature = "parallel")]
    {
        items
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, item)| f(index, item));
    }
    #[cfg(not(feature = "parallel"))]
    {
        items
            .iter_mut()
            .enumerate()
            .for_each(|(index, item)| f(index, item));
    }
}

fn scan_chunk_size(len: usize) -> usize {
    #[cfg(feature = "parallel")]
    {
        len.div_ceil(rayon::current_num_threads()).max(1024)
    }
    #[cfg(not(feature = "parallel"))]
    {
        len.max(1)
    }
}

// --- kernel ---------------------------------------------------------------

/// One RAF prefix–suffix decomposition — same shape and binding as the
/// reference kernel's.
struct RafDecomposition<F: Field> {
    prefix: Polynomial<F>,
    q_shift: Polynomial<F>,
    q_value: Polynomial<F>,
    checkpoint: F,
}

impl<F: Field> RafDecomposition<F> {
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
fn extension_pair<F: Field>(evals: &[F], b: usize, half: usize) -> (F, F) {
    let lo = evals[b];
    let hi = evals[b + half];
    (lo, hi + hi - lo)
}

/// Cycle-round state: the Gruen-split eq factor plus the dense cycle tables.
struct CycleState<F: Field> {
    gruen: GruenSplitEqPolynomial<F>,
    tables: CycleTablesDriver<F>,
}

/// Where the dense cycle tables live. `Host` is the ordinary CPU state;
/// `Device` means a [`PhaseScanner`] adopted them at the address→cycle
/// boundary and folds/evaluates them itself ([`PhaseScanner::cycle_round`]).
enum CycleTablesDriver<F: Field> {
    Host(HostCycleTables<F>),
    /// `pending_bind` is a challenge already bound into `gruen` but not yet
    /// folded into the tables — the device folds it fused with the next
    /// round's evaluation, or the handoff path applies it on the CPU.
    Device {
        pending_bind: Option<F>,
    },
}

struct HostCycleTables<F: Field> {
    combined_val: Polynomial<F>,
    ra: Vec<Polynomial<F>>,
    /// Reused low-to-high binding buffer (swapped through every bind).
    bind_scratch: Vec<F>,
}

impl<F: Field> HostCycleTables<F> {
    fn new(tables: CycleTables<F>) -> Self {
        Self {
            combined_val: Polynomial::new(tables.combined_val),
            ra: tables.ra.into_iter().map(Polynomial::new).collect(),
            bind_scratch: Vec::new(),
        }
    }

    fn bind(&mut self, challenge: F) {
        self.combined_val
            .bind_low_to_high_reusing_scratch(challenge, &mut self.bind_scratch);
        for ra in &mut self.ra {
            ra.bind_low_to_high_reusing_scratch(challenge, &mut self.bind_scratch);
        }
    }
}

/// Per-thread RAF scan accumulators over one phase's chunk domain, in
/// deferred-reduction form.
struct RafScan<F: Field> {
    shift_half: Vec<F::Accumulator>,
    left: Vec<F::Accumulator>,
    right: Vec<F::Accumulator>,
    shift_full: Vec<F::Accumulator>,
    identity: Vec<F::Accumulator>,
    upper_all_ones: Vec<F::Accumulator>,
}

/// The reduced (field-element) form of one thread's [`RafScan`] — and the
/// RAF half of a [`PhaseScanSums`], whichever tier produced it.
pub(crate) struct RafSums<F> {
    pub(crate) shift_half: Vec<F>,
    pub(crate) left: Vec<F>,
    pub(crate) right: Vec<F>,
    pub(crate) shift_full: Vec<F>,
    pub(crate) identity: Vec<F>,
    pub(crate) upper_all_ones: Vec<F>,
}

impl<F: Field> RafScan<F> {
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

impl<F: Field> RafSums<F> {
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

// --- phase-scan seam --------------------------------------------------------
//
// The three big per-phase trace passes (condensation, the fused RAF scan,
// the per-table suffix scan) are the only `O(T)` work in the address
// rounds; everything downstream operates on 256-sized chunk tables. The
// seam below lets a device tier substitute those passes: field sums are
// exact, so ANY scan regrouping produces the same field elements the CPU
// scan does, and the shared assembly ([`assemble_phase`]
// (OptimizedInstructionReadRafKernel::assemble_phase)) keeps the round
// polynomials byte-identical by construction.

/// One lookup table present in the trace: its kind and its contiguous slice
/// of the flat bucket array (ascending cycle indices).
pub(crate) struct PresentTable {
    pub(crate) table: LookupTableKind<RISCV_XLEN>,
    pub(crate) range: Range<usize>,
}

/// The static scan inputs a scanner captures at construction (all shared —
/// the kernel keeps using them for its CPU paths).
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    expect(dead_code, reason = "read only by the Metal scanner")
)]
pub(crate) struct ScannerInputs<'a> {
    pub(crate) rows: &'a Arc<Vec<InstructionCycleRow>>,
    pub(crate) bucket_flat: &'a Arc<Vec<u32>>,
    pub(crate) present: &'a [PresentTable],
}

/// One phase's scan request. When `condense` is `Some((v_prev, shift))` the
/// scanner must first fold `v_prev[(lookup_index >> shift) & 255]` into
/// `u_evals` in place (phase-boundary condensation), exactly as the CPU
/// path does, before accumulating the phase sums against the updated
/// weights.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    expect(dead_code, reason = "read only by the Metal scanner")
)]
pub(crate) struct PhaseScanRequest<'a, F> {
    pub(crate) suffix_len: usize,
    pub(crate) condense: Option<(&'a [F], usize)>,
    pub(crate) u_evals: &'a mut [F],
}

/// Raw per-phase scan sums, CPU- and device-produced alike: the fused RAF
/// buckets plus, per [`PresentTable`] in order, the flat suffix-major
/// read-checking buckets (`[s * CHUNK_SIZE + chunk]`, `s` indexing
/// `table.suffixes()`).
pub(crate) struct PhaseScanSums<F> {
    pub(crate) raf: RafSums<F>,
    pub(crate) suffix: Vec<Vec<F>>,
}

#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    expect(dead_code, reason = "constructed only by the Metal scanner")
)]
pub(crate) enum ScanOutcome<F> {
    /// Scan complete; `u_evals` holds the post-condensation weights.
    Scanned(PhaseScanSums<F>),
    /// Gate declined or buffers ineligible; `u_evals` untouched — the CPU
    /// path runs this phase (the scanner stays live for later ones).
    Declined,
    /// A dispatch failed after device work may have started: `u_evals` is
    /// unreliable (condensation writes in place) and must be rebuilt; the
    /// scanner is dead.
    Corrupt,
}

/// Inputs for the cycle-table materialization at the address→cycle
/// boundary: the collapsed per-table values and RAF constants (host-derived
/// from the bound checkpoints) plus the per-phase bound-challenge tables the
/// `ra` products gather from.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    expect(dead_code, reason = "read only by the Metal scanner")
)]
pub(crate) struct CycleInitRequest<'a, F> {
    pub(crate) table_values: &'a [F],
    pub(crate) raf_interleaved: F,
    pub(crate) raf_identity: F,
    pub(crate) v_tables: &'a [Vec<F>],
    pub(crate) ra_count: usize,
    pub(crate) phases_per_ra: usize,
    pub(crate) address_bits: usize,
}

/// Materialized cycle tables: `combined_val` then the `ra_count` virtual RA
/// products, each one entry per cycle.
pub(crate) struct CycleTables<F> {
    pub(crate) combined_val: Vec<F>,
    pub(crate) ra: Vec<Vec<F>>,
}

/// Device seam for the per-phase trace scans, the cycle-table
/// materialization, and the cycle product rounds.
pub(crate) trait PhaseScanner<F: Field> {
    fn scan_phase(&mut self, request: PhaseScanRequest<'_, F>) -> ScanOutcome<F>;

    /// Materialize the cycle tables on the device, or `None` for the CPU
    /// path (materialization is pure — a failed attempt discards its
    /// buffers, so unlike condensation there is no corrupt state).
    fn materialize_cycle(&mut self, request: CycleInitRequest<'_, F>) -> Option<CycleTables<F>> {
        let _ = request;
        None
    }

    /// Materialize AND keep the cycle tables for device rounds. `true` moves
    /// the kernel's table driver to [`CycleTablesDriver::Device`]: rounds go
    /// through [`cycle_round`](Self::cycle_round) until the scanner steps
    /// aside, and the kernel keeps the scanner alive past the address
    /// phases. `false` (with no side effects) keeps the ordinary
    /// [`materialize_cycle`](Self::materialize_cycle)/CPU path.
    fn adopt_cycle(&mut self, request: &CycleInitRequest<'_, F>) -> bool {
        let _ = request;
        false
    }

    /// One fused cycle round over the adopted tables: fold `bind` (when
    /// present) low-to-high, then return the product-grid evaluations
    /// `[q(1), …, q(F−1), q(∞)]` against the CURRENT (post-`bind`) gruen
    /// levels. `None` steps aside with NO effect — the live tables are
    /// still the pre-`bind` state and the caller reclaims them through
    /// [`take_cycle_tables`](Self::take_cycle_tables).
    fn cycle_round(&mut self, bind: Option<F>, e_in: &[F], e_out: &[F]) -> Option<Vec<F>> {
        let _ = (bind, e_in, e_out);
        None
    }

    /// Hand the adopted tables back at their current (post-last-successful-
    /// round) length. `None` only if nothing was adopted — an invariant
    /// violation when the driver is [`CycleTablesDriver::Device`].
    fn take_cycle_tables(&mut self) -> Option<CycleTables<F>> {
        None
    }
}

pub struct OptimizedInstructionReadRafKernel<F: Field> {
    dimensions: InstructionReadRafDimensions,
    gamma: F,
    r_reduction: Vec<F>,
    rows: Arc<Vec<InstructionCycleRow>>,
    /// All present tables' cycle buckets, concatenated in
    /// `LookupTableKind::iter()` order; per-table slices in `present`.
    bucket_flat: Arc<Vec<u32>>,
    present: Vec<PresentTable>,
    scanner: Option<Box<dyn PhaseScanner<F>>>,
    /// Condensed per-cycle eq weights (see the reference kernel).
    u_evals: Vec<F>,
    prefix_checkpoints: Vec<PrefixEval<F>>,
    /// Materialized prefix chunk polynomials for the current phase, in
    /// `ALL_PREFIXES` order.
    prefix_tables: Vec<Polynomial<F>>,
    /// Per present table: enum value + suffix `Q` polynomials in
    /// `table.suffixes()` order.
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
    rounds_bound: usize,
}

impl<F: Field> OptimizedInstructionReadRafKernel<F> {
    pub(crate) fn new(
        dimensions: InstructionReadRafDimensions,
        r_reduction: &[F],
        rows: Arc<Vec<InstructionCycleRow>>,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        Self::new_with_scanner(dimensions, r_reduction, rows, gamma, |_| None)
    }

    /// As [`new`](Self::new), with a [`PhaseScanner`] factory invoked once
    /// the static scan inputs (rows, buckets) exist; `None` keeps every
    /// phase on the CPU.
    pub(crate) fn new_with_scanner(
        dimensions: InstructionReadRafDimensions,
        r_reduction: &[F],
        rows: Arc<Vec<InstructionCycleRow>>,
        gamma: F,
        scanner: impl FnOnce(ScannerInputs<'_>) -> Option<Box<dyn PhaseScanner<F>>>,
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
        if log_t >= 32 {
            return Err(KernelError::Unsupported {
                reason: "cycle bucket indices are u32",
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

        let mut buckets = vec![Vec::new(); LookupTableKind::<RISCV_XLEN>::COUNT];
        for (j, row) in rows.iter().enumerate() {
            if let Some(table_index) = row.table_index() {
                buckets
                    .get_mut(table_index)
                    .ok_or(KernelError::InvariantViolation {
                        reason: "stage-5 row selects an unknown lookup table",
                    })?
                    .push(j as u32);
            }
        }
        let mut bucket_flat = Vec::with_capacity(buckets.iter().map(Vec::len).sum());
        let mut present = Vec::new();
        for table in LookupTableKind::<RISCV_XLEN>::iter() {
            let bucket = &buckets[table.index()];
            if bucket.is_empty() {
                continue;
            }
            let start = bucket_flat.len();
            bucket_flat.extend_from_slice(bucket);
            present.push(PresentTable {
                table,
                range: start..bucket_flat.len(),
            });
        }
        let bucket_flat = Arc::new(bucket_flat);
        let scanner = scanner(ScannerInputs {
            rows: &rows,
            bucket_flat: &bucket_flat,
            present: &present,
        });

        let mut kernel = Self {
            dimensions,
            gamma,
            r_reduction: r_reduction.to_vec(),
            rows,
            bucket_flat,
            present,
            scanner,
            u_evals: eq_table(r_reduction),
            prefix_checkpoints: ALL_PREFIXES
                .iter()
                .map(|prefix| prefix.default_checkpoint::<F>())
                .collect(),
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
            rounds_bound: 0,
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
        let sums = self.phase_sums(phase, suffix_len);
        self.assemble_phase(phase, suffix_len, sums);
        self.phase_challenges.clear();
    }

    /// The phase's scan sums: device scanner when installed and willing,
    /// CPU otherwise. Either way `u_evals` ends up condensed and the sums
    /// are the same field elements (exact arithmetic — see the seam docs).
    fn phase_sums(&mut self, phase: usize, suffix_len: usize) -> PhaseScanSums<F> {
        let prev_shift = phase.checked_sub(1).map(|prev| self.suffix_len(prev));
        if let Some(scanner) = self.scanner.as_mut() {
            let request = PhaseScanRequest {
                suffix_len,
                condense: prev_shift.map(|shift| (self.v_tables[phase - 1].as_slice(), shift)),
                u_evals: &mut self.u_evals,
            };
            match scanner.scan_phase(request) {
                ScanOutcome::Scanned(sums) => return sums,
                ScanOutcome::Declined => {}
                ScanOutcome::Corrupt => {
                    self.scanner = None;
                    self.rebuild_u_evals(phase);
                    return self.cpu_phase_sums(suffix_len);
                }
            }
        }
        self.condense_cpu(phase);
        self.cpu_phase_sums(suffix_len)
    }

    /// Condensation: fold the previous phase's bound-challenge eq weights
    /// into the per-cycle mass.
    fn condense_cpu(&mut self, phase: usize) {
        if phase == 0 {
            return;
        }
        let shift = self.suffix_len(phase - 1);
        let rows = Arc::clone(&self.rows);
        let v_prev = std::mem::take(&mut self.v_tables[phase - 1]);
        for_each_index_mut(&mut self.u_evals, |j, u| {
            *u *= v_prev[((rows[j].lookup_index >> shift) as usize) & (CHUNK_SIZE - 1)];
        });
        self.v_tables[phase - 1] = v_prev;
    }

    /// Recompute the per-cycle weights from scratch: `eq(r_reduction, ·)`
    /// with every completed phase's bound-challenge table folded back in.
    /// Device condensation updates `u_evals` in place, so a failed dispatch
    /// may leave it partially updated; the inputs (rows, `v_tables`) are
    /// intact, so this rebuild is exact.
    fn rebuild_u_evals(&mut self, phase: usize) {
        self.u_evals = eq_table(&self.r_reduction);
        for prev in 0..phase {
            let shift = self.suffix_len(prev);
            let rows = Arc::clone(&self.rows);
            let v_prev = std::mem::take(&mut self.v_tables[prev]);
            for_each_index_mut(&mut self.u_evals, |j, u| {
                *u *= v_prev[((rows[j].lookup_index >> shift) as usize) & (CHUNK_SIZE - 1)];
            });
            self.v_tables[prev] = v_prev;
        }
    }

    fn suffix_mask(suffix_len: usize) -> u128 {
        if suffix_len == 128 {
            u128::MAX
        } else {
            (1u128 << suffix_len) - 1
        }
    }

    /// CPU scan of one phase (post-condensation): the fused RAF pass plus
    /// the per-table suffix passes.
    fn cpu_phase_sums(&self, suffix_len: usize) -> PhaseScanSums<F> {
        PhaseScanSums {
            raf: self.cpu_raf_sums(suffix_len),
            suffix: self.cpu_suffix_sums(suffix_len),
        }
    }

    /// Fused RAF scan over the whole trace (deferred-reduction sums,
    /// primitive-scalar multiplies).
    fn cpu_raf_sums(&self, suffix_len: usize) -> RafSums<F> {
        let suffix_mask = Self::suffix_mask(suffix_len);
        let upper_suffix_bits = suffix_len.saturating_sub(self.address_bits() / 2);
        let rows = self.rows.as_slice();
        let u_evals = self.u_evals.as_slice();
        map_reduce_chunks(
            rows.len(),
            scan_chunk_size(rows.len()),
            |range| {
                let mut scan = RafScan::<F>::new();
                for (row, &u) in rows[range.clone()].iter().zip(&u_evals[range]) {
                    let chunk = ((row.lookup_index >> suffix_len) as usize) & (CHUNK_SIZE - 1);
                    let suffix_bits = row.lookup_index & suffix_mask;
                    if CANONICAL_INSTRUCTION_ADDRESS
                        && row.raf_flag
                        && (upper_suffix_bits == 0
                            || (suffix_bits >> (suffix_len - upper_suffix_bits))
                                == (1u128 << upper_suffix_bits) - 1)
                    {
                        scan.upper_all_ones[chunk].add(u);
                    }
                    if !row.raf_flag {
                        scan.shift_half[chunk].add(u);
                        let (left, right) = LookupBits::new(suffix_bits, suffix_len).uninterleave();
                        let left = u64::from(left);
                        if left != 0 {
                            scan.left[chunk].add(u.mul_u64(left));
                        }
                        let right = u64::from(right);
                        if right != 0 {
                            scan.right[chunk].add(u.mul_u64(right));
                        }
                    } else {
                        scan.shift_full[chunk].add(u);
                        if suffix_bits != 0 {
                            scan.identity[chunk].add(u.mul_u128(suffix_bits));
                        }
                    }
                }
                scan.reduce()
            },
            RafSums::merge,
            RafSums::zero,
        )
    }

    /// Build the phase's chunk tables from its scan sums — the shared
    /// assembly both scan tiers feed, so round polynomials downstream are
    /// byte-identical whichever tier scanned.
    fn assemble_phase(&mut self, phase: usize, suffix_len: usize, sums: PhaseScanSums<F>) {
        let PhaseScanSums { raf, suffix } = sums;
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

        self.suffix_tables = self
            .present
            .iter()
            .zip(suffix)
            .map(|(present, flat)| {
                let polynomials = flat
                    .chunks_exact(CHUNK_SIZE)
                    .map(|coefficients| Polynomial::new(coefficients.to_vec()))
                    .collect();
                (present.table, polynomials)
            })
            .collect();

        // Table-prefix chunk polynomials from the checkpoints, one prefix per
        // parallel task.
        let checkpoints = self.prefix_checkpoints.as_slice();
        self.prefix_tables = map_indices(ALL_PREFIXES.len(), |index| {
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
    }

    /// Read-checking suffix accumulators for the phase, per present table:
    /// tables in parallel, each over parallel bucket chunks, suffixes
    /// classified once (`One` adds, {0,1}-valued adds conditionally, general
    /// ones use the primitive-scalar multiply).
    fn cpu_suffix_sums(&self, suffix_len: usize) -> Vec<Vec<F>> {
        let suffix_mask = Self::suffix_mask(suffix_len);
        let rows = self.rows.as_slice();
        let u_evals = self.u_evals.as_slice();
        let bucket_flat = self.bucket_flat.as_slice();
        let present = self.present.as_slice();
        let suffix_shift = suffix_len;
        map_indices(present.len(), |table_position| {
            let table = present[table_position].table;
            let suffixes = table.suffixes();
            let num_suffixes = suffixes.len();
            let one_position = suffixes
                .iter()
                .position(|suffix| matches!(suffix, Suffixes::One));
            let bucket = &bucket_flat[present[table_position].range.clone()];

            map_reduce_chunks(
                bucket.len(),
                scan_chunk_size(bucket.len()),
                |range| {
                    let mut accumulators =
                        vec![F::Accumulator::default(); num_suffixes * CHUNK_SIZE];
                    for &j in &bucket[range] {
                        let row = &rows[j as usize];
                        let u = u_evals[j as usize];
                        let chunk =
                            ((row.lookup_index >> suffix_shift) as usize) & (CHUNK_SIZE - 1);
                        let suffix_bits =
                            LookupBits::new(row.lookup_index & suffix_mask, suffix_len);
                        for (s_index, suffix) in suffixes.iter().enumerate() {
                            let slot = &mut accumulators[s_index * CHUNK_SIZE + chunk];
                            if one_position == Some(s_index) {
                                slot.add(u);
                            } else if suffix.is_01_valued() {
                                if suffix.suffix_mle(suffix_bits) == 1 {
                                    slot.add(u);
                                }
                            } else {
                                let value = suffix.suffix_mle(suffix_bits);
                                if value != 0 {
                                    slot.add(u.mul_u64(value));
                                }
                            }
                        }
                    }
                    accumulators
                        .into_iter()
                        .map(|accumulator| accumulator.reduce())
                        .collect::<Vec<F>>()
                },
                |mut a, b| {
                    for (a, b) in a.iter_mut().zip(&b) {
                        *a += *b;
                    }
                    a
                },
                || vec![F::zero(); num_suffixes * CHUNK_SIZE],
            )
        })
    }

    /// The address-round quadratic, evaluated at `c ∈ {0, 2}` with
    /// `s(1) = previous_claim − s(0)` (the engine-checked hint), emitted
    /// through the same `from_evals` constructor as the reference.
    fn address_message(&self, previous_claim: F) -> UnivariatePoly<F> {
        let half = self.prefix_tables[0].evals().len() / 2;
        // Field-only borrows: the parallel closure must not capture `&self`
        // (the scanner slot is not `Sync`).
        let prefix_tables = self.prefix_tables.as_slice();
        let suffix_tables = self.suffix_tables.as_slice();
        let raf_left = &self.raf_left;
        let raf_right = &self.raf_right;
        let raf_identity = &self.raf_identity;
        let raf_upper_all_ones = &self.raf_upper_all_ones;
        // Partial sums: [read, left, right, identity, upper] × {c=0, c=2}.
        let sums = map_reduce_chunks(
            half,
            (half / 8).max(8),
            |range| {
                let mut sums = [F::zero(); 10];
                // Per-thread scratch: full prefix eval rows (indexed by the
                // `Prefixes` discriminant, as `combine` expects) plus suffix
                // eval rows reused across tables.
                let mut p0: Vec<PrefixEval<F>> = Vec::with_capacity(prefix_tables.len());
                let mut p2: Vec<PrefixEval<F>> = Vec::with_capacity(prefix_tables.len());
                let mut s0: Vec<SuffixEval<F>> = Vec::new();
                let mut s2: Vec<SuffixEval<F>> = Vec::new();
                for b in range {
                    p0.clear();
                    p2.clear();
                    for table in prefix_tables {
                        let (lo, ext) = extension_pair(table.evals(), b, half);
                        p0.push(PrefixEval::from(lo));
                        p2.push(PrefixEval::from(ext));
                    }
                    for (table, suffixes) in suffix_tables {
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
                    let (left0, left2) = raf_left.message_evals(b, half);
                    let (right0, right2) = raf_right.message_evals(b, half);
                    let (id0, id2) = raf_identity.message_evals(b, half);
                    sums[2] += left0;
                    sums[3] += left2;
                    sums[4] += right0;
                    sums[5] += right2;
                    sums[6] += id0;
                    sums[7] += id2;
                    if CANONICAL_INSTRUCTION_ADDRESS {
                        let (upper0, upper2) = raf_upper_all_ones.message_evals(b, half);
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
        &mut self,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        {
            let cycle = self
                .cycle
                .as_mut()
                .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
            if let CycleTablesDriver::Device { pending_bind } = &cycle.tables {
                let bind = *pending_bind;
                let gruen = &cycle.gruen;
                let lanes = self.scanner.as_mut().and_then(|scanner| {
                    scanner.cycle_round(bind, gruen.e_in_current(), gruen.e_out_current())
                });
                match lanes {
                    Some(lanes) => {
                        cycle.tables = CycleTablesDriver::Device { pending_bind: None };
                        return Ok(cycle.gruen.gruen_poly_from_evals(&lanes, previous_claim));
                    }
                    // The device stepped aside pre-round: reclaim the tables
                    // (and any pending fold) and finish on the CPU.
                    None => self.ensure_host_cycle_tables()?,
                }
            }
        }
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        let CycleTablesDriver::Host(tables) = &cycle.tables else {
            return Err(SumcheckError::MissingEvaluationSource { kind: "opening" });
        };
        let factors = 1 + tables.ra.len();

        struct Scratch<F: Field> {
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
                {
                    let val = tables.combined_val.evals();
                    let lo = e_in * val[2 * row];
                    let hi = e_in * val[2 * row + 1];
                    scratch.evals[0] = hi;
                    scratch.steps[0] = hi - lo;
                }
                for ((ra, eval), step) in tables
                    .ra
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
                accumulate_product(&scratch.evals, &mut scratch.lanes[0]);
                for lane in 1..factors - 1 {
                    for (eval, step) in scratch.evals.iter_mut().zip(&scratch.steps) {
                        *eval += *step;
                    }
                    accumulate_product(&scratch.evals, &mut scratch.lanes[lane]);
                }
                accumulate_product(&scratch.steps, &mut scratch.lanes[factors - 1]);
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

        let ra_count = self.dimensions.num_virtual_ra_polys();
        let phases_per_ra = self.phases() / ra_count;
        let address_bits = self.address_bits();

        let request = CycleInitRequest {
            table_values: &table_values,
            raf_interleaved,
            raf_identity,
            v_tables: &self.v_tables,
            ra_count,
            phases_per_ra,
            address_bits,
        };
        let adopted = self
            .scanner
            .as_mut()
            .is_some_and(|scanner| scanner.adopt_cycle(&request));
        let tables = if adopted {
            CycleTablesDriver::Device { pending_bind: None }
        } else {
            let device_tables = self
                .scanner
                .as_mut()
                .and_then(|scanner| scanner.materialize_cycle(request));
            let tables = device_tables.unwrap_or_else(|| {
                let rows = self.rows.as_slice();
                let v_tables = self.v_tables.as_slice();
                let combined_val: Vec<F> = map_indices(rows.len(), |j| {
                    let row = &rows[j];
                    let table_value = row
                        .table_index()
                        .map_or_else(F::zero, |index| table_values[index]);
                    let raf_value = if !row.raf_flag {
                        raf_interleaved
                    } else {
                        raf_identity
                    };
                    table_value + raf_value
                });
                let ra = (0..ra_count)
                    .map(|i| {
                        map_indices(rows.len(), |j| {
                            let index = rows[j].lookup_index;
                            let mut phase = i * phases_per_ra;
                            let mut shift = address_bits - (phase + 1) * CHUNK_LEN;
                            let mut product =
                                v_tables[phase][((index >> shift) as usize) & (CHUNK_SIZE - 1)];
                            for _ in 1..phases_per_ra {
                                phase += 1;
                                shift -= CHUNK_LEN;
                                product *=
                                    v_tables[phase][((index >> shift) as usize) & (CHUNK_SIZE - 1)];
                            }
                            product
                        })
                    })
                    .collect();
                CycleTables { combined_val, ra }
            });
            CycleTablesDriver::Host(HostCycleTables::new(tables))
        };

        self.cycle = Some(CycleState {
            gruen: GruenSplitEqPolynomial::new(&self.r_reduction, BindingOrder::LowToHigh),
            tables,
        });

        // The address-phase state is dead past this point.
        self.u_evals = Vec::new();
        self.prefix_tables = Vec::new();
        self.suffix_tables = Vec::new();
        self.v_tables = Vec::new();
        self.bucket_flat = Arc::new(Vec::new());
        self.present = Vec::new();
        // Without adopted tables the scanner is dead too — dropping it
        // releases the device buffers (it holds `Arc` clones of the rows and
        // buckets) with the address-phase state. An adopting scanner stays
        // for the cycle rounds and is dropped at the CPU handoff.
        if !adopted {
            self.scanner = None;
        }
    }

    /// Reclaim device-adopted cycle tables into ordinary host state,
    /// applying any pending fold. No-op when the tables are already
    /// host-side.
    fn ensure_host_cycle_tables(&mut self) -> Result<(), SumcheckError<F>> {
        let cycle = self
            .cycle
            .as_mut()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        let CycleTablesDriver::Device { pending_bind } = &cycle.tables else {
            return Ok(());
        };
        let pending = *pending_bind;
        let tables = self
            .scanner
            .as_mut()
            .and_then(|scanner| scanner.take_cycle_tables())
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        let mut host = HostCycleTables::new(tables);
        if let Some(challenge) = pending {
            host.bind(challenge);
        }
        cycle.tables = CycleTablesDriver::Host(host);
        self.scanner = None;
        Ok(())
    }

    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        if self.rounds_bound < self.address_bits() {
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
                let phase = self.rounds_bound / CHUNK_LEN;
                self.v_tables.push(eq_table(&self.phase_challenges));
                for (checkpoint, table) in
                    self.prefix_checkpoints.iter_mut().zip(&self.prefix_tables)
                {
                    *checkpoint = PrefixEval::from(table.evals()[0]);
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
            let cycle = self
                .cycle
                .as_mut()
                .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
            cycle.gruen.bind(challenge);
            match &mut cycle.tables {
                CycleTablesDriver::Host(tables) => tables.bind(challenge),
                // The device folds fused with the next round's evaluation
                // (or the handoff applies it) — never two challenges deep,
                // since every bind is followed by a message or the handoff.
                CycleTablesDriver::Device { pending_bind } => {
                    debug_assert!(pending_bind.is_none());
                    *pending_bind = Some(challenge);
                }
            }
            self.cycle_challenges.push(challenge);
        }
        self.rounds_bound += 1;
        Ok(())
    }
}

impl<F: Field> ProveRounds<F> for OptimizedInstructionReadRafKernel<F> {
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
        if self.rounds_bound < self.address_bits() {
            Ok(self.address_message(previous_claim))
        } else {
            self.cycle_message(round, previous_claim)
        }
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: Field> SumcheckKernel<F> for OptimizedInstructionReadRafKernel<F> {
    type Relation = InstructionReadRaf<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<InstructionReadRafOutputClaims<F>, SumcheckKernelError<F>> {
        if self.rounds_bound != self.num_rounds() {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.num_rounds() - self.rounds_bound,
            });
        }
        // finish_rounds' final challenge may still be pending device-side.
        self.ensure_host_cycle_tables()
            .map_err(|_| SumcheckKernelError::InvariantViolation {
                reason: "device-adopted cycle tables unavailable after full binding",
            })?;
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "cycle tables absent after full binding",
            })?;
        let CycleTablesDriver::Host(tables) = &cycle.tables else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "cycle tables absent after full binding",
            });
        };

        // Flag claims at the normalized (big-endian) cycle point via the
        // split-eq factorization `eq(r_cycle, j) = E_hi[j_hi] · E_lo[j_lo]`:
        // per-table masses accumulate over the low half and scale by `E_hi`
        // once per block (exact by distributivity).
        let r_cycle: Vec<F> = self.cycle_challenges.iter().rev().copied().collect();
        let eq_cycle = TensorEqTable::<F>::new(&r_cycle);
        let num_tables = LookupTableKind::<RISCV_XLEN>::COUNT;
        let rows = self.rows.as_slice();
        let (lookup_table_flags, instruction_raf_flag) = eq_cycle.par_fold_out_in(
            || vec![F::Accumulator::default(); num_tables + 1],
            |accumulators, row_index, _x_in, e_in| {
                let row = &rows[row_index];
                if let Some(index) = row.table_index() {
                    accumulators[index].add(e_in);
                }
                if row.raf_flag {
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

        Ok(InstructionReadRafOutputClaims {
            lookup_table_flags,
            instruction_ra: tables.ra.iter().map(|ra| ra.evals()[0]).collect(),
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
    use jolt_field::{Fr, FromPrimitiveInt, MulPow2};
    use jolt_lookup_tables::{LookupBits, LookupTableKind, XLEN as RISCV_XLEN};
    use jolt_sumcheck::ProveRounds;
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
                    None,
                    None,
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
        let count = LookupTableKind::<RISCV_XLEN>::COUNT;
        let tables = [0, 3 % count, 7 % count, 11 % count, count - 1];
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
