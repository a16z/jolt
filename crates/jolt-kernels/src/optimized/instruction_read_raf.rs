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
//! - **Shared witness rows**: re-emulating sources keep a strong
//!   `SharedInstructionRows` carry in the [`ProofSession`]; slice-backed
//!   sources keep only `SharedInstructionRowsWeak`, sharing inside one stage
//!   and rebuilding index-parallel later so `40 B × T` rows do not survive
//!   through the prover's peak window.

#[cfg(feature = "parallel")]
use std::mem::MaybeUninit;
use std::sync::Arc;

#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::metal::solinas::{
    AddressPhaseSequence, AddressPhaseSequenceConfig, AddressPhaseSums, AddressRafScanRow,
    BooleanityRow, BooleanityRows, Fp128, InstructionReadRafStage1Lease, MetalError,
    Product5Sequence, Product5SequenceConfig, SolinasMetal, PRODUCT5_FACTORS,
};
use jolt_claims::protocols::jolt::geometry::instruction::{
    InstructionReadRafDimensions, CANONICAL_INSTRUCTION_ADDRESS,
};
use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafOutputClaims;
#[cfg(all(feature = "metal", target_os = "macos"))]
use jolt_field::Prime128OffsetA7F7 as AkitaField;
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
    accumulate_product_grid, collect_par_map, for_each_index_mut, map_indices, map_reduce_chunks,
    scan_chunk_size, RoundProgress,
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
#[repr(C)]
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
const INSTRUCTION_READ_RAF_TOPOLOGY_RANK_HIGH_BIT: u8 = 1 << 6;

const _: () = assert!(LookupTableKind::<RISCV_XLEN>::COUNT < PACKED_TABLE_MASK as usize);

pub(crate) const fn canonical_instruction_read_raf_claim(claim: u8) -> u8 {
    claim & !INSTRUCTION_READ_RAF_TOPOLOGY_RANK_HIGH_BIT
}

pub(crate) const fn instruction_read_raf_claim_table_plus_one(claim: u8) -> u8 {
    canonical_instruction_read_raf_claim(claim) & PACKED_TABLE_MASK as u8
}

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
        assert!(
            bytecode_pc < PACKED_PC_MASK as usize,
            "bytecode PC exceeds packed row"
        );
        let pc_plus_one = bytecode_pc as u64 + 1;
        let table_plus_one = table_index.map_or(0, |index| index as u64 + 1);
        let packed_pc_and_flags = pc_plus_one
            | (table_plus_one << PACKED_TABLE_SHIFT)
            | (u64::from(raf_flag) << PACKED_RAF_SHIFT);
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

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) fn metal_booleanity_rows(rows: &[Self]) -> &[BooleanityRow] {
        // SAFETY: both repr(C) row types are five u64 words in the same order.
        unsafe { std::slice::from_raw_parts(rows.as_ptr().cast(), rows.len()) }
    }

    #[inline]
    pub(crate) fn table_index(&self) -> Option<usize> {
        let table_plus_one =
            ((self.packed_pc_and_flags >> PACKED_TABLE_SHIFT) & PACKED_TABLE_MASK) as usize;
        table_plus_one.checked_sub(1)
    }

    #[inline]
    pub(crate) fn bytecode_pc(&self) -> usize {
        ((self.packed_pc_and_flags & PACKED_PC_MASK) - 1) as usize
    }

    #[cfg(test)]
    pub(crate) fn mapped_pc(&self) -> Option<usize> {
        Some(self.bytecode_pc())
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

#[cfg(test)]
pub(crate) fn collect_instruction_cycle_rows<F: JoltField>(
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
) -> Result<Vec<InstructionCycleRow>, KernelError<F>> {
    InstructionCycleRow::collect(witness, cycles)
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

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn prepare_metal_instruction_read_raf(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    inputs: ProverInputs<'_, AkitaField, InstructionReadRaf<AkitaField>>,
    external_address_phases: bool,
) -> Result<OptimizedInstructionReadRafKernel<AkitaField>, KernelError<AkitaField>> {
    let dimensions = inputs.relation.dimensions();
    let rows = Arc::new(InstructionCycleRow::collect(
        witness,
        1 << dimensions.log_t(),
    )?);
    if witness.random_access().is_some() {
        session.park(SharedInstructionRowsWeak(Arc::downgrade(&rows)));
    } else {
        session.park(SharedInstructionRows(Arc::clone(&rows)));
    }
    OptimizedInstructionReadRafKernel::new_inner(
        dimensions,
        &inputs.points.lookup_output,
        rows,
        inputs.challenges.gamma,
        external_address_phases,
    )
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
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Offloaded,
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

enum InstructionReadRafClaimColumns {
    Uninitialized,
    Owned(Vec<u8>),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Stage1(InstructionReadRafStage1Lease),
}

impl InstructionReadRafClaimColumns {
    fn as_slice(&self) -> Option<&[u8]> {
        match self {
            Self::Uninitialized => None,
            Self::Owned(claims) => Some(claims),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Stage1(lease) => Some(lease.claim_slice()),
        }
    }

    fn len(&self) -> usize {
        self.as_slice().map_or(0, <[u8]>::len)
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    const fn is_stage1(&self) -> bool {
        matches!(self, Self::Stage1(_))
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
    /// Per-table cycle buckets (`u32` cycle indices), by
    /// `LookupTableKind::index()`.
    buckets: Vec<Vec<u32>>,
    /// Condensed per-cycle eq weights (see the reference kernel).
    u_evals: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    prefix_checkpoints: Vec<PrefixEval<F>>,
    /// `ALL_PREFIXES` indices referenced by tables with non-empty buckets.
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
    #[cfg_attr(feature = "allocative", allocative(skip))]
    claim_columns: InstructionReadRafClaimColumns,
    progress: RoundProgress,
    external_address_phases: bool,
}

fn build_cycle_buckets<F: JoltField>(
    rows: &[InstructionCycleRow],
) -> Result<Vec<Vec<u32>>, KernelError<F>> {
    let num_tables = LookupTableKind::<RISCV_XLEN>::COUNT;

    #[cfg(not(feature = "parallel"))]
    {
        let mut buckets = vec![Vec::new(); num_tables];
        for (cycle_index, row) in rows.iter().enumerate() {
            if let Some(table_index) = row.table_index() {
                buckets
                    .get_mut(table_index)
                    .ok_or(KernelError::InvariantViolation {
                        reason: "stage-5 row selects an unknown lookup table",
                    })?
                    .push(cycle_index as u32);
            }
        }
        Ok(buckets)
    }

    #[cfg(feature = "parallel")]
    {
        const CHUNKS_PER_THREAD: usize = 16;
        let chunk_count = rayon::current_num_threads().saturating_mul(CHUNKS_PER_THREAD);
        let chunk_size = rows.len().div_ceil(chunk_count).max(1);
        let counts_by_chunk: Vec<Vec<usize>> = rows
            .par_chunks(chunk_size)
            .map(|chunk| -> Result<Vec<usize>, KernelError<F>> {
                let mut counts = vec![0; num_tables];
                for row in chunk {
                    if let Some(table_index) = row.table_index() {
                        *counts
                            .get_mut(table_index)
                            .ok_or(KernelError::InvariantViolation {
                                reason: "stage-5 row selects an unknown lookup table",
                            })? += 1;
                    }
                }
                Ok(counts)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut totals = vec![0; num_tables];
        for counts in &counts_by_chunk {
            for (total, count) in totals.iter_mut().zip(counts) {
                *total += count;
            }
        }
        let mut uninit_buckets: Vec<Vec<MaybeUninit<u32>>> = totals
            .into_iter()
            .map(|len| {
                let mut bucket = Vec::with_capacity(len);
                bucket.resize_with(len, MaybeUninit::uninit);
                bucket
            })
            .collect();
        let mut outputs_by_chunk: Vec<Vec<&mut [MaybeUninit<u32>]>> = (0..counts_by_chunk.len())
            .map(|_| Vec::with_capacity(num_tables))
            .collect();
        for (table_index, bucket) in uninit_buckets.iter_mut().enumerate() {
            let mut remaining = bucket.as_mut_slice();
            for (chunk_index, counts) in counts_by_chunk.iter().enumerate() {
                let (output, rest) = remaining.split_at_mut(counts[table_index]);
                outputs_by_chunk[chunk_index].push(output);
                remaining = rest;
            }
            debug_assert!(remaining.is_empty());
        }
        rows.par_chunks(chunk_size)
            .enumerate()
            .zip(outputs_by_chunk.into_par_iter())
            .for_each(|((chunk_index, chunk), mut outputs)| {
                let mut positions = vec![0; num_tables];
                let cycle_index_start = chunk_index * chunk_size;
                for (offset, row) in chunk.iter().enumerate() {
                    if let Some(table_index) = row.table_index() {
                        let _ = outputs[table_index][positions[table_index]]
                            .write((cycle_index_start + offset) as u32);
                        positions[table_index] += 1;
                    }
                }
                debug_assert!(positions
                    .iter()
                    .zip(&outputs)
                    .all(|(position, output)| *position == output.len()));
            });
        Ok(uninit_buckets
            .into_iter()
            .map(|bucket| {
                // SAFETY: each chunk writes exactly the per-table count used to
                // partition every bucket, checked above in debug builds.
                unsafe { bucket.into_boxed_slice().assume_init().into_vec() }
            })
            .collect())
    }
}

impl<F: JoltField> OptimizedInstructionReadRafKernel<F> {
    pub(crate) fn new(
        dimensions: InstructionReadRafDimensions,
        r_reduction: &[F],
        rows: Arc<Vec<InstructionCycleRow>>,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        Self::new_inner(dimensions, r_reduction, rows, gamma, false)
    }

    fn new_inner(
        dimensions: InstructionReadRafDimensions,
        r_reduction: &[F],
        rows: Arc<Vec<InstructionCycleRow>>,
        gamma: F,
        external_address_phases: bool,
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

        let buckets = build_cycle_buckets(&rows)?;
        let mut present_prefixes = vec![false; ALL_PREFIXES.len()];
        for table in
            LookupTableKind::<RISCV_XLEN>::iter().filter(|table| !buckets[table.index()].is_empty())
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
            claim_columns: InstructionReadRafClaimColumns::Uninitialized,
            progress: RoundProgress::new(dimensions.sumcheck_rounds()),
            external_address_phases,
        };
        if !external_address_phases {
            kernel.init_phase(0);
        }
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
        // Condensation: fold the previous phase's bound-challenge eq weights
        // into the per-cycle mass.
        if phase != 0 {
            let shift = self.suffix_len(phase - 1);
            let rows = Arc::clone(&self.rows);
            let v_prev = std::mem::take(&mut self.v_tables[phase - 1]);
            for_each_index_mut(&mut self.u_evals, |j, u| {
                *u *= v_prev[((rows[j].lookup_index() >> shift) as usize) & (CHUNK_SIZE - 1)];
            });
            self.v_tables[phase - 1] = v_prev;
        }

        let suffix_len = self.suffix_len(phase);
        let suffix_mask = if suffix_len == 128 {
            u128::MAX
        } else {
            (1u128 << suffix_len) - 1
        };
        let upper_suffix_bits = suffix_len.saturating_sub(self.address_bits() / 2);

        // Fused RAF scan over the whole trace (deferred-reduction sums,
        // primitive-scalar multiplies).
        let rows = self.rows.as_slice();
        let u_evals = self.u_evals.as_slice();
        let raf = map_reduce_chunks(
            rows.len(),
            scan_chunk_size(rows.len()),
            |range| {
                let mut scan = RafScan::<F>::new();
                for (row, &u) in rows[range.clone()].iter().zip(&u_evals[range]) {
                    let lookup_index = row.lookup_index();
                    let chunk = ((lookup_index >> suffix_len) as usize) & (CHUNK_SIZE - 1);
                    let suffix_bits = lookup_index & suffix_mask;
                    if CANONICAL_INSTRUCTION_ADDRESS
                        && row.raf_flag()
                        && (upper_suffix_bits == 0
                            || (suffix_bits >> (suffix_len - upper_suffix_bits))
                                == (1u128 << upper_suffix_bits) - 1)
                    {
                        scan.upper_all_ones[chunk].add(u);
                    }
                    if !row.raf_flag() {
                        scan.shift_half[chunk].add(u);
                        let (left, right) = LookupBits::new(suffix_bits, suffix_len).uninterleave();
                        let left = u64::from(left);
                        if left != 0 {
                            scan.left[chunk].fmadd_u64(u, left);
                        }
                        let right = u64::from(right);
                        if right != 0 {
                            scan.right[chunk].fmadd_u64(u, right);
                        }
                    } else {
                        scan.shift_full[chunk].add(u);
                        if suffix_bits != 0 {
                            scan.identity[chunk].fmadd_u128(u, suffix_bits);
                        }
                    }
                }
                scan.reduce()
            },
            RafSums::merge,
            RafSums::zero,
        );

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

        self.init_suffix_tables(suffix_len, suffix_mask);

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

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn install_address_phase(
        &mut self,
        phase: usize,
        raf: RafSums<F>,
        suffix_tables: Vec<(LookupTableKind<RISCV_XLEN>, Vec<Polynomial<F>>)>,
    ) {
        let suffix_len = self.suffix_len(phase);
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

        self.suffix_tables = suffix_tables;
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

    /// Read-checking suffix accumulators for the phase, per present table:
    /// tables in parallel, each over parallel bucket chunks, suffixes
    /// classified once (`One` adds, {0,1}-valued adds conditionally, general
    /// ones use the primitive-scalar multiply).
    fn init_suffix_tables(&mut self, suffix_len: usize, suffix_mask: u128) {
        let rows = self.rows.as_slice();
        let u_evals = self.u_evals.as_slice();
        let present: Vec<LookupTableKind<RISCV_XLEN>> = LookupTableKind::<RISCV_XLEN>::iter()
            .filter(|table| !self.buckets[table.index()].is_empty())
            .collect();
        let buckets = self.buckets.as_slice();
        let suffix_shift = suffix_len;
        let new_tables = map_indices(present.len(), |table_position| {
            let table = present[table_position];
            let suffixes = table.suffixes();
            let num_suffixes = suffixes.len();
            let one_position = suffixes
                .iter()
                .position(|suffix| matches!(suffix, Suffixes::One));
            let bucket = &buckets[table.index()];

            let flat = map_reduce_chunks(
                bucket.len(),
                scan_chunk_size(bucket.len()),
                |range| {
                    let mut accumulators =
                        vec![F::Accumulator::default(); num_suffixes * CHUNK_SIZE];
                    for &j in &bucket[range] {
                        let row = &rows[j as usize];
                        let u = u_evals[j as usize];
                        let lookup_index = row.lookup_index();
                        let chunk = ((lookup_index >> suffix_shift) as usize) & (CHUNK_SIZE - 1);
                        let suffix_bits = LookupBits::new(lookup_index & suffix_mask, suffix_len);
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
                                    slot.fmadd_u64(u, value);
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
            );

            let polynomials = flat
                .chunks_exact(CHUNK_SIZE)
                .map(|coefficients| Polynomial::new(coefficients.to_vec()))
                .collect();
            (table, polynomials)
        });
        self.suffix_tables = new_tables;
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
        let claim_columns =
            self.claim_columns
                .as_slice()
                .ok_or(SumcheckError::MissingEvaluationSource {
                    kind: "instruction read-RAF claim columns",
                })?;
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
                            let lo =
                                e_in * Self::pending_combined_base(pending, claim_columns, 2 * row);
                            let hi = e_in
                                * Self::pending_combined_base(pending, claim_columns, 2 * row + 1);
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
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    CycleTables::Offloaded => return,
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
    fn init_cycle_rounds(&mut self) -> Result<(), SumcheckError<F>> {
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
        const {
            assert!(
                LookupTableKind::<RISCV_XLEN>::COUNT < 0x3f,
                "table indices must fit the packed claim byte"
            );
        }
        match &self.claim_columns {
            InstructionReadRafClaimColumns::Uninitialized => {
                let rows = self.rows.as_slice();
                self.claim_columns =
                    InstructionReadRafClaimColumns::Owned(map_indices(rows.len(), |j| {
                        let row = &rows[j];
                        let table = row.table_index().map_or(0, |index| index as u8 + 1);
                        table | (u8::from(row.raf_flag()) << 7)
                    }));
            }
            InstructionReadRafClaimColumns::Owned(_) => {}
            #[cfg(all(feature = "metal", target_os = "macos"))]
            InstructionReadRafClaimColumns::Stage1(_) => {}
        }
        if self.claim_columns.len() != 1usize << self.dimensions.log_t() {
            return Err(SumcheckError::MissingEvaluationSource {
                kind: "instruction read-RAF claim columns",
            });
        }

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
        Ok(())
    }

    /// The pending combined-value base at cycle `j` (packed-byte lookup).
    #[inline]
    fn pending_combined_base(pending: &PendingCycleTables<F>, claim_columns: &[u8], j: usize) -> F {
        let packed = canonical_instruction_read_raf_claim(claim_columns[j]);
        let table_value = match instruction_read_raf_claim_table_plus_one(packed) {
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
                    if !self.external_address_phases {
                        self.init_phase(phase + 1);
                    }
                } else {
                    self.init_cycle_rounds()?;
                }
            }
        } else {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            if self.claim_columns.is_stage1()
                && self
                    .cycle
                    .as_ref()
                    .is_some_and(|cycle| matches!(cycle.tables, CycleTables::Pending(_)))
            {
                return Err(SumcheckError::ComputeBackend {
                    backend: "metal",
                    message:
                        "resident Stage-1 first cycle bind requires the retained address sequence"
                            .to_owned(),
                });
            }
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
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    CycleTables::Offloaded => {
                        return Err(SumcheckError::ComputeBackend {
                            backend: "metal",
                            message: "CPU bind requires resident cycle tables".to_owned(),
                        });
                    }
                }
            };
            if let Some(pending) = pending {
                // First cycle bind: materialize the half-domain tables
                // straight from the bases under this challenge — the same
                // values a full-T materialization would bind to, without
                // the full-T tables ever existing.
                let claim_columns = self.claim_columns.as_slice().ok_or(
                    SumcheckError::MissingEvaluationSource {
                        kind: "instruction read-RAF claim columns",
                    },
                )?;
                let half = claim_columns.len() / 2;
                let combined_val: Vec<F> = map_indices(half, |position| {
                    let lo = Self::pending_combined_base(&pending, claim_columns, 2 * position);
                    let hi = Self::pending_combined_base(&pending, claim_columns, 2 * position + 1);
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

#[cfg(all(feature = "metal", target_os = "macos"))]
impl OptimizedInstructionReadRafKernel<AkitaField> {
    pub(crate) fn new_metal_resident(
        dimensions: InstructionReadRafDimensions,
        r_reduction: &[AkitaField],
        claims: InstructionReadRafStage1Lease,
        gamma: AkitaField,
    ) -> Result<Self, KernelError<AkitaField>> {
        let address_bits = dimensions.instruction_address_bits();
        let log_t = dimensions.log_t();
        let ra_count = dimensions.num_virtual_ra_polys();
        if address_bits != 2 * RISCV_XLEN {
            return Err(KernelError::Unsupported {
                reason: "instruction read-RAF supports only the 2·XLEN interleaved-operand address width",
            });
        }
        if !address_bits.is_multiple_of(ra_count)
            || !(address_bits / ra_count).is_multiple_of(CHUNK_LEN)
        {
            return Err(KernelError::Unsupported {
                reason: "virtual RA chunk width must be a multiple of the phase width",
            });
        }
        if ra_count + 1 != PRODUCT5_FACTORS {
            return Err(KernelError::Unsupported {
                reason: "resident instruction read-RAF requires the four-RA Product5 geometry",
            });
        }
        if log_t >= 32 {
            return Err(KernelError::Unsupported {
                reason: "resident instruction read-RAF cycle indices are u32",
            });
        }
        if r_reduction.len() != log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "instruction claim-reduction point".to_owned(),
                expected: log_t,
                got: r_reduction.len(),
            });
        }
        let rows = 1usize << log_t;
        if claims.receipt().rows() != rows || claims.claim_slice().len() != rows {
            return Err(KernelError::TableSizeMismatch {
                table: "resident instruction read-RAF claims".to_owned(),
                expected: rows,
                got: claims.claim_slice().len(),
            });
        }

        Ok(Self {
            dimensions,
            gamma,
            r_reduction: r_reduction.to_vec(),
            rows: Arc::new(Vec::new()),
            buckets: Vec::new(),
            u_evals: Vec::new(),
            prefix_checkpoints: ALL_PREFIXES
                .iter()
                .map(|prefix| prefix.default_checkpoint::<AkitaField>())
                .collect(),
            prefix_indices: (0..ALL_PREFIXES.len()).collect(),
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
            claim_columns: InstructionReadRafClaimColumns::Stage1(claims),
            progress: RoundProgress::new(dimensions.sumcheck_rounds()),
            external_address_phases: true,
        })
    }

    pub(crate) fn metal_prepare_booleanity_rows(
        &self,
        context: &SolinasMetal,
    ) -> Result<BooleanityRows, MetalError> {
        if self.claim_columns.is_stage1() {
            return Err(MetalError::InvalidInstructionReadRafGrouped(
                "resident Stage-1 rows must be borrowed from their owner".to_owned(),
            ));
        }
        context.prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&self.rows))
    }

    pub(crate) fn metal_prepare_address_sequence(
        &mut self,
        context: &SolinasMetal,
        config: AddressPhaseSequenceConfig,
    ) -> Result<AddressPhaseSequence, SumcheckError<AkitaField>> {
        if self.claim_columns.is_stage1() {
            return Err(metal_state_error(
                "resident Stage-1 state requires prebuilt grouped planes",
            ));
        }
        if !self.external_address_phases || self.progress.bound() != 0 {
            return Err(metal_state_error(
                "resident address handoff requires an unbound external address state",
            ));
        }
        let sequence = context
            .prepare_address_phase_sequence_from_buckets(
                self.rows.len(),
                &self.buckets,
                config,
                |index| {
                    let row = &self.rows[index];
                    (
                        AddressRafScanRow::new_with_table(
                            row.lookup_index(),
                            row.table_index(),
                            row.raf_flag(),
                        ),
                        Fp128::from_jolt_field(&self.u_evals[index]),
                    )
                },
            )
            .map_err(metal_sumcheck_error)?;
        self.u_evals = Vec::new();
        self.buckets = Vec::new();
        Ok(sequence)
    }

    pub(crate) fn metal_address_phase_request(
        &self,
    ) -> Result<(u32, Option<[Fp128; CHUNK_SIZE]>), SumcheckError<AkitaField>> {
        if !self.external_address_phases
            || self.progress.bound() >= self.address_bits()
            || !self.progress.bound().is_multiple_of(CHUNK_LEN)
        {
            return Err(metal_state_error(
                "resident address phase requested outside a phase boundary",
            ));
        }
        let phase = self.progress.bound() / CHUNK_LEN;
        let previous = if phase == 0 {
            None
        } else {
            let table = self.v_tables.get(phase - 1).ok_or_else(|| {
                metal_state_error("resident address condensation table is absent")
            })?;
            if table.len() != CHUNK_SIZE {
                return Err(metal_state_error(
                    "resident address condensation table has the wrong length",
                ));
            }
            Some(std::array::from_fn(|index| {
                Fp128::from_jolt_field(&table[index])
            }))
        };
        Ok((self.suffix_len(phase) as u32, previous))
    }

    pub(crate) fn metal_install_address_phase(
        &mut self,
        sums: AddressPhaseSums,
    ) -> Result<(), SumcheckError<AkitaField>> {
        if !self.external_address_phases
            || self.progress.bound() >= self.address_bits()
            || !self.progress.bound().is_multiple_of(CHUNK_LEN)
        {
            return Err(metal_state_error(
                "resident address output arrived outside a phase boundary",
            ));
        }
        let convert = |values: &[Fp128]| {
            values
                .iter()
                .copied()
                .map(Fp128::into_jolt_field::<AkitaField>)
                .collect::<Vec<_>>()
        };
        let raf = RafSums {
            shift_half: convert(sums.raf().shift_half()),
            left: convert(sums.raf().left()),
            right: convert(sums.raf().right()),
            shift_full: convert(sums.raf().shift_full()),
            identity: convert(sums.raf().identity()),
            upper_all_ones: convert(sums.raf().upper_all_ones()),
        };
        let mut suffix_tables = Vec::with_capacity(LookupTableKind::<RISCV_XLEN>::COUNT);
        for table in LookupTableKind::<RISCV_XLEN>::iter() {
            let flat = sums
                .suffix()
                .table(table.index())
                .ok_or_else(|| metal_state_error("resident address suffix table is absent"))?;
            let polynomials = flat
                .chunks_exact(CHUNK_SIZE)
                .map(|coefficients| Polynomial::new(convert(coefficients)))
                .collect();
            suffix_tables.push((table, polynomials));
        }
        self.install_address_phase(self.progress.bound() / CHUNK_LEN, raf, suffix_tables);
        Ok(())
    }

    pub(crate) fn metal_address_active(&self) -> bool {
        self.external_address_phases && self.progress.bound() < self.address_bits()
    }

    pub(crate) fn metal_address_phase_pending(&self) -> bool {
        self.metal_address_active() && self.progress.bound().is_multiple_of(CHUNK_LEN)
    }

    pub(crate) fn metal_bind_address(
        &mut self,
        challenge: AkitaField,
    ) -> Result<(), SumcheckError<AkitaField>> {
        if !self.metal_address_active() {
            return Err(metal_state_error(
                "resident address bind requested after the address rounds",
            ));
        }
        self.bind(challenge)
    }

    pub(crate) fn metal_address_message(
        &self,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if !self.metal_address_active() {
            return Err(metal_state_error(
                "resident address message requested after the address rounds",
            ));
        }
        Ok(self.address_message(previous_claim))
    }

    pub(crate) fn metal_resident_cycle_message(
        &self,
        sequence: &mut AddressPhaseSequence,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        let CycleTables::Pending(pending) = &cycle.tables else {
            return Err(metal_state_error(
                "resident first cycle message requires pending tables",
            ));
        };
        let q_evals = sequence
            .cycle_message(
                &self.v_tables,
                &pending.table_values,
                pending.raf_interleaved,
                pending.raf_identity,
                cycle.gruen.e_in_current(),
                cycle.gruen.e_out_current(),
            )
            .map_err(metal_sumcheck_error)?;
        Ok(cycle.gruen.gruen_poly_from_evals(&q_evals, previous_claim))
    }

    pub(crate) fn metal_resident_cycle_available(&self) -> bool {
        self.dimensions.num_virtual_ra_polys() + 1 == PRODUCT5_FACTORS
    }

    pub(crate) fn metal_offload_resident_bind(
        &mut self,
        challenge: AkitaField,
        sequence: AddressPhaseSequence,
        config: Product5SequenceConfig,
    ) -> Result<(Product5Sequence, [AkitaField; PRODUCT5_FACTORS]), SumcheckError<AkitaField>> {
        let pending = {
            let cycle = self
                .cycle
                .as_mut()
                .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
            cycle.gruen.bind(challenge);
            let tables = core::mem::replace(&mut cycle.tables, CycleTables::Offloaded);
            let CycleTables::Pending(pending) = tables else {
                return Err(metal_state_error(
                    "resident cycle handoff requires pending tables",
                ));
            };
            pending
        };
        self.cycle_challenges.push(challenge);
        self.progress.advance();
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        let result = sequence
            .fused_cycle_transition(
                &self.v_tables,
                &pending.table_values,
                pending.raf_interleaved,
                pending.raf_identity,
                challenge,
                cycle.gruen.e_in_current(),
                cycle.gruen.e_out_current(),
                config,
            )
            .map_err(metal_sumcheck_error)?;
        self.rows = Arc::new(Vec::new());
        self.v_tables = Vec::new();
        Ok(result)
    }

    pub(crate) fn metal_handoff_available(&self, cutoff: usize) -> bool {
        self.dimensions.num_virtual_ra_polys() + 1 == PRODUCT5_FACTORS
            && !self.claim_columns.is_stage1()
            && self.claim_columns.len() / 2 > cutoff
            && self
                .cycle
                .as_ref()
                .is_some_and(|cycle| matches!(cycle.tables, CycleTables::Pending(_)))
    }

    pub(crate) fn metal_offload_pending_bind(
        &mut self,
        challenge: AkitaField,
        context: &SolinasMetal,
        config: Product5SequenceConfig,
    ) -> Result<Product5Sequence, SumcheckError<AkitaField>> {
        if self.claim_columns.is_stage1() {
            return Err(metal_state_error(
                "resident Stage-1 state cannot use the CPU pending-table handoff",
            ));
        }
        let (pending, e_in, e_out) = {
            let cycle = self
                .cycle
                .as_mut()
                .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
            cycle.gruen.bind(challenge);
            let tables = core::mem::replace(&mut cycle.tables, CycleTables::Offloaded);
            let CycleTables::Pending(pending) = tables else {
                return Err(metal_state_error(
                    "dense-cycle handoff requires pending CPU tables",
                ));
            };
            (
                pending,
                cycle.gruen.e_in_current().to_vec(),
                cycle.gruen.e_out_current().to_vec(),
            )
        };

        let claim_columns = self
            .claim_columns
            .as_slice()
            .ok_or_else(|| metal_state_error("cycle claim columns are absent"))?;
        let elements = claim_columns.len() / 2;
        let sequence = context
            .prepare_product5_sequence_from_fn(elements, &e_in, &e_out, config, |index| {
                let factor = index / elements;
                let position = index % elements;
                let source = 2 * position;
                let (lo, hi) = if factor == 0 {
                    (
                        Self::pending_combined_base(&pending, claim_columns, source),
                        Self::pending_combined_base(&pending, claim_columns, source + 1),
                    )
                } else {
                    (
                        self.pending_ra_base(factor - 1, source),
                        self.pending_ra_base(factor - 1, source + 1),
                    )
                };
                lo + challenge * (hi - lo)
            })
            .map_err(metal_sumcheck_error)?;

        self.rows = Arc::new(Vec::new());
        self.v_tables = Vec::new();
        self.cycle_challenges.push(challenge);
        self.progress.advance();
        Ok(sequence)
    }

    pub(crate) fn metal_bind_offloaded(
        &mut self,
        challenge: AkitaField,
    ) -> Result<(), SumcheckError<AkitaField>> {
        let cycle = self
            .cycle
            .as_mut()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        if !matches!(cycle.tables, CycleTables::Offloaded) {
            return Err(metal_state_error(
                "device bind requires offloaded cycle tables",
            ));
        }
        cycle.gruen.bind(challenge);
        self.cycle_challenges.push(challenge);
        self.progress.advance();
        Ok(())
    }

    pub(crate) fn metal_cycle_weights(
        &self,
    ) -> Result<(&[AkitaField], &[AkitaField]), SumcheckError<AkitaField>> {
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        Ok((cycle.gruen.e_in_current(), cycle.gruen.e_out_current()))
    }

    pub(crate) fn metal_cycle_message(
        &self,
        q_evals: &[AkitaField; PRODUCT5_FACTORS],
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        if !matches!(cycle.tables, CycleTables::Offloaded) {
            return Err(metal_state_error(
                "device message requires offloaded cycle tables",
            ));
        }
        Ok(cycle.gruen.gruen_poly_from_evals(q_evals, previous_claim))
    }

    pub(crate) fn metal_restore_dense(
        &mut self,
        tables: [Vec<AkitaField>; PRODUCT5_FACTORS],
    ) -> Result<(), SumcheckError<AkitaField>> {
        let cycle = self
            .cycle
            .as_mut()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        if !matches!(cycle.tables, CycleTables::Offloaded) {
            return Err(metal_state_error(
                "device readback requires offloaded cycle tables",
            ));
        }
        let [combined_val, ra_0, ra_1, ra_2, ra_3] = tables;
        cycle.tables = CycleTables::Dense {
            combined_val: Polynomial::new(combined_val),
            ra: vec![
                Polynomial::new(ra_0),
                Polynomial::new(ra_1),
                Polynomial::new(ra_2),
                Polynomial::new(ra_3),
            ],
        };
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_sumcheck_error(error: MetalError) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_state_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
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
        let claim_columns =
            self.claim_columns
                .as_slice()
                .ok_or(SumcheckKernelError::InvariantViolation {
                    reason: "instruction read-RAF claim columns are absent",
                })?;
        if claim_columns.len() != 1usize << self.dimensions.log_t() {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "instruction read-RAF claim columns have the wrong length",
            });
        }
        let (lookup_table_flags, instruction_raf_flag) = eq_cycle.par_fold_out_in(
            || vec![F::Accumulator::default(); num_tables + 1],
            |accumulators, row_index, _x_in, e_in| {
                let packed = canonical_instruction_read_raf_claim(claim_columns[row_index]);
                let table_plus_one = instruction_read_raf_claim_table_plus_one(packed);
                if table_plus_one != 0 {
                    accumulators[usize::from(table_plus_one) - 1].add(e_in);
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

    use super::{build_cycle_buckets, InstructionCycleRow, OptimizedInstructionReadRafKernel};

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
    fn cycle_buckets_preserve_cycle_order() {
        let rows = pack(&fixture_rows(9, 0xB0C7));
        let actual = build_cycle_buckets::<Fr>(&rows).unwrap();
        let mut expected = vec![Vec::new(); LookupTableKind::<RISCV_XLEN>::COUNT];
        for (cycle, row) in rows.iter().enumerate() {
            if let Some(table) = row.table_index() {
                expected[table].push(cycle as u32);
            }
        }
        assert_eq!(actual, expected);
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
