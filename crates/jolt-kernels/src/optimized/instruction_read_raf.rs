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
//!   sources keep only `SharedInstructionRowsWeak` and rebuild index-parallel
//!   once the last owner drops. Queued destruction can extend sharing across
//!   stages.

mod cycle_groups;

use cycle_groups::CyclePairGroups;
use std::num::NonZeroU16;
use std::sync::{Arc, Weak};

use jolt_claims::protocols::jolt::geometry::instruction::{
    InstructionReadRafDimensions, CANONICAL_INSTRUCTION_ADDRESS,
};
use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafOutputClaims;
use jolt_field::{Accumulator, JoltField};
use jolt_lookup_tables::tables::prefixes::{PrefixEval, ALL_PREFIXES};
use jolt_lookup_tables::tables::suffixes::{SuffixEval, Suffixes};
use jolt_lookup_tables::{LookupBits, LookupTableKind, XLEN as RISCV_XLEN};
use jolt_poly::{
    BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, TensorEqTable, UnivariatePoly,
};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::SumcheckInputClaims;
use jolt_verifier::stages::stage5::InstructionReadRaf;
#[cfg(feature = "akita")]
use jolt_witness::witnesses::FusedInc;
use jolt_witness::witnesses::{
    BytecodePc, InstructionRafFlag, LookupIndex, RemappedRamAddress, TableIndex,
};
use jolt_witness::{stream_witnesses, JoltWitnessPlane, StreamConsumer, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::constant_suffix::ConstantSuffix;
use super::lazy_ra::{
    ChunkIndexSource, LazyFoldedRa, PAIRED_MATERIALIZE_ADDRESSES, PAIRED_MATERIALIZE_BRANCHES,
};
use super::support::{
    accumulate_product_grid, collect_par_map, map_indices, map_reduce_chunks,
    product_grid_scratch_len, repeat_hash, repeats_often, scan_chunk_size, BundleAccess,
    GruenRoundMessage, LowDegreeProductSum, PrefixCycleWeights, RepeatCache, RoundProgress,
    TailFactorProducts,
};
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Address variables bound per phase — identical to the reference kernel (and
/// to the legacy prover below its 2^24-cycle threshold).
const CHUNK_LEN: usize = 8;
const CHUNK_SIZE: usize = 1 << CHUNK_LEN;

/// Widest lazy branch set of the cycle tables: the 2^16-entry RA columns
/// outgrow the caches as their branch tables double, so past four branches
/// a gather round costs more than binding the `T/8` dense tables it saves.
const LAZY_MAX_WIDTH: usize = 4;

const _: () = assert!(
    LookupTableKind::<RISCV_XLEN>::COUNT < u8::MAX as usize,
    "InstructionCycleRow packs lookup table indices as u8"
);

/// One packed per-cycle row: the stage-5 facts plus the bytecode/RAM and
/// packed fused-inc sources used by later one-hot kernels. The lookup index
/// is split into native limbs and the PC/table/flags share one word, keeping
/// the retained row at 40 bytes in Akita mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
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
    pub(crate) fn collect<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<ConstantSuffix<Self>, KernelError<F>> {
        if let Some(access) = witness.random_access() {
            if cycles <= access.cycles() {
                let retained = BundleAccess::<WideInstructionRow>::View(&access)
                    .constant_suffix_start(cycles)?
                    .saturating_add(1)
                    .min(cycles);
                let rows = collect_par_map(&access, retained, |row: WideInstructionRow| {
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
                return Ok(ConstantSuffix::new(rows, cycles));
            }
        }
        let mut consumers = (PackRows {
            rows: Vec::with_capacity(cycles),
        },);
        stream_witnesses(witness, 0..cycles, 1 << 12, &mut consumers)?;
        Ok(consumers.0.rows.into())
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
pub(crate) struct SharedInstructionRows(pub(crate) Arc<ConstantSuffix<InstructionCycleRow>>);

/// Weak cache for slice-backed rows; it does not keep a collection alive.
/// A queued kernel drop can retain the last strong reference across stages.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SharedInstructionRowsWeak(pub(crate) Weak<ConstantSuffix<InstructionCycleRow>>);

impl InstructionCycleRow {
    /// Reuse a length-matched carry; weak rows are valid only for random-access
    /// witnesses, whose collection does not replay execution.
    pub(crate) fn cached<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Option<Arc<ConstantSuffix<Self>>> {
        if let Some(SharedInstructionRows(rows)) = session.state::<SharedInstructionRows>() {
            if rows.len() == cycles {
                return Some(Arc::clone(rows));
            }
            drop(session.take::<SharedInstructionRows>());
        }
        let _ = witness.random_access()?;
        session
            .state::<SharedInstructionRowsWeak>()
            .and_then(|weak| weak.0.upgrade())
            .filter(|rows| rows.len() == cycles)
    }

    /// Reuse cached rows or collect them, retaining a strong carry only for
    /// witnesses whose next collection would replay execution.
    pub(crate) fn shared<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<Arc<ConstantSuffix<Self>>, KernelError<F>> {
        let rows = match Self::cached(session, witness, cycles) {
            Some(rows) => rows,
            None => Arc::new(Self::collect(witness, cycles)?),
        };
        if witness.random_access().is_some() {
            drop(session.take::<SharedInstructionRows>());
            session.park(SharedInstructionRowsWeak(Arc::downgrade(&rows)));
        } else {
            session.park(SharedInstructionRows(Arc::clone(&rows)));
        }
        Ok(rows)
    }
}

/// Lazy-RA index source: column `i` is the `i`-th most significant
/// `chunk_bits`-bit chunk of the per-cycle lookup index (always hot), off the
/// stage-5 rows.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct LookupIndexChunks {
    rows: Arc<ConstantSuffix<InstructionCycleRow>>,
    prefix_len: usize,
    chunks: usize,
    chunk_bits: usize,
}

impl LookupIndexChunks {
    /// The packed address before extracting the committed selector chunks.
    #[inline]
    pub(crate) fn packed_index(&self, cycle: usize) -> u128 {
        self.rows[cycle].lookup_index()
    }

    pub(crate) fn new(
        rows: Arc<ConstantSuffix<InstructionCycleRow>>,
        chunks: usize,
        chunk_bits: usize,
    ) -> Self {
        let prefix_len = rows.prefix_len();
        Self {
            rows,
            prefix_len,
            chunks,
            chunk_bits,
        }
    }
}

pub(super) fn visit_paired_four_bit_values<F: JoltField>(
    packed_index: impl Fn(usize) -> u128,
    tables: &[Vec<F>],
    mut visit: impl FnMut(usize, F),
) {
    const DIGITS_PER_LIMB: usize = u64::BITS as usize / 4;
    let radix = PAIRED_MATERIALIZE_ADDRESSES + 1;
    // Four-bit digits never cross a 64-bit limb boundary.
    for limb in 0..tables.len().div_ceil(DIGITS_PER_LIMB) {
        let end = tables.len() - limb * DIGITS_PER_LIMB;
        let start = end.saturating_sub(DIGITS_PER_LIMB);
        let mut indices = [0u64; PAIRED_MATERIALIZE_BRANCHES];
        for (branch, index) in indices.iter_mut().enumerate() {
            *index = (packed_index(branch) >> (limb * u64::BITS as usize)) as u64;
        }
        for (column, table) in tables[start..end].iter().enumerate().rev() {
            let mut sum = F::Accumulator::default();
            for (pair, indices) in indices.chunks_exact_mut(2).enumerate() {
                let lo = (indices[0] & 15) as usize;
                let hi = (indices[1] & 15) as usize;
                sum.add(table[pair * radix * radix + lo * radix + hi]);
                indices[0] >>= 4;
                indices[1] >>= 4;
            }
            visit(start + column, sum.reduce());
        }
    }
}

impl LookupIndexChunks {
    fn visit_paired_values_inner<F: JoltField>(
        &self,
        tables: &[Vec<F>],
        row: usize,
        mut visit: impl FnMut(usize, F),
    ) {
        let bits = self.chunk_bits;
        let mut indices = [0u128; PAIRED_MATERIALIZE_BRANCHES];
        for (branch, index) in indices.iter_mut().enumerate() {
            *index = self.rows[row * PAIRED_MATERIALIZE_BRANCHES + branch].lookup_index();
        }
        let mask = (1u128 << bits) - 1;
        let radix = PAIRED_MATERIALIZE_ADDRESSES + 1;
        for (column, table) in tables.iter().enumerate().rev() {
            let mut sum = F::Accumulator::default();
            for (pair, indices) in indices.chunks_exact_mut(2).enumerate() {
                let lo = (indices[0] & mask) as usize;
                let hi = (indices[1] & mask) as usize;
                sum.add(table[pair * radix * radix + lo * radix + hi]);
                indices[0] >>= bits;
                indices[1] >>= bits;
            }
            visit(column, sum.reduce());
        }
    }
}

impl ChunkIndexSource for LookupIndexChunks {
    fn constant_suffix_start(&self) -> usize {
        self.prefix_len
    }
    fn num_columns(&self) -> usize {
        self.chunks
    }

    fn cycles(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    fn index(&self, i: usize, j: usize) -> Option<usize> {
        let shift = (self.chunks - 1 - i) * self.chunk_bits;
        let mask = (1u128 << self.chunk_bits) - 1;
        Some(((self.packed_index(j) >> shift) & mask) as usize)
    }

    fn gather_values<F: JoltField>(
        &self,
        tables: &[Vec<F>],
        width: usize,
        row: usize,
        out: &mut [&mut [F]],
        output_row: usize,
    ) {
        let mut local = [0u128; 16];
        let mut heap;
        let indices = if width <= local.len() {
            &mut local[..width]
        } else {
            heap = vec![0; width];
            heap.as_mut_slice()
        };
        for (branch, index) in indices.iter_mut().enumerate() {
            *index = self.rows[row * width + branch].lookup_index();
        }
        let mask = (1u128 << self.chunk_bits) - 1;
        let stride = 1usize << self.chunk_bits;
        for (column, table) in out.iter_mut().zip(tables).rev() {
            let mut sum = F::Accumulator::default();
            for (branch, index) in indices.iter_mut().enumerate() {
                sum.add(table[branch * stride + (*index & mask) as usize]);
                *index >>= self.chunk_bits;
            }
            column[output_row] = sum.reduce();
        }
    }

    fn visit_paired_values<F: JoltField>(
        &self,
        tables: &[Vec<F>],
        row: usize,
        visit: impl FnMut(usize, F),
    ) {
        match self.chunk_bits {
            4 => visit_paired_four_bit_values(
                |branch| self.packed_index(row * PAIRED_MATERIALIZE_BRANCHES + branch),
                tables,
                visit,
            ),
            _ => self.visit_paired_values_inner(tables, row, visit),
        }
    }

    #[inline]
    fn gather_pairs<F: JoltField>(
        &self,
        tables: &[Vec<F>],
        width: usize,
        row: usize,
        out: &mut [(F, F)],
    ) {
        let mask = (1u128 << self.chunk_bits) - 1;
        let stride = 1usize << self.chunk_bits;
        for branch in 0..width {
            let mut lo = self.rows[2 * row * width + branch].lookup_index();
            let mut hi = self.rows[(2 * row + 1) * width + branch].lookup_index();
            let base = branch * stride;
            for (out, table) in out.iter_mut().zip(tables).rev() {
                let pair = (
                    table[base + (lo & mask) as usize],
                    table[base + (hi & mask) as usize],
                );
                if branch == 0 {
                    *out = pair;
                } else {
                    out.0 += pair.0;
                    out.1 += pair.1;
                }
                lo >>= self.chunk_bits;
                hi >>= self.chunk_bits;
            }
        }
    }
}

/// Lazy-RA index source for the combined cycle value: the packed claim byte
/// (see the kernel's `claim_columns`) keys a 256-entry value table.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct ClaimBytes(Arc<ConstantSuffix<u8>>);

impl ChunkIndexSource for ClaimBytes {
    fn constant_suffix_start(&self) -> usize {
        self.0.prefix_len()
    }

    fn num_columns(&self) -> usize {
        1
    }

    fn cycles(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn index(&self, _i: usize, j: usize) -> Option<usize> {
        Some(usize::from(self.0[j]))
    }
}

pub struct OptimizedInstructionReadRaf;

impl<F: JoltField> PrepareKernel<F, InstructionReadRaf<F>> for OptimizedInstructionReadRaf {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionReadRaf<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionReadRaf<F>>>, KernelError<F>> {
        let dimensions = inputs.relation.dimensions();
        let rows: Arc<ConstantSuffix<InstructionCycleRow>> = Arc::new(
            InstructionCycleRow::collect(witness, 1 << dimensions.log_t())?,
        );
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

#[inline]
fn extension_pair<F: JoltField>(evals: &[F], b: usize, half: usize) -> (F, F) {
    let lo = evals[b];
    let hi = evals[b + half];
    (lo, hi + hi - lo)
}

/// Cycle-round state: the Gruen-split eq factor plus the cycle tables. Both
/// tables are point masses over compact per-cycle columns, served
/// index-encoded until the third cycle bind materializes them at `T/8`
/// ([`LazyFoldedRa`]): the combined value is categorical in the packed claim
/// byte, and each virtual `ra_i` is the product of its phases' eq tables at
/// the lookup-index chunks. No dense cycle table is ever longer than `T/8`
/// (dense `(1 + ra_count) × T/2` tables would be the stage-5 peak), and no
/// gather multiplies unless a virtual chunk is wider than 16 bits.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct CycleState<F: JoltField> {
    gruen: GruenSplitEqPolynomial<F>,
    /// The per-cycle `Val + γ·RafVal` at the bound address point, keyed by
    /// the packed claim byte.
    combined_val: LazyFoldedRa<F, ClaimBytes>,
    /// Virtual RA polynomials over lookup-index chunks of at most 16 bits
    /// (the tensor product of those phases' eq tables); wider virtual
    /// chunks are products of such columns.
    ra: LazyFoldedRa<F, LookupIndexChunks>,
}

struct RafScan<F: JoltField> {
    shift_half: Vec<F::SumAccumulator>,
    left: Vec<F::Accumulator>,
    right: Vec<F::Accumulator>,
    shift_full: Vec<F::SumAccumulator>,
    identity: Vec<F::Accumulator>,
    upper_all_ones: Vec<F::SumAccumulator>,
}

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
            shift_half: vec![F::SumAccumulator::default(); CHUNK_SIZE],
            left: vec![F::Accumulator::default(); CHUNK_SIZE],
            right: vec![F::Accumulator::default(); CHUNK_SIZE],
            shift_full: vec![F::SumAccumulator::default(); CHUNK_SIZE],
            identity: vec![F::Accumulator::default(); CHUNK_SIZE],
            upper_all_ones: vec![F::SumAccumulator::default(); CHUNK_SIZE],
        }
    }

    fn reduce(self) -> RafSums<F> {
        fn reduce<A: Accumulator>(accumulators: Vec<A>) -> Vec<A::Element> {
            accumulators
                .into_iter()
                .map(|accumulator| accumulator.reduce())
                .collect()
        }
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
    /// `OptimizedInstructionReadRafKernel::new` rejects rows whose
    /// `table_index()` is not below `LookupTableKind::COUNT` before it builds
    /// the blocks; such a row's class would index `starts` out of bounds.
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

    fn should_coalesce(rows: &[InstructionCycleRow]) -> bool {
        repeats_often(
            rows.len(),
            SCAN_BLOCK,
            |row| Self::key_hash(&rows[row]),
            |a, b| Self::same_key(&rows[a], &rows[b]),
        )
    }

    /// Address phases are linear in the cycle weights. Equal lookup keys
    /// within a class receive identical updates, so retain their summed weight
    /// at one offset. The cycle rounds still use the original witness rows.
    fn coalesced<F: JoltField>(
        rows: &[InstructionCycleRow],
        weights: &mut [F],
        cache: &mut RepeatCache<NonZeroU16>,
    ) -> Self {
        let mut block = Self::new(rows);
        let starts = block.starts;
        let mut end = 0;
        cache.clear();
        for (class, range) in starts.windows(2).enumerate() {
            block.starts[class] = end as u16;
            for position in usize::from(range[0])..usize::from(range[1]) {
                let offset = usize::from(block.offsets[position]);
                let row = &rows[offset];
                let representative =
                    cache.find_or_insert(Self::key_hash(row), offset, |previous| {
                        Self::same_key(&rows[previous], row)
                    });
                if let Some(previous) = representative {
                    let weight = weights[offset];
                    weights[previous] += weight;
                } else {
                    block.offsets[end] = offset as u16;
                    end += 1;
                }
            }
        }
        block.starts[SCAN_CLASSES] = end as u16;
        block
    }

    fn key_hash(row: &InstructionCycleRow) -> u64 {
        let key = row.lookup_index();
        repeat_hash([key as u64, (key >> 64) as u64, Self::class(row) as u64])
    }

    fn same_key(left: &InstructionCycleRow, right: &InstructionCycleRow) -> bool {
        left.lookup_index() == right.lookup_index() && Self::class(left) == Self::class(right)
    }

    fn class(row: &InstructionCycleRow) -> usize {
        2 * row.table_index().map_or(0, |table| table + 1) + usize::from(row.raf_flag())
    }
}

/// Scan tasks per pool thread, so work stealing balances the phase scan
/// across heterogeneous cores.
const SCAN_TASKS_PER_THREAD: usize = 8;

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
/// `LookupTableKind::index()`, `None` for tables no cycle selects.
struct PhaseSums<F> {
    raf: RafSums<F>,
    suffixes: Vec<Option<Vec<F>>>,
}

impl<F: JoltField> PhaseSums<F> {
    fn zero() -> Self {
        Self {
            raf: RafSums::zero(),
            suffixes: vec![None; LookupTableKind::<RISCV_XLEN>::COUNT],
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.raf = self.raf.merge(other.raf);
        for (sums, other) in self.suffixes.iter_mut().zip(other.suffixes) {
            let Some(other) = other else {
                continue;
            };
            match sums {
                Some(sums) => {
                    for (a, b) in sums.iter_mut().zip(&other) {
                        *a += *b;
                    }
                }
                None => *sums = Some(other),
            }
        }
        self
    }
}

#[derive(Clone, Copy)]
enum SuffixRow {
    One,
    ZeroOne(Suffixes),
    General(Suffixes),
}

impl SuffixRow {
    fn new(suffix: Suffixes) -> Self {
        if matches!(suffix, Suffixes::One) {
            Self::One
        } else if suffix.is_01_valued() {
            Self::ZeroOne(suffix)
        } else {
            Self::General(suffix)
        }
    }

    #[inline]
    fn accumulate<F: JoltField, A: Accumulator<Element = F>>(
        self,
        row: &mut [A],
        chunk: usize,
        bits: LookupBits,
        value: F,
    ) {
        match self {
            Self::One => row[chunk].add(value),
            Self::ZeroOne(suffix) => {
                if suffix.suffix_mle(bits) == 1 {
                    row[chunk].add(value);
                }
            }
            Self::General(suffix) => {
                let scalar = suffix.suffix_mle(bits);
                if scalar != 0 {
                    row[chunk].fmadd_u64(value, scalar);
                }
            }
        }
    }
}

enum SuffixScan<F: JoltField> {
    One(Vec<F::SumAccumulator>),
    ZeroOne(Suffixes, Vec<F::SumAccumulator>),
    General(Suffixes, Vec<F::Accumulator>),
}

/// Accumulators in `table.suffixes()` order. Boolean rows use separate compact
/// allocations only when the field supplies a smaller sum accumulator.
enum TableScan<F: JoltField> {
    Flat {
        suffixes: Vec<SuffixRow>,
        accumulators: Vec<F::Accumulator>,
    },
    Compact(Vec<SuffixScan<F>>),
}

impl<F: JoltField> TableScan<F> {
    /// Separate row allocations must reduce the accumulator footprint;
    /// otherwise keep one contiguous allocation for the table.
    const fn uses_compact_rows() -> bool {
        std::mem::size_of::<F::SumAccumulator>() < std::mem::size_of::<F::Accumulator>()
    }

    fn new(table: LookupTableKind<RISCV_XLEN>) -> Self {
        if Self::uses_compact_rows() {
            Self::Compact(
                table
                    .suffixes()
                    .iter()
                    .map(|&suffix| match SuffixRow::new(suffix) {
                        SuffixRow::One => {
                            SuffixScan::One(vec![F::SumAccumulator::default(); CHUNK_SIZE])
                        }
                        SuffixRow::ZeroOne(suffix) => SuffixScan::ZeroOne(
                            suffix,
                            vec![F::SumAccumulator::default(); CHUNK_SIZE],
                        ),
                        SuffixRow::General(suffix) => {
                            SuffixScan::General(suffix, vec![F::Accumulator::default(); CHUNK_SIZE])
                        }
                    })
                    .collect(),
            )
        } else {
            let suffixes: Vec<_> = table
                .suffixes()
                .iter()
                .copied()
                .map(SuffixRow::new)
                .collect();
            let accumulators = vec![F::Accumulator::default(); suffixes.len() * CHUNK_SIZE];
            Self::Flat {
                suffixes,
                accumulators,
            }
        }
    }

    #[inline]
    fn accumulate(&mut self, chunk: usize, bits: LookupBits, value: F) {
        match self {
            Self::Flat {
                suffixes,
                accumulators,
            } => {
                for (&suffix, row) in suffixes
                    .iter()
                    .zip(accumulators.chunks_exact_mut(CHUNK_SIZE))
                {
                    suffix.accumulate(row, chunk, bits, value);
                }
            }
            Self::Compact(suffixes) => {
                for suffix in suffixes {
                    match suffix {
                        SuffixScan::One(row) => SuffixRow::One.accumulate(row, chunk, bits, value),
                        SuffixScan::ZeroOne(suffix, row) => {
                            SuffixRow::ZeroOne(*suffix).accumulate(row, chunk, bits, value);
                        }
                        SuffixScan::General(suffix, row) => {
                            SuffixRow::General(*suffix).accumulate(row, chunk, bits, value);
                        }
                    }
                }
            }
        }
    }

    fn reduce(self) -> Vec<F> {
        match self {
            Self::Flat { accumulators, .. } => accumulators
                .into_iter()
                .map(|value| value.reduce())
                .collect(),
            Self::Compact(suffixes) => {
                let mut sums = Vec::with_capacity(suffixes.len() * CHUNK_SIZE);
                for suffix in suffixes {
                    match suffix {
                        SuffixScan::One(row) | SuffixScan::ZeroOne(_, row) => {
                            sums.extend(row.into_iter().map(|value| value.reduce()));
                        }
                        SuffixScan::General(_, row) => {
                            sums.extend(row.into_iter().map(|value| value.reduce()));
                        }
                    }
                }
                sums
            }
        }
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
                let active = usize::from(block.starts[SCAN_CLASSES]);
                if active == rows.len() {
                    for (row, u) in rows.iter().zip(u_evals.iter_mut()) {
                        *u *= v_prev[((row.lookup_index() >> shift) as usize) & (CHUNK_SIZE - 1)];
                    }
                } else {
                    for &offset in &block.offsets[..active] {
                        let offset = usize::from(offset);
                        u_evals[offset] *= v_prev
                            [((rows[offset].lookup_index() >> shift) as usize) & (CHUNK_SIZE - 1)];
                    }
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
                .map(|scan| scan.map(TableScan::reduce))
                .collect(),
        }
    }

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
    rows: Arc<ConstantSuffix<InstructionCycleRow>>,
    blocks: Vec<ScanBlock>,
    /// The final weight aggregates the constant row suffix. After coalescing,
    /// only offsets retained in `blocks` are live; merged-away weights are stale.
    u_evals: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    prefix_checkpoints: Vec<PrefixEval<F>>,
    /// `ALL_PREFIXES` indices referenced by tables some cycle selects.
    prefix_indices: Vec<usize>,
    prefix_tables: Vec<Polynomial<F>>,
    /// Per present table: enum value + suffix `Q` polynomials in
    /// `table.suffixes()` order.
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_keyed_polys))]
    suffix_tables: Vec<(LookupTableKind<RISCV_XLEN>, Vec<Polynomial<F>>)>,
    raf_left: RafDecomposition<F>,
    raf_right: RafDecomposition<F>,
    raf_identity: RafDecomposition<F>,
    raf_upper_all_ones: RafDecomposition<F>,
    v_tables: Vec<Vec<F>>,
    phase_challenges: Vec<F>,
    cycle_challenges: Vec<F>,
    cycle: Option<CycleState<F>>,
    /// Packed per-cycle output-claim facts (bits 0..=6: `table_index + 1`,
    /// 0 for none; bit 7: the RAF flag), snapped at the address/cycle
    /// handoff and shared with the combined value's index source — the
    /// final flag walk needs only this byte per cycle.
    claim_columns: Arc<ConstantSuffix<u8>>,
    progress: RoundProgress,
}

impl<F: JoltField> OptimizedInstructionReadRafKernel<F> {
    pub(crate) fn new(
        dimensions: InstructionReadRafDimensions,
        r_reduction: &[F],
        rows: Arc<ConstantSuffix<InstructionCycleRow>>,
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

        let mut u_evals = PrefixCycleWeights::new(r_reduction, rows.prefix_len()).to_dense();
        let scan_rows = &rows.representatives()[..u_evals.len()];
        let num_tables = LookupTableKind::<RISCV_XLEN>::COUNT;
        let present_tables = map_reduce_chunks(
            scan_rows.len(),
            scan_chunk_size(scan_rows.len()),
            |range| {
                let mut present = 0u64;
                for row in &scan_rows[range] {
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

        let blocks = if ScanBlock::should_coalesce(scan_rows) {
            #[cfg(feature = "parallel")]
            let chunks = u_evals.par_chunks_mut(SCAN_BLOCK);
            #[cfg(not(feature = "parallel"))]
            let chunks = u_evals.chunks_mut(SCAN_BLOCK);
            let coalesce = |cache: &mut RepeatCache<NonZeroU16>,
                            (block, weights): (usize, &mut [F])| {
                ScanBlock::coalesced(
                    &scan_rows[block * SCAN_BLOCK..block * SCAN_BLOCK + weights.len()],
                    weights,
                    cache,
                )
            };
            #[cfg(feature = "parallel")]
            let blocks = chunks
                .enumerate()
                .map_init(|| RepeatCache::new(SCAN_BLOCK), coalesce)
                .collect();
            #[cfg(not(feature = "parallel"))]
            let blocks = {
                let mut cache = RepeatCache::new(SCAN_BLOCK);
                chunks
                    .enumerate()
                    .map(|item| coalesce(&mut cache, item))
                    .collect()
            };
            blocks
        } else {
            map_indices(scan_rows.len().div_ceil(SCAN_BLOCK), |block| {
                ScanBlock::new(
                    &scan_rows[block * SCAN_BLOCK..((block + 1) * SCAN_BLOCK).min(scan_rows.len())],
                )
            })
        };
        let mut kernel = Self {
            dimensions,
            gamma,
            r_reduction: r_reduction.to_vec(),
            rows,
            blocks,
            u_evals,
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
            claim_columns: Arc::new(Vec::new().into()),
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
        let PhaseSums { raf, suffixes } = scan.run(
            &self.blocks,
            &self.rows.representatives()[..self.u_evals.len()],
            &mut self.u_evals,
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
            .zip(suffixes)
            .filter_map(|(table, sums)| {
                let polynomials = sums?
                    .chunks_exact(CHUNK_SIZE)
                    .map(|coefficients| Polynomial::new(coefficients.to_vec()))
                    .collect();
                Some((table, polynomials))
            })
            .collect();

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
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let cycle = self
            .cycle
            .as_ref()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
        let factors = 1 + self.dimensions.num_virtual_ra_polys();
        let tail_products = TailFactorProducts::new(cycle.ra.tail_values());

        struct Scratch<F: JoltField> {
            /// Cross-row lanes for `q(1), …, q(F−1), q(∞)` — `e_in` rides in
            /// the `Val` factor, so these stay unreduced across the block.
            lanes: Vec<F::Accumulator>,
            pairs: Vec<(F, F)>,
            evals: Vec<F>,
            steps: Vec<F>,
            grid: Vec<F>,
            low_degree: LowDegreeProductSum<F>,
        }

        let prefix_pairs = cycle
            .combined_val
            .prefix_len()
            .max(cycle.ra.prefix_len())
            .div_ceil(2);
        let tail_mass = cycle.gruen.suffix_weight(prefix_pairs)
            * cycle
                .combined_val
                .tail_values()
                .first()
                .copied()
                .unwrap_or(F::zero())
            * cycle.ra.tail_values().iter().copied().product::<F>();
        let groups = match (&cycle.ra, &cycle.combined_val) {
            (
                LazyFoldedRa::Lazy {
                    source, width: 1, ..
                },
                LazyFoldedRa::Lazy {
                    source: claims,
                    width: 1,
                    ..
                },
            ) => CyclePairGroups::new(source, claims, prefix_pairs, cycle.gruen.e_in_current_len()),
            _ => None,
        };
        let make_scratch = || Scratch {
            lanes: vec![F::Accumulator::default(); factors],
            pairs: vec![(F::zero(), F::zero()); factors],
            evals: vec![F::zero(); factors],
            steps: vec![F::zero(); factors],
            grid: vec![F::zero(); product_grid_scratch_len(factors)],
            low_degree: LowDegreeProductSum::default(),
        };
        let eval_row = |scratch: &mut Scratch<F>, row: usize, e_in: F| {
            let (val, ra) = scratch.pairs.split_at_mut(1);
            cycle.combined_val.lo_hi_all(row, val);
            cycle.ra.lo_hi_all(row, ra);
            let (lo, hi) = val[0];
            val[0] = (e_in * lo, e_in * hi);
            for ((&(lo, hi), eval), step) in scratch
                .pairs
                .iter()
                .zip(scratch.evals.iter_mut())
                .zip(scratch.steps.iter_mut())
            {
                *eval = hi;
                *step = hi - lo;
            }
            if !scratch.low_degree.try_accumulate(
                &scratch.evals,
                &scratch.steps,
                tail_products.as_ref(),
            ) {
                accumulate_product_grid(
                    &scratch.evals,
                    &scratch.steps,
                    &mut scratch.lanes,
                    &mut scratch.grid,
                );
            }
        };
        let finish_block = |_x_out: usize, e_out: F, mut scratch: Scratch<F>| {
            scratch
                .low_degree
                .merge_into(&mut scratch.lanes, &mut scratch.grid);
            let mut out = vec![F::Accumulator::default(); factors];
            for (out, lane) in out.iter_mut().zip(scratch.lanes) {
                out.fmadd(e_out, lane.reduce());
            }
            out
        };
        let merge = |mut a: Vec<F::Accumulator>, b: Vec<F::Accumulator>| {
            for (a, b) in a.iter_mut().zip(b) {
                a.merge(b);
            }
            a
        };
        let block_lanes = if let Some(groups) = groups {
            groups.fold(
                &cycle.gruen,
                prefix_pairs,
                make_scratch,
                eval_row,
                finish_block,
                merge,
            )
        } else {
            cycle.gruen.par_fold_out_in_prefix(
                prefix_pairs,
                make_scratch,
                |scratch, row, _, weight| eval_row(scratch, row, weight),
                finish_block,
                merge,
            )
        };
        let mut q_evals: Vec<F> = block_lanes.into_iter().map(|lane| lane.reduce()).collect();
        for value in &mut q_evals[..factors - 1] {
            *value += tail_mass;
        }
        cycle
            .gruen
            .checked_toom(&q_evals, previous_claim, round, || {
                tail_mass
                    + cycle.gruen.par_fold_out_in_prefix(
                        prefix_pairs,
                        || {
                            (
                                vec![(F::zero(), F::zero()); factors],
                                F::Accumulator::default(),
                            )
                        },
                        |(pairs, sum), row, _, weight| {
                            let (val, ra) = pairs.split_at_mut(1);
                            cycle.combined_val.lo_hi_all(row, val);
                            cycle.ra.lo_hi_all(row, ra);
                            let value = pairs
                                .iter()
                                .fold(F::one(), |product, pair| product * pair.0);
                            sum.fmadd(weight, value);
                        },
                        |_, weight, (_, sum)| weight * sum.reduce(),
                        |a, b| a + b,
                    )
            })
    }

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
        let rows = self.rows.representatives();
        const {
            assert!(
                LookupTableKind::<RISCV_XLEN>::COUNT < 0x7f,
                "table indices must fit the packed claim byte"
            );
        }
        let claim_columns = Arc::new(ConstantSuffix::new(
            map_indices(rows.len(), |j| {
                let row = &rows[j];
                let table = row.table_index().map_or(0, |index| index as u8 + 1);
                table | (u8::from(row.raf_flag()) << 7)
            }),
            self.rows.len(),
        ));
        self.claim_columns = Arc::clone(&claim_columns);
        let combined_table: Vec<F> = (0..=u8::MAX)
            .map(|packed| {
                let table_value = usize::from(packed & 0x7f)
                    .checked_sub(1)
                    .and_then(|table| table_values.get(table))
                    .map_or_else(F::zero, |value| *value);
                let raf_value = if packed & 0x80 == 0 {
                    raf_interleaved
                } else {
                    raf_identity
                };
                table_value + raf_value
            })
            .collect();

        // `ra_i = Π_{phases p of i} v_p[chunk_p]`: phase pairs tensor into
        // 2^16-entry columns over 16-bit lookup-index chunks, so 16-bit
        // virtual chunks gather without multiplying; wider ones factor into
        // several columns.
        let phases_per_ra = self.phases() / self.dimensions.num_virtual_ra_polys();
        let phases_per_column = phases_per_ra.min(2);
        let v_tables = std::mem::take(&mut self.v_tables);
        let column_tables = map_indices(self.phases() / phases_per_column, |column| {
            let mut table = vec![F::one()];
            for v in &v_tables[column * phases_per_column..(column + 1) * phases_per_column] {
                table = table
                    .iter()
                    .flat_map(|high| v.iter().map(move |low| *high * *low))
                    .collect();
            }
            table
        });
        let rows = std::mem::replace(&mut self.rows, Arc::new(Vec::new().into()));
        let chunks =
            LookupIndexChunks::new(rows, column_tables.len(), phases_per_column * CHUNK_LEN);

        self.cycle = Some(CycleState {
            gruen: GruenSplitEqPolynomial::new(&self.r_reduction, BindingOrder::LowToHigh),
            combined_val: LazyFoldedRa::factored(
                vec![combined_table],
                1,
                LAZY_MAX_WIDTH,
                ClaimBytes(claim_columns),
            ),
            ra: LazyFoldedRa::factored(
                column_tables,
                phases_per_ra / phases_per_column,
                LAZY_MAX_WIDTH,
                chunks,
            ),
        });

        // The address-phase state is dead past this point; the rows live on
        // in the lazy RA source until the third cycle bind.
        self.u_evals = Vec::new();
        self.prefix_tables = Vec::new();
        self.suffix_tables = Vec::new();
        self.blocks = Vec::new();
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
            let cycle = self
                .cycle
                .as_mut()
                .ok_or(SumcheckError::MissingEvaluationSource { kind: "opening" })?;
            cycle.gruen.bind(challenge);
            cycle.combined_val.bind(challenge);
            cycle.ra.bind(challenge);
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
        let claim_columns = &self.claim_columns;
        let (mut lookup_table_flags, mut instruction_raf_flag) = eq_cycle.par_fold_out_in_prefix(
            claim_columns.prefix_len(),
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

        let tail = claim_columns[claim_columns.prefix_len()];
        let tail_mass = EqPolynomial::suffix_sum(&r_cycle, claim_columns.prefix_len());
        if tail & 0x7f != 0 {
            lookup_table_flags[usize::from(tail & 0x7f) - 1] += tail_mass;
        }
        if tail & 0x80 != 0 {
            instruction_raf_flag += tail_mass;
        }

        Ok(InstructionReadRafOutputClaims {
            lookup_table_flags,
            instruction_ra: cycle.ra.final_values(),
            instruction_raf_flag,
        })
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use crate::optimized::parity::ExceptionalEq;
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use jolt_claims::protocols::jolt::geometry::instruction::{
        InstructionReadRafDimensions, CANONICAL_INSTRUCTION_ADDRESS,
    };
    use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafInputClaims;
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Fr, JoltField, Ring};
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

    use super::{
        InstructionCycleRow, LookupIndexChunks, OptimizedInstructionReadRafKernel, RepeatCache,
        ScanBlock,
    };

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

    fn challenge<F: JoltField>(round: usize) -> F {
        F::from_u64(
            0x9E37_79B9_7F4A_7C15 ^ (round as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ 0x11,
        )
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fixture_rows(log_t: usize, seed: u64) -> Vec<InstructionReadRafWitness> {
        let tables = [
            LookupTableKind::<RISCV_XLEN>::And(Default::default()).index(),
            LookupTableKind::<RISCV_XLEN>::Andn(Default::default()).index(),
            LookupTableKind::<RISCV_XLEN>::Or(Default::default()).index(),
            LookupTableKind::<RISCV_XLEN>::Xor(Default::default()).index(),
            LookupTableKind::<RISCV_XLEN>::VirtualXORROTW7(Default::default()).index(),
            LookupTableKind::<RISCV_XLEN>::UnsignedLessThan(Default::default()).index(),
            LookupTableKind::<RISCV_XLEN>::Equal(Default::default()).index(),
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

    #[test]
    fn packed_lookup_binding_matches_multilinear_weights() {
        fn check<F: JoltField>(columns: usize, bits: usize) {
            use super::super::{constant_suffix::ConstantSuffix, lazy_ra::LazyFoldedRa};
            use rand_chacha::ChaCha20Rng;
            use rand_core::{RngCore, SeedableRng};

            let mut rng = ChaCha20Rng::seed_from_u64(314_159);
            let cycles = 1 << 16;
            let prefix = cycles - 7;
            let indices: Vec<u128> = (0..cycles)
                .map(|i| {
                    if i >= prefix {
                        0x1234_5678_9abc_def0_fedc_ba98_7654_3210
                    } else {
                        (u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64())
                    }
                })
                .collect();
            let rows = indices
                .iter()
                .map(|&index| {
                    InstructionCycleRow::new(
                        index,
                        None,
                        false,
                        0,
                        None,
                        #[cfg(feature = "akita")]
                        FusedInc::default(),
                    )
                })
                .collect();
            let source =
                LookupIndexChunks::new(Arc::new(ConstantSuffix::new(rows, cycles)), columns, bits);
            let tables: Vec<Vec<F>> = (0..columns)
                .map(|_| (0..16).map(|_| F::random(&mut rng)).collect())
                .collect();
            let challenges = [3, 5, 13, 29].map(F::from_u64);
            let mut bound = LazyFoldedRa::new(tables.clone(), source);
            for challenge in challenges {
                bound.bind(challenge);
            }
            let weights: Vec<F> = (0..16)
                .map(|branch| {
                    challenges
                        .iter()
                        .enumerate()
                        .fold(F::one(), |weight, (bit, r)| {
                            weight
                                * if (branch >> bit) & 1 == 1 {
                                    *r
                                } else {
                                    F::one() - *r
                                }
                        })
                })
                .collect();
            for (column, table) in tables.iter().enumerate() {
                for row in 0..cycles / 16 {
                    let expected: F = weights
                        .iter()
                        .enumerate()
                        .map(|(branch, weight)| {
                            let digit = ((indices[row * 16 + branch]
                                >> (bits * (columns - 1 - column)))
                                & ((1u128 << bits) - 1))
                                as usize;
                            *weight * table[digit]
                        })
                        .sum();
                    assert_eq!(bound.value(column, row), expected);
                }
            }
        }
        check::<Fr>(16, 2);
        check::<Fr>(16, 4);
        check::<Fr>(20, 4);
        #[cfg(feature = "akita")]
        check::<Prime128OffsetA7F7>(32, 4);
    }

    /// The sumcheck input claim from first principles:
    /// `Σ_j eq(r_reduction, j) · (Val_j(k_j) + γ·RafVal_j(k_j))` with the
    /// point-mass `ra` collapsed at each cycle's lookup index. Pins both
    /// kernels to the protocol, not merely to each other (each kernel's own
    /// `s(0) + s(1) = claim` self-check would reject a drifted round 0).
    fn input_claim<F: JoltField>(
        rows: &[InstructionReadRafWitness],
        r_reduction: &[F],
        gamma: F,
    ) -> F {
        let tables: Vec<LookupTableKind<RISCV_XLEN>> = LookupTableKind::iter().collect();
        let gamma_sqr = gamma * gamma;
        let address_bits = 2 * RISCV_XLEN;
        eq_table(r_reduction)
            .iter()
            .zip(rows)
            .map(|(&u, row)| {
                let k = row.lookup_index.0;
                let value = row.table_index.0.map_or_else(
                    || F::from_u64(0),
                    |index| F::from_u64(tables[index].materialize_entry(k)),
                );
                let raf = if !row.raf_flag.0 {
                    let (left, right) = LookupBits::new(k, address_bits).uninterleave();
                    gamma * F::from_u64(u64::from(left)) + gamma_sqr * F::from_u64(u64::from(right))
                } else {
                    let mut raf = gamma_sqr
                        * (F::from_u64(k as u64) + F::from_u64((k >> 64) as u64).mul_pow_2(64));
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

    fn assert_parity(log_t: usize, num_virtual_ra_polys: usize, seed: u64) {
        assert_parity_case(log_t, num_virtual_ra_polys, seed, None);
    }

    fn assert_parity_case(
        log_t: usize,
        num_virtual_ra_polys: usize,
        seed: u64,
        exceptional: Option<ExceptionalEq>,
    ) {
        assert_parity_prefix(log_t, num_virtual_ra_polys, seed, exceptional, None);
    }

    #[test]
    fn constant_suffix_rounds_match_reference() {
        for prefix in [0, 1, 15, 16, 17, 33, 63] {
            for exceptional in [None].into_iter().chain(ExceptionalEq::ALL.map(Some)) {
                for zero_tail in [true, false] {
                    assert_parity_prefix(6, 8, 123, exceptional, Some((prefix, zero_tail)));
                }
            }
        }
    }

    fn assert_parity_prefix(
        log_t: usize,
        num_virtual_ra_polys: usize,
        seed: u64,
        exceptional: Option<ExceptionalEq>,
        prefix: Option<(usize, bool)>,
    ) {
        assert_parity_prefix_field::<Fr>(log_t, num_virtual_ra_polys, seed, exceptional, prefix);
        #[cfg(feature = "akita")]
        assert_parity_prefix_field::<Prime128OffsetA7F7>(
            log_t,
            num_virtual_ra_polys,
            seed,
            exceptional,
            prefix,
        );
    }

    fn assert_parity_prefix_field<F: JoltField>(
        log_t: usize,
        num_virtual_ra_polys: usize,
        seed: u64,
        exceptional: Option<ExceptionalEq>,
        prefix: Option<(usize, bool)>,
    ) {
        let dimensions = InstructionReadRafDimensions::new(
            log_t,
            2 * RISCV_XLEN,
            NonZeroUsize::new(num_virtual_ra_polys).unwrap(),
        );
        let mut rows = fixture_rows(log_t, seed);
        if let Some((prefix, zero_tail)) = prefix {
            let mut tail = rows[rows.len() - 1];
            if zero_tail {
                tail.lookup_index = LookupIndex(0);
                tail.table_index = TableIndex(None);
                tail.raf_flag = InstructionRafFlag(false);
            }
            for row in &mut rows[prefix..] {
                *row = tail;
            }
        }
        let r_reduction: Vec<F> = exceptional.map_or_else(
            || {
                (0..log_t)
                    .map(|i| F::from_u64(1000 + 37 * i as u64))
                    .collect()
            },
            |case| case.point(log_t, challenge(dimensions.instruction_address_bits())),
        );
        assert_parity_rows(dimensions, &r_reduction, rows);
    }

    fn assert_parity_rows<F: JoltField>(
        dimensions: InstructionReadRafDimensions,
        r_reduction: &[F],
        rows: Vec<InstructionReadRafWitness>,
    ) {
        let gamma = F::from_u64(0xACE1_57EF);

        let mut reference =
            InstructionReadRafKernel::new(dimensions, r_reduction, rows.clone(), gamma).unwrap();
        let mut optimized = OptimizedInstructionReadRafKernel::new(
            dimensions,
            r_reduction,
            Arc::new(pack(&rows).into()),
            gamma,
        )
        .unwrap();

        let rounds = reference.num_rounds();
        assert_eq!(rounds, optimized.num_rounds());
        let mut claim = input_claim(&rows, r_reduction, gamma);
        for round in 0..rounds {
            let bind = round.checked_sub(1).map(challenge);
            let reference_poly = reference.prove_round(bind, round, claim).unwrap();
            let optimized_poly = optimized.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                reference_poly.coefficients(),
                optimized_poly.coefficients(),
                "round {round} polynomial mismatch"
            );
            claim = reference_poly.evaluate(challenge(round));
        }
        reference.finish_rounds(challenge(rounds - 1)).unwrap();
        optimized.finish_rounds(challenge(rounds - 1)).unwrap();

        let inputs = InstructionReadRafInputClaims {
            lookup_output: F::from_u64(0),
            left_lookup_operand: F::from_u64(0),
            right_lookup_operand: F::from_u64(0),
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

    fn check_repeated_lookup_rows<F: JoltField>(log_t: usize, prefix: usize, constant: Option<F>) {
        let dimensions =
            InstructionReadRafDimensions::new(log_t, 2 * RISCV_XLEN, NonZeroUsize::new(8).unwrap());
        let mut pattern = fixture_rows(5, 97531);
        pattern[1].lookup_index = pattern[0].lookup_index;
        pattern[2].lookup_index = pattern[0].lookup_index;
        pattern[2].table_index = pattern[0].table_index;
        pattern[2].raf_flag = InstructionRafFlag(!pattern[0].raf_flag.0);
        let rows: Vec<_> = (0..1 << log_t)
            .map(|i| pattern[if i < prefix { i % pattern.len() } else { 0 }])
            .collect();
        assert!(ScanBlock::should_coalesce(&pack(&rows)));
        let point: Vec<F> = (0..log_t)
            .map(|i| constant.unwrap_or_else(|| challenge(i)))
            .collect();
        assert_parity_rows(dimensions, &point, rows);
    }

    #[test]
    fn coalescing_preserves_weight_for_colliding_keys_and_classes() {
        fn check<F: JoltField>() {
            // Four slots for up to ten distinct (key, class) pairs force collisions.
            let mut cache = RepeatCache::new(1);
            for length in [64, 17] {
                let rows: Vec<_> = (0..length)
                    .map(|i| {
                        let high = [0u64, 0, 1, 2, 0, 3, 3, 4][i % 8];
                        let key = (u128::from(high) << 64) | u128::from(high.rotate_left(32));
                        InstructionCycleRow::new(
                            key,
                            None,
                            i >= length / 2,
                            0,
                            None,
                            #[cfg(feature = "akita")]
                            FusedInc::default(),
                        )
                    })
                    .collect();
                let original: Vec<F> = (0..length)
                    .map(|i| F::from_u64((i * i + 1) as u64))
                    .collect();
                let mut weights = original.clone();
                let block = ScanBlock::coalesced(&rows, &mut weights, &mut cache);
                for (class, range) in block.starts.windows(2).enumerate() {
                    for high in 0..5u64 {
                        let key = (u128::from(high) << 64) | u128::from(high.rotate_left(32));
                        let expected: F = rows
                            .iter()
                            .zip(&original)
                            .filter(|(row, _)| {
                                ScanBlock::class(row) == class && row.lookup_index() == key
                            })
                            .map(|(_, weight)| *weight)
                            .sum();
                        let actual: F = block.offsets[usize::from(range[0])..usize::from(range[1])]
                            .iter()
                            .map(|offset| usize::from(*offset))
                            .filter(|offset| rows[*offset].lookup_index() == key)
                            .map(|offset| weights[offset])
                            .sum();
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
        check::<Fr>();
        #[cfg(feature = "akita")]
        check::<Prime128OffsetA7F7>();
    }

    #[test]
    fn repeated_lookup_rows_match_reference() {
        fn check<F: JoltField>() {
            for prefix in [1027, 1 << 11] {
                for constant in [None, Some(F::zero()), Some(F::one())] {
                    check_repeated_lookup_rows(11, prefix, constant);
                }
            }
        }
        check::<Fr>();
        #[cfg(feature = "akita")]
        check::<Prime128OffsetA7F7>();
    }

    #[test]
    fn repeated_lookup_rows_match_reference_across_scan_tasks() {
        for prefix in [6003, 1 << 14] {
            check_repeated_lookup_rows::<Fr>(14, prefix, None);
            #[cfg(feature = "akita")]
            check_repeated_lookup_rows::<Prime128OffsetA7F7>(14, prefix, None);
        }
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

    /// 8-, 32- and 64-bit virtual chunks across the third cycle bind's dense
    /// switch: single 8-bit columns, and products of two and four 16-bit
    /// columns.
    #[test]
    fn parity_virtual_chunk_widths_across_dense_switch() {
        for (num_virtual_ra_polys, seed) in [(16, 11), (4, 22), (2, 33)] {
            assert_parity(6, num_virtual_ra_polys, seed);
        }
    }

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
            Arc::new(pack(&rows).into()),
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
    #[test]
    fn parity_exceptional_eq_in_lazy_and_dense_cycle_tables() {
        for virtuals in [4usize, 8] {
            for case in ExceptionalEq::ALL {
                assert_parity_case(6, virtuals, 257, Some(case));
            }
        }
    }
}
