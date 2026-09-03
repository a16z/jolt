//! Optimized bytecode read+RAF kernels: the stage-6a address phase and the
//! stage-6b cycle phase, byte-parity twins of the reference kernels in
//! [`crate::reference::bytecode_read_raf`].
//!
//! Ported legacy techniques (`jolt-prover-legacy/src/zkvm/bytecode/read_raf_checking.rs`):
//!
//! - **Split-eq two-table pushforwards** (address phase): the per-stage
//!   `F_s(k) = Σ_{j: pc(j)=k} eq(r_cycle_s, j)` tables are accumulated as
//!   `Σ_{j_hi} E_hi_s[j_hi] · (Σ_{j_lo: pc=k} E_lo_s[j_lo])` — the inner sums
//!   are additions only and the base stages share one trace walk, so the eq
//!   tables cost `O(√T)` each instead of `O(T)` and the `O(T)` walk pays one
//!   PC lookup per cycle for all stages. In the packed protocol, four more
//!   pushforwards use the same walk with the fused-increment row weight
//!   (legacy `BytecodeReadRafAddressSumcheckProver::initialize`).
//! - **Sparse one-hot RA** (cycle phase): the committed `BytecodeRa(i)`
//!   factors are materialized as `ra_i(j) = eq(chunk_i)[chunk_i(pc_j)]` from
//!   per-cycle mapped PCs (cold cycles are zero), never as `K x T` one-hot
//!   grids — the legacy `RaPolynomial` trick, `O(T)` per chunk
//!   (legacy `BytecodeReadRafCycleSumcheckProver::initialize`, the 1.13s
//!   member of legacy stage 6b at `2^23`).
//! - **Linear-leaf fusion** (cycle phase): the five `StageCycleEq · Val`
//!   pairs, both RAF terms (`int(r_address)`-scaled stage-1/3 eqs), and the
//!   entry selector all enter the summand linearly, so they collapse into ONE
//!   combined coefficient table `C(j)`; the summand becomes
//!   `Π_i ra_i(j) · C(j)` — `d+2` bound tables instead of the reference's
//!   `13`, with field-identical round messages (multilinear extension is
//!   linear in the table argument). The RAF weights fold as the legacy
//!   `raf_int_weight` does: stage 1 carries `γ⁵·int_r`, stage 3 `γ⁶·int_r`.
//! - **Lazy packed fused increment** (cycle phase, Akita): the extra stages
//!   remain `Π_i ra_i · FusedInc · C_fused`, preserving the relation's extra
//!   degree. `FusedInc` binds from shared sign-magnitude rows for three
//!   rounds and materializes at `T / 16` instead of allocating a dense field
//!   column at prepare.
//! - **Eval-at-1 recovery** and **rayon walks**, both phases (see the module
//!   docs on [`crate::optimized`]).
//!
//! Both phases read the stage-5 [`InstructionCycleRow`] carry their Booleanity
//! and RA-virtualization peers already retain, so neither protocol pays for a
//! second per-cycle PC cache. Both phases read the same total `BytecodePc`
//! column: the pushforward slot and the committed one-hot hot index are the
//! same value on every row.
//!
//! Two prover-side offloads layer on top:
//!
//! - [`spawn_bytecode_stage_pushforwards`] builds the stage-1..4 base
//!   pushforward tables on a capped background pool during stage 5 (their
//!   points exist by the end of stage 4); the 6a prepare reclaims them and
//!   walks only the stage-5 point (plus any fused stages) inline — the same
//!   per-stage lanes either way, so the values are identical.
//! - The cycle phase's Metal member drives the rounds through
//!   [`prepare_bytecode_read_raf_cycle`]'s driver factory; the device reads
//!   the packed 8-byte [`PcRow`] lanes (the trace record's co-produced scan)
//!   and publishes the live combined table through a recovery cell on any
//!   decline, so the optimized kernel resumes on the CPU mid-sumcheck.

use std::sync::{Arc, Mutex};

use jolt_claims::protocols::jolt::geometry::bytecode::{
    self, read_raf_stage_values, BytecodeReadRafStageValueInputs,
};
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use jolt_claims::protocols::jolt::geometry::dimensions::{
    committed_address_chunks, REGISTER_ADDRESS_BITS,
};
use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_claims::OutputClaims;
use jolt_field::JoltField;
#[cfg(feature = "akita")]
use jolt_poly::BindingOrder;
use jolt_poly::{IdentityPolynomial, MultilinearEvaluation, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage6a::bytecode_read_raf::{
    BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_witness::witnesses::RaChunkSelector;
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::instruction_read_raf::{shared_instruction_rows, InstructionCycleRow, InstructionRows};
use super::lazy_ra::{ChunkIndexSource, LazyFoldedRa, LazyRaDevice};
use super::lifetime_trace::LifetimeTag;
use super::support::{
    bind_all, eq_table, gamma_powers, pair, par_sum_pair_groups, par_sum_pair_groups_reusing,
    round_poly_from_skipped_evals, scaled_eq_table, RoundProgress,
};
use crate::mmap_vec::MmapVec;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Sentinel for a cold cycle (no bytecode mapping) in [`PcRow::mapped_pc`].
/// With the total `BytecodePc` column no current producer emits it; the
/// device shader still treats it as a zero row.
const COLD: u32 = u32::MAX;

/// One cycle's packed bytecode PC lanes for the Metal cycle driver — 2×u32
/// per row, the shader ABI. The device gathers committed one-hot chunk
/// indices from `mapped_pc`; with the total `BytecodePc` column both lanes
/// carry the same PC on every mapped row (`push_pc` survives as the record
/// walk's pack slot, zeroed on no-ops).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PcRow {
    pub(crate) push_pc: u32,
    pub(crate) mapped_pc: u32,
}

impl PcRow {
    /// The row from the trace record's lanes. On any trace the record builds
    /// from, `Pc` extraction succeeded for every row, so `mapped_pc` is the
    /// total per-cycle PC (with the u32-range guards applied).
    pub(crate) fn from_lanes<F: JoltField>(pc: u64, is_noop: bool) -> Result<Self, KernelError<F>> {
        let mapped = match pc {
            pc if pc as u32 as u64 == pc && pc as u32 != COLD => pc as u32,
            _ => {
                return Err(KernelError::InvariantViolation {
                    reason: "bytecode PC exceeds the packed u32 range",
                })
            }
        };
        let push_pc = if is_noop { 0 } else { pc };
        if push_pc as u32 as u64 != push_pc {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode PC exceeds the packed u32 range",
            });
        }
        Ok(Self {
            push_pc: push_pc as u32,
            mapped_pc: mapped,
        })
    }
}

/// The session key of the shared per-cycle PC scan.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct PcRowsKey(
    #[cfg_attr(feature = "allocative", allocative(skip))]
    #[cfg_attr(
        all(not(all(feature = "metal", target_os = "macos")), not(test)),
        expect(dead_code, reason = "read by the metal bytecode slot and tests")
    )]
    Arc<PcRows>,
);

/// The scan behind [`PcRowsKey`] — the packed rows plus the lifetime tag that
/// logs the last-`Arc`-drop site under `JOLT_LIFETIME_TRACE=1`.
pub(crate) struct PcRows {
    rows: MmapVec<PcRow>,
    _lifetime: LifetimeTag,
}

impl PcRows {
    #[cfg_attr(
        not(all(feature = "metal", target_os = "macos")),
        expect(dead_code, reason = "used by the metal bytecode slot")
    )]
    pub(crate) fn as_slice(&self) -> &[PcRow] {
        &self.rows
    }

    fn new(rows: MmapVec<PcRow>) -> Self {
        let bytes = rows.len() * size_of::<PcRow>();
        Self {
            rows,
            _lifetime: LifetimeTag::new("PcRows", bytes),
        }
    }
}

impl std::ops::Deref for PcRows {
    type Target = [PcRow];

    fn deref(&self) -> &Self::Target {
        &self.rows
    }
}

/// Park a lane-packed PC scan (the trace record's walk co-produces it), so
/// the Metal cycle driver is served without a second trace pass.
pub(crate) fn park_pc_rows(session: &mut ProofSession, rows: MmapVec<PcRow>) {
    session.park(PcRowsKey(Arc::new(PcRows::new(rows))));
}

/// The parked PC scan, if a record walk co-produced one in this session.
#[cfg(test)]
pub(crate) fn parked_pc_rows(session: &ProofSession) -> Option<Arc<PcRows>> {
    session.state::<PcRowsKey>().map(|key| Arc::clone(&key.0))
}

/// The Metal driver's packed row lane: reclaim the record walk's parked scan,
/// or pack one from the shared stage-5 rows (both lanes carry the total
/// bytecode PC).
#[cfg(all(feature = "metal", target_os = "macos", not(feature = "akita")))]
fn device_pc_rows<F: JoltField>(
    session: &mut ProofSession,
    rows: &[InstructionCycleRow],
) -> Result<Arc<PcRows>, KernelError<F>> {
    if let Some(key) = session.state::<PcRowsKey>() {
        if key.0.len() != rows.len() {
            return Err(KernelError::TableSizeMismatch {
                table: "bytecode cycle PC rows".to_owned(),
                expected: rows.len(),
                got: key.0.len(),
            });
        }
        return Ok(Arc::clone(&key.0));
    }
    let pack = |slot: &mut PcRow, row: &InstructionCycleRow| {
        let pc = row_pc(row);
        if pc as u32 as usize != pc || pc as u32 == COLD {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode PC exceeds the packed u32 range",
            });
        }
        *slot = PcRow {
            push_pc: pc as u32,
            mapped_pc: pc as u32,
        };
        Ok::<(), KernelError<F>>(())
    };
    let mut packed: MmapVec<PcRow> = MmapVec::zeroed(rows.len());
    #[cfg(feature = "parallel")]
    packed
        .par_iter_mut()
        .zip(rows.par_iter())
        .try_for_each(|(slot, row)| pack(slot, row))?;
    #[cfg(not(feature = "parallel"))]
    packed
        .iter_mut()
        .zip(rows.iter())
        .try_for_each(|(slot, row)| pack(slot, row))?;
    Ok(Arc::new(PcRows::new(packed)))
}

/// Total bytecode PC of a packed stage-5 row. The row's `Option` lane is
/// sentinel ABI; the packing pass writes the total `BytecodePc` on every
/// cycle, and an unset lane decodes to the reserved noop row 0.
#[inline]
fn row_pc(row: &InstructionCycleRow) -> usize {
    row.mapped_pc().unwrap_or_default()
}

/// Per-stage cycle-eq pushforwards onto the bytecode address domain. Base and
/// row-weighted stages share one trace walk over the split-eq decomposition.
/// Each stage's table accumulates in its own lanes, so a subset walk produces
/// the same field values as the full walk —
/// [`spawn_bytecode_stage_pushforwards`] builds the stage-1..4 base tables in
/// the background and the 6a prepare completes the rest inline.
fn stage_pushforwards<F: JoltField, R: Sync>(
    base_cycle_points: &[Vec<F>],
    weighted_cycle_points: &[Vec<F>],
    rows: &[R],
    addresses: usize,
    pc: impl Fn(&R) -> usize + Sync,
    row_weight: impl Fn(&R) -> F + Sync,
) -> Vec<Vec<F>> {
    let Some(first_stage) = base_cycle_points
        .first()
        .or_else(|| weighted_cycle_points.first())
    else {
        return Vec::new();
    };
    let log_t = first_stage.len();
    let base_stages = base_cycle_points.len();
    let num_stages = base_stages + weighted_cycle_points.len();
    let lo_bits = log_t / 2;
    let hi_bits = log_t - lo_bits;
    let in_len = 1usize << lo_bits;
    let out_len = 1usize << hi_bits;

    // Big-endian points split as eq(r, j) = eq(r[..hi], j_hi) · eq(r[hi..], j_lo)
    // with j = (j_hi << lo_bits) | j_lo.
    let e_hi = base_cycle_points
        .iter()
        .chain(weighted_cycle_points)
        .map(|point| eq_table(&point[..hi_bits]))
        .collect::<Vec<_>>();
    let e_lo = base_cycle_points
        .iter()
        .chain(weighted_cycle_points)
        .map(|point| eq_table(&point[hi_bits..]))
        .collect::<Vec<_>>();

    let block = |range: std::ops::Range<usize>| -> Vec<Vec<F>> {
        let mut partial = (0..num_stages)
            .map(|_| vec![F::zero(); addresses])
            .collect::<Vec<_>>();
        let mut inner = (0..num_stages)
            .map(|_| vec![F::zero(); addresses])
            .collect::<Vec<_>>();
        let mut seen = vec![false; addresses];
        let mut touched: Vec<usize> = Vec::with_capacity(in_len);
        for j_hi in range {
            for &k in &touched {
                for stage_inner in &mut inner {
                    stage_inner[k] = F::zero();
                }
                seen[k] = false;
            }
            touched.clear();
            let base = j_hi << lo_bits;
            for j_lo in 0..in_len {
                let row = &rows[base + j_lo];
                let pc = pc(row);
                if !seen[pc] {
                    seen[pc] = true;
                    touched.push(pc);
                }
                let (base_inner, weighted_inner) = inner.split_at_mut(base_stages);
                for (stage_inner, stage_lo) in base_inner.iter_mut().zip(&e_lo[..base_stages]) {
                    stage_inner[pc] += stage_lo[j_lo];
                }
                if !weighted_inner.is_empty() {
                    let weight = row_weight(row);
                    for (stage_inner, stage_lo) in
                        weighted_inner.iter_mut().zip(&e_lo[base_stages..])
                    {
                        stage_inner[pc] += stage_lo[j_lo] * weight;
                    }
                }
            }
            for &k in &touched {
                for ((stage_partial, stage_inner), stage_hi) in
                    partial.iter_mut().zip(&inner).zip(&e_hi)
                {
                    stage_partial[k] += stage_hi[j_hi] * stage_inner[k];
                }
            }
        }
        partial
    };

    #[cfg(feature = "parallel")]
    {
        let num_threads = rayon::current_num_threads();
        let chunk = out_len.div_ceil(num_threads).max(1);
        (0..out_len)
            .into_par_iter()
            .step_by(chunk)
            .map(|start| block(start..(start + chunk).min(out_len)))
            .reduce(
                || {
                    (0..num_stages)
                        .map(|_| vec![F::zero(); addresses])
                        .collect()
                },
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                    }
                    left
                },
            )
    }
    #[cfg(not(feature = "parallel"))]
    {
        block(0..out_len)
    }
}

/// Session carry: the stage-1..4 base pushforward tables, built ahead of
/// stage 6a on a dedicated thread. The points are stored for validation — any
/// mismatch (or a panicked worker, e.g. an out-of-domain PC that the
/// prepare's own validation would reject anyway) falls back to the inline
/// full walk, which produces bit-identical values.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct PrebuiltBytecodePushforwards<F> {
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalar_rows))]
    points: [Vec<F>; 4],
    addresses: usize,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    handle: std::thread::JoinHandle<Vec<Vec<F>>>,
}

/// The background pool width for [`spawn_bytecode_stage_pushforwards`].
const BYTECODE_BACKGROUND_THREADS: usize = 4;

/// Spawn the stage-1..4 bytecode pushforward walk on a dedicated capped
/// thread pool and park the join handle in the session. Call once the
/// stage-4 output points exist (the stage-5 point does not yet); the 6a
/// address-phase prepare reclaims the four tables and walks only the
/// stage-5 point (plus any fused stages) inline.
pub fn spawn_bytecode_stage_pushforwards<F: JoltField>(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    log_t: usize,
    addresses: usize,
    early_points: [Vec<F>; 4],
) -> Result<(), KernelError<F>> {
    if early_points.iter().any(|point| point.len() != log_t) {
        return Err(KernelError::InvariantViolation {
            reason: "bytecode early stage point has the wrong variable count",
        });
    }
    let rows = shared_instruction_rows(session, witness, 1usize << log_t)?;
    let points = early_points.clone();
    let handle = std::thread::spawn(move || {
        let _token = super::BACKGROUND_BUILD_TOKEN
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let build = || {
            stage_pushforwards::<F, _>(&early_points, &[], &rows, addresses, row_pc, |_| F::one())
        };
        #[cfg(feature = "parallel")]
        if let Ok(pool) = rayon::ThreadPoolBuilder::new()
            .num_threads(BYTECODE_BACKGROUND_THREADS)
            .build()
        {
            return pool.install(build);
        }
        build()
    });
    session.park(PrebuiltBytecodePushforwards {
        points,
        addresses,
        handle,
    });
    Ok(())
}

/// Stage-6a address phase: `PrepareKernel` front of the optimized kernel.
pub struct OptimizedBytecodeReadRafAddress;

impl<F: JoltField> PrepareKernel<F, BytecodeReadRafAddressPhase<F>>
    for OptimizedBytecodeReadRafAddress
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafAddressPhase<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let addresses = 1usize << dimensions.log_k();
        let cycles = 1usize << dimensions.log_t();

        let program = witness.program_preprocessing();
        if program.bytecode.bytecode.len() != addresses {
            return Err(KernelError::TableSizeMismatch {
                table: "bytecode stage values".to_owned(),
                expected: addresses,
                got: program.bytecode.bytecode.len(),
            });
        }
        let stage_gammas = inputs.challenges.stage_gamma_powers();
        let stage_values = read_raf_stage_values(BytecodeReadRafStageValueInputs {
            bytecode: &program.bytecode.bytecode,
            register_read_write_point: &relation.register_read_write_point()
                [..REGISTER_ADDRESS_BITS],
            register_val_evaluation_point: &relation.register_val_evaluation_point()
                [..REGISTER_ADDRESS_BITS],
            stage1_gammas: &stage_gammas[0],
            stage2_gammas: &stage_gammas[1],
            stage3_gammas: &stage_gammas[2],
            stage4_gammas: &stage_gammas[3],
            stage5_gammas: &stage_gammas[4],
        });

        let stage_cycle_points = relation.stage_cycle_points();
        let fused_cycle_points = relation.fused_inc_cycle_points();
        for point in stage_cycle_points.iter().chain(fused_cycle_points) {
            if point.len() != dimensions.log_t() {
                return Err(KernelError::InvariantViolation {
                    reason: "bytecode stage cycle point has the wrong variable count",
                });
            }
        }
        let rows = shared_instruction_rows(session, witness, cycles)?;
        let push_pc = row_pc;
        let entry_bytecode_index = relation.entry_bytecode_index();
        if entry_bytecode_index >= addresses || rows.iter().any(|row| push_pc(row) >= addresses) {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode index outside the padded bytecode domain",
            });
        }

        let base_stages = stage_cycle_points.len();
        let num_stages = base_stages + fused_cycle_points.len();
        let gamma_powers = gamma_powers(inputs.challenges.gamma, num_stages + 3);

        #[cfg(not(feature = "akita"))]
        let row_weight = |_: &InstructionCycleRow| F::one();
        #[cfg(feature = "akita")]
        let row_weight = InstructionCycleRow::fused_inc::<F>;
        // Reclaim the background-built stage-1..4 base tables when they match
        // this relation's points; walk only the remaining stages inline.
        // Otherwise (no spawn, stale points, panicked worker) run the full
        // walk — the same per-stage lanes either way, so the values are
        // identical.
        let prebuilt = session
            .take::<PrebuiltBytecodePushforwards<F>>()
            .filter(|carry| {
                carry.addresses == addresses && carry.points[..] == stage_cycle_points[..4]
            })
            .and_then(|carry| carry.handle.join().ok());
        let pushforwards = if let Some(mut early) = prebuilt {
            early.extend(stage_pushforwards::<F, _>(
                &stage_cycle_points[4..],
                fused_cycle_points,
                &rows,
                addresses,
                push_pc,
                row_weight,
            ));
            early
        } else {
            stage_pushforwards::<F, _>(
                stage_cycle_points,
                fused_cycle_points,
                &rows,
                addresses,
                push_pc,
                row_weight,
            )
        };
        if pushforwards.len() != num_stages {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode pushforward table count drifted",
            });
        }
        let pushforwards = pushforwards
            .into_iter()
            .map(Polynomial::new)
            .collect::<Vec<_>>();
        let values = (0..NUM_BYTECODE_VAL_STAGES)
            .map(|stage| Polynomial::new(stage_values.iter().map(|row| row[stage]).collect()))
            .collect::<Vec<_>>();
        let mut stage_values = (0..base_stages).map(StageVal::Table).collect::<Vec<_>>();
        if !fused_cycle_points.is_empty() {
            if fused_cycle_points.len() != bytecode::LATTICE_FUSED_INC_STAGES
                || NUM_BYTECODE_VAL_STAGES != base_stages + 1
            {
                return Err(KernelError::InvariantViolation {
                    reason: "packed bytecode read-raf stage shape is inconsistent",
                });
            }
            stage_values.extend([
                StageVal::Table(base_stages),
                StageVal::Table(base_stages),
                StageVal::Complement(base_stages),
                StageVal::Complement(base_stages),
            ]);
        }
        let mut raf_weights = vec![F::zero(); num_stages];
        raf_weights[0] = gamma_powers[num_stages];
        raf_weights[2] = gamma_powers[num_stages - 1];
        let int_table = Polynomial::new((0..addresses).map(|k| F::from_u64(k as u64)).collect());
        let one_hot = |index: usize| {
            let mut table = vec![F::zero(); addresses];
            table[index] = F::one();
            Polynomial::new(table)
        };

        Ok(Box::new(AddressKernel {
            progress: RoundProgress::new(relation.rounds()),
            committed_program: relation.committed_program(),
            stage_weights: gamma_powers[..num_stages].to_vec(),
            entry_weight: gamma_powers[num_stages + 2],
            raf_weights,
            pushforwards,
            values,
            stage_values,
            int_table,
            entry_trace: one_hot(push_pc(&rows[0])),
            entry_expected: one_hot(entry_bytecode_index),
        }))
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum StageVal {
    Table(usize),
    Complement(usize),
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct AddressKernel<F: JoltField> {
    progress: RoundProgress,
    committed_program: bool,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    stage_weights: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    entry_weight: F,
    /// Within-stage RAF `Int` weights, divided by the stage batching weight.
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    raf_weights: Vec<F>,
    pushforwards: Vec<Polynomial<F>>,
    /// RAW stage-value tables — the RAF identity binds separately so
    /// committed mode can stage the raw bound `Val_s` wire claims.
    values: Vec<Polynomial<F>>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    stage_values: Vec<StageVal>,
    int_table: Polynomial<F>,
    entry_trace: Polynomial<F>,
    entry_expected: Polynomial<F>,
}
impl<F: JoltField> AddressKernel<F> {
    #[inline]
    fn stage_pair(&self, stage: usize, y: usize) -> (F, F) {
        match self.stage_values[stage] {
            StageVal::Table(index) => pair(&self.values[index], y),
            StageVal::Complement(index) => {
                let (lo, hi) = pair(&self.values[index], y);
                (F::one() - lo, F::one() - hi)
            }
        }
    }

    fn bound_stage_value(&self, stage: usize) -> F {
        match self.stage_values[stage] {
            StageVal::Table(index) => self.values[index].evals()[0],
            StageVal::Complement(index) => F::one() - self.values[index].evals()[0],
        }
    }

    fn bind(&mut self, challenge: F) {
        bind_all(
            self.pushforwards
                .iter_mut()
                .chain(self.values.iter_mut())
                .chain([
                    &mut self.int_table,
                    &mut self.entry_trace,
                    &mut self.entry_expected,
                ]),
            challenge,
        );
        self.progress.advance();
    }

    /// The summand's evaluations at `t ∈ {0, 2}` summed over group `y`.
    #[inline]
    fn group_evals(&self, y: usize) -> [F; 2] {
        let (int_lo, int_hi) = pair(&self.int_table, y);
        let int_delta = int_hi - int_lo;
        let int_at_2 = int_hi + int_delta;
        let mut out = [F::zero(); 2];
        for s in 0..self.stage_values.len() {
            let (f_lo, f_hi) = pair(&self.pushforwards[s], y);
            let (v_lo, v_hi) = self.stage_pair(s, y);
            let raf = self.raf_weights[s];
            let val_at_0 = v_lo + raf * int_lo;
            let val_at_2 = (v_hi + v_hi - v_lo) + raf * int_at_2;
            out[0] += self.stage_weights[s] * f_lo * val_at_0;
            out[1] += self.stage_weights[s] * (f_hi + f_hi - f_lo) * val_at_2;
        }
        let (t_lo, t_hi) = pair(&self.entry_trace, y);
        let (e_lo, e_hi) = pair(&self.entry_expected, y);
        out[0] += self.entry_weight * t_lo * e_lo;
        out[1] += self.entry_weight * (t_hi + t_hi - t_lo) * (e_hi + e_hi - e_lo);
        out
    }
}

impl<F: JoltField> ProveRounds<F> for AddressKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.entry_trace.len() / 2;

        let evals = par_sum_pair_groups(half, 2, |acc, y| {
            let group = self.group_evals(y);
            acc[0] += group[0];
            acc[1] += group[1];
        });

        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for AddressKernel<F> {
    type Relation = BytecodeReadRafAddressPhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<BytecodeReadRafAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let mut intermediate =
            self.entry_weight * self.entry_trace.evals()[0] * self.entry_expected.evals()[0];
        let bound_int = self.int_table.evals()[0];
        for s in 0..self.stage_values.len() {
            intermediate += self.stage_weights[s]
                * self.pushforwards[s].evals()[0]
                * (self.bound_stage_value(s) + self.raf_weights[s] * bound_int);
        }
        let val_stages = if self.committed_program {
            self.values.iter().map(|table| table.evals()[0]).collect()
        } else {
            Vec::new()
        };
        Ok(BytecodeReadRafAddressPhaseOutputClaims {
            intermediate,
            val_stages,
        })
    }
}

/// The packed fused-increment cycle column, bound lazily while the bytecode
/// RA factors still retain their shared compact rows. The fourth bind
/// materializes only `T / 16` field elements and releases this handle.
#[cfg(feature = "akita")]
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
enum LazyFusedInc<F: JoltField> {
    Lazy {
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        branch_weights: Vec<F>,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        rows: Arc<InstructionRows>,
    },
    Dense(Polynomial<F>),
}

#[cfg(feature = "akita")]
impl<F: JoltField> LazyFusedInc<F> {
    fn new(rows: Arc<InstructionRows>) -> Self {
        Self::Lazy {
            branch_weights: vec![F::one()],
            rows,
        }
    }

    #[inline]
    fn value(&self, index: usize) -> F {
        match self {
            Self::Lazy {
                branch_weights,
                rows,
            } => branch_weights
                .iter()
                .enumerate()
                .fold(F::zero(), |acc, (offset, weight)| {
                    acc + *weight * rows[index * branch_weights.len() + offset].fused_inc::<F>()
                }),
            Self::Dense(polynomial) => polynomial.evals()[index],
        }
    }

    #[inline]
    fn lo_hi(&self, row: usize) -> (F, F) {
        (self.value(2 * row), self.value(2 * row + 1))
    }

    fn bind(&mut self, challenge: F) {
        *self = match std::mem::replace(self, Self::Dense(Polynomial::zeros(0))) {
            Self::Lazy {
                branch_weights,
                rows,
            } => {
                let one_minus = F::one() - challenge;
                let mut next = Vec::with_capacity(2 * branch_weights.len());
                next.extend(branch_weights.iter().map(|weight| one_minus * *weight));
                next.extend(branch_weights.iter().map(|weight| challenge * *weight));
                if next.len() < 16 {
                    Self::Lazy {
                        branch_weights: next,
                        rows,
                    }
                } else {
                    let len = rows.len() / next.len();
                    let evaluate = |index: usize| {
                        next.iter()
                            .enumerate()
                            .fold(F::zero(), |acc, (offset, weight)| {
                                acc + *weight * rows[index * next.len() + offset].fused_inc::<F>()
                            })
                    };
                    #[cfg(feature = "parallel")]
                    let evals = (0..len).into_par_iter().map(evaluate).collect();
                    #[cfg(not(feature = "parallel"))]
                    let evals = (0..len).map(evaluate).collect();
                    Self::Dense(Polynomial::new(evals))
                }
            }
            Self::Dense(mut polynomial) => {
                polynomial.bind_with_order(challenge, BindingOrder::LowToHigh);
                Self::Dense(polynomial)
            }
        };
    }
}

/// Stage-6b cycle phase: `PrepareKernel` front of the optimized kernel.
pub struct OptimizedBytecodeReadRafCycle;

impl<F: JoltField> PrepareKernel<F, BytecodeReadRafCycle<F>> for OptimizedBytecodeReadRafCycle {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafCycle<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafCycle<F>>>, KernelError<F>>
    {
        prepare_bytecode_read_raf_cycle(session, witness, inputs, |_| None)
    }
}

/// Device factory inputs for the stage-6b cycle member. The factory copies
/// what it retains; every borrow is prepare-scoped.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    expect(dead_code, reason = "read only by the Metal driver factory")
)]
pub(crate) struct BytecodeCycleDeviceInputs<'a, F> {
    pub(crate) rows: &'a Arc<PcRows>,
    pub(crate) stage_cycle_points: &'a [Vec<F>],
    pub(crate) stage_weights: &'a [F],
    pub(crate) entry_term: F,
    pub(crate) selector_shifts: Vec<u32>,
    pub(crate) committed_chunk_bits: usize,
    pub(crate) degree: usize,
}

/// A device driver plus its fail-closed combined-table recovery channel.
pub(crate) struct BytecodeCycleDevice<F: JoltField> {
    pub(crate) driver: Box<dyn LazyRaDevice<F>>,
    pub(crate) combined_recovery: Arc<Mutex<Option<Vec<F>>>>,
}

pub(crate) fn prepare_bytecode_read_raf_cycle<F: JoltField>(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    inputs: ProverInputs<'_, F, BytecodeReadRafCycle<F>>,
    driver_factory: impl FnOnce(BytecodeCycleDeviceInputs<'_, F>) -> Option<BytecodeCycleDevice<F>>,
) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafCycle<F>>>, KernelError<F>> {
    let relation = inputs.relation;
    let dimensions = relation.dimensions();
    let r_address = relation.r_address();
    let stage_cycle_points = relation.stage_cycle_points();
    let cycles = 1usize << dimensions.log_t();
    let num_ra = dimensions.num_committed_ra_polys();

    let chunks = committed_address_chunks(r_address, relation.committed_chunk_bits());
    if chunks.len() != num_ra {
        return Err(KernelError::InvariantViolation {
            reason: "bytecode address chunk count disagrees with the committed RA count",
        });
    }
    let rows = shared_instruction_rows(session, witness, cycles)?;

    // ra_i(j) = eq(chunk_i)[chunk_i(pc_j)] — the address fold of the
    // one-hot grid, served lazily off the sparse per-cycle indices for
    // the first four binds instead of `d × T` dense.
    let chunk_eqs: Vec<Vec<F>> = chunks.iter().map(|chunk| eq_table(chunk)).collect();
    let selectors = (0..num_ra)
        .map(|index| {
            RaChunkSelector::new(index, num_ra, relation.committed_chunk_bits())
                .map_err(KernelError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(feature = "akita")]
    let fused_inc = LazyFusedInc::new(Arc::clone(&rows));

    // The combined coefficient table: every non-RA factor of the summand
    // is linear in one cycle table, so
    //   C(j) = Σ_s (γ^s·val_s + raf_s·int_r)·eq_s(j) + γ⁷·entry·[j = 0]
    // with raf_0 = γ⁵·int_r, raf_2 = γ⁶·int_r (SpartanOuterRaf rides the
    // stage-1 cycle point, SpartanShiftRaf the stage-3 one).
    let stage_values = relation.stage_values_at_r_address()?;
    let num_stages = stage_cycle_points.len();
    let base_stages = bytecode::BYTECODE_STAGE_GAMMA_COUNTS.len();
    let gamma_powers = gamma_powers(inputs.challenges.gamma, num_stages + 3);
    let int_at_r_address = IdentityPolynomial::new(r_address.len()).evaluate(r_address);
    let mut stage_weights = (0..base_stages)
        .map(|stage| gamma_powers[stage] * stage_values[stage])
        .collect::<Vec<_>>();
    stage_weights[0] += gamma_powers[num_stages] * int_at_r_address;
    stage_weights[2] += gamma_powers[num_stages + 1] * int_at_r_address;

    for point in stage_cycle_points {
        if point.len() != dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode stage cycle point has the wrong variable count",
            });
        }
    }
    let entry_scalar = eq_table(r_address)[relation.entry_bytecode_index()];
    let entry_term = gamma_powers[num_stages + 2] * entry_scalar;

    // The Metal member builds the combined table on-device from the same
    // stage weights. The packed protocol's fused stages stay on the CPU, so
    // the device tier is not offered under `akita`.
    #[cfg(all(feature = "metal", target_os = "macos", not(feature = "akita")))]
    let device = {
        let device_rows = device_pc_rows::<F>(session, &rows)?;
        driver_factory(BytecodeCycleDeviceInputs {
            rows: &device_rows,
            stage_cycle_points: &stage_cycle_points[..base_stages],
            stage_weights: &stage_weights,
            entry_term,
            selector_shifts: selectors
                .iter()
                .map(|selector| selector.shift() as u32)
                .collect(),
            committed_chunk_bits: relation.committed_chunk_bits(),
            degree: relation.degree(),
        })
    };
    #[cfg(not(all(feature = "metal", target_os = "macos", not(feature = "akita"))))]
    let device: Option<BytecodeCycleDevice<F>> = {
        let _ = driver_factory;
        None
    };
    // The PC scan's last consumer: remove the session's copy so the rows
    // free with the driver instead of living to the end of the proof.
    let _ = session.take::<PcRowsKey>();

    let launch = device.is_some();
    let (driver, combined_recovery, combined) = if let Some(device) = device {
        (Some(device.driver), Some(device.combined_recovery), None)
    } else {
        let mut combined = vec![F::zero(); cycles];
        for (point, weight) in stage_cycle_points[..base_stages].iter().zip(stage_weights) {
            let scaled = scaled_eq_table(point, weight);
            #[cfg(feature = "parallel")]
            combined
                .par_iter_mut()
                .zip(scaled.par_iter())
                .for_each(|(acc, term)| *acc += *term);
            #[cfg(not(feature = "parallel"))]
            combined
                .iter_mut()
                .zip(scaled.iter())
                .for_each(|(acc, term)| *acc += *term);
        }
        combined[0] += entry_term;
        (None, None, Some(Polynomial::new(combined)))
    };

    #[cfg(feature = "akita")]
    let fused_combined = {
        let store = stage_values[base_stages];
        let mut combined = vec![F::zero(); cycles];
        for stage in base_stages..num_stages {
            let value = if stage < base_stages + 2 {
                store
            } else {
                F::one() - store
            };
            let scaled = scaled_eq_table(&stage_cycle_points[stage], gamma_powers[stage] * value);
            #[cfg(feature = "parallel")]
            combined
                .par_iter_mut()
                .zip(scaled.par_iter())
                .for_each(|(acc, term)| *acc += *term);
            #[cfg(not(feature = "parallel"))]
            combined
                .iter_mut()
                .zip(scaled.iter())
                .for_each(|(acc, term)| *acc += *term);
        }
        Polynomial::new(combined)
    };

    let ra = LazyFoldedRa::new_with_driver(chunk_eqs, BytecodePcChunks { rows, selectors }, driver);

    let output_openings = bytecode::read_raf_output_openings(dimensions).bytecode_ra;
    if output_openings.len() != num_ra {
        return Err(KernelError::InvariantViolation {
            reason: "bytecode RA output opening count disagrees with the committed RA count",
        });
    }

    Ok(Box::new(CycleKernel {
        progress: RoundProgress::new(relation.rounds()),
        degree: relation.degree(),
        ra,
        combined,
        combined_recovery,
        #[cfg(feature = "akita")]
        fused_inc,
        #[cfg(feature = "akita")]
        fused_combined,
        output_openings,
        launch,
        launched: false,
    }))
}

/// Lazy-RA index source: chunk `i` of the per-cycle total bytecode PC, off
/// the shared stage-5 rows (every cycle is hot — the committed one-hot has a
/// row for no-ops too).
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct BytecodePcChunks {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    rows: Arc<InstructionRows>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    selectors: Vec<RaChunkSelector>,
}

impl ChunkIndexSource for BytecodePcChunks {
    fn num_polys(&self) -> usize {
        self.selectors.len()
    }

    fn cycles(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    fn index(&self, i: usize, j: usize) -> Option<usize> {
        Some(self.selectors[i].chunk_usize(row_pc(&self.rows[j])))
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct CycleKernel<F: JoltField> {
    progress: RoundProgress,
    degree: usize,
    ra: LazyFoldedRa<F, BytecodePcChunks>,
    /// `None` while the device owns the combined table; recovered through
    /// `combined_recovery` on any device decline or failure.
    combined: Option<Polynomial<F>>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    combined_recovery: Option<Arc<Mutex<Option<Vec<F>>>>>,
    #[cfg(feature = "akita")]
    fused_inc: LazyFusedInc<F>,
    #[cfg(feature = "akita")]
    fused_combined: Polynomial<F>,
    /// The produced `BytecodeRa` opening ids, in `read_raf_output_openings`
    /// order (index-aligned with `ra`).
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    output_openings: Vec<JoltOpeningId>,
    /// A device driver is installed: rounds run two-phase (launch in
    /// `begin_round`, collect in `collect_round`).
    launch: bool,
    /// A device round is in flight for the current batch round.
    launched: bool,
}
impl<F: JoltField> CycleKernel<F> {
    fn bind(&mut self, challenge: F) {
        if let Some(combined) = &mut self.combined {
            bind_all([combined], challenge);
        }
        #[cfg(feature = "akita")]
        bind_all([&mut self.fused_combined], challenge);
        self.ra.bind(challenge);
        #[cfg(feature = "akita")]
        self.fused_inc.bind(challenge);
        self.recover_combined();
        self.progress.advance();
    }

    fn recover_combined(&mut self) {
        if self.combined.is_some() {
            return;
        }
        let Some(recovery) = &self.combined_recovery else {
            return;
        };
        let mut recovery = recovery
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(table) = recovery.take() {
            self.combined = Some(Polynomial::new(table));
            self.ra.disable_device();
        }
    }

    /// The summand's evaluations at `t ∈ {0, 2, 3, .., degree}` summed over
    /// group `y`, written into `acc` (length `degree`); `ra_pairs` is the
    /// caller's per-group `(lo, hi)` scratch (each RA pair is gathered once
    /// per group, not once per sample point).
    #[inline]
    #[expect(
        clippy::expect_used,
        reason = "device decline must publish the combined recovery table"
    )]
    fn accumulate_group(&self, y: usize, acc: &mut [F], ra_pairs: &mut [(F, F)]) {
        let (c_lo, c_hi) = pair(
            self.combined
                .as_ref()
                .expect("combined table unavailable after device recovery"),
            y,
        );
        let c_delta = c_hi - c_lo;
        #[cfg(feature = "akita")]
        let (fused_coefficient_lo, fused_coefficient_hi) = pair(&self.fused_combined, y);
        #[cfg(feature = "akita")]
        let fused_coefficient_delta = fused_coefficient_hi - fused_coefficient_lo;
        #[cfg(feature = "akita")]
        let (fused_inc_lo, fused_inc_hi) = self.fused_inc.lo_hi(y);
        #[cfg(feature = "akita")]
        let fused_inc_delta = fused_inc_hi - fused_inc_lo;
        for (i, slot) in ra_pairs.iter_mut().enumerate() {
            *slot = self.ra.lo_hi(i, y);
        }
        #[cfg(not(feature = "akita"))]
        let coefficient_at_zero = c_lo;
        #[cfg(feature = "akita")]
        let coefficient_at_zero = c_lo + fused_inc_lo * fused_coefficient_lo;
        acc[0] += ra_pairs
            .iter()
            .fold(coefficient_at_zero, |acc, (lo, _)| acc * *lo);
        for (slot, t) in (2..=self.degree).enumerate() {
            let t_value = F::from_u64(t as u64);
            let coefficient = c_lo + t_value * c_delta;
            #[cfg(feature = "akita")]
            let coefficient = coefficient
                + (fused_inc_lo + t_value * fused_inc_delta)
                    * (fused_coefficient_lo + t_value * fused_coefficient_delta);
            acc[slot + 1] += ra_pairs.iter().fold(coefficient, |acc, (lo, hi)| {
                acc * (*lo + t_value * (*hi - *lo))
            });
        }
    }
}

impl<F: JoltField> ProveRounds<F> for CycleKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        self.message(previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }

    fn begin_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        _previous_claim: F,
    ) -> Result<bool, SumcheckError<F>> {
        if !self.launch {
            return Ok(false);
        }
        // A prepare-time prelaunched round 0 (only round 0 arrives bindless)
        // is already in flight: report launched without re-launching.
        if bind.is_none() && self.launched {
            return Ok(true);
        }
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        if self.combined.is_none() {
            self.launched = self.ra.launch_device_lanes(&[], &[]);
            if !self.launched {
                // A decline may have normalized the device tier away
                // (published recovery / reclaimed tables) — pick it up so
                // the collect-side CPU compute finds the combined table.
                self.recover_combined();
            }
        }
        Ok(self.launched)
    }

    fn collect_round(
        &mut self,
        _bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if std::mem::take(&mut self.launched) {
            if let Some(evals) = self.ra.collect_device_lanes() {
                debug_assert_eq!(evals.len(), self.degree);
                return Ok(round_poly_from_skipped_evals(&evals, previous_claim));
            }
            // Wait failure: the driver published the recovery and latched
            // off — recompute the SAME round on the CPU below.
            self.recover_combined();
        }
        if self.launch {
            // `begin_round` already bound.
            return self.message(previous_claim);
        }
        self.prove_round(_bind, _round, previous_claim)
    }
}

impl<F: JoltField> CycleKernel<F> {
    /// The round message from current (post-bind) state; the device tier's
    /// synchronous shot first, the CPU loop otherwise.
    fn message(&mut self, previous_claim: F) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if self.combined.is_none() {
            if let Some(evals) = self.ra.device_lanes(&[], &[]) {
                debug_assert_eq!(evals.len(), self.degree);
                return Ok(round_poly_from_skipped_evals(&evals, previous_claim));
            }
            self.recover_combined();
        }
        let half = self
            .combined
            .as_ref()
            .map(|combined| combined.len())
            .unwrap_or_default()
            / 2;
        let slots = self.degree;
        let num_ra = self.ra.num_polys();

        let evals = par_sum_pair_groups_reusing(
            half,
            slots,
            || vec![(F::zero(), F::zero()); num_ra],
            |acc, ra_pairs, y| self.accumulate_group(y, acc, ra_pairs),
        );

        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }
}

impl<F: JoltField> SumcheckKernel<F> for CycleKernel<F> {
    type Relation = BytecodeReadRafCycle<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        self.ra.ensure_host();
        let ra = &self.ra;
        let output_openings = &self.output_openings;
        SumcheckOutputClaims::<F, Self::Relation>::from_opening_values(|id: &JoltOpeningId| {
            #[cfg(feature = "akita")]
            if *id == bytecode::fused_inc_read_raf_opening() {
                return Some(self.fused_inc.value(0));
            }
            output_openings
                .iter()
                .position(|opening| opening == id)
                .map(|index| ra.value(index, 0))
        })
        .map_err(SumcheckKernelError::from)
    }
}

#[cfg(test)]
mod stage_pushforward_tests {
    use jolt_field::{Fr, One, Ring, Zero};

    use super::*;
    use crate::optimized::parity::synthetic_point;

    #[derive(Clone, Copy)]
    struct Row {
        pc: usize,
        weight: Fr,
    }

    #[test]
    fn membership_is_independent_of_zero_contributions() {
        let log_t = 4;
        let addresses = 4;
        let mut points: [Vec<Fr>; 5] =
            std::array::from_fn(|stage| synthetic_point(log_t, 101 + stage as u64));
        points[0][2] = Fr::one();

        let pcs = [2, 0, 2, 1, 3, 3, 0, 2, 1, 1, 1, 1, 0, 3, 2, 0];
        let rows: Vec<Row> = pcs
            .into_iter()
            .enumerate()
            .map(|(index, pc)| Row {
                pc,
                weight: if index.is_multiple_of(3) {
                    Fr::zero()
                } else {
                    Fr::from_u64(index as u64 + 1)
                },
            })
            .collect();

        let expected = |weighted: bool| {
            points
                .iter()
                .map(|point| {
                    let eq = eq_table(point);
                    let mut values = vec![Fr::zero(); addresses];
                    for (index, row) in rows.iter().enumerate() {
                        values[row.pc] += eq[index] * if weighted { row.weight } else { Fr::one() };
                    }
                    values
                })
                .collect::<Vec<_>>()
        };

        let unweighted = stage_pushforwards::<Fr, _>(
            &points,
            &[],
            &rows,
            addresses,
            |row| row.pc,
            |_| Fr::one(),
        );
        let weighted = stage_pushforwards::<Fr, _>(
            &[],
            &points,
            &rows,
            addresses,
            |row| row.pc,
            |row| row.weight,
        );
        let combined = stage_pushforwards::<Fr, _>(
            &points,
            &points,
            &rows,
            addresses,
            |row| row.pc,
            |row| row.weight,
        );
        assert_eq!(unweighted, expected(false));
        assert_eq!(weighted, expected(true));
        assert_eq!(
            combined,
            unweighted
                .iter()
                .chain(&weighted)
                .cloned()
                .collect::<Vec<_>>()
        );
    }
}

/// Byte-parity of both phases against the reference kernels, run as a PAIR
/// through one shared [`ProofSession`] so the parked per-cycle rows flow the
/// way production stages would exercise them.
///
/// Fixture honesty: `with_sample_backend` is the only witness plane
/// constructible without a `jolt-program` dependency. At its scale the
/// bytecode decomposition has a single committed chunk (`d = 1`), so the
/// cycle kernel's multi-factor `Π_i ra_i` loop runs with one factor; the
/// degree-`d+1` sampling, cold-cycle zeroing, entry/RAF fusion, and both
/// pushforward paths are exercised for real.
#[cfg(all(test, not(feature = "akita")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
    use jolt_field::{Fr, Ring};
    use jolt_verifier::stages::stage6a::bytecode_read_raf::{
        BytecodeReadRafAddressPhaseInputClaims, BytecodeStagePoints,
    };
    use jolt_verifier::stages::stage6b::bytecode_read_raf::{
        BytecodeReadRafCycleInputs, BytecodeReadRafCyclePhaseCommittedChallenges,
        BytecodeReadRafInputClaims, BytecodeReadRafTableFoldInputs,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::{JoltWitnessOracle, ProgramSource};

    use super::super::instruction_read_raf::SharedInstructionRows;
    use super::*;
    use crate::optimized::parity::{
        probe_input_claim, probe_one_hot_family, run_lockstep, synthetic_point,
    };
    use crate::ReferenceBackend;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn run_pair(committed_program: bool) {
        with_sample_backend(|backend| {
            let log_t = JoltWitnessOracle::<Fr>::shape(
                backend,
                JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc),
            )
            .unwrap()
            .rows()
            .ilog2() as usize;
            let (bytecode_d, chunk_bits) =
                probe_one_hot_family(backend, JoltCommittedPolynomial::BytecodeRa, log_t);
            assert!(bytecode_d > 0, "fixture serves no BytecodeRa polynomials");
            let program = backend.program_preprocessing();
            let bytecode_len = program.bytecode.bytecode.len();
            assert!(bytecode_len.is_power_of_two());
            let log_k = bytecode_len.ilog2() as usize;
            let dimensions = BytecodeReadRafDimensions::new(log_t, log_k, bytecode_d);
            let entry_bytecode_index = 1.min(bytecode_len - 1);

            let stage_cycle_points: [Vec<Fr>; 5] =
                std::array::from_fn(|s| synthetic_point(log_t, 11 + s as u64));
            let stage_points = BytecodeStagePoints {
                stage_cycle_points: stage_cycle_points.clone(),
                register_read_write_point: synthetic_point(REGISTER_ADDRESS_BITS + log_t, 31),
                register_val_evaluation_point: synthetic_point(REGISTER_ADDRESS_BITS + log_t, 37),
                fused_inc_cycle_points: Vec::new(),
            };

            // ---- Stage 6a: address phase.
            let address_relation = BytecodeReadRafAddressPhase::new(
                dimensions,
                committed_program,
                stage_points.clone(),
                entry_bytecode_index,
            );
            let address_challenges = BytecodeReadRafAddressPhaseChallenges {
                gamma: fr(3),
                stage1_gamma: fr(5),
                stage2_gamma: fr(7),
                stage3_gamma: fr(11),
                stage4_gamma: fr(13),
                stage5_gamma: fr(17),
            };
            let address_claims = BytecodeReadRafAddressPhaseInputClaims::<Fr>::default();
            let address_input_points = BytecodeReadRafAddressPhaseInputClaims::<Vec<Fr>>::default();

            let mut session = ProofSession::default();
            let mut reference =
                <ReferenceBackend as PrepareKernel<Fr, BytecodeReadRafAddressPhase<Fr>>>::prepare(
                    &ReferenceBackend,
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &address_relation,
                        claims: &address_claims,
                        points: &address_input_points,
                        challenges: &address_challenges,
                    },
                )
                .unwrap();
            let mut optimized = OptimizedBytecodeReadRafAddress
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &address_relation,
                        claims: &address_claims,
                        points: &address_input_points,
                        challenges: &address_challenges,
                    },
                )
                .unwrap();
            assert!(
                session.state::<SharedInstructionRows>().is_some(),
                "the address phase must park the shared cycle rows"
            );

            let claim = probe_input_claim(reference.as_mut());
            let address_sumcheck_challenges = synthetic_point(log_k, 101);
            run_lockstep(
                reference.as_mut(),
                optimized.as_mut(),
                claim,
                &address_sumcheck_challenges,
            );
            let reference_outputs = reference.output_claims(&address_claims).unwrap();
            let optimized_outputs = optimized.output_claims(&address_claims).unwrap();
            assert_eq!(reference_outputs, optimized_outputs);
            assert_eq!(
                reference_outputs.val_stages.len(),
                if committed_program { 5 } else { 0 }
            );

            // ---- Stage 6b: cycle phase, from the same session (the parked
            // cycle rows are reused, mirroring the production 6a→6b flow) and
            // the production r_address wiring (the reversed 6a point).
            let r_address: Vec<Fr> = address_sumcheck_challenges.iter().rev().copied().collect();
            let stage_gammas = address_challenges.stage_gamma_powers();
            let cycle_relation = BytecodeReadRafCycle::full(BytecodeReadRafCycleInputs {
                dimensions,
                r_address,
                stage_cycle_points: stage_cycle_points.clone(),
                entry_bytecode_index,
                committed_chunk_bits: chunk_bits,
                table_fold: Some(BytecodeReadRafTableFoldInputs {
                    bytecode: &program.bytecode.bytecode,
                    register_read_write_point: &stage_points.register_read_write_point
                        [..REGISTER_ADDRESS_BITS],
                    register_val_evaluation_point: &stage_points.register_val_evaluation_point
                        [..REGISTER_ADDRESS_BITS],
                    stage_gammas: std::array::from_fn(|s| stage_gammas[s].as_slice()),
                }),
            })
            .unwrap();
            let cycle_challenges = BytecodeReadRafCyclePhaseCommittedChallenges { gamma: fr(19) };
            let cycle_claims = BytecodeReadRafInputClaims::<Fr>::default();
            let cycle_input_points = BytecodeReadRafInputClaims::<Vec<Fr>>::default();

            let mut reference =
                <ReferenceBackend as PrepareKernel<Fr, BytecodeReadRafCycle<Fr>>>::prepare(
                    &ReferenceBackend,
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &cycle_relation,
                        claims: &cycle_claims,
                        points: &cycle_input_points,
                        challenges: &cycle_challenges,
                    },
                )
                .unwrap();
            let mut optimized = OptimizedBytecodeReadRafCycle
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &cycle_relation,
                        claims: &cycle_claims,
                        points: &cycle_input_points,
                        challenges: &cycle_challenges,
                    },
                )
                .unwrap();

            let claim = probe_input_claim(reference.as_mut());
            let cycle_sumcheck_challenges = synthetic_point(log_t, 211);
            run_lockstep(
                reference.as_mut(),
                optimized.as_mut(),
                claim,
                &cycle_sumcheck_challenges,
            );
            assert_eq!(
                reference.output_claims(&cycle_claims).unwrap(),
                optimized.output_claims(&cycle_claims).unwrap()
            );
        });
    }

    /// The background stage-1..4 walk plus the inline stage-5 walk equals
    /// the five-stage walk (matching carry consumed, same round polys and
    /// claims), and a stale carry (wrong points) falls back to the identical
    /// inline build.
    #[test]
    fn background_pushforwards_match_inline_build() {
        with_sample_backend(|backend| {
            let log_t = JoltWitnessOracle::<Fr>::shape(
                backend,
                JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc),
            )
            .unwrap()
            .rows()
            .ilog2() as usize;
            let (bytecode_d, _) =
                probe_one_hot_family(backend, JoltCommittedPolynomial::BytecodeRa, log_t);
            assert!(bytecode_d > 0, "fixture serves no BytecodeRa polynomials");
            let program = backend.program_preprocessing();
            let bytecode_len = program.bytecode.bytecode.len();
            let log_k = bytecode_len.ilog2() as usize;
            let dimensions = BytecodeReadRafDimensions::new(log_t, log_k, bytecode_d);
            let stage_cycle_points: [Vec<Fr>; 5] =
                std::array::from_fn(|s| synthetic_point(log_t, 11 + s as u64));
            let stage_points = BytecodeStagePoints {
                stage_cycle_points: stage_cycle_points.clone(),
                register_read_write_point: synthetic_point(REGISTER_ADDRESS_BITS + log_t, 31),
                register_val_evaluation_point: synthetic_point(REGISTER_ADDRESS_BITS + log_t, 37),
                fused_inc_cycle_points: Vec::new(),
            };
            let relation = BytecodeReadRafAddressPhase::new(
                dimensions,
                false,
                stage_points,
                1.min(bytecode_len - 1),
            );
            let challenges = BytecodeReadRafAddressPhaseChallenges {
                gamma: fr(3),
                stage1_gamma: fr(5),
                stage2_gamma: fr(7),
                stage3_gamma: fr(11),
                stage4_gamma: fr(13),
                stage5_gamma: fr(17),
            };
            let claims = BytecodeReadRafAddressPhaseInputClaims::<Fr>::default();
            let input_points = BytecodeReadRafAddressPhaseInputClaims::<Vec<Fr>>::default();
            let prepare = |session: &mut ProofSession| {
                OptimizedBytecodeReadRafAddress
                    .prepare(
                        session,
                        backend,
                        ProverInputs {
                            relation: &relation,
                            claims: &claims,
                            points: &input_points,
                            challenges: &challenges,
                        },
                    )
                    .unwrap()
            };

            let mut spawned_session = ProofSession::default();
            spawn_bytecode_stage_pushforwards(
                &mut spawned_session,
                backend,
                log_t,
                bytecode_len,
                std::array::from_fn(|s| stage_cycle_points[s].clone()),
            )
            .unwrap();
            let mut spawned = prepare(&mut spawned_session);
            assert!(
                spawned_session
                    .state::<PrebuiltBytecodePushforwards<Fr>>()
                    .is_none(),
                "the prepare must consume the background carry"
            );

            let mut stale_session = ProofSession::default();
            spawn_bytecode_stage_pushforwards(
                &mut stale_session,
                backend,
                log_t,
                bytecode_len,
                std::array::from_fn(|s| synthetic_point(log_t, 900 + s as u64)),
            )
            .unwrap();
            let mut stale = prepare(&mut stale_session);

            let mut inline_a = prepare(&mut ProofSession::default());
            let mut inline_b = prepare(&mut ProofSession::default());

            // The optimized kernel recovers eval-at-1 from the claim instead
            // of checking it, so the honest input claim probes off a
            // reference twin (the run_pair idiom).
            let mut reference =
                <ReferenceBackend as PrepareKernel<Fr, BytecodeReadRafAddressPhase<Fr>>>::prepare(
                    &ReferenceBackend,
                    &mut ProofSession::default(),
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let input_claim = probe_input_claim(reference.as_mut());
            let drive: Vec<Fr> = (0..relation.rounds())
                .map(|round| fr(1000 + round as u64))
                .collect();
            run_lockstep(&mut *inline_a, &mut *spawned, input_claim, &drive);
            run_lockstep(&mut *inline_b, &mut *stale, input_claim, &drive);
        });
    }

    #[test]
    fn bytecode_phases_match_reference_through_a_shared_session() {
        run_pair(false);
    }

    #[test]
    fn bytecode_phases_match_reference_in_committed_program_mode() {
        run_pair(true);
    }
}

#[cfg(all(test, feature = "akita"))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod akita_tests {
    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use jolt_claims::protocols::jolt::lattice::relations::read_raf::LatticeReadRafAddressPhaseInputClaims;
    use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
    use jolt_field::{Fr, Ring};
    use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeStagePoints;
    use jolt_verifier::stages::stage6b::bytecode_read_raf::{
        BytecodeReadRafCycleInputs, BytecodeReadRafCyclePhaseCommittedChallenges,
        BytecodeReadRafInputClaims, BytecodeReadRafTableFoldInputs,
    };
    use jolt_witness::ProgramSource;

    use super::*;
    use crate::optimized::booleanity::testing::with_booleanity_backend;
    use crate::optimized::parity::{probe_input_claim, run_lockstep, synthetic_point};
    use crate::ReferenceBackend;

    fn address_parity(log_t: usize, log_k_chunk: u8, committed_program: bool) {
        with_booleanity_backend(log_t, log_k_chunk, |backend, base_dimensions| {
            let bytecode_len = backend.program_preprocessing().bytecode.bytecode.len();
            let dimensions = BytecodeReadRafDimensions::new(
                log_t,
                bytecode_len.ilog2() as usize,
                base_dimensions.layout.bytecode(),
            );
            let relation = BytecodeReadRafAddressPhase::new(
                dimensions,
                committed_program,
                BytecodeStagePoints {
                    stage_cycle_points: std::array::from_fn(|stage| {
                        synthetic_point(log_t, 11 + stage as u64)
                    }),
                    register_read_write_point: synthetic_point(REGISTER_ADDRESS_BITS + log_t, 23),
                    register_val_evaluation_point: synthetic_point(
                        REGISTER_ADDRESS_BITS + log_t,
                        29,
                    ),
                    fused_inc_cycle_points: (0..bytecode::LATTICE_FUSED_INC_STAGES)
                        .map(|stage| synthetic_point(log_t, 41 + stage as u64))
                        .collect(),
                },
                0,
            );
            let challenges = BytecodeReadRafAddressPhaseChallenges {
                gamma: Fr::from_u64(3),
                stage1_gamma: Fr::from_u64(5),
                stage2_gamma: Fr::from_u64(7),
                stage3_gamma: Fr::from_u64(11),
                stage4_gamma: Fr::from_u64(13),
                stage5_gamma: Fr::from_u64(17),
            };
            let claims = LatticeReadRafAddressPhaseInputClaims::<Fr>::default();
            let input_points = LatticeReadRafAddressPhaseInputClaims::<Vec<Fr>>::default();

            let mut session = ProofSession::default();
            let mut reference = ReferenceBackend
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let mut optimized = OptimizedBytecodeReadRafAddress
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();

            let claim = probe_input_claim(reference.as_mut());
            run_lockstep(
                reference.as_mut(),
                optimized.as_mut(),
                claim,
                &synthetic_point(dimensions.log_k(), 101),
            );
            let reference = reference.output_claims(&claims).unwrap();
            let optimized = optimized.output_claims(&claims).unwrap();
            assert_eq!(reference, optimized);
            assert_eq!(
                optimized.val_stages.len(),
                if committed_program {
                    NUM_BYTECODE_VAL_STAGES
                } else {
                    0
                }
            );
        });
    }

    fn cycle_parity(log_t: usize, log_k_chunk: u8) {
        with_booleanity_backend(log_t, log_k_chunk, |backend, base_dimensions| {
            let program = backend.program_preprocessing();
            let bytecode_len = program.bytecode.bytecode.len();
            let dimensions = BytecodeReadRafDimensions::new(
                log_t,
                bytecode_len.ilog2() as usize,
                base_dimensions.layout.bytecode(),
            );
            let stage_cycle_points =
                std::array::from_fn(|stage| synthetic_point(log_t, 11 + stage as u64));
            let address_challenges = BytecodeReadRafAddressPhaseChallenges {
                gamma: Fr::from_u64(3),
                stage1_gamma: Fr::from_u64(5),
                stage2_gamma: Fr::from_u64(7),
                stage3_gamma: Fr::from_u64(11),
                stage4_gamma: Fr::from_u64(13),
                stage5_gamma: Fr::from_u64(17),
            };
            let stage_gammas = address_challenges.stage_gamma_powers();
            let relation = BytecodeReadRafCycle::full(BytecodeReadRafCycleInputs {
                dimensions,
                r_address: synthetic_point(dimensions.log_k(), 19),
                stage_cycle_points,
                entry_bytecode_index: 0,
                committed_chunk_bits: usize::from(log_k_chunk),
                table_fold: Some(BytecodeReadRafTableFoldInputs {
                    bytecode: &program.bytecode.bytecode,
                    register_read_write_point: &synthetic_point(REGISTER_ADDRESS_BITS, 23),
                    register_val_evaluation_point: &synthetic_point(REGISTER_ADDRESS_BITS, 29),
                    stage_gammas: std::array::from_fn(|stage| stage_gammas[stage].as_slice()),
                }),
            })
            .unwrap();
            let challenges = BytecodeReadRafCyclePhaseCommittedChallenges {
                gamma: Fr::from_u64(31),
            };
            let claims = BytecodeReadRafInputClaims::<Fr>::default();
            let input_points = BytecodeReadRafInputClaims::<Vec<Fr>>::default();

            let mut session = ProofSession::default();
            let mut reference = ReferenceBackend
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let mut optimized = OptimizedBytecodeReadRafCycle
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();

            let claim = probe_input_claim(reference.as_mut());
            run_lockstep(
                reference.as_mut(),
                optimized.as_mut(),
                claim,
                &synthetic_point(log_t, 211),
            );
            assert_eq!(
                reference.output_claims(&claims).unwrap(),
                optimized.output_claims(&claims).unwrap()
            );
        });
    }

    #[test]
    fn bytecode_cycle_matches_reference_k16() {
        cycle_parity(2, 4);
    }

    #[test]
    fn bytecode_cycle_matches_reference_k256() {
        cycle_parity(3, 8);
    }

    #[test]
    fn bytecode_address_matches_reference_k16() {
        address_parity(2, 4, false);
    }

    #[test]
    fn bytecode_address_matches_reference_k256_committed() {
        address_parity(3, 8, true);
    }
}
