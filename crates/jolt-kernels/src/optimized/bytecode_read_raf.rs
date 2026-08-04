//! Optimized bytecode read+RAF kernels: the stage-6a address phase and the
//! stage-6b cycle phase, byte-parity twins of the reference kernels in
//! [`crate::reference::bytecode_read_raf`].
//!
//! Ported legacy techniques (`jolt-prover-legacy/src/zkvm/bytecode/read_raf_checking.rs`):
//!
//! - **Split-eq two-table pushforwards** (address phase): the per-stage
//!   `F_s(k) = Σ_{j: pc(j)=k} eq(r_cycle_s, j)` tables are accumulated as
//!   `Σ_{j_hi} E_hi_s[j_hi] · (Σ_{j_lo: pc=k} E_lo_s[j_lo])` — the inner sums
//!   are additions only and the five stages share one trace walk, so the eq
//!   tables cost `O(√T)` each instead of `O(T)` and the `O(T)` walk pays one
//!   PC lookup per cycle for all stages
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
//! - **Eval-at-1 recovery** and **rayon walks**, both phases (see the module
//!   docs on [`crate::optimized`]).
//!
//! The two phases share one per-proof trace scan through the
//! [`ProofSession`]: whichever phase prepares first parks the packed
//! per-cycle PC rows; the other reclaims them (`state_or_insert_with` keyed
//! by the private [`PcRowsKey`] type — the modular equivalent of legacy's
//! shared `Arc<Vec<Cycle>>`).

use std::sync::{Arc, Mutex};

use jolt_claims::protocols::jolt::geometry::bytecode::{
    self, read_raf_stage_values, BytecodeReadRafStageValueInputs,
};
use jolt_claims::protocols::jolt::geometry::dimensions::{
    committed_address_chunks, REGISTER_ADDRESS_BITS,
};
use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_claims::OutputClaims;
use jolt_field::Field;
use jolt_poly::{IdentityPolynomial, MultilinearEvaluation, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage6a::bytecode_read_raf::{
    BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_witness::witnesses::{BytecodePc, MappedPc, RaChunkSelector};
use jolt_witness::{JoltWitnessPlane, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::lazy_ra::{ChunkIndexSource, LazyFoldedRa};
use super::lifetime_trace::LifetimeTag;
#[cfg(feature = "parallel")]
use super::support::merge_evals;
use super::support::{
    bind_all, collect_rows, eq_table, pair, round_poly_from_skipped_evals, scaled_eq_table,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Sentinel for a cold cycle (no bytecode mapping) in [`PcRow::mapped_pc`].
const COLD: u32 = u32::MAX;

/// One cycle's packed bytecode PC facts: the pushforward slot (no-ops and
/// unmapped rows land on 0 — the address-phase convention) and the committed
/// one-hot hot index (unmapped rows are cold — the cycle-phase convention).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PcRow {
    pub(crate) push_pc: u32,
    pub(crate) mapped_pc: u32,
}

impl PcRow {
    /// The row from the trace record's lanes. On any trace the record builds
    /// from, `Pc` extraction succeeded for every row, so `MappedPc` is
    /// `Some(pc)` everywhere and `BytecodePc` is `pc` gated on the no-op
    /// flag — the packed values (and the u32-range guards) match
    /// [`pc_rows`]'s bundle pack exactly.
    pub(crate) fn from_lanes<F: Field>(pc: u64, is_noop: bool) -> Result<Self, KernelError<F>> {
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
struct PcRowsKey(Arc<PcRows>);

/// The scan behind [`PcRowsKey`] — a `Vec` plus the lifetime tag that logs
/// the last-`Arc`-drop site under `JOLT_LIFETIME_TRACE=1`.
pub(crate) struct PcRows {
    rows: Vec<PcRow>,
    _lifetime: LifetimeTag,
}

impl PcRows {
    fn new(rows: Vec<PcRow>) -> Self {
        let bytes = rows.len() * size_of::<PcRow>();
        Self {
            rows,
            _lifetime: LifetimeTag::new("PcRows", bytes),
        }
    }
}

impl std::ops::Deref for PcRows {
    type Target = Vec<PcRow>;

    fn deref(&self) -> &Self::Target {
        &self.rows
    }
}

/// Park a lane-packed PC scan (the trace record's walk co-produces it), so
/// [`pc_rows`] serves it without a second trace pass.
pub(crate) fn park_pc_rows(session: &mut ProofSession, rows: Vec<PcRow>) {
    session.park(PcRowsKey(Arc::new(PcRows::new(rows))));
}

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct PcBundle {
    bytecode_pc: BytecodePc,
    mapped_pc: MappedPc,
}

/// One trace scan per proof, shared by both phases through the session.
pub(crate) fn pc_rows<F: Field>(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
) -> Result<Arc<PcRows>, KernelError<F>> {
    if session.state::<PcRowsKey>().is_none() {
        let bundles: Vec<PcBundle> = collect_rows(witness, cycles)?;
        let pack = |bundle: &PcBundle| {
            let mapped = match bundle.mapped_pc.0 {
                Some(pc) if pc as u32 as usize == pc && pc as u32 != COLD => pc as u32,
                Some(_) => {
                    return Err(KernelError::InvariantViolation {
                        reason: "bytecode PC exceeds the packed u32 range",
                    })
                }
                None => COLD,
            };
            if bundle.bytecode_pc.0 as u32 as usize != bundle.bytecode_pc.0 {
                return Err(KernelError::InvariantViolation {
                    reason: "bytecode PC exceeds the packed u32 range",
                });
            }
            Ok(PcRow {
                push_pc: bundle.bytecode_pc.0 as u32,
                mapped_pc: mapped,
            })
        };
        #[cfg(feature = "parallel")]
        let rows = bundles
            .par_iter()
            .map(pack)
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(not(feature = "parallel"))]
        let rows = bundles.iter().map(pack).collect::<Result<Vec<_>, _>>()?;
        session.park(PcRowsKey(Arc::new(PcRows::new(rows))));
    }
    let rows = session
        .state::<PcRowsKey>()
        .map(|key| Arc::clone(&key.0))
        .ok_or(KernelError::InvariantViolation {
            reason: "bytecode PC rows vanished from the session",
        })?;
    if rows.len() != cycles {
        return Err(KernelError::TableSizeMismatch {
            table: "bytecode cycle PC rows".to_owned(),
            expected: cycles,
            got: rows.len(),
        });
    }
    Ok(rows)
}

/// Per-stage cycle-eq pushforwards onto the bytecode address domain for any
/// point subset, all stages in one trace walk over the split-eq two-table
/// decomposition. Each stage's table is accumulated in its own lanes, so a
/// subset walk produces the same field values as the five-stage walk — the
/// stage-1..4 tables build in the background during stage 5 and only the
/// stage-5 table's walk stays on the 6a critical path.
fn stage_pushforwards_for<F: Field>(
    stage_cycle_points: &[Vec<F>],
    rows: &[PcRow],
    addresses: usize,
) -> Vec<Vec<F>> {
    let stages = stage_cycle_points.len();
    let log_t = stage_cycle_points[0].len();
    let lo_bits = log_t / 2;
    let hi_bits = log_t - lo_bits;
    let in_len = 1usize << lo_bits;
    let out_len = 1usize << hi_bits;

    // Big-endian points split as eq(r, j) = eq(r[..hi], j_hi) · eq(r[hi..], j_lo)
    // with j = (j_hi << lo_bits) | j_lo.
    let e_hi: Vec<Vec<F>> = stage_cycle_points
        .iter()
        .map(|point| eq_table(&point[..hi_bits]))
        .collect();
    let e_lo: Vec<Vec<F>> = stage_cycle_points
        .iter()
        .map(|point| eq_table(&point[hi_bits..]))
        .collect();

    let block = |range: std::ops::Range<usize>| -> Vec<Vec<F>> {
        let mut partial: Vec<Vec<F>> = (0..stages).map(|_| vec![F::zero(); addresses]).collect();
        let mut inner: Vec<Vec<F>> = (0..stages).map(|_| vec![F::zero(); addresses]).collect();
        let mut touched: Vec<usize> = Vec::with_capacity(in_len);
        for j_hi in range {
            for &k in &touched {
                for stage_inner in &mut inner {
                    stage_inner[k] = F::zero();
                }
            }
            touched.clear();
            let base = j_hi << lo_bits;
            for j_lo in 0..in_len {
                let pc = rows[base + j_lo].push_pc as usize;
                if inner[0][pc].is_zero() {
                    touched.push(pc);
                }
                for (stage_inner, stage_lo) in inner.iter_mut().zip(&e_lo) {
                    stage_inner[pc] += stage_lo[j_lo];
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
                || (0..stages).map(|_| vec![F::zero(); addresses]).collect(),
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

/// Session carry: the stage-1..4 pushforward tables, built ahead of stage 6a
/// on a dedicated thread. The points are stored for validation — any mismatch
/// (or a panicked worker, e.g. an out-of-domain PC that the prepare's own
/// validation would reject anyway) falls back to the inline five-stage walk,
/// which produces bit-identical values.
struct PrebuiltBytecodePushforwards<F> {
    points: [Vec<F>; 4],
    addresses: usize,
    handle: std::thread::JoinHandle<Vec<Vec<F>>>,
}

/// The background pool width for [`spawn_bytecode_stage_pushforwards`].
const BYTECODE_BACKGROUND_THREADS: usize = 4;

/// Spawn the stage-1..4 bytecode pushforward walk on a dedicated capped
/// thread pool and park the join handle in the session. Call once the
/// stage-4 output points exist (the stage-5 point does not yet); the 6a
/// address-phase prepare reclaims the four tables and walks only the
/// stage-5 point inline.
pub fn spawn_bytecode_stage_pushforwards<F: Field>(
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
    let rows = pc_rows(session, witness, 1usize << log_t)?;
    let points = early_points.clone();
    let handle = std::thread::spawn(move || {
        let _token = super::BACKGROUND_BUILD_TOKEN
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let build = || stage_pushforwards_for::<F>(&early_points, &rows, addresses);
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

impl<F: Field> PrepareKernel<F, BytecodeReadRafAddressPhase<F>>
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
        for point in stage_cycle_points {
            if point.len() != dimensions.log_t() {
                return Err(KernelError::InvariantViolation {
                    reason: "bytecode stage cycle point has the wrong variable count",
                });
            }
        }
        let rows = pc_rows(session, witness, cycles)?;
        let entry_bytecode_index = relation.entry_bytecode_index();
        if entry_bytecode_index >= addresses
            || rows.iter().any(|row| row.push_pc as usize >= addresses)
        {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode index outside the padded bytecode domain",
            });
        }

        let gamma = inputs.challenges.gamma;
        let mut gamma_powers = [F::one(); 8];
        for i in 1..8 {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        // Reclaim the background-built stage-1..4 tables when they match this
        // relation's points; walk only the stage-5 point inline. Otherwise
        // (no spawn, stale points, panicked worker) run the five-stage walk —
        // the same per-stage lanes in either case, so the values are
        // identical.
        let prebuilt = session
            .take::<PrebuiltBytecodePushforwards<F>>()
            .filter(|carry| {
                carry.addresses == addresses && carry.points[..] == stage_cycle_points[..4]
            })
            .and_then(|carry| carry.handle.join().ok());
        let tables: Vec<Vec<F>> = if let Some(mut early) = prebuilt {
            early.extend(stage_pushforwards_for(
                std::slice::from_ref(&stage_cycle_points[4]),
                &rows,
                addresses,
            ));
            early
        } else {
            stage_pushforwards_for(stage_cycle_points.as_slice(), &rows, addresses)
        };
        let tables: [Vec<F>; 5] =
            tables
                .try_into()
                .map_err(|_| KernelError::InvariantViolation {
                    reason: "bytecode pushforward table count drifted",
                })?;
        let pushforwards = tables.map(Polynomial::new);
        let values: [Polynomial<F>; 5] = std::array::from_fn(|s| {
            Polynomial::new(stage_values.iter().map(|row| row[s]).collect())
        });
        let int_table = Polynomial::new((0..addresses).map(|k| F::from_u64(k as u64)).collect());
        let one_hot = |index: usize| {
            let mut table = vec![F::zero(); addresses];
            table[index] = F::one();
            Polynomial::new(table)
        };

        Ok(Box::new(AddressKernel {
            rounds: relation.rounds(),
            committed_program: relation.committed_program(),
            stage_weights: std::array::from_fn(|s| gamma_powers[s]),
            entry_weight: gamma_powers[7],
            raf_weights: [
                gamma_powers[5],
                F::zero(),
                gamma_powers[4],
                F::zero(),
                F::zero(),
            ],
            pushforwards,
            values,
            int_table,
            entry_trace: one_hot(rows[0].push_pc as usize),
            entry_expected: one_hot(entry_bytecode_index),
            rounds_bound: 0,
        }))
    }
}

struct AddressKernel<F: Field> {
    rounds: usize,
    committed_program: bool,
    stage_weights: [F; 5],
    entry_weight: F,
    /// Within-stage RAF `Int` weights: the overall γ⁵/γ⁶ RAF weights divided
    /// by the γ⁰/γ² stage weights (legacy `raf_int_weight`).
    raf_weights: [F; 5],
    pushforwards: [Polynomial<F>; 5],
    /// RAW stage-value tables — the RAF identity binds separately so
    /// committed mode can stage the raw bound `Val_s` wire claims.
    values: [Polynomial<F>; 5],
    int_table: Polynomial<F>,
    entry_trace: Polynomial<F>,
    entry_expected: Polynomial<F>,
    rounds_bound: usize,
}

impl<F: Field> AddressKernel<F> {
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
        self.rounds_bound += 1;
    }

    /// The summand's evaluations at `t ∈ {0, 2}` summed over group `y`.
    #[inline]
    fn group_evals(&self, y: usize) -> [F; 2] {
        let (int_lo, int_hi) = pair(&self.int_table, y);
        let int_delta = int_hi - int_lo;
        let int_at_2 = int_hi + int_delta;
        let mut out = [F::zero(); 2];
        for s in 0..5 {
            let (f_lo, f_hi) = pair(&self.pushforwards[s], y);
            let (v_lo, v_hi) = pair(&self.values[s], y);
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

impl<F: Field> ProveRounds<F> for AddressKernel<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
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

        #[cfg(feature = "parallel")]
        let evals = (0..half)
            .into_par_iter()
            .fold(
                || vec![F::zero(); 2],
                |mut acc, y| {
                    let group = self.group_evals(y);
                    acc[0] += group[0];
                    acc[1] += group[1];
                    acc
                },
            )
            .reduce(|| vec![F::zero(); 2], merge_evals);
        #[cfg(not(feature = "parallel"))]
        let evals = (0..half).fold(vec![F::zero(); 2], |mut acc, y| {
            let group = self.group_evals(y);
            acc[0] += group[0];
            acc[1] += group[1];
            acc
        });

        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: Field> SumcheckKernel<F> for AddressKernel<F> {
    type Relation = BytecodeReadRafAddressPhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<BytecodeReadRafAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        if self.rounds_bound != self.rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds - self.rounds_bound,
            });
        }
        let mut intermediate =
            self.entry_weight * self.entry_trace.evals()[0] * self.entry_expected.evals()[0];
        let bound_int = self.int_table.evals()[0];
        for s in 0..5 {
            intermediate += self.stage_weights[s]
                * self.pushforwards[s].evals()[0]
                * (self.values[s].evals()[0] + self.raf_weights[s] * bound_int);
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

/// Stage-6b cycle phase: `PrepareKernel` front of the optimized kernel.
pub struct OptimizedBytecodeReadRafCycle;

impl<F: Field> PrepareKernel<F, BytecodeReadRafCycle<F>> for OptimizedBytecodeReadRafCycle {
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
    pub(crate) stage_cycle_points: &'a [Vec<F>; 5],
    pub(crate) stage_weights: [F; 5],
    pub(crate) entry_term: F,
    pub(crate) selector_shifts: Vec<u32>,
    pub(crate) committed_chunk_bits: usize,
    pub(crate) degree: usize,
}

/// A device driver plus its fail-closed combined-table recovery channel.
pub(crate) struct BytecodeCycleDevice<F: Field> {
    pub(crate) driver: Box<dyn super::lazy_ra::LazyRaDevice<F>>,
    pub(crate) combined_recovery: Arc<Mutex<Option<Vec<F>>>>,
}

pub(crate) fn prepare_bytecode_read_raf_cycle<F: Field>(
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
    let rows = pc_rows(session, witness, cycles)?;
    let _ = session.take::<PcRowsKey>();

    let chunk_eqs: Vec<Vec<F>> = chunks.iter().map(|chunk| eq_table(chunk)).collect();
    let selectors = (0..num_ra)
        .map(|index| {
            RaChunkSelector::new(index, num_ra, relation.committed_chunk_bits())
                .map_err(KernelError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let stage_values = relation.stage_values_at_r_address()?;
    let gamma = inputs.challenges.gamma;
    let mut gamma_powers = [F::one(); 8];
    for i in 1..8 {
        gamma_powers[i] = gamma_powers[i - 1] * gamma;
    }
    let int_at_r_address = IdentityPolynomial::new(r_address.len()).evaluate(r_address);
    let mut stage_weights: [F; 5] = std::array::from_fn(|s| gamma_powers[s] * stage_values[s]);
    stage_weights[0] += gamma_powers[5] * int_at_r_address;
    stage_weights[2] += gamma_powers[6] * int_at_r_address;
    for point in stage_cycle_points {
        if point.len() != dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode stage cycle point has the wrong variable count",
            });
        }
    }
    let entry_term = gamma_powers[7] * eq_table(r_address)[relation.entry_bytecode_index()];

    let device = driver_factory(BytecodeCycleDeviceInputs {
        rows: &rows,
        stage_cycle_points,
        stage_weights,
        entry_term,
        selector_shifts: selectors
            .iter()
            .map(|selector| selector.shift() as u32)
            .collect(),
        committed_chunk_bits: relation.committed_chunk_bits(),
        degree: relation.degree(),
    });

    let (driver, combined_recovery, combined) = if let Some(device) = device {
        (Some(device.driver), Some(device.combined_recovery), None)
    } else {
        let mut combined = vec![F::zero(); cycles];
        for (point, weight) in stage_cycle_points.iter().zip(stage_weights) {
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
    let ra = LazyFoldedRa::new_with_driver(chunk_eqs, BytecodePcChunks { rows, selectors }, driver);

    let output_openings = bytecode::read_raf_output_openings(dimensions).bytecode_ra;
    if output_openings.len() != num_ra {
        return Err(KernelError::InvariantViolation {
            reason: "bytecode RA output opening count disagrees with the committed RA count",
        });
    }

    Ok(Box::new(CycleKernel {
        rounds: relation.rounds(),
        degree: relation.degree(),
        ra,
        combined,
        combined_recovery,
        output_openings,
        rounds_bound: 0,
    }))
}

/// Lazy-RA index source: chunk `i` of the per-cycle mapped bytecode PC,
/// cold on unmapped cycles, off the session-shared PC scan.
struct BytecodePcChunks {
    rows: Arc<PcRows>,
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
        let row = &self.rows[j];
        if row.mapped_pc == COLD {
            return None;
        }
        Some(self.selectors[i].chunk_usize(row.mapped_pc as usize))
    }
}

struct CycleKernel<F: Field> {
    rounds: usize,
    degree: usize,
    ra: LazyFoldedRa<F, BytecodePcChunks>,
    combined: Option<Polynomial<F>>,
    combined_recovery: Option<Arc<Mutex<Option<Vec<F>>>>>,
    /// The produced `BytecodeRa` opening ids, in `read_raf_output_openings`
    /// order (index-aligned with `ra`).
    output_openings: Vec<JoltOpeningId>,
    rounds_bound: usize,
}

impl<F: Field> CycleKernel<F> {
    fn bind(&mut self, challenge: F) {
        if let Some(combined) = &mut self.combined {
            bind_all([combined], challenge);
        }
        self.ra.bind(challenge);
        self.recover_combined();
        self.rounds_bound += 1;
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
        for (i, slot) in ra_pairs.iter_mut().enumerate() {
            *slot = self.ra.lo_hi(i, y);
        }
        acc[0] += ra_pairs.iter().fold(c_lo, |acc, (lo, _)| acc * *lo);
        for (slot, t) in (2..=self.degree).enumerate() {
            let t_value = F::from_u64(t as u64);
            acc[slot + 1] += ra_pairs
                .iter()
                .fold(c_lo + t_value * c_delta, |acc, (lo, hi)| {
                    acc * (*lo + t_value * (*hi - *lo))
                });
        }
    }
}

impl<F: Field> ProveRounds<F> for CycleKernel<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
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

        #[cfg(feature = "parallel")]
        let evals = (0..half)
            .into_par_iter()
            .fold(
                || (vec![F::zero(); slots], vec![(F::zero(), F::zero()); num_ra]),
                |(mut acc, mut ra_pairs), y| {
                    self.accumulate_group(y, &mut acc, &mut ra_pairs);
                    (acc, ra_pairs)
                },
            )
            .map(|(acc, _)| acc)
            .reduce(|| vec![F::zero(); slots], merge_evals);
        #[cfg(not(feature = "parallel"))]
        let evals = {
            let mut ra_pairs = vec![(F::zero(), F::zero()); num_ra];
            (0..half).fold(vec![F::zero(); slots], |mut acc, y| {
                self.accumulate_group(y, &mut acc, &mut ra_pairs);
                acc
            })
        };

        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: Field> SumcheckKernel<F> for CycleKernel<F> {
    type Relation = BytecodeReadRafCycle<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        if self.rounds_bound != self.rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds - self.rounds_bound,
            });
        }
        self.ra.ensure_host();
        let ra = &self.ra;
        let output_openings = &self.output_openings;
        SumcheckOutputClaims::<F, Self::Relation>::from_opening_values(|id: &JoltOpeningId| {
            output_openings
                .iter()
                .position(|opening| opening == id)
                .map(|index| ra.value(index, 0))
        })
        .map_err(SumcheckKernelError::from)
    }
}

/// Byte-parity of both phases against the reference kernels, run as a PAIR
/// through one shared [`ProofSession`] so the parked per-cycle PC scan flows
/// the way production stages would exercise it.
///
/// Fixture honesty: `with_sample_backend` is the only witness plane
/// constructible without a `jolt-program` dependency. At its scale the
/// bytecode decomposition has a single committed chunk (`d = 1`), so the
/// cycle kernel's multi-factor `Π_i ra_i` loop runs with one factor; the
/// degree-`d+1` sampling, cold-cycle zeroing, entry/RAF fusion, and both
/// pushforward paths are exercised for real.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage6a::bytecode_read_raf::{
        BytecodeReadRafAddressPhaseInputClaims, BytecodeStagePoints,
    };
    use jolt_verifier::stages::stage6b::bytecode_read_raf::{
        BytecodeReadRafCycleInputs, BytecodeReadRafCyclePhaseCommittedChallenges,
        BytecodeReadRafInputClaims, BytecodeReadRafTableFoldInputs,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::{JoltWitnessOracle, ProgramSource};

    use super::*;
    use crate::optimized::harness::{
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
                session.state::<PcRowsKey>().is_some(),
                "the address phase must park the shared PC scan"
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
            // PC scan is reclaimed, mirroring the production 6a→6b flow) and
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
