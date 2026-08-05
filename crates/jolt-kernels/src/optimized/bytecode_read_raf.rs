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
//! In the base protocol the two phases share one compact PC scan through the
//! [`ProofSession`]. The packed phases instead share the wider rows their
//! Booleanity peers already retain, avoiding a second cycle cache.

use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::bytecode::{
    self, read_raf_stage_values, BytecodeReadRafStageValueInputs,
};
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use jolt_claims::protocols::jolt::geometry::dimensions::{
    committed_address_chunks, REGISTER_ADDRESS_BITS,
};
use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_claims::OutputClaims;
#[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
use jolt_field::AkitaField;
use jolt_field::Field;
#[cfg(feature = "akita")]
use jolt_field::{AdditiveAccumulator, RingAccumulator};
#[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
use jolt_poly::eq_index_msb;
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
#[cfg(not(feature = "akita"))]
use jolt_witness::witnesses::{BytecodePc, MappedPc};
use jolt_witness::JoltWitnessPlane;
#[cfg(not(feature = "akita"))]
use jolt_witness::WitnessBundle;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "akita")]
use super::instruction_read_raf::{shared_instruction_rows, InstructionCycleRow};
use super::lazy_ra::{ChunkIndexSource, LazyFoldedRa};
#[cfg(not(feature = "akita"))]
use super::support::collect_rows;
#[cfg(feature = "parallel")]
use super::support::merge_evals;
use super::support::{bind_all, eq_table, pair, round_poly_from_skipped_evals, scaled_eq_table};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Sentinel for a cold cycle (no bytecode mapping) in [`PcRow::mapped_pc`].
#[cfg(not(feature = "akita"))]
const COLD: u32 = u32::MAX;

/// One cycle's packed bytecode PC facts: the pushforward slot (no-ops and
/// unmapped rows land on 0 — the address-phase convention) and the committed
/// one-hot hot index (unmapped rows are cold — the cycle-phase convention).
#[derive(Clone, Copy, Debug)]
#[cfg(not(feature = "akita"))]
pub(crate) struct PcRow {
    push_pc: u32,
    #[cfg(not(feature = "akita"))]
    mapped_pc: u32,
}

/// The session key of the shared per-cycle PC scan.
#[cfg(not(feature = "akita"))]
struct PcRowsKey(Arc<Vec<PcRow>>);

#[cfg(all(feature = "allocative", not(feature = "akita")))]
impl allocative::Allocative for PcRowsKey {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        crate::backend::visit_arc_vec(&mut visitor, allocative::Key::new("rows"), &self.0);
        visitor.exit();
    }
}

#[cfg(not(feature = "akita"))]
#[derive(Clone, Copy, Debug, WitnessBundle)]
struct PcBundle {
    bytecode_pc: BytecodePc,
    #[cfg(not(feature = "akita"))]
    mapped_pc: MappedPc,
}

/// One trace scan per proof, shared by both phases through the session.
#[cfg(not(feature = "akita"))]
fn pc_rows<F: Field>(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
) -> Result<Arc<Vec<PcRow>>, KernelError<F>> {
    if session.state::<PcRowsKey>().is_none() {
        let bundles: Vec<PcBundle> = collect_rows(witness, cycles)?;
        let pack = |bundle: &PcBundle| {
            #[cfg(not(feature = "akita"))]
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
                #[cfg(not(feature = "akita"))]
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
        session.park(PcRowsKey(Arc::new(rows)));
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

/// Per-stage cycle-eq pushforwards onto the bytecode address domain, with all
/// stages sharing one trace walk over the split-eq two-table decomposition.
fn stage_pushforwards<F: Field, R: Sync, const WEIGHTED: bool>(
    stage_cycle_points: &[Vec<F>],
    rows: &[R],
    addresses: usize,
    pc: impl Fn(&R) -> usize + Sync,
    row_weight: impl Fn(&R) -> F + Sync,
) -> Vec<Vec<F>> {
    let log_t = stage_cycle_points[0].len();
    let lo_bits = log_t / 2;
    let hi_bits = log_t - lo_bits;
    let in_len = 1usize << lo_bits;
    let out_len = 1usize << hi_bits;

    // Big-endian points split as eq(r, j) = eq(r[..hi], j_hi) · eq(r[hi..], j_lo)
    // with j = (j_hi << lo_bits) | j_lo.
    let e_hi = stage_cycle_points
        .iter()
        .map(|point| eq_table(&point[..hi_bits]))
        .collect::<Vec<_>>();
    let e_lo = stage_cycle_points
        .iter()
        .map(|point| eq_table(&point[hi_bits..]))
        .collect::<Vec<_>>();

    let block = |range: std::ops::Range<usize>| -> Vec<Vec<F>> {
        let mut partial = (0..stage_cycle_points.len())
            .map(|_| vec![F::zero(); addresses])
            .collect::<Vec<_>>();
        let mut inner = (0..stage_cycle_points.len())
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
                let weight = if WEIGHTED { row_weight(row) } else { F::one() };
                for (stage_inner, stage_lo) in inner.iter_mut().zip(&e_lo) {
                    if WEIGHTED {
                        stage_inner[pc] += stage_lo[j_lo] * weight;
                    } else {
                        stage_inner[pc] += stage_lo[j_lo];
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
                    (0..stage_cycle_points.len())
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
        let fused_cycle_points = relation.fused_inc_cycle_points();
        for point in stage_cycle_points.iter().chain(fused_cycle_points) {
            if point.len() != dimensions.log_t() {
                return Err(KernelError::InvariantViolation {
                    reason: "bytecode stage cycle point has the wrong variable count",
                });
            }
        }
        #[cfg(not(feature = "akita"))]
        let rows = pc_rows(session, witness, cycles)?;
        #[cfg(feature = "akita")]
        let rows = shared_instruction_rows(session, witness, cycles)?;
        #[cfg(not(feature = "akita"))]
        let push_pc = |row: &PcRow| row.push_pc as usize;
        #[cfg(feature = "akita")]
        let push_pc = |row: &InstructionCycleRow| row.mapped_pc().unwrap_or(0);
        let entry_bytecode_index = relation.entry_bytecode_index();
        if entry_bytecode_index >= addresses || rows.iter().any(|row| push_pc(row) >= addresses) {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode index outside the padded bytecode domain",
            });
        }

        let base_stages = stage_cycle_points.len();
        let num_stages = base_stages + fused_cycle_points.len();
        let gamma = inputs.challenges.gamma;
        let mut gamma_powers = vec![F::one(); num_stages + 3];
        for i in 1..gamma_powers.len() {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        let pushforwards = stage_pushforwards::<F, _, false>(
            stage_cycle_points,
            &rows,
            addresses,
            push_pc,
            |_| F::one(),
        );
        #[cfg(feature = "akita")]
        let pushforwards = {
            let mut pushforwards = pushforwards;
            pushforwards.extend(stage_pushforwards::<F, _, true>(
                fused_cycle_points,
                &rows,
                addresses,
                push_pc,
                InstructionCycleRow::fused_inc::<F>,
            ));
            pushforwards
        };
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
            rounds: relation.rounds(),
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
            rounds_bound: 0,
        }))
    }
}

#[derive(Clone, Copy)]
enum StageVal {
    Table(usize),
    Complement(usize),
}

struct AddressKernel<F: Field> {
    rounds: usize,
    committed_program: bool,
    stage_weights: Vec<F>,
    entry_weight: F,
    /// Within-stage RAF `Int` weights, divided by the stage batching weight.
    raf_weights: Vec<F>,
    pushforwards: Vec<Polynomial<F>>,
    /// RAW stage-value tables — the RAF identity binds separately so
    /// committed mode can stage the raw bound `Val_s` wire claims.
    values: Vec<Polynomial<F>>,
    stage_values: Vec<StageVal>,
    int_table: Polynomial<F>,
    entry_trace: Polynomial<F>,
    entry_expected: Polynomial<F>,
    rounds_bound: usize,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for AddressKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::{poly_heap_bytes, polys_heap_bytes, vec_heap_bytes};
        let mut visitor = visitor.enter_self_sized::<Self>();
        for (key, bytes) in [
            ("stage_weights", vec_heap_bytes(&self.stage_weights)),
            ("raf_weights", vec_heap_bytes(&self.raf_weights)),
            ("pushforwards", polys_heap_bytes(&self.pushforwards)),
            ("values", polys_heap_bytes(&self.values)),
            ("stage_values", vec_heap_bytes(&self.stage_values)),
            ("int_table", poly_heap_bytes(&self.int_table)),
            ("entry_trace", poly_heap_bytes(&self.entry_trace)),
            ("entry_expected", poly_heap_bytes(&self.entry_expected)),
        ] {
            visitor.visit_simple(allocative::Key::new(key), bytes);
        }
        visitor.exit();
    }
}

impl<F: Field> AddressKernel<F> {
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
        self.rounds_bound += 1;
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
enum LazyFusedInc<F: Field> {
    Lazy {
        branch_weights: Vec<F>,
        rows: Arc<Vec<InstructionCycleRow>>,
    },
    Dense(Polynomial<F>),
}

#[cfg(feature = "akita")]
impl<F: Field> LazyFusedInc<F> {
    fn new(rows: Arc<Vec<InstructionCycleRow>>) -> Self {
        Self::Lazy {
            branch_weights: vec![F::one()],
            rows,
        }
    }

    #[cfg(feature = "allocative")]
    fn heap_bytes(&self) -> usize {
        match self {
            Self::Lazy { branch_weights, .. } => crate::backend::vec_heap_bytes(branch_weights),
            Self::Dense(polynomial) => crate::backend::poly_heap_bytes(polynomial),
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
                polynomial.bind_with_order(challenge, jolt_poly::BindingOrder::LowToHigh);
                Self::Dense(polynomial)
            }
        };
    }
}

/// Inner-loop algebra used by the stage-6b cycle kernel.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BytecodeCycleAlgebra {
    /// Evaluate every sample point independently.
    Generic,
    /// Specialize the Akita two-RA relation to ten field products per group.
    #[default]
    Q10,
    /// Keep the four terminal Q10 products in wide accumulators until reduction.
    Q10Accum,
}

/// Stage-6b cycle phase: `PrepareKernel` front of the optimized kernel.
#[derive(Clone, Copy, Debug, Default)]
pub struct OptimizedBytecodeReadRafCycle {
    #[cfg(feature = "akita")]
    algebra: BytecodeCycleAlgebra,
}

impl OptimizedBytecodeReadRafCycle {
    pub const fn new(algebra: BytecodeCycleAlgebra) -> Self {
        #[cfg(not(feature = "akita"))]
        let _ = algebra;
        Self {
            #[cfg(feature = "akita")]
            algebra,
        }
    }
}

impl<F: Field> PrepareKernel<F, BytecodeReadRafCycle<F>> for OptimizedBytecodeReadRafCycle {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafCycle<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafCycle<F>>>, KernelError<F>>
    {
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
        #[cfg(not(feature = "akita"))]
        let rows = pc_rows(session, witness, cycles)?;
        #[cfg(feature = "akita")]
        let rows = shared_instruction_rows(session, witness, cycles)?;
        // In base mode this is the compact PC scan's last consumer.
        #[cfg(not(feature = "akita"))]
        let _ = session.take::<PcRowsKey>();

        // ra_i(j) = eq(chunk_i)[chunk_i(pc_j)] — the address fold of the
        // one-hot grid, served lazily off the sparse per-cycle indices for
        // the first three binds instead of `d × T` dense.
        let chunk_eqs: Vec<Vec<F>> = chunks.iter().map(|chunk| eq_table(chunk)).collect();
        let selectors = (0..num_ra)
            .map(|index| {
                RaChunkSelector::new(index, num_ra, relation.committed_chunk_bits())
                    .map_err(KernelError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(feature = "akita")]
        let fused_inc = LazyFusedInc::new(Arc::clone(&rows));
        let ra = LazyFoldedRa::new(chunk_eqs, BytecodePcChunks { rows, selectors });

        // The combined coefficient table: every non-RA factor of the summand
        // is linear in one cycle table, so
        //   C(j) = Σ_s (γ^s·val_s + raf_s·int_r)·eq_s(j)
        //          + γ^(n+2)·entry·[j = 0]
        // where n is the stage count, raf_0 = γ^n·int_r, and
        // raf_2 = γ^(n+1)·int_r.
        let stage_values = relation.stage_values_at_r_address()?;
        let gamma = inputs.challenges.gamma;
        let num_stages = stage_cycle_points.len();
        let base_stages = bytecode::BYTECODE_STAGE_GAMMA_COUNTS.len();
        let mut gamma_powers = vec![F::one(); num_stages + 3];
        for i in 1..gamma_powers.len() {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
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
        let entry_scalar = eq_table(r_address)[relation.entry_bytecode_index()];
        combined[0] += gamma_powers[num_stages + 2] * entry_scalar;

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
                let scaled =
                    scaled_eq_table(&stage_cycle_points[stage], gamma_powers[stage] * value);
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

        let output_openings = bytecode::read_raf_output_openings(dimensions).bytecode_ra;
        if output_openings.len() != num_ra {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode RA output opening count disagrees with the committed RA count",
            });
        }

        Ok(Box::new(CycleKernel {
            rounds: relation.rounds(),
            degree: relation.degree(),
            #[cfg(feature = "akita")]
            algebra: self.algebra,
            tables: CycleTableState::Cpu(CycleCpuTables {
                ra,
                combined: Polynomial::new(combined),
                #[cfg(feature = "akita")]
                fused_inc,
                #[cfg(feature = "akita")]
                fused_combined,
            }),
            output_openings,
            rounds_bound: 0,
        }))
    }
}

#[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
pub(crate) struct MetalBytecodeCycleInputs {
    pub stage_points: Vec<Vec<AkitaField>>,
    pub stage_weights: Vec<AkitaField>,
    pub entry_weight: AkitaField,
    pub ra0: Vec<AkitaField>,
    pub ra1: Vec<AkitaField>,
}

#[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
pub(crate) fn prepare_metal_bytecode_cycle_shell(
    inputs: ProverInputs<'_, AkitaField, BytecodeReadRafCycle<AkitaField>>,
    cpu_tail_algebra: BytecodeCycleAlgebra,
) -> Result<(CycleKernel<AkitaField>, MetalBytecodeCycleInputs), KernelError<AkitaField>> {
    let relation = inputs.relation;
    let dimensions = relation.dimensions();
    let cycles = 1usize << dimensions.log_t();
    let num_ra = dimensions.num_committed_ra_polys();
    if relation.degree() != 4 || num_ra != 2 || relation.committed_chunk_bits() != 8 {
        return Err(KernelError::InvariantViolation {
            reason: "Metal bytecode cycle shell requires degree four and two 8-bit RA chunks",
        });
    }

    let stage_points = relation.stage_cycle_points();
    let base_stages = bytecode::BYTECODE_STAGE_GAMMA_COUNTS.len();
    let num_stages = stage_points.len();
    if base_stages != 5 || num_stages != 9 {
        return Err(KernelError::InvariantViolation {
            reason: "Metal bytecode cycle shell requires five base and four fused stages",
        });
    }
    for point in stage_points {
        if point.len() != dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode stage cycle point has the wrong variable count",
            });
        }
    }

    let stage_values = relation.stage_values_at_r_address()?;
    let gamma = inputs.challenges.gamma;
    let mut gamma_powers = vec![AkitaField::one(); num_stages + 3];
    for index in 1..gamma_powers.len() {
        gamma_powers[index] = gamma_powers[index - 1] * gamma;
    }
    let r_address = relation.r_address();
    let int_at_r_address = IdentityPolynomial::new(r_address.len()).evaluate(r_address);
    let mut stage_weights = (0..base_stages)
        .map(|stage| gamma_powers[stage] * stage_values[stage])
        .collect::<Vec<_>>();
    stage_weights[0] += gamma_powers[num_stages] * int_at_r_address;
    stage_weights[2] += gamma_powers[num_stages + 1] * int_at_r_address;
    let store = stage_values[base_stages];
    stage_weights.extend((base_stages..num_stages).map(|stage| {
        let value = if stage < base_stages + 2 {
            store
        } else {
            AkitaField::one() - store
        };
        gamma_powers[stage] * value
    }));

    let entry_index = u128::try_from(relation.entry_bytecode_index()).map_err(|_| {
        KernelError::InvariantViolation {
            reason: "bytecode entry index exceeds u128",
        }
    })?;
    let entry_weight =
        gamma_powers[num_stages + 2] * eq_index_msb::<AkitaField>(r_address, entry_index);
    let chunks = committed_address_chunks(r_address, relation.committed_chunk_bits());
    if chunks.len() != 2 || chunks.iter().any(|chunk| chunk.len() != 8) {
        return Err(KernelError::InvariantViolation {
            reason: "Metal bytecode cycle RA chunk geometry is not two by eight",
        });
    }
    let selectors = (0..2)
        .map(|index| RaChunkSelector::new(index, 2, 8).map_err(KernelError::from))
        .collect::<Result<Vec<_>, _>>()?;
    if selectors[0].shift() != 8 || selectors[1].shift() != 0 {
        return Err(KernelError::InvariantViolation {
            reason: "Metal bytecode cycle RA chunks are not most-significant first",
        });
    }
    let ra0 = eq_table(&chunks[0]);
    let ra1 = eq_table(&chunks[1]);
    let output_openings = bytecode::read_raf_output_openings(dimensions).bytecode_ra;
    if output_openings.len() != 2 {
        return Err(KernelError::InvariantViolation {
            reason: "Metal bytecode cycle output opening count is not two",
        });
    }

    Ok((
        CycleKernel {
            rounds: relation.rounds(),
            degree: relation.degree(),
            algebra: cpu_tail_algebra,
            tables: CycleTableState::Offloaded { elements: cycles },
            output_openings,
            rounds_bound: 0,
        },
        MetalBytecodeCycleInputs {
            stage_points: stage_points.to_vec(),
            stage_weights,
            entry_weight,
            ra0,
            ra1,
        },
    ))
}

/// Lazy-RA index source: chunk `i` of the per-cycle mapped bytecode PC,
/// cold on unmapped cycles, off the session-shared PC scan.
struct BytecodePcChunks {
    #[cfg(not(feature = "akita"))]
    rows: Arc<Vec<PcRow>>,
    #[cfg(feature = "akita")]
    rows: Arc<Vec<InstructionCycleRow>>,
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
        #[cfg(not(feature = "akita"))]
        {
            if row.mapped_pc == COLD {
                return None;
            }
            Some(self.selectors[i].chunk_usize(row.mapped_pc as usize))
        }
        #[cfg(feature = "akita")]
        {
            row.mapped_pc().map(|pc| self.selectors[i].chunk_usize(pc))
        }
    }

    #[cfg(feature = "allocative")]
    fn heap_bytes(&self) -> usize {
        crate::backend::vec_heap_bytes(&self.selectors)
    }
}

pub(crate) struct CycleKernel<F: Field> {
    rounds: usize,
    degree: usize,
    #[cfg(feature = "akita")]
    algebra: BytecodeCycleAlgebra,
    tables: CycleTableState<F>,
    /// The produced `BytecodeRa` opening ids, in `read_raf_output_openings`
    /// order (index-aligned with the RA tables).
    output_openings: Vec<JoltOpeningId>,
    rounds_bound: usize,
}

struct CycleCpuTables<F: Field> {
    ra: LazyFoldedRa<F, BytecodePcChunks>,
    combined: Polynomial<F>,
    #[cfg(feature = "akita")]
    fused_inc: LazyFusedInc<F>,
    #[cfg(feature = "akita")]
    fused_combined: Polynomial<F>,
}

enum CycleTableState<F: Field> {
    Cpu(CycleCpuTables<F>),
    #[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
    Offloaded {
        elements: usize,
    },
}

#[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
pub(crate) struct BytecodeCycleDenseState {
    pub combined: Vec<jolt_field::AkitaField>,
    pub fused_combined: Vec<jolt_field::AkitaField>,
    pub fused_inc: Vec<jolt_field::AkitaField>,
    pub ra0: Vec<jolt_field::AkitaField>,
    pub ra1: Vec<jolt_field::AkitaField>,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for CycleKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::{poly_heap_bytes, vec_heap_bytes};
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Ok(tables) = self.cpu_tables() {
            visitor.visit_simple(allocative::Key::new("ra"), tables.ra.heap_bytes());
            visitor.visit_simple(
                allocative::Key::new("combined"),
                poly_heap_bytes(&tables.combined),
            );
            #[cfg(feature = "akita")]
            visitor.visit_simple(
                allocative::Key::new("fused_inc"),
                tables.fused_inc.heap_bytes(),
            );
            #[cfg(feature = "akita")]
            visitor.visit_simple(
                allocative::Key::new("fused_combined"),
                poly_heap_bytes(&tables.fused_combined),
            );
        }
        visitor.visit_simple(
            allocative::Key::new("output_openings"),
            vec_heap_bytes(&self.output_openings),
        );
        visitor.exit();
    }
}

impl<F: Field> CycleCpuTables<F> {
    fn bind(&mut self, challenge: F) {
        bind_all([&mut self.combined], challenge);
        #[cfg(feature = "akita")]
        bind_all([&mut self.fused_combined], challenge);
        self.ra.bind(challenge);
        #[cfg(feature = "akita")]
        self.fused_inc.bind(challenge);
    }

    /// The summand's evaluations at `t ∈ {0, 2, 3, .., degree}` summed over
    /// group `y`, written into `acc` (length `degree`); `ra_pairs` is the
    /// caller's per-group `(lo, hi)` scratch (each RA pair is gathered once
    /// per group, not once per sample point).
    #[inline]
    fn accumulate_group(&self, y: usize, acc: &mut [F], ra_pairs: &mut [(F, F)]) {
        let (c_lo, c_hi) = pair(&self.combined, y);
        let c_delta = c_hi - c_lo;
        #[cfg(feature = "akita")]
        let (fused_coefficient_lo, fused_coefficient_hi) = pair(&self.fused_combined, y);
        #[cfg(feature = "akita")]
        let fused_coefficient_delta = fused_coefficient_hi - fused_coefficient_lo;
        #[cfg(feature = "akita")]
        let (fused_inc_lo, fused_inc_hi) = self.fused_inc.lo_hi(y);
        #[cfg(feature = "akita")]
        let fused_inc_delta = fused_inc_hi - fused_inc_lo;
        self.ra.lo_hi_all(y, ra_pairs);
        #[cfg(not(feature = "akita"))]
        let coefficient_at_zero = c_lo;
        #[cfg(feature = "akita")]
        let coefficient_at_zero = c_lo + fused_inc_lo * fused_coefficient_lo;
        acc[0] += ra_pairs
            .iter()
            .fold(coefficient_at_zero, |acc, (lo, _)| acc * *lo);
        for (slot, t) in (2..=acc.len()).enumerate() {
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

    #[cfg(feature = "akita")]
    #[inline]
    fn q10_factors(&self, y: usize, ra_pairs: &mut [(F, F)]) -> ([F; 4], [F; 4]) {
        let (c_0, c_1) = pair(&self.combined, y);
        let (inc_0, inc_1) = self.fused_inc.lo_hi(y);
        let (fused_0, fused_1) = pair(&self.fused_combined, y);

        self.ra.lo_hi_all(y, ra_pairs);
        let ra = quadratic_grid(ra_pairs[0], ra_pairs[1]);
        let coefficient_0 = c_0 + inc_0 * fused_0;
        let coefficient_1 = c_1 + inc_1 * fused_1;
        let coefficient_leading = (inc_1 - inc_0) * (fused_1 - fused_0);
        let coefficient =
            quadratic_grid_from_anchors(coefficient_0, coefficient_1, coefficient_leading);
        (ra, coefficient)
    }

    #[cfg(feature = "akita")]
    #[inline]
    fn accumulate_group_q10(&self, y: usize, acc: &mut [F; 4], ra_pairs: &mut [(F, F)]) {
        let (ra, coefficient) = self.q10_factors(y, ra_pairs);
        for ((acc, ra), coefficient) in acc.iter_mut().zip(ra).zip(coefficient) {
            *acc += ra * coefficient;
        }
    }

    #[cfg(feature = "akita")]
    #[inline]
    fn accumulate_group_q10_wide(
        &self,
        y: usize,
        acc: &mut [F::Accumulator; 4],
        ra_pairs: &mut [(F, F)],
    ) {
        let (ra, coefficient) = self.q10_factors(y, ra_pairs);
        for ((acc, ra), coefficient) in acc.iter_mut().zip(ra).zip(coefficient) {
            acc.fmadd(ra, coefficient);
        }
    }

    #[cfg(feature = "akita")]
    fn q10_wide_evals(&self, half: usize) -> Vec<F> {
        let num_ra = self.ra.num_polys();
        #[cfg(feature = "parallel")]
        let acc = (0..half)
            .into_par_iter()
            .fold(
                || {
                    (
                        [F::Accumulator::default(); 4],
                        vec![(F::zero(), F::zero()); num_ra],
                    )
                },
                |(mut acc, mut ra_pairs), y| {
                    self.accumulate_group_q10_wide(y, &mut acc, &mut ra_pairs);
                    (acc, ra_pairs)
                },
            )
            .map(|(acc, _)| acc)
            .reduce(
                || [F::Accumulator::default(); 4],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        left.merge(right);
                    }
                    left
                },
            );
        #[cfg(not(feature = "parallel"))]
        let acc = {
            let mut acc = [F::Accumulator::default(); 4];
            let mut ra_pairs = vec![(F::zero(), F::zero()); num_ra];
            for y in 0..half {
                self.accumulate_group_q10_wide(y, &mut acc, &mut ra_pairs);
            }
            acc
        };
        acc.map(F::Accumulator::reduce).into()
    }

    #[cfg(feature = "akita")]
    fn target_evals<const Q10: bool>(&self, half: usize) -> Vec<F> {
        let num_ra = self.ra.num_polys();
        #[cfg(feature = "parallel")]
        let acc = (0..half)
            .into_par_iter()
            .fold(
                || ([F::zero(); 4], vec![(F::zero(), F::zero()); num_ra]),
                |(mut acc, mut ra_pairs), y| {
                    if Q10 {
                        self.accumulate_group_q10(y, &mut acc, &mut ra_pairs);
                    } else {
                        self.accumulate_group(y, &mut acc, &mut ra_pairs);
                    }
                    (acc, ra_pairs)
                },
            )
            .map(|(acc, _)| acc)
            .reduce(
                || [F::zero(); 4],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                    left
                },
            );
        #[cfg(not(feature = "parallel"))]
        let acc = {
            let mut acc = [F::zero(); 4];
            let mut ra_pairs = vec![(F::zero(), F::zero()); num_ra];
            for y in 0..half {
                if Q10 {
                    self.accumulate_group_q10(y, &mut acc, &mut ra_pairs);
                } else {
                    self.accumulate_group(y, &mut acc, &mut ra_pairs);
                }
            }
            acc
        };
        acc.into()
    }
}

impl<F: Field> CycleKernel<F> {
    fn cpu_tables(&self) -> Result<&CycleCpuTables<F>, SumcheckError<F>> {
        match &self.tables {
            CycleTableState::Cpu(tables) => Ok(tables),
            #[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
            CycleTableState::Offloaded { .. } => Err(bytecode_cycle_state_error(
                "CPU bytecode cycle operation reached offloaded tables",
            )),
        }
    }

    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        match &mut self.tables {
            CycleTableState::Cpu(tables) => tables.bind(challenge),
            #[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
            CycleTableState::Offloaded { .. } => {
                return Err(bytecode_cycle_state_error(
                    "CPU bytecode cycle bind reached offloaded tables",
                ));
            }
        }
        self.rounds_bound += 1;
        Ok(())
    }
}

#[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
impl CycleKernel<AkitaField> {
    pub(crate) fn metal_message(
        &self,
        evals: [AkitaField; 4],
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if self.degree != 4 {
            return Err(bytecode_cycle_state_error(
                "Metal bytecode cycle message requires degree four",
            ));
        }
        let _ = self.metal_elements()?;
        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    pub(crate) fn metal_commit_bind(
        &mut self,
        device_elements: usize,
    ) -> Result<(), SumcheckError<AkitaField>> {
        let CycleTableState::Offloaded { elements } = &mut self.tables else {
            return Err(bytecode_cycle_state_error(
                "Metal bytecode cycle bind commit requires offloaded tables",
            ));
        };
        if *elements < 2 || device_elements != *elements / 2 || self.rounds_bound >= self.rounds {
            return Err(bytecode_cycle_state_error(
                "Metal bytecode cycle bind commit disagrees with the resident length",
            ));
        }
        *elements = device_elements;
        self.rounds_bound += 1;
        Ok(())
    }

    pub(crate) fn metal_restore_dense(
        &mut self,
        state: BytecodeCycleDenseState,
    ) -> Result<(), SumcheckError<AkitaField>> {
        let expected = self.metal_elements()?;
        for (name, got) in [
            ("combined", state.combined.len()),
            ("fused_combined", state.fused_combined.len()),
            ("fused_inc", state.fused_inc.len()),
            ("ra0", state.ra0.len()),
            ("ra1", state.ra1.len()),
        ] {
            if got != expected {
                return Err(bytecode_cycle_state_error(format!(
                    "Metal bytecode cycle {name} readback has length {got}, expected {expected}"
                )));
            }
        }
        self.tables = CycleTableState::Cpu(CycleCpuTables {
            ra: LazyFoldedRa::Dense(vec![Polynomial::new(state.ra0), Polynomial::new(state.ra1)]),
            combined: Polynomial::new(state.combined),
            fused_inc: LazyFusedInc::Dense(Polynomial::new(state.fused_inc)),
            fused_combined: Polynomial::new(state.fused_combined),
        });
        Ok(())
    }

    pub(crate) fn metal_elements(&self) -> Result<usize, SumcheckError<AkitaField>> {
        match &self.tables {
            CycleTableState::Offloaded { elements } => Ok(*elements),
            CycleTableState::Cpu(_) => Err(bytecode_cycle_state_error(
                "Metal bytecode cycle operation reached restored CPU tables",
            )),
        }
    }

    #[cfg(test)]
    pub(crate) const fn metal_rounds_bound(&self) -> usize {
        self.rounds_bound
    }
}

#[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
fn bytecode_cycle_state_error<F: Field>(message: impl Into<String>) -> SumcheckError<F> {
    SumcheckError::ComputeBackend {
        backend: "bytecode-cycle",
        message: message.into(),
    }
}

#[cfg(feature = "akita")]
#[inline]
fn quadratic_grid<F: Field>(left: (F, F), right: (F, F)) -> [F; 4] {
    let leading = (left.1 - left.0) * (right.1 - right.0);
    let at_0 = left.0 * right.0;
    let at_1 = left.1 * right.1;
    quadratic_grid_from_anchors(at_0, at_1, leading)
}

#[cfg(feature = "akita")]
#[inline]
fn quadratic_grid_from_anchors<F: Field>(at_0: F, at_1: F, leading: F) -> [F; 4] {
    let second_difference = leading + leading;
    let delta_2 = at_1 - at_0 + second_difference;
    let at_2 = at_1 + delta_2;
    let delta_3 = delta_2 + second_difference;
    let at_3 = at_2 + delta_3;
    let at_4 = at_3 + delta_3 + second_difference;
    [at_0, at_2, at_3, at_4]
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
            self.bind(challenge)?;
        }
        let tables = self.cpu_tables()?;
        let half = tables.combined.len() / 2;
        let slots = self.degree;
        let num_ra = tables.ra.num_polys();

        #[cfg(feature = "akita")]
        if num_ra == 2 && slots == 4 {
            let evals = match self.algebra {
                BytecodeCycleAlgebra::Generic => tables.target_evals::<false>(half),
                BytecodeCycleAlgebra::Q10 => tables.target_evals::<true>(half),
                BytecodeCycleAlgebra::Q10Accum => tables.q10_wide_evals(half),
            };
            return Ok(round_poly_from_skipped_evals(&evals, previous_claim));
        }

        #[cfg(feature = "parallel")]
        let evals = (0..half)
            .into_par_iter()
            .fold(
                || (vec![F::zero(); slots], vec![(F::zero(), F::zero()); num_ra]),
                |(mut acc, mut ra_pairs), y| {
                    tables.accumulate_group(y, &mut acc, &mut ra_pairs);
                    (acc, ra_pairs)
                },
            )
            .map(|(acc, _)| acc)
            .reduce(|| vec![F::zero(); slots], merge_evals);
        #[cfg(not(feature = "parallel"))]
        let evals = {
            let mut ra_pairs = vec![(F::zero(), F::zero()); num_ra];
            (0..half).fold(vec![F::zero(); slots], |mut acc, y| {
                tables.accumulate_group(y, &mut acc, &mut ra_pairs);
                acc
            })
        };

        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
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
        #[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
        let tables = match &self.tables {
            CycleTableState::Cpu(tables) => tables,
            CycleTableState::Offloaded { .. } => {
                return Err(SumcheckKernelError::InvariantViolation {
                    reason: "bytecode cycle output requested while tables remain offloaded",
                });
            }
        };
        #[cfg(not(all(feature = "akita", feature = "metal", target_os = "macos")))]
        let CycleTableState::Cpu(tables) = &self.tables;
        let ra = &tables.ra;
        let output_openings = &self.output_openings;
        SumcheckOutputClaims::<F, Self::Relation>::from_opening_values(|id: &JoltOpeningId| {
            #[cfg(feature = "akita")]
            if *id == bytecode::fused_inc_read_raf_opening() {
                return Some(tables.fused_inc.value(0));
            }
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
#[cfg(all(test, not(feature = "akita")))]
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
            let mut optimized = OptimizedBytecodeReadRafCycle::default()
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
    use jolt_field::{AkitaField, Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage6a::bytecode_read_raf::BytecodeStagePoints;
    use jolt_verifier::stages::stage6b::bytecode_read_raf::{
        BytecodeReadRafCommittedCycleInputs, BytecodeReadRafCycleInputs,
        BytecodeReadRafCyclePhaseCommittedChallenges, BytecodeReadRafInputClaims,
        BytecodeReadRafTableFoldInputs,
    };
    use jolt_witness::ProgramSource;

    use super::*;
    use crate::optimized::booleanity::testing::with_booleanity_backend;
    use crate::optimized::harness::{probe_input_claim, run_lockstep, synthetic_point};
    use crate::ReferenceBackend;

    fn akita_point(len: usize, seed: u64) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

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

    fn cycle_parity(log_t: usize, log_k_chunk: u8, committed_program: bool) {
        with_booleanity_backend(log_t, log_k_chunk, |backend, base_dimensions| {
            let program = backend.program_preprocessing();
            let bytecode_len = program.bytecode.bytecode.len();
            if log_k_chunk == 1 {
                assert_eq!(base_dimensions.layout.bytecode(), 2);
            }
            let dimensions = BytecodeReadRafDimensions::new(
                log_t,
                bytecode_len.ilog2() as usize,
                base_dimensions.layout.bytecode(),
            );
            if log_k_chunk == 1 {
                assert_eq!(dimensions.num_committed_ra_polys(), 2);
                assert_eq!(dimensions.num_committed_ra_polys() + 2, 4);
            }
            let stage_cycle_points =
                std::array::from_fn(|stage| akita_point(log_t, 11 + stage as u64));
            let address_challenges = BytecodeReadRafAddressPhaseChallenges {
                gamma: AkitaField::from_u64(3),
                stage1_gamma: AkitaField::from_u64(5),
                stage2_gamma: AkitaField::from_u64(7),
                stage3_gamma: AkitaField::from_u64(11),
                stage4_gamma: AkitaField::from_u64(13),
                stage5_gamma: AkitaField::from_u64(17),
            };
            let stage_gammas = address_challenges.stage_gamma_powers();
            let r_address = akita_point(dimensions.log_k(), 19);
            let relation = if committed_program {
                BytecodeReadRafCycle::committed(BytecodeReadRafCommittedCycleInputs {
                    dimensions,
                    r_address,
                    stage_cycle_points,
                    entry_bytecode_index: 0,
                    committed_chunk_bits: usize::from(log_k_chunk),
                    val_stages: (0..NUM_BYTECODE_VAL_STAGES)
                        .map(|stage| AkitaField::from_u64(101 + stage as u64))
                        .collect(),
                })
            } else {
                let register_read_write_point = akita_point(REGISTER_ADDRESS_BITS, 23);
                let register_val_evaluation_point = akita_point(REGISTER_ADDRESS_BITS, 29);
                BytecodeReadRafCycle::full(BytecodeReadRafCycleInputs {
                    dimensions,
                    r_address,
                    stage_cycle_points,
                    entry_bytecode_index: 0,
                    committed_chunk_bits: usize::from(log_k_chunk),
                    table_fold: Some(BytecodeReadRafTableFoldInputs {
                        bytecode: &program.bytecode.bytecode,
                        register_read_write_point: &register_read_write_point,
                        register_val_evaluation_point: &register_val_evaluation_point,
                        stage_gammas: std::array::from_fn(|stage| stage_gammas[stage].as_slice()),
                    }),
                })
                .unwrap()
            };
            if log_k_chunk == 1 {
                assert_eq!(relation.degree(), 4);
            }
            let challenges = BytecodeReadRafCyclePhaseCommittedChallenges {
                gamma: AkitaField::from_u64(31),
            };
            let claims = BytecodeReadRafInputClaims::<AkitaField>::default();
            let input_points = BytecodeReadRafInputClaims::<Vec<AkitaField>>::default();

            for algebra in [
                BytecodeCycleAlgebra::Generic,
                BytecodeCycleAlgebra::Q10,
                BytecodeCycleAlgebra::Q10Accum,
            ] {
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
                let mut optimized = OptimizedBytecodeReadRafCycle::new(algebra)
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
                    &akita_point(log_t, 211),
                );
                assert_eq!(
                    reference.output_claims(&claims).unwrap(),
                    optimized.output_claims(&claims).unwrap(),
                    "output mismatch for {algebra:?}"
                );
            }
        });
    }

    #[test]
    fn quadratic_grid_matches_direct_evaluation() {
        let left = (Fr::from_u64(7), Fr::from_u64(29));
        let right = (Fr::from_u64(11), Fr::from_u64(41));
        let actual = quadratic_grid(left, right);
        for (actual, t) in actual.into_iter().zip([0, 2, 3, 4]) {
            let t = Fr::from_u64(t);
            let left_at_t = left.0 + t * (left.1 - left.0);
            let right_at_t = right.0 + t * (right.1 - right.0);
            assert_eq!(actual, left_at_t * right_at_t);
        }
    }

    #[test]
    fn q10_coefficient_grid_matches_direct_evaluation() {
        let c = (Fr::from_u64(5), Fr::from_u64(17));
        let inc = (Fr::from_u64(7), Fr::from_u64(29));
        let fused = (Fr::from_u64(11), Fr::from_u64(41));
        let actual = quadratic_grid_from_anchors(
            c.0 + inc.0 * fused.0,
            c.1 + inc.1 * fused.1,
            (inc.1 - inc.0) * (fused.1 - fused.0),
        );
        for (actual, t) in actual.into_iter().zip([0, 2, 3, 4]) {
            let t = Fr::from_u64(t);
            let affine = |pair: (Fr, Fr)| pair.0 + t * (pair.1 - pair.0);
            assert_eq!(actual, affine(c) + affine(inc) * affine(fused));
        }
    }

    #[test]
    fn bytecode_cycle_matches_reference_k16() {
        cycle_parity(2, 4, false);
    }

    #[test]
    fn bytecode_cycle_matches_reference_k256() {
        cycle_parity(3, 8, false);
    }

    #[test]
    fn bytecode_cycle_q10_matches_reference_two_chunks() {
        cycle_parity(5, 1, false);
    }

    #[test]
    fn bytecode_cycle_q10_matches_committed_reference_two_chunks() {
        cycle_parity(5, 1, true);
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
