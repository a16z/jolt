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

use super::instruction_read_raf::InstructionCycleRow;
use super::lazy_ra::{ChunkIndexSource, LazyFoldedRa};
use super::support::{
    bind_all, eq_table, gamma_powers, pair, par_sum_pair_groups, par_sum_pair_groups_reusing,
    round_poly_from_skipped_evals, scaled_eq_table, RoundProgress,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Per-stage cycle-eq pushforwards onto the bytecode address domain. Base and
/// row-weighted stages share one trace walk over the split-eq decomposition.
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
        // FR-on, the jolt fold sees the ordinary x-register slots only (the
        // FR-operand slots ride the side table) — the same mask the
        // reference kernel and the verifier's own fold apply.
        #[cfg(feature = "field-inline")]
        let masked_bytecode =
            jolt_verifier::stages::field_inline_bytecode::suppress_field_operand_slots(
                &program.bytecode.bytecode,
            );
        #[cfg(feature = "field-inline")]
        let bytecode_rows: &[jolt_riscv::JoltInstructionRow] = &masked_bytecode;
        #[cfg(not(feature = "field-inline"))]
        let bytecode_rows = &program.bytecode.bytecode;
        let stage_values = read_raf_stage_values(BytecodeReadRafStageValueInputs {
            bytecode: bytecode_rows,
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
        let rows = InstructionCycleRow::shared(session, witness, cycles)?;
        let push_pc = InstructionCycleRow::bytecode_pc;
        let entry_bytecode_index = relation.entry_bytecode_index();
        if entry_bytecode_index >= addresses || rows.iter().any(|row| push_pc(row) >= addresses) {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode index outside the padded bytecode domain",
            });
        }

        let base_stages = stage_cycle_points.len();
        let num_stages = base_stages + fused_cycle_points.len();
        let gamma_powers = gamma_powers(inputs.challenges.gamma, num_stages + 3);

        // The FR extension's fold geometry: the side-table row values under
        // the extended per-stage gamma powers, each leg over its own cycle
        // binding (see the reference kernel's `FieldInlineAddressLegs`).
        #[cfg(feature = "field-inline")]
        let (fr_values, fr_read_write_cycle, fr_val_evaluation_cycle) = {
            use jolt_claims::protocols::field_inline::geometry::bytecode as field_inline_bytecode;
            use jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K;

            let geometry = relation.field_inline_geometry()?;
            let table = &geometry.table;
            if table.rows.len() != addresses {
                return Err(KernelError::TableSizeMismatch {
                    table: "field-inline bytecode side table".to_owned(),
                    expected: addresses,
                    got: table.rows.len(),
                });
            }
            if geometry.read_write_point.len() != FIELD_REGISTERS_LOG_K + dimensions.log_t()
                || geometry.val_evaluation_point.len() != FIELD_REGISTERS_LOG_K + dimensions.log_t()
            {
                return Err(KernelError::InvariantViolation {
                    reason: "FR opening point has the wrong variable count",
                });
            }
            let (read_write_address, read_write_cycle) =
                geometry.read_write_point.split_at(FIELD_REGISTERS_LOG_K);
            let (val_evaluation_address, val_evaluation_cycle) = geometry
                .val_evaluation_point
                .split_at(FIELD_REGISTERS_LOG_K);
            let gammas =
                jolt_verifier::stages::field_inline_bytecode::field_inline_stage_gamma_powers(
                    inputs.challenges,
                );
            let fr_rows = field_inline_bytecode::read_raf_stage_values(
                field_inline_bytecode::FieldInlineBytecodeReadRafStageValueInputs {
                    bytecode: &table.rows,
                    field_register_read_write_point: read_write_address,
                    field_register_val_evaluation_point: val_evaluation_address,
                    stage1_gammas: &gammas.stage1,
                    stage4_gammas: &gammas.stage4,
                    stage5_gammas: &gammas.stage5,
                },
            );
            let column =
                |s: usize| Polynomial::new(fr_rows.iter().map(|row| row[s]).collect::<Vec<F>>());
            (
                [column(0), column(3), column(4)],
                read_write_cycle.to_vec(),
                val_evaluation_cycle.to_vec(),
            )
        };

        #[cfg(not(feature = "akita"))]
        let row_weight = |_: &InstructionCycleRow| F::one();
        #[cfg(feature = "akita")]
        let row_weight = InstructionCycleRow::fused_inc::<F>;
        // One walk computes every pushforward: the base stages (packed: plus
        // the fused stages) and, FR-on, the two FR cycle sub-points appended
        // as unweighted base stages; the FR stage-1 leg shares the ordinary
        // stage-1 pushforward (identical cycle binding).
        #[cfg(feature = "field-inline")]
        let composed_points: Vec<Vec<F>> = stage_cycle_points
            .iter()
            .cloned()
            .chain([fr_read_write_cycle, fr_val_evaluation_cycle])
            .collect();
        #[cfg(feature = "field-inline")]
        let walk_points: &[Vec<F>] = &composed_points;
        #[cfg(not(feature = "field-inline"))]
        let walk_points: &[Vec<F>] = stage_cycle_points;
        #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
        let mut pushforwards = stage_pushforwards::<F, _>(
            walk_points,
            fused_cycle_points,
            &rows,
            addresses,
            push_pc,
            row_weight,
        );
        #[cfg(feature = "field-inline")]
        let fr_pushforwards: [Polynomial<F>; 3] = {
            let missing = || KernelError::InvariantViolation {
                reason: "FR pushforwards missing from the shared walk",
            };
            // The two FR walks sit between the base and fused blocks (the
            // walk's output order is [base..., weighted...] and the FR points
            // ride the base list) — on the packed shape the fused
            // pushforwards follow them, so the global tail is wrong there.
            let fr_start = stage_cycle_points.len();
            if pushforwards.len() < fr_start + 2 {
                return Err(missing());
            }
            let val_evaluation = Polynomial::new(pushforwards.remove(fr_start + 1));
            let read_write = Polynomial::new(pushforwards.remove(fr_start));
            let stage1 = Polynomial::new(pushforwards.first().cloned().ok_or_else(missing)?);
            [stage1, read_write, val_evaluation]
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
            #[cfg(feature = "field-inline")]
            field_inline: FieldInlineAddressLegs {
                weights: [gamma_powers[0], gamma_powers[3], gamma_powers[4]],
                pushforwards: fr_pushforwards,
                values: fr_values,
            },
        }))
    }
}

/// The address phase's FR extension: three additional
/// `weight · pushforward · row-table` products over the bytecode address
/// domain, mirroring the reference kernel's legs — the stage-1 op-flag leg
/// (over the ordinary stage-1 cycle binding), the stage-4 leg (over the FR
/// read-write cycle sub-point), the stage-5 leg (over the FR val-evaluation
/// cycle sub-point), at the ordinary γ⁰/γ³/γ⁴ stage weights.
#[cfg(feature = "field-inline")]
struct FieldInlineAddressLegs<F: JoltField> {
    weights: [F; 3],
    pushforwards: [Polynomial<F>; 3],
    values: [Polynomial<F>; 3],
}

#[cfg(feature = "field-inline")]
impl<F: JoltField> FieldInlineAddressLegs<F> {
    fn bind(&mut self, challenge: F) {
        bind_all(
            self.pushforwards.iter_mut().chain(self.values.iter_mut()),
            challenge,
        );
    }

    /// The legs' `[t = 0, t = 2]` contributions summed over group `y`.
    #[inline]
    fn group_evals(&self, y: usize) -> [F; 2] {
        let mut out = [F::zero(); 2];
        for (weight, (pushforward, value)) in self
            .weights
            .iter()
            .zip(self.pushforwards.iter().zip(&self.values))
        {
            let (f_lo, f_hi) = pair(pushforward, y);
            let (v_lo, v_hi) = pair(value, y);
            out[0] += *weight * f_lo * v_lo;
            out[1] += *weight * (f_hi + f_hi - f_lo) * (v_hi + v_hi - v_lo);
        }
        out
    }

    /// The legs' contribution to the fully bound intermediate.
    fn bound_term(&self) -> F {
        self.weights
            .iter()
            .zip(self.pushforwards.iter().zip(&self.values))
            .map(|(weight, (pushforward, value))| {
                *weight * pushforward.evals()[0] * value.evals()[0]
            })
            .sum()
    }
}

// Hand impl: the array-of-table fields have no derive-visitable shape.
#[cfg(all(feature = "field-inline", feature = "allocative"))]
impl<F: JoltField> allocative::Allocative for FieldInlineAddressLegs<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        for table in self.pushforwards.iter().chain(&self.values) {
            table.visit(&mut visitor);
        }
        visitor.exit();
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
    #[cfg(feature = "field-inline")]
    field_inline: FieldInlineAddressLegs<F>,
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
        #[cfg(feature = "field-inline")]
        self.field_inline.bind(challenge);
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
        #[cfg(feature = "field-inline")]
        {
            let legs = self.field_inline.group_evals(y);
            out[0] += legs[0];
            out[1] += legs[1];
        }
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
        #[cfg(feature = "field-inline")]
        {
            intermediate += self.field_inline.bound_term();
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
        rows: Arc<Vec<InstructionCycleRow>>,
    },
    Dense(Polynomial<F>),
}

#[cfg(feature = "akita")]
impl<F: JoltField> LazyFusedInc<F> {
    fn new(rows: Arc<Vec<InstructionCycleRow>>) -> Self {
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
        let rows = InstructionCycleRow::shared(session, witness, cycles)?;

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
        let ra = LazyFoldedRa::new(chunk_eqs, BytecodePcChunks { rows, selectors });

        // The combined coefficient table: every non-RA factor of the summand
        // is linear in one cycle table, so
        //   C(j) = Σ_s (γ^s·val_s + raf_s·int_r)·eq_s(j) + γ⁷·entry·[j = 0]
        // with raf_0 = γ⁵·int_r, raf_2 = γ⁶·int_r (SpartanOuterRaf rides the
        // stage-1 cycle point, SpartanShiftRaf the stage-3 one). FR-on the
        // composed FR terms pre-fold in too: the FR stage-1 leg shares the
        // ordinary stage-1 cycle binding (its fold merges into the stage-0
        // weight), and the stage-4/5 legs ride the FR read-write /
        // val-evaluation cycle sub-points at γ³/γ⁴ (the reference kernel's
        // composed pre-fold, term for term).
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

        let eq_address: Vec<F> = eq_table(r_address);
        #[cfg(feature = "field-inline")]
        let (fr_folds, fr_read_write_cycle, fr_val_evaluation_cycle) = {
            use jolt_claims::protocols::field_inline::geometry::bytecode as field_inline_bytecode;
            use jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K;

            let fold = relation.field_inline_fold()?;
            let addresses = 1usize << dimensions.log_k();
            if fold.table.rows.len() != addresses {
                return Err(KernelError::TableSizeMismatch {
                    table: "field-inline bytecode side table".to_owned(),
                    expected: addresses,
                    got: fold.table.rows.len(),
                });
            }
            if fold.read_write_address.len() != FIELD_REGISTERS_LOG_K
                || fold.val_evaluation_address.len() != FIELD_REGISTERS_LOG_K
                || fold.read_write_cycle.len() != dimensions.log_t()
                || fold.val_evaluation_cycle.len() != dimensions.log_t()
            {
                return Err(KernelError::InvariantViolation {
                    reason: "FR bytecode fold points have the wrong variable counts",
                });
            }
            let fr_rows = field_inline_bytecode::read_raf_stage_values(
                field_inline_bytecode::FieldInlineBytecodeReadRafStageValueInputs {
                    bytecode: &fold.table.rows,
                    field_register_read_write_point: &fold.read_write_address,
                    field_register_val_evaluation_point: &fold.val_evaluation_address,
                    stage1_gammas: &fold.gammas.stage1,
                    stage4_gammas: &fold.gammas.stage4,
                    stage5_gammas: &fold.gammas.stage5,
                },
            );
            let mut fr_folds = [F::zero(); 5];
            for (row, eq) in fr_rows.iter().zip(&eq_address) {
                for (fr_fold, value) in fr_folds.iter_mut().zip(row) {
                    *fr_fold += *value * *eq;
                }
            }
            (
                fr_folds,
                fold.read_write_cycle.clone(),
                fold.val_evaluation_cycle.clone(),
            )
        };
        #[cfg(feature = "field-inline")]
        {
            stage_weights[0] += fr_folds[0];
        }

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
        // The FR stage-4/5 legs ride their own cycle sub-points at γ³/γ⁴.
        #[cfg(feature = "field-inline")]
        for (point, weight) in [
            (
                fr_read_write_cycle.as_slice(),
                gamma_powers[3] * fr_folds[3],
            ),
            (
                fr_val_evaluation_cycle.as_slice(),
                gamma_powers[4] * fr_folds[4],
            ),
        ] {
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
        let entry_scalar = eq_address[relation.entry_bytecode_index()];
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
            progress: RoundProgress::new(relation.rounds()),
            degree: relation.degree(),
            ra,
            combined: Polynomial::new(combined),
            #[cfg(feature = "akita")]
            fused_inc,
            #[cfg(feature = "akita")]
            fused_combined,
            output_openings,
        }))
    }
}

/// Lazy-RA index source: chunk `i` of the per-cycle mapped bytecode PC,
/// cold on unmapped cycles, off the shared stage-5 rows.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct BytecodePcChunks {
    rows: Arc<Vec<InstructionCycleRow>>,
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
        Some(self.selectors[i].chunk_usize(self.rows[j].bytecode_pc()))
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
    combined: Polynomial<F>,
    #[cfg(feature = "akita")]
    fused_inc: LazyFusedInc<F>,
    #[cfg(feature = "akita")]
    fused_combined: Polynomial<F>,
    /// The produced `BytecodeRa` opening ids, in `read_raf_output_openings`
    /// order (index-aligned with `ra`).
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    output_openings: Vec<JoltOpeningId>,
}
impl<F: JoltField> CycleKernel<F> {
    fn bind(&mut self, challenge: F) {
        bind_all([&mut self.combined], challenge);
        #[cfg(feature = "akita")]
        bind_all([&mut self.fused_combined], challenge);
        self.ra.bind(challenge);
        #[cfg(feature = "akita")]
        self.fused_inc.bind(challenge);
        self.progress.advance();
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
        let half = self.combined.len() / 2;
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

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for CycleKernel<F> {
    type Relation = BytecodeReadRafCycle<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
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

    use super::super::instruction_read_raf::{SharedInstructionRows, SharedInstructionRowsWeak};
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
            // An all-inactive, well-formed FR geometry (the sample backend's
            // program is FR-free): every FR row value is zero, so the
            // reference kernels' FR legs vanish and byte-parity with the
            // (FR-less) optimized kernels is exact.
            #[cfg(feature = "field-inline")]
            let field_inline_table =
                jolt_verifier::stages::field_inline_bytecode::FieldInlineBytecodeTable {
                    rows: vec![Default::default(); bytecode_len],
                    field_register_log_k:
                        jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K,
                };
            #[cfg(feature = "field-inline")]
            let field_read_write_point = synthetic_point(
                jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K + log_t,
                41,
            );
            #[cfg(feature = "field-inline")]
            let field_val_evaluation_point = synthetic_point(
                jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K + log_t,
                43,
            );

            // ---- Stage 6a: address phase.
            let address_relation = BytecodeReadRafAddressPhase::new(
                dimensions,
                committed_program,
                stage_points.clone(),
                entry_bytecode_index,
            );
            #[cfg(feature = "field-inline")]
            address_relation
                .set_field_inline_geometry(
                    jolt_verifier::stages::stage6a::field_inline::FieldInlineBytecodeReadRafGeometry {
                        table: field_inline_table.clone(),
                        read_write_point: field_read_write_point.clone(),
                        val_evaluation_point: field_val_evaluation_point.clone(),
                    },
                )
                .unwrap();
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
                session.state::<SharedInstructionRows>().is_some()
                    || session.state::<SharedInstructionRowsWeak>().is_some(),
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
                // All-inactive and well-formed: the composed reference cycle
                // kernel folds the FR rows (all zero) at these points, so its
                // messages stay byte-equal to the FR-less optimized kernel's.
                #[cfg(feature = "field-inline")]
                field_inline:
                    jolt_verifier::stages::field_inline_bytecode::FieldInlineBytecodeFold {
                        table: field_inline_table.clone(),
                        read_write_address: field_read_write_point
                            [..jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K]
                            .to_vec(),
                        read_write_cycle: field_read_write_point
                            [jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K..]
                            .to_vec(),
                        val_evaluation_address: field_val_evaluation_point
                            [..jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K]
                            .to_vec(),
                        val_evaluation_cycle: field_val_evaluation_point
                            [jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K..]
                            .to_vec(),
                        gammas:
                            jolt_verifier::stages::field_inline_bytecode::field_inline_stage_gamma_powers(
                                &address_challenges,
                            ),
                    },
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
            // All-inactive and well-formed, mirroring `run_pair`: the geometry
            // attach is required fail-closed under `field-inline`.
            #[cfg(feature = "field-inline")]
            relation
                .set_field_inline_geometry(
                    jolt_verifier::stages::stage6a::field_inline::FieldInlineBytecodeReadRafGeometry {
                        table: jolt_verifier::stages::field_inline_bytecode::FieldInlineBytecodeTable {
                            rows: vec![Default::default(); bytecode_len],
                            field_register_log_k:
                                jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K,
                        },
                        read_write_point: synthetic_point(
                            jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K + log_t,
                            61,
                        ),
                        val_evaluation_point: synthetic_point(
                            jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K + log_t,
                            67,
                        ),
                    },
                )
                .unwrap();
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
            #[cfg(feature = "field-inline")]
            let field_read_write_point = synthetic_point(
                jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K + log_t,
                41,
            );
            #[cfg(feature = "field-inline")]
            let field_val_evaluation_point = synthetic_point(
                jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K + log_t,
                43,
            );
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
                // All-inactive and well-formed, mirroring `run_pair`: the
                // composed reference cycle kernel folds the FR rows (all zero)
                // at these points, so parity with the optimized kernel holds.
                #[cfg(feature = "field-inline")]
                field_inline:
                    jolt_verifier::stages::field_inline_bytecode::FieldInlineBytecodeFold {
                        table: jolt_verifier::stages::field_inline_bytecode::FieldInlineBytecodeTable {
                            rows: vec![Default::default(); bytecode_len],
                            field_register_log_k:
                                jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K,
                        },
                        read_write_address: field_read_write_point
                            [..jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K]
                            .to_vec(),
                        read_write_cycle: field_read_write_point
                            [jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K..]
                            .to_vec(),
                        val_evaluation_address: field_val_evaluation_point
                            [..jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K]
                            .to_vec(),
                        val_evaluation_cycle: field_val_evaluation_point
                            [jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K..]
                            .to_vec(),
                        gammas:
                            jolt_verifier::stages::field_inline_bytecode::field_inline_stage_gamma_powers(
                                &address_challenges,
                            ),
                    },
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
