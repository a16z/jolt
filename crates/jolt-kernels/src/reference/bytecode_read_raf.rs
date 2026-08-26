//! The bytecode read+RAF checking (stage 6a) address-phase kernel — a hand
//! kernel: the relation's output `Expr` is the bare staged
//! `BytecodeReadRafAddrClaim` intermediate, hiding the true summand
//!
//! `Σ_k Σ_s γ^s · F_s(k) · Val'_s(k) + γ^{S+2} · E_trace(k) · E_expected(k)`
//!
//! from the naive interpreter (each term is a product of two multilinears).
//! `F_s(k) = Σ_{j: pc(j)=k} eq(r_cycle_s, j)` are the per-stage cycle-eq
//! pushforwards onto the bytecode address domain, `Val'_s` are the per-row
//! stage-value tables (from the verifier's own `read_raf_stage_values`
//! fold) with the RAF address-identity added at stages 1 and 3
//! (`Val'_1 = Val_1 + γ^{S}·Int`, `Val'_3 = Val_3 + γ^{S-1}·Int` — the overall
//! `γ^S`/`γ^{S+1}` RAF weights divided by the stage weights `γ^0`/`γ^2`,
//! `S` the stage count), and the entry term is the product of two one-hots
//! (the trace's first PC, the preprocessing entry index). Each term is
//! quadratic per variable, so the true round polynomial is quadratic, sampled
//! at three points; binding is `LowToHigh` over the `log_K` bytecode address
//! variables.
//!
//! On the packed (lattice) shape the fold extends to nine stages: the four
//! fused-inc consumer stages (`γ^5..8`) discharge the reduced `Inc` claims —
//! each is a read-raf-shaped stage whose pushforward weights every cycle's
//! eq contribution by its fused delta
//! (`F'_s(k) = Σ_{j: pc(j)=k} eq(r_cycle_s, j) · FusedInc(j)`) against the
//! staged store column (`Val_5 = Val_6 = store`, `Val_7 = Val_8 = 1−store`);
//! the RAF/entry weights shift to `γ^9..11`. The kernel adapts to the
//! jolt-claims shape at runtime (the relation carries the four consumer
//! cycle points exactly on that build).
//!
//! The raw `Val_s` tables and the `Int` identity table bind SEPARATELY (the
//! per-round extension is linear, so the split computes field-identical
//! messages to a pre-folded table): committed-program mode stages the raw
//! bound `Val_s` values as `BytecodeValClaim` wire claims, which the folded
//! table cannot produce.
//!
//! Under `field-inline` both phases carry the FR extension (spec:
//! `field-inline-protocol.md`, "Stage 6 Composition") the composed way the
//! Spartan kernels were extended — the jolt symbolic expressions cannot name
//! the FR terms, so the kernels materialize them from the pinned jolt-claims
//! composed helpers (the FR `read_raf_stage_values` row fold under the
//! extended stage-1/4/5 gamma powers), never restating the row formula. The
//! address phase adds three (pushforward, FR row table) legs at the ordinary
//! γ⁰/γ³/γ⁴ stage weights — the stage-1 leg over the ordinary stage-1 cycle
//! binding, the stage-4/5 legs over the FR read-write / val-evaluation cycle
//! sub-points. The cycle phase swaps the naive prover for a composed hand
//! kernel over `C(j) · Π_i BytecodeRa_i(j)`, with every scalar-weighted eq /
//! RAF / entry / FR term pre-folded into the single coefficient multilinear
//! `C` (each term is a scalar times one eq table, so the pre-fold is exact).

#[cfg(not(feature = "field-inline"))]
use std::collections::BTreeMap;

use crate::ProverInputs;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    geometry::bytecode as field_inline_bytecode, FIELD_REGISTERS_LOG_K,
};
use jolt_claims::protocols::jolt::geometry::bytecode::{
    bytecode_ra, read_raf_stage_values, BytecodeReadRafDimensions, BytecodeReadRafStageValueInputs,
    LATTICE_FUSED_INC_STAGES,
};
#[cfg(not(feature = "field-inline"))]
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::bytecode_val_stage_opening;
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use jolt_claims::protocols::jolt::geometry::dimensions::{
    committed_address_chunks, REGISTER_ADDRESS_BITS,
};
use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
#[cfg(not(feature = "field-inline"))]
use jolt_claims::protocols::jolt::{BytecodeReadRafPublic, JoltDerivedId};
use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
#[cfg(not(feature = "field-inline"))]
use jolt_claims::{Source, SymbolicSumcheck};
use jolt_field::JoltField;
use jolt_poly::{
    BindingOrder, IdentityPolynomial, MultilinearEvaluation, Polynomial, UnivariatePoly,
};
use jolt_sumcheck::{ProveRounds, SumcheckError};
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::relations::SumcheckOutputClaims;
use jolt_verifier::stages::relations::{ConcreteSumcheck, SumcheckInputClaims};
use jolt_verifier::stages::stage6a::bytecode_read_raf::{
    BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycleOutputClaims;
use jolt_witness::witnesses::BytecodePc;
use jolt_witness::{collect_bundles, JoltWitnessPlane, WitnessBundle};

#[cfg(not(feature = "field-inline"))]
use super::views::dense_view;
use super::views::{address_fold, eq_table};
#[cfg(not(feature = "field-inline"))]
use crate::NaiveSumcheckProver;
use crate::{
    KernelError, PrepareKernel, ProofSession, ReferenceBackend, SumcheckKernel, SumcheckKernelError,
};

/// The base flag stages of the read-raf fold (the lattice shape appends the
/// four fused-inc consumer stages).
const BASE_STAGES: usize = 5;

/// The per-cycle witness of the bytecode read+RAF address phase: the PC
/// pushforward source (no-ops and unmapped rows land on slot 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
pub struct BytecodeReadRafWitness {
    pub bytecode_pc: BytecodePc,
}

/// One stage's raw-value source: a raw table by index, or its pointwise
/// complement (the fused register legs read `1 − store`; the extension of
/// `1 − f` is `1 − ext(f)`, so no second table binds).
#[derive(Clone, Copy)]
enum StageVal {
    Table(usize),
    Complement(usize),
}

impl<F: JoltField> PrepareKernel<F, BytecodeReadRafAddressPhase<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafAddressPhase<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        // The per-row stage-value tables: the verifier's own fold over the
        // padded bytecode (carrying the lattice store stage as its last
        // element on the packed shape). FR-on, the jolt fold sees the
        // ordinary x-register slots only (the FR-operand slots ride the side
        // table): see `field_inline_bytecode::suppress_field_operand_slots`.
        let program = witness.program_preprocessing();
        let stage_gammas = inputs.challenges.stage_gamma_powers();
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
        // The PC pushforward source: the per-cycle bytecode indices,
        // collected as typed bundles off the witness plane's row source.
        let rows: Vec<BytecodeReadRafWitness> =
            collect_bundles(witness, 1 << relation.dimensions().log_t())?;
        let bytecode_indices: Vec<usize> = rows.iter().map(|row| row.bytecode_pc.0).collect();
        // The packed fused stages' cycle factor: the per-cycle fused deltas,
        // fetched exactly when the relation carries the consumer points.
        let fused_values: Vec<F> = if relation.fused_inc_cycle_points().is_empty() {
            Vec::new()
        } else {
            witness.oracle_table(JoltPolynomialId::Virtual(JoltVirtualPolynomial::FusedInc))?
        };
        Ok(Box::new(BytecodeReadRafAddressKernel::new(
            relation,
            relation.dimensions(),
            stage_values,
            relation.stage_cycle_points(),
            relation.fused_inc_cycle_points(),
            fused_values,
            bytecode_indices,
            relation.entry_bytecode_index(),
            inputs.challenges,
        )?))
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
pub struct BytecodeReadRafAddressKernel<F: JoltField> {
    rounds: usize,
    /// Committed-program mode stages the raw bound `Val_s` wire claims.
    committed_program: bool,
    /// `γ^s` batching weights for the stage products, then `γ^{S+2}` for the
    /// entry product.
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    stage_weights: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    entry_weight: F,
    /// The per-stage `Int` weights inside `Val'_s = Val_s + raf_weight_s·Int`.
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    raf_weights: Vec<F>,
    /// The per-stage cycle-eq pushforwards `F_s` (the fused stages weighted
    /// by the fused deltas).
    pushforwards: Vec<Polynomial<F>>,
    /// The RAW distinct value tables (no RAF fold — see the module doc); the
    /// staged `BytecodeValClaim` wire set on the packed shape includes the
    /// store column the fused stages read.
    values: Vec<Polynomial<F>>,
    /// Each stage's raw-value source over `values`.
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    stage_vals: Vec<StageVal>,
    /// The RAF address identity `Int(k) = k`, bound alongside.
    int_table: Polynomial<F>,
    entry_trace: Polynomial<F>,
    entry_expected: Polynomial<F>,
    /// The FR extension's three (pushforward, row-table) legs.
    #[cfg(feature = "field-inline")]
    field_inline: FieldInlineAddressLegs<F>,
    rounds_bound: usize,
}

/// The address phase's FR extension: three additional
/// `weight · pushforward · row-table` products over the bytecode address
/// domain (see the module doc). Leg order: the stage-1 op-flag leg (over the
/// ordinary stage-1 cycle binding), the stage-4 leg (over the FR read-write
/// cycle sub-point), the stage-5 leg (over the FR val-evaluation cycle
/// sub-point).
#[cfg(feature = "field-inline")]
struct FieldInlineAddressLegs<F: JoltField> {
    /// γ⁰ / γ³ / γ⁴ — each leg rides the same outer stage weight as its
    /// ordinary stage claim.
    weights: [F; 3],
    pushforwards: [Polynomial<F>; 3],
    /// The FR side-table row values under the extended per-stage gamma
    /// powers (the jolt-claims FR `read_raf_stage_values` columns 0/3/4).
    values: [Polynomial<F>; 3],
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

#[cfg(feature = "field-inline")]
impl<F: JoltField> FieldInlineAddressLegs<F> {
    fn bind(&mut self, challenge: F) {
        for table in self.pushforwards.iter_mut().chain(self.values.iter_mut()) {
            table.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
    }

    /// The legs' contribution to one round-message sample at `point`, summed
    /// over pair `y`.
    fn round_term(&self, y: usize, point: F) -> F {
        let ext = |table: &Polynomial<F>| {
            table.sumcheck_round_eval_with_order(y, point, BindingOrder::LowToHigh)
        };
        self.weights
            .iter()
            .zip(self.pushforwards.iter().zip(&self.values))
            .map(|(weight, (pushforward, value))| *weight * ext(pushforward) * ext(value))
            .sum()
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

impl<F: JoltField> BytecodeReadRafAddressKernel<F> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the address phase's full geometry, spelled per source"
    )]
    pub fn new(
        relation: &BytecodeReadRafAddressPhase<F>,
        dimensions: BytecodeReadRafDimensions,
        stage_values: Vec<[F; NUM_BYTECODE_VAL_STAGES]>,
        stage_cycle_points: &[Vec<F>; BASE_STAGES],
        fused_cycle_points: &[Vec<F>],
        fused_values: Vec<F>,
        bytecode_indices: Vec<usize>,
        entry_bytecode_index: usize,
        challenges: &BytecodeReadRafAddressPhaseChallenges<F>,
    ) -> Result<Self, KernelError<F>> {
        // The packed (lattice) shape appends one store val stage and four
        // fused-inc consumer stages; anything else is an unknown shape.
        let (num_stages, lattice) = match (NUM_BYTECODE_VAL_STAGES, fused_cycle_points.len()) {
            (5, 0) => (BASE_STAGES, false),
            (6, LATTICE_FUSED_INC_STAGES) => (BASE_STAGES + LATTICE_FUSED_INC_STAGES, true),
            _ => {
                return Err(KernelError::InvariantViolation {
                    reason:
                        "the bytecode read-raf stage shape matches neither the base five-stage \
                             fold nor the packed nine-stage fold",
                })
            }
        };
        let addresses = 1usize << dimensions.log_k();
        let cycles = 1usize << dimensions.log_t();
        if stage_values.len() != addresses {
            return Err(KernelError::TableSizeMismatch {
                table: "bytecode stage values".to_owned(),
                expected: addresses,
                got: stage_values.len(),
            });
        }
        if bytecode_indices.len() != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: "bytecode cycle indices".to_owned(),
                expected: cycles,
                got: bytecode_indices.len(),
            });
        }
        if lattice && fused_values.len() != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: "fused increment values".to_owned(),
                expected: cycles,
                got: fused_values.len(),
            });
        }
        for point in stage_cycle_points.iter().chain(fused_cycle_points) {
            if point.len() != dimensions.log_t() {
                return Err(KernelError::InvariantViolation {
                    reason: "bytecode stage cycle point has the wrong variable count",
                });
            }
        }
        if entry_bytecode_index >= addresses || bytecode_indices.iter().any(|&pc| pc >= addresses) {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode index outside the padded bytecode domain",
            });
        }

        let gamma = challenges.gamma;
        let mut gamma_powers = vec![F::one(); num_stages + 3];
        for i in 1..gamma_powers.len() {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        // F_s pushforwards: one trace scan per stage; the fused stages weight
        // each cycle's eq contribution by its fused delta.
        let pushforward = |point: &[F], fused: bool| {
            let eq_cycle = eq_table(point);
            let mut table = vec![F::zero(); addresses];
            for (j, &pc) in bytecode_indices.iter().enumerate() {
                if fused {
                    table[pc] += eq_cycle[j] * fused_values[j];
                } else {
                    table[pc] += eq_cycle[j];
                }
            }
            Polynomial::new(table)
        };
        let pushforwards: Vec<Polynomial<F>> = stage_cycle_points
            .iter()
            .map(|point| pushforward(point, false))
            .chain(
                fused_cycle_points
                    .iter()
                    .map(|point| pushforward(point, true)),
            )
            .collect();

        // The FR legs: the side-table row values under the extended gamma
        // powers (the composed jolt-claims fold), each leg over its own cycle
        // binding. Fail-closed on a missing side table or malformed FR
        // opening points.
        #[cfg(feature = "field-inline")]
        let field_inline = {
            let geometry = relation.field_inline_geometry()?;
            let table = &geometry.table;
            if table.rows.len() != addresses {
                return Err(KernelError::TableSizeMismatch {
                    table: "field-inline bytecode side table".to_owned(),
                    expected: addresses,
                    got: table.rows.len(),
                });
            }
            fn split_fr_point<F: JoltField>(
                point: &[F],
                log_t: usize,
            ) -> Result<(&[F], &[F]), KernelError<F>> {
                if point.len() != FIELD_REGISTERS_LOG_K + log_t {
                    return Err(KernelError::InvariantViolation {
                        reason: "FR opening point has the wrong variable count",
                    });
                }
                Ok(point.split_at(FIELD_REGISTERS_LOG_K))
            }
            let (read_write_address, read_write_cycle) =
                split_fr_point(&geometry.read_write_point, dimensions.log_t())?;
            let (val_evaluation_address, val_evaluation_cycle) =
                split_fr_point(&geometry.val_evaluation_point, dimensions.log_t())?;
            let gammas =
                jolt_verifier::stages::field_inline_bytecode::field_inline_stage_gamma_powers(
                    challenges,
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
            FieldInlineAddressLegs {
                weights: [gamma_powers[0], gamma_powers[3], gamma_powers[4]],
                pushforwards: [
                    pushforward(&stage_cycle_points[0], false),
                    pushforward(read_write_cycle, false),
                    pushforward(val_evaluation_cycle, false),
                ],
                values: [column(0), column(3), column(4)],
            }
        };

        // The RAW stage-value tables; the RAF identity `Int(k) = k` binds as
        // its own table with the within-stage weights `γ^S` (stage 1) and
        // `γ^{S-1}` (stage 3) applied at message time.
        let mut raf_weights = vec![F::zero(); num_stages];
        raf_weights[0] = gamma_powers[num_stages];
        raf_weights[2] = gamma_powers[num_stages - 1];
        let values: Vec<Polynomial<F>> = (0..NUM_BYTECODE_VAL_STAGES)
            .map(|s| Polynomial::new(stage_values.iter().map(|row| row[s]).collect()))
            .collect();
        // The fused RAM legs read the staged store column, the register legs
        // its complement.
        let mut stage_vals: Vec<StageVal> = (0..BASE_STAGES).map(StageVal::Table).collect();
        if lattice {
            stage_vals.extend([
                StageVal::Table(BASE_STAGES),
                StageVal::Table(BASE_STAGES),
                StageVal::Complement(BASE_STAGES),
                StageVal::Complement(BASE_STAGES),
            ]);
        }
        let int_table = Polynomial::new((0..addresses).map(|k| F::from_u64(k as u64)).collect());

        let one_hot = |index: usize| {
            let mut table = vec![F::zero(); addresses];
            table[index] = F::one();
            Polynomial::new(table)
        };

        Ok(Self {
            rounds: relation.rounds(),
            committed_program: relation.committed_program(),
            stage_weights: gamma_powers[..num_stages].to_vec(),
            entry_weight: gamma_powers[num_stages + 2],
            raf_weights,
            pushforwards,
            values,
            stage_vals,
            int_table,
            entry_trace: one_hot(bytecode_indices[0]),
            entry_expected: one_hot(entry_bytecode_index),
            #[cfg(feature = "field-inline")]
            field_inline,
            rounds_bound: 0,
        })
    }
}

impl<F: JoltField> BytecodeReadRafAddressKernel<F> {
    fn bind(&mut self, challenge: F) {
        for table in self.pushforwards.iter_mut().chain(self.values.iter_mut()) {
            table.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        self.int_table
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.entry_trace
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.entry_expected
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        #[cfg(feature = "field-inline")]
        self.field_inline.bind(challenge);
        self.rounds_bound += 1;
    }

    /// Stage `s`'s raw value at a fully bound table (`evals()[0]`).
    fn bound_stage_val(&self, stage: usize) -> F {
        match self.stage_vals[stage] {
            StageVal::Table(index) => self.values[index].evals()[0],
            StageVal::Complement(index) => F::one() - self.values[index].evals()[0],
        }
    }
}

impl<F: JoltField> ProveRounds<F> for BytecodeReadRafAddressKernel<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.entry_trace.evals().len() / 2;
        let mut evals = [F::zero(); 3];
        for (c, eval) in evals.iter_mut().enumerate() {
            let point = F::from_u64(c as u64);
            let ext = |table: &Polynomial<F>, y: usize| {
                table.sumcheck_round_eval_with_order(y, point, BindingOrder::LowToHigh)
            };
            let mut sum = F::zero();
            for y in 0..half {
                let int_ext = ext(&self.int_table, y);
                for (s, stage_val) in self.stage_vals.iter().enumerate() {
                    let val_ext = match stage_val {
                        StageVal::Table(index) => ext(&self.values[*index], y),
                        StageVal::Complement(index) => F::one() - ext(&self.values[*index], y),
                    };
                    sum += self.stage_weights[s]
                        * ext(&self.pushforwards[s], y)
                        * (val_ext + self.raf_weights[s] * int_ext);
                }
                sum += self.entry_weight * ext(&self.entry_trace, y) * ext(&self.entry_expected, y);
                #[cfg(feature = "field-inline")]
                {
                    sum += self.field_inline.round_term(y, point);
                }
            }
            *eval = sum;
        }

        let round_sum = evals[0] + evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for BytecodeReadRafAddressKernel<F> {
    type Relation = BytecodeReadRafAddressPhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<BytecodeReadRafAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        if self.rounds_bound != self.num_rounds() {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.num_rounds() - self.rounds_bound,
            });
        }
        let mut intermediate =
            self.entry_weight * self.entry_trace.evals()[0] * self.entry_expected.evals()[0];
        let bound_int = self.int_table.evals()[0];
        for s in 0..self.stage_vals.len() {
            intermediate += self.stage_weights[s]
                * self.pushforwards[s].evals()[0]
                * (self.bound_stage_val(s) + self.raf_weights[s] * bound_int);
        }
        #[cfg(feature = "field-inline")]
        {
            intermediate += self.field_inline.bound_term();
        }
        // Committed mode stages the RAW bound `Val_s` values (the distinct
        // tables — the fused stages dedup through the store column).
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

impl<F: JoltField> PrepareKernel<F, BytecodeReadRafCycle<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafCycle<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafCycle<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let r_address = relation.r_address();
        let stage_cycle_points = relation.stage_cycle_points();
        let entry_bytecode_index = relation.entry_bytecode_index();
        let committed_chunk_bits = relation.committed_chunk_bits();
        // The address-only stage-value fold, off the relation: full mode
        // computed it at construction; committed mode's constants ARE the
        // stage-6a staged raw values.
        let stage_values_at_r_address = relation.stage_values_at_r_address()?;
        let cycles = 1usize << dimensions.log_t();

        let chunks = committed_address_chunks(r_address, committed_chunk_bits);
        if chunks.len() != dimensions.num_committed_ra_polys() {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode address chunk count disagrees with the committed RA count",
            });
        }
        let ra_folds: Vec<Vec<F>> = chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                address_fold(witness, bytecode_ra(index), dimensions.log_t(), chunk)
            })
            .collect::<Result<_, _>>()?;

        let int_at_r_address = IdentityPolynomial::new(r_address.len()).evaluate(r_address);
        let entry_scalar = eq_table(r_address)[entry_bytecode_index];

        // rv64: the naive prover over the anchor committed expression — every
        // stage value a constant table, every eq/RAF/entry public a derived
        // multilinear.
        #[cfg(not(feature = "field-inline"))]
        {
            let mut opening_tables = BTreeMap::new();
            for (index, fold) in ra_folds.into_iter().enumerate() {
                let _ = opening_tables.insert(bytecode_ra(index), Polynomial::new(fold));
            }
            for (stage, value) in stage_values_at_r_address.into_iter().enumerate() {
                let _ = opening_tables.insert(
                    bytecode_val_stage_opening(stage),
                    Polynomial::new(vec![value; cycles]),
                );
            }
            // The packed fused stages carry the `FusedInc` opening as their
            // cycle factor: serve its dense trace column when the relation's
            // expression references it (the base expression never does).
            for term in &relation.symbolic().output_expression::<F>().terms {
                for factor in &term.factors {
                    let Source::Opening(id) = factor else {
                        continue;
                    };
                    if matches!(
                        id.polynomial_id(),
                        JoltPolynomialId::Virtual(JoltVirtualPolynomial::FusedInc)
                    ) && !opening_tables.contains_key(id)
                    {
                        let _ =
                            opening_tables.insert(*id, Polynomial::new(dense_view(witness, *id)?));
                    }
                }
            }

            let scaled_eq = |point: &[F], scalar: F| -> Vec<F> {
                eq_table(point).into_iter().map(|eq| scalar * eq).collect()
            };
            // eq(zero cycle, ·): the cycle-0 boundary selector.
            let mut entry_cycle = vec![F::zero(); cycles];
            entry_cycle[0] = entry_scalar;
            let mut derived_tables = BTreeMap::new();
            for (stage, point) in stage_cycle_points.iter().enumerate() {
                let _ = derived_tables.insert(
                    JoltDerivedId::from(BytecodeReadRafPublic::StageCycleEq(stage)),
                    Polynomial::new(eq_table(point)),
                );
            }
            let _ = derived_tables.insert(
                JoltDerivedId::from(BytecodeReadRafPublic::SpartanOuterRaf),
                Polynomial::new(scaled_eq(&stage_cycle_points[0], int_at_r_address)),
            );
            let _ = derived_tables.insert(
                JoltDerivedId::from(BytecodeReadRafPublic::SpartanShiftRaf),
                Polynomial::new(scaled_eq(&stage_cycle_points[2], int_at_r_address)),
            );
            let _ = derived_tables.insert(
                JoltDerivedId::from(BytecodeReadRafPublic::Entry),
                Polynomial::new(entry_cycle),
            );

            Ok(Box::new(NaiveSumcheckProver::new(
                &inputs,
                opening_tables,
                derived_tables,
                BindingOrder::LowToHigh,
            )?))
        }

        // FR-on: the composed hand kernel over `C(j) · Π_i BytecodeRa_i(j)`
        // (see the module doc) — the anchor expression cannot name the FR
        // terms' distinct cycle-eq factors, so every scalar-weighted term
        // pre-folds into the single coefficient multilinear. The driver's
        // `expected_final_claim` (the full-mode `expected_output`, which
        // composes the FR public stage values) and the round-0 check against
        // the stage-6a intermediate pin it from both ends.
        #[cfg(feature = "field-inline")]
        {
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
            // The FR row values at `r_address`: the composed jolt-claims row
            // fold under the carried extended gamma powers (stages 1/4/5;
            // stages 2/3 gain no FR terms).
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
            let eq_address = eq_table(r_address);
            let mut fr_folds = [F::zero(); 5];
            for (row, eq) in fr_rows.iter().zip(&eq_address) {
                for (fr_fold, value) in fr_folds.iter_mut().zip(row) {
                    *fr_fold += *value * *eq;
                }
            }

            let gamma = inputs.challenges.gamma;
            // γ^0..γ^{S+2}: the S stage weights (5, or 9 on the packed
            // shape), then the RAF outer/shift and entry weights.
            let num_stages = stage_cycle_points.len();
            let mut gamma_powers = vec![F::one(); num_stages + 3];
            for i in 1..gamma_powers.len() {
                gamma_powers[i] = gamma_powers[i - 1] * gamma;
            }
            fn accumulate<F: JoltField>(coefficient: &mut [F], cycle_point: &[F], weight: F) {
                for (slot, eq) in coefficient.iter_mut().zip(eq_table(cycle_point)) {
                    *slot += weight * eq;
                }
            }
            let mut coefficient = vec![F::zero(); cycles];
            for (s, cycle_point) in stage_cycle_points.iter().enumerate().take(BASE_STAGES) {
                // The FR stage-1 leg shares the ordinary stage-1 cycle
                // binding, so its fold merges into the stage-0 weight; the
                // RAF terms ride the stage-1/3 cycle eq tables at γ^S/γ^{S+1}.
                let mut weight = gamma_powers[s] * stage_values_at_r_address[s];
                if s == 0 {
                    weight += fr_folds[0] + gamma_powers[num_stages] * int_at_r_address;
                }
                if s == 2 {
                    weight += gamma_powers[num_stages + 1] * int_at_r_address;
                }
                accumulate(&mut coefficient, cycle_point, weight);
            }
            accumulate(
                &mut coefficient,
                &fold.read_write_cycle,
                gamma_powers[3] * fr_folds[3],
            );
            accumulate(
                &mut coefficient,
                &fold.val_evaluation_cycle,
                gamma_powers[4] * fr_folds[4],
            );
            coefficient[0] += gamma_powers[num_stages + 2] * entry_scalar;

            // The packed shape's four fused-inc consumer stages cannot
            // pre-fold into `C` (`FusedInc(j)` is a second cycle multilinear):
            // they get their own coefficient, the staged store fold for the
            // two RAM legs and its complement for the two register legs
            // (the verifier's `fused_stage_value` resolution).
            #[cfg(feature = "akita")]
            let fused = {
                let store = stage_values_at_r_address[BASE_STAGES];
                let mut fused_coefficient = vec![F::zero(); cycles];
                for (offset, cycle_point) in stage_cycle_points.iter().skip(BASE_STAGES).enumerate()
                {
                    let address_fold = if offset < 2 { store } else { F::one() - store };
                    accumulate(
                        &mut fused_coefficient,
                        cycle_point,
                        gamma_powers[BASE_STAGES + offset] * address_fold,
                    );
                }
                let fused_values: Vec<F> = witness
                    .oracle_table(JoltPolynomialId::Virtual(JoltVirtualPolynomial::FusedInc))?;
                FusedIncCycleLeg {
                    coefficient: Polynomial::new(fused_coefficient),
                    values: Polynomial::new(fused_values),
                }
            };

            Ok(Box::new(ComposedBytecodeReadRafCycleKernel {
                rounds: relation.rounds(),
                degree: relation.degree(),
                coefficient: Polynomial::new(coefficient),
                bytecode_ra: ra_folds.into_iter().map(Polynomial::new).collect(),
                #[cfg(feature = "akita")]
                fused,
                rounds_bound: 0,
            }))
        }
    }
}

/// The FR-on cycle-phase kernel: `Σ_j C(j) · Π_i BytecodeRa_i(j)` with the
/// composed coefficient multilinear `C` (see the module doc and the prepare
/// arm above); the packed shape adds `C_fused(j) · FusedInc(j)` to `C(j)`.
/// Samples the anchor relation's `degree() + 1` points per round — with the
/// coefficients pre-folded the true degree is exactly the anchor degree
/// (`num_ra + 1`, or `num_ra + 2` packed).
#[cfg(feature = "field-inline")]
struct ComposedBytecodeReadRafCycleKernel<F: JoltField> {
    rounds: usize,
    degree: usize,
    coefficient: Polynomial<F>,
    bytecode_ra: Vec<Polynomial<F>>,
    #[cfg(feature = "akita")]
    fused: FusedIncCycleLeg<F>,
    rounds_bound: usize,
}

/// The packed fused-inc leg of the composed cycle kernel: the four consumer
/// stages' scalar-folded coefficient and the `FusedInc` trace column it
/// multiplies (also the source of the lattice `fused_inc` output claim).
#[cfg(all(feature = "field-inline", feature = "akita"))]
struct FusedIncCycleLeg<F: JoltField> {
    coefficient: Polynomial<F>,
    values: Polynomial<F>,
}

// Size arithmetic rather than a derive, like the sibling kernels.
#[cfg(all(feature = "allocative", feature = "field-inline"))]
impl<F: JoltField> allocative::Allocative for ComposedBytecodeReadRafCycleKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("coefficient"),
            self.coefficient.len() * size_of::<F>(),
        );
        visitor.visit_simple(
            allocative::Key::new("bytecode_ra"),
            self.bytecode_ra
                .iter()
                .map(|table| table.len() * size_of::<F>())
                .sum::<usize>(),
        );
        #[cfg(feature = "akita")]
        visitor.visit_simple(
            allocative::Key::new("fused"),
            (self.fused.coefficient.len() + self.fused.values.len()) * size_of::<F>(),
        );
        visitor.exit();
    }
}

#[cfg(feature = "field-inline")]
impl<F: JoltField> ComposedBytecodeReadRafCycleKernel<F> {
    fn bind(&mut self, challenge: F) {
        self.coefficient
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        for table in &mut self.bytecode_ra {
            table.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        #[cfg(feature = "akita")]
        {
            self.fused
                .coefficient
                .bind_with_order(challenge, BindingOrder::LowToHigh);
            self.fused
                .values
                .bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        self.rounds_bound += 1;
    }
}

#[cfg(feature = "field-inline")]
impl<F: JoltField> ProveRounds<F> for ComposedBytecodeReadRafCycleKernel<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.coefficient.evals().len() / 2;
        let order = BindingOrder::LowToHigh;
        let mut evals = Vec::with_capacity(self.degree + 1);
        for sample in 0..=self.degree {
            let point = F::from_u64(sample as u64);
            let sum = (0..half)
                .map(|y| {
                    #[cfg_attr(not(feature = "akita"), expect(unused_mut))]
                    let mut coefficient = self
                        .coefficient
                        .sumcheck_round_eval_with_order(y, point, order);
                    #[cfg(feature = "akita")]
                    {
                        coefficient += self
                            .fused
                            .coefficient
                            .sumcheck_round_eval_with_order(y, point, order)
                            * self
                                .fused
                                .values
                                .sumcheck_round_eval_with_order(y, point, order);
                    }
                    self.bytecode_ra.iter().fold(coefficient, |product, table| {
                        product * table.sumcheck_round_eval_with_order(y, point, order)
                    })
                })
                .sum::<F>();
            evals.push(sum);
        }
        let round_sum = evals[0] + evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

#[cfg(feature = "field-inline")]
impl<F: JoltField> SumcheckKernel<F> for ComposedBytecodeReadRafCycleKernel<F> {
    type Relation = BytecodeReadRafCycle<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, BytecodeReadRafCycle<F>>,
    ) -> Result<SumcheckOutputClaims<F, BytecodeReadRafCycle<F>>, SumcheckKernelError<F>> {
        if self.rounds_bound != self.rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds - self.rounds_bound,
            });
        }
        Ok(BytecodeReadRafCycleOutputClaims {
            bytecode_ra: self
                .bytecode_ra
                .iter()
                .map(|table| table.evals()[0])
                .collect(),
            #[cfg(feature = "akita")]
            fused_inc: self.fused.values.evals()[0],
        })
    }
}
