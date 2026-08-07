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

use std::collections::BTreeMap;

use crate::ProverInputs;
use jolt_claims::protocols::jolt::geometry::bytecode::{
    bytecode_ra, read_raf_stage_values, BytecodeReadRafDimensions, BytecodeReadRafStageValueInputs,
    LATTICE_FUSED_INC_STAGES,
};
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::{
    bytecode_val_stage_opening, NUM_BYTECODE_VAL_STAGES,
};
use jolt_claims::protocols::jolt::geometry::dimensions::{
    committed_address_chunks, REGISTER_ADDRESS_BITS,
};
use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
use jolt_claims::protocols::jolt::{
    BytecodeReadRafPublic, JoltDerivedId, JoltPolynomialId, JoltVirtualPolynomial,
};
use jolt_claims::{Source, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::{
    BindingOrder, IdentityPolynomial, MultilinearEvaluation, Polynomial, UnivariatePoly,
};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{ConcreteSumcheck, SumcheckInputClaims};
use jolt_verifier::stages::stage6a::bytecode_read_raf::{
    BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_witness::witnesses::BytecodePc;
use jolt_witness::{collect_bundles, JoltWitnessPlane, WitnessBundle};

use super::views::{address_fold, dense_view, eq_table};
use crate::{
    KernelError, NaiveSumcheckProver, PrepareKernel, ProofSession, ReferenceBackend,
    SumcheckKernel, SumcheckKernelError,
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

impl<F: Field> PrepareKernel<F, BytecodeReadRafAddressPhase<F>> for ReferenceBackend {
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
        // element on the packed shape).
        let program = witness.program_preprocessing();
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

pub struct BytecodeReadRafAddressKernel<F: Field> {
    rounds: usize,
    /// Committed-program mode stages the raw bound `Val_s` wire claims.
    committed_program: bool,
    /// `γ^s` batching weights for the stage products, then `γ^{S+2}` for the
    /// entry product.
    stage_weights: Vec<F>,
    entry_weight: F,
    /// The per-stage `Int` weights inside `Val'_s = Val_s + raf_weight_s·Int`.
    raf_weights: Vec<F>,
    /// The per-stage cycle-eq pushforwards `F_s` (the fused stages weighted
    /// by the fused deltas).
    pushforwards: Vec<Polynomial<F>>,
    /// The RAW distinct value tables (no RAF fold — see the module doc); the
    /// staged `BytecodeValClaim` wire set on the packed shape includes the
    /// store column the fused stages read.
    values: Vec<Polynomial<F>>,
    /// Each stage's raw-value source over `values`.
    stage_vals: Vec<StageVal>,
    /// The RAF address identity `Int(k) = k`, bound alongside.
    int_table: Polynomial<F>,
    entry_trace: Polynomial<F>,
    entry_expected: Polynomial<F>,
    rounds_bound: usize,
}

// Size arithmetic rather than a derive, so `F` stays unbounded; `Polynomial`
// sizing is by `len()`, exact at the mid-stage snapshot.
#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for BytecodeReadRafAddressKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::poly_heap_bytes;
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("pushforwards"),
            self.pushforwards.iter().map(poly_heap_bytes).sum::<usize>(),
        );
        visitor.visit_simple(
            allocative::Key::new("values"),
            self.values.iter().map(poly_heap_bytes).sum::<usize>(),
        );
        visitor.visit_simple(
            allocative::Key::new("int_table"),
            poly_heap_bytes(&self.int_table),
        );
        visitor.visit_simple(
            allocative::Key::new("entry_trace"),
            poly_heap_bytes(&self.entry_trace),
        );
        visitor.visit_simple(
            allocative::Key::new("entry_expected"),
            poly_heap_bytes(&self.entry_expected),
        );
        visitor.exit();
    }
}

impl<F: Field> BytecodeReadRafAddressKernel<F> {
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
            rounds_bound: 0,
        })
    }
}

impl<F: Field> BytecodeReadRafAddressKernel<F> {
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

impl<F: Field> ProveRounds<F> for BytecodeReadRafAddressKernel<F> {
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

impl<F: Field> SumcheckKernel<F> for BytecodeReadRafAddressKernel<F> {
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

impl<F: Field> PrepareKernel<F, BytecodeReadRafCycle<F>> for ReferenceBackend {
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
        let mut opening_tables = BTreeMap::new();
        for (index, chunk) in chunks.iter().enumerate() {
            let _ = opening_tables.insert(
                bytecode_ra(index),
                Polynomial::new(address_fold(
                    witness,
                    bytecode_ra(index),
                    dimensions.log_t(),
                    chunk,
                )?),
            );
        }
        for (stage, value) in stage_values_at_r_address.into_iter().enumerate() {
            let _ = opening_tables.insert(
                bytecode_val_stage_opening(stage),
                Polynomial::new(vec![value; cycles]),
            );
        }
        // The packed fused stages carry the `FusedInc` opening as their cycle
        // factor: serve its dense trace column when the relation's expression
        // references it (the base expression never does).
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
                    let _ = opening_tables.insert(*id, Polynomial::new(dense_view(witness, *id)?));
                }
            }
        }

        let int_at_r_address = IdentityPolynomial::new(r_address.len()).evaluate(r_address);
        let entry_scalar = eq_table(r_address)[entry_bytecode_index];
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
}
