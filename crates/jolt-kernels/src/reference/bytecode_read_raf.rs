//! The bytecode read+RAF checking (stage 6a) address-phase kernel — a hand
//! kernel: the relation's output `Expr` is the bare staged
//! `BytecodeReadRafAddrClaim` intermediate, hiding the true summand
//!
//! `Σ_k Σ_s γ^s · F_s(k) · Val'_s(k) + γ⁷ · E_trace(k) · E_expected(k)`
//!
//! from the naive interpreter (each term is a product of two multilinears).
//! `F_s(k) = Σ_{j: pc(j)=k} eq(r_cycle_s, j)` are the per-stage cycle-eq
//! pushforwards onto the bytecode address domain, `Val'_s` are the per-row
//! stage-value tables (from the verifier's own `read_raf_stage_values`
//! fold) with the RAF address-identity added at stages 1 and 3
//! (`Val'_1 = Val_1 + γ⁵·Int`, `Val'_3 = Val_3 + γ⁴·Int` — the overall γ⁵/γ⁶
//! RAF weights divided by the stage weights γ⁰/γ²), and the entry term is the
//! product of two one-hots (the trace's first PC, the preprocessing entry
//! index). Each term is quadratic per variable, so the true round polynomial
//! is quadratic, sampled at three points; binding is `LowToHigh` over the
//! `log_K` bytecode address variables.
//!
//! The raw `Val_s` tables and the `Int` identity table bind SEPARATELY (the
//! per-round extension is linear, so the split computes field-identical
//! messages to a pre-folded table): committed-program mode stages the five
//! raw bound `Val_s` values as `BytecodeValStage` wire claims, which the
//! folded table cannot produce.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::bytecode::{bytecode_ra, BytecodeReadRafDimensions};
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::bytecode_val_stage_opening;
use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::relations::bytecode::{
    BytecodeReadRafAddressPhaseChallenges, BytecodeReadRafCyclePhaseCommittedChallenges,
};
use jolt_claims::protocols::jolt::{BytecodeReadRafPublic, JoltDerivedId};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::{
    BindingOrder, IdentityPolynomial, MultilinearEvaluation, Polynomial, UnivariatePoly,
};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6a::bytecode_read_raf::{
    BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage6b::bytecode_read_raf::{
    BytecodeReadRafCycle, BytecodeReadRafCycleInputs,
};
use jolt_witness::JoltWitnessOracle;

use super::views::{address_fold, eq_table};
use crate::bytecode_read_raf::{
    BytecodeReadRafAddressProver, BytecodeReadRafCycleProver, BytecodeReadRafWitness,
};
use crate::{KernelError, NaiveSumcheckProver, ProofSession, ProveSumcheck, ReferenceBackend};

impl<F: Field> BytecodeReadRafAddressProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        dimensions: BytecodeReadRafDimensions,
        committed_program: bool,
        stage_values: Vec<[F; 5]>,
        stage_cycle_points: &[Vec<F>; 5],
        rows: Vec<BytecodeReadRafWitness>,
        entry_bytecode_index: usize,
        challenges: &BytecodeReadRafAddressPhaseChallenges<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = BytecodeReadRafAddressPhase<F>>>, KernelError<F>>
    {
        let bytecode_indices = rows.iter().map(|row| row.bytecode_pc.0).collect();
        Ok(Box::new(BytecodeReadRafAddressKernel::new(
            dimensions,
            committed_program,
            stage_values,
            stage_cycle_points,
            bytecode_indices,
            entry_bytecode_index,
            challenges,
        )?))
    }
}

pub struct BytecodeReadRafAddressKernel<F: Field> {
    relation: BytecodeReadRafAddressPhase<F>,
    /// Committed-program mode stages the five raw bound `Val_s` wire claims.
    committed_program: bool,
    /// `γ^{s}` batching weights for the five stage products, then `γ⁷` for the
    /// entry product.
    stage_weights: [F; 5],
    entry_weight: F,
    /// The per-stage `Int` weights inside `Val'_s = Val_s + raf_weight_s·Int`.
    raf_weights: [F; 5],
    /// The per-stage cycle-eq pushforwards `F_s`.
    pushforwards: [Polynomial<F>; 5],
    /// The RAW per-stage value tables (no RAF fold — see the module doc).
    values: [Polynomial<F>; 5],
    /// The RAF address identity `Int(k) = k`, bound alongside.
    int_table: Polynomial<F>,
    entry_trace: Polynomial<F>,
    entry_expected: Polynomial<F>,
    rounds_bound: usize,
}

impl<F: Field> BytecodeReadRafAddressKernel<F> {
    pub fn new(
        dimensions: BytecodeReadRafDimensions,
        committed_program: bool,
        stage_values: Vec<[F; 5]>,
        stage_cycle_points: &[Vec<F>; 5],
        bytecode_indices: Vec<usize>,
        entry_bytecode_index: usize,
        challenges: &BytecodeReadRafAddressPhaseChallenges<F>,
    ) -> Result<Self, KernelError<F>> {
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
        for point in stage_cycle_points {
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
        let mut gamma_powers = [F::one(); 8];
        for i in 1..8 {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        // F_s pushforwards: one trace scan per stage.
        let pushforwards = std::array::from_fn(|s| {
            let eq_cycle = eq_table(&stage_cycle_points[s]);
            let mut table = vec![F::zero(); addresses];
            for (j, &pc) in bytecode_indices.iter().enumerate() {
                table[pc] += eq_cycle[j];
            }
            Polynomial::new(table)
        });

        // The RAW stage-value tables; the RAF identity `Int(k) = k` binds as
        // its own table with the within-stage weights γ⁵ (stage 1) and γ⁴
        // (stage 3) applied at message time.
        let raf_weights = [
            gamma_powers[5],
            F::zero(),
            gamma_powers[4],
            F::zero(),
            F::zero(),
        ];
        let values = std::array::from_fn(|s| {
            Polynomial::new(stage_values.iter().map(|row| row[s]).collect())
        });
        let int_table = Polynomial::new((0..addresses).map(|k| F::from_u64(k as u64)).collect());

        let one_hot = |index: usize| {
            let mut table = vec![F::zero(); addresses];
            table[index] = F::one();
            Polynomial::new(table)
        };

        Ok(Self {
            relation: BytecodeReadRafAddressPhase::new(dimensions, committed_program),
            committed_program,
            stage_weights: std::array::from_fn(|s| gamma_powers[s]),
            entry_weight: gamma_powers[7],
            raf_weights,
            pushforwards,
            values,
            int_table,
            entry_trace: one_hot(bytecode_indices[0]),
            entry_expected: one_hot(entry_bytecode_index),
            rounds_bound: 0,
        })
    }
}

impl<F: Field> ProveRounds<F> for BytecodeReadRafAddressKernel<F> {
    fn num_rounds(&self) -> usize {
        self.relation.symbolic().rounds()
    }

    fn compute_message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
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
                for s in 0..5 {
                    sum += self.stage_weights[s]
                        * ext(&self.pushforwards[s], y)
                        * (ext(&self.values[s], y) + self.raf_weights[s] * int_ext);
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

    fn ingest_challenge(&mut self, challenge: F, _round: usize) -> Result<(), SumcheckError<F>> {
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
        Ok(())
    }
}

impl<F: Field> ProveSumcheck<F> for BytecodeReadRafAddressKernel<F> {
    type Relation = BytecodeReadRafAddressPhase<F>;

    fn relation(&self) -> &BytecodeReadRafAddressPhase<F> {
        &self.relation
    }

    fn output_claims(
        &mut self,
    ) -> Result<BytecodeReadRafAddressPhaseOutputClaims<F>, KernelError<F>> {
        if self.rounds_bound != self.num_rounds() {
            return Err(KernelError::NotFullyBound {
                remaining: self.num_rounds() - self.rounds_bound,
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
        // Committed mode stages the five RAW bound `Val_s` values.
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

impl<F: Field> BytecodeReadRafCycleProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        dimensions: BytecodeReadRafDimensions,
        r_address: &[F],
        stage_cycle_points: &[Vec<F>; 5],
        entry_bytecode_index: usize,
        committed_chunk_bits: usize,
        stage_values_at_r_address: [F; 5],
        challenges: &BytecodeReadRafCyclePhaseCommittedChallenges<F>,
        witness: &dyn JoltWitnessOracle<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = BytecodeReadRafCycle<F>>>, KernelError<F>> {
        let cycles = 1usize << dimensions.log_t();
        // The table fold feeds only `expected_output`, which the kernel's
        // relation copy never runs (the recipe's own batch instance does).
        let relation = BytecodeReadRafCycle::full(BytecodeReadRafCycleInputs {
            dimensions,
            r_address: r_address.to_vec(),
            stage_cycle_points: stage_cycle_points.clone(),
            entry_bytecode_index,
            committed_chunk_bits,
            table_fold: None,
        })?;

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
            relation,
            challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
