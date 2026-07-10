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
//! fold) with the RAF address-identity folded into stages 1 and 3
//! (`Val'_1 = Val_1 + γ⁵·Int`, `Val'_3 = Val_3 + γ⁴·Int` — the overall γ⁵/γ⁶
//! RAF weights divided by the stage weights γ⁰/γ²), and the entry term is the
//! product of two one-hots (the trace's first PC, the preprocessing entry
//! index). Each term is quadratic per variable, so the true round polynomial
//! is quadratic, sampled at three points; binding is `LowToHigh` over the
//! `log_K` bytecode address variables.

use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::stage6a::bytecode_read_raf::{
    BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseOutputClaims,
};

use crate::views::eq_table;
use crate::{KernelError, ProofSession, ProveSumcheck, ReferenceBackend};

/// The stage-6a bytecode read+RAF address-phase slot. The typed relation data
/// is the per-row stage-value table (the verifier's `read_raf_stage_values`
/// output over the padded bytecode), the five upstream stage cycle points,
/// the per-cycle bytecode indices (the PC pushforward source), and the
/// preprocessing entry index.
pub trait BytecodeReadRafAddressProver<F: Field> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the relation's construction data"
    )]
    fn prepare(
        &self,
        session: &mut ProofSession,
        dimensions: BytecodeReadRafDimensions,
        stage_values: Vec<[F; 5]>,
        stage_cycle_points: &[Vec<F>; 5],
        bytecode_indices: Vec<usize>,
        entry_bytecode_index: usize,
        challenges: &BytecodeReadRafAddressPhaseChallenges<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = BytecodeReadRafAddressPhase<F>>>, KernelError<F>>;
}

impl<F: Field> BytecodeReadRafAddressProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        dimensions: BytecodeReadRafDimensions,
        stage_values: Vec<[F; 5]>,
        stage_cycle_points: &[Vec<F>; 5],
        bytecode_indices: Vec<usize>,
        entry_bytecode_index: usize,
        challenges: &BytecodeReadRafAddressPhaseChallenges<F>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = BytecodeReadRafAddressPhase<F>>>, KernelError<F>>
    {
        Ok(Box::new(BytecodeReadRafAddressKernel::new(
            dimensions,
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
    /// `γ^{s}` batching weights for the five stage products, then `γ⁷` for the
    /// entry product.
    stage_weights: [F; 5],
    entry_weight: F,
    /// The per-stage cycle-eq pushforwards `F_s`.
    pushforwards: [Polynomial<F>; 5],
    /// The per-stage value tables with the RAF identity folded in.
    values: [Polynomial<F>; 5],
    entry_trace: Polynomial<F>,
    entry_expected: Polynomial<F>,
    rounds_bound: usize,
}

impl<F: Field> BytecodeReadRafAddressKernel<F> {
    pub fn new(
        dimensions: BytecodeReadRafDimensions,
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
                return Err(KernelError::Unsupported {
                    reason: "bytecode stage cycle point has the wrong variable count",
                });
            }
        }
        if entry_bytecode_index >= addresses || bytecode_indices.iter().any(|&pc| pc >= addresses) {
            return Err(KernelError::Unsupported {
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

        // Stage-value tables, with the RAF identity `Int(k) = k` folded into
        // stages 1 and 3 at the within-stage weights.
        let raf_weights = [
            gamma_powers[5],
            F::zero(),
            gamma_powers[4],
            F::zero(),
            F::zero(),
        ];
        let values = std::array::from_fn(|s| {
            Polynomial::new(
                stage_values
                    .iter()
                    .enumerate()
                    .map(|(k, row)| row[s] + raf_weights[s] * F::from_u64(k as u64))
                    .collect(),
            )
        });

        let one_hot = |index: usize| {
            let mut table = vec![F::zero(); addresses];
            table[index] = F::one();
            Polynomial::new(table)
        };

        Ok(Self {
            relation: BytecodeReadRafAddressPhase::new(dimensions, 0),
            stage_weights: std::array::from_fn(|s| gamma_powers[s]),
            entry_weight: gamma_powers[7],
            pushforwards,
            values,
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
                for s in 0..5 {
                    sum += self.stage_weights[s]
                        * ext(&self.pushforwards[s], y)
                        * ext(&self.values[s], y);
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
        for s in 0..5 {
            intermediate +=
                self.stage_weights[s] * self.pushforwards[s].evals()[0] * self.values[s].evals()[0];
        }
        Ok(BytecodeReadRafAddressPhaseOutputClaims {
            intermediate,
            val_stages: Vec::new(),
        })
    }
}
