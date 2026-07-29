//! Optimized instruction claim-reduction (stage 2) kernel.
//!
//! The summand is `eq(τ_low, j) · (o₁ + γ·o₂ + γ²·o₃ + γ³·o₄ + γ⁴·o₅)(j)`
//! over the five instruction-lookup operand tables. The reference tier
//! interprets the relation expression through the naive prover (per-point
//! `BTreeMap` leaf resolution, a materialized and bound `T`-sized eq table).
//! This kernel keeps the five opening tables (their individual bound values
//! are the output claims) but:
//!
//! - folds the γ-combination once per index pair before the round-point
//!   interpolation (exact by linearity of binding), and
//! - factors the eq weight out via `GruenSplitEqPolynomial` — the round
//!   message is `s(t) = ℓ(t) · Σ_y E_out·E_in · combo(t, y)` at the naive
//!   prover's `t ∈ {0, 1, 2}` sample points, and no eq table is ever
//!   materialized or bound.
//!
//! The bound eq factor is pinned to the verifier's scalar path by
//! [`validate_derived_tables`](crate::SumcheckKernel::validate_derived_tables)
//! (Gruen scalar vs `derive_output_term(EqSpartan)`), exactly as the naive
//! tier's materialized table is.

use jolt_claims::protocols::jolt::geometry::claim_reductions::instruction::{
    left_instruction_input_reduced, left_lookup_operand_reduced, lookup_output_reduced,
    right_instruction_input_reduced, right_lookup_operand_reduced,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::InstructionClaimReductionOutputClaims;
use jolt_claims::protocols::jolt::{InstructionClaimReductionPublic, JoltDerivedId};
use jolt_field::Field;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_witness::JoltWitnessPlane;

use crate::reference::views::dense_view;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// The five reduced tables, in output-claim declaration order.
const NUM_TABLES: usize = 5;

/// Optimized [`PrepareKernel`] implementor for the
/// `instruction_claim_reduction` slot.
pub struct OptimizedInstructionClaimReduction;

impl<F: Field> PrepareKernel<F, InstructionClaimReduction<F>>
    for OptimizedInstructionClaimReduction
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionClaimReduction<F>>>, KernelError<F>>
    {
        let ids = [
            lookup_output_reduced(),
            left_lookup_operand_reduced(),
            right_lookup_operand_reduced(),
            left_instruction_input_reduced(),
            right_instruction_input_reduced(),
        ];
        let mut tables = Vec::with_capacity(NUM_TABLES);
        for id in ids {
            tables.push(Polynomial::new(dense_view(witness, id)?));
        }
        Ok(Box::new(OptimizedInstructionClaimReductionKernel::new(
            inputs.relation.tau_low(),
            tables,
            inputs.challenges.gamma,
        )?))
    }
}

pub struct OptimizedInstructionClaimReductionKernel<F: Field> {
    log_t: usize,
    gamma_powers: [F; NUM_TABLES],
    /// `[lookup_output, left/right lookup operand, left/right instruction
    /// input]` — output-claim declaration order.
    tables: Vec<Polynomial<F>>,
    gruen: GruenSplitEqPolynomial<F>,
    bind_scratch: Vec<F>,
    rounds_bound: usize,
}

impl<F: Field> OptimizedInstructionClaimReductionKernel<F> {
    pub fn new(
        tau_low: &[F],
        tables: Vec<Polynomial<F>>,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        let log_t = tau_low.len();
        if tables.len() != NUM_TABLES {
            return Err(KernelError::InvariantViolation {
                reason: "instruction claim reduction expects five reduced tables",
            });
        }
        for table in &tables {
            if table.len() != 1 << log_t {
                return Err(KernelError::TableSizeMismatch {
                    table: "instruction claim-reduction operand table".to_owned(),
                    expected: 1 << log_t,
                    got: table.len(),
                });
            }
        }
        let gamma_sqr = gamma * gamma;
        Ok(Self {
            log_t,
            gamma_powers: [
                F::one(),
                gamma,
                gamma_sqr,
                gamma_sqr * gamma,
                gamma_sqr * gamma_sqr,
            ],
            tables,
            gruen: GruenSplitEqPolynomial::new(tau_low, BindingOrder::LowToHigh),
            bind_scratch: Vec::new(),
            rounds_bound: 0,
        })
    }

    /// `s(t) = ℓ(t) · Σ_y E(y) · combo(t, y)` at `t ∈ {0, 1, 2}`, with the
    /// γ-combination folded before the point interpolation (exact by
    /// linearity of binding).
    fn message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        const POINTS: usize = 3;
        let q_evals = self.gruen.par_fold_out_in(
            || [F::zero(); POINTS],
            |acc, row, _x_in, e_in| {
                let mut lo = F::zero();
                let mut step = F::zero();
                for (table, gamma_power) in self.tables.iter().zip(&self.gamma_powers) {
                    let evals = table.evals();
                    let table_lo = evals[2 * row];
                    lo += *gamma_power * table_lo;
                    step += *gamma_power * (evals[2 * row + 1] - table_lo);
                }
                let mut eval = lo;
                for value in acc.iter_mut() {
                    *value += e_in * eval;
                    eval += step;
                }
            },
            |_x_out, e_out, mut acc| {
                for value in &mut acc {
                    *value *= e_out;
                }
                acc
            },
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(&b) {
                    *a += *b;
                }
                a
            },
        );

        let (l_at_0, l_at_1) = self.gruen.current_linear_evals();
        let l_step = l_at_1 - l_at_0;
        let mut l_eval = l_at_0;
        let mut evals = [F::zero(); POINTS];
        for (eval, q) in evals.iter_mut().zip(&q_evals) {
            *eval = l_eval * *q;
            l_eval += l_step;
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

    fn bind(&mut self, challenge: F) {
        self.gruen.bind(challenge);
        for table in &mut self.tables {
            table.bind_low_to_high_reusing_scratch(challenge, &mut self.bind_scratch);
        }
        self.rounds_bound += 1;
    }
}

impl<F: Field> ProveRounds<F> for OptimizedInstructionClaimReductionKernel<F> {
    fn num_rounds(&self) -> usize {
        self.log_t
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
        self.message(round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: Field> SumcheckKernel<F> for OptimizedInstructionClaimReductionKernel<F> {
    type Relation = InstructionClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<InstructionClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        if self.rounds_bound != self.log_t {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.log_t - self.rounds_bound,
            });
        }
        Ok(InstructionClaimReductionOutputClaims {
            lookup_output: self.tables[0].evals()[0],
            left_lookup_operand: self.tables[1].evals()[0],
            right_lookup_operand: self.tables[2].evals()[0],
            left_instruction_input: self.tables[3].evals()[0],
            right_instruction_input: self.tables[4].evals()[0],
        })
    }

    /// Pin the fully-bound Gruen scalar to the verifier's
    /// `derive_output_term(EqSpartan)`, exactly as the naive tier's
    /// materialized eq table is pinned.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        if self.rounds_bound != self.log_t {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.log_t - self.rounds_bound,
            });
        }
        let id = JoltDerivedId::from(InstructionClaimReductionPublic::EqSpartan);
        let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
        let got = self.gruen.current_scalar();
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use std::collections::BTreeMap;

    use jolt_claims::protocols::jolt::geometry::claim_reductions::instruction::{
        left_instruction_input_reduced, left_lookup_operand_reduced, lookup_output_reduced,
        right_instruction_input_reduced, right_lookup_operand_reduced,
    };
    use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::{
        InstructionClaimReductionChallenges, InstructionClaimReductionInputClaims,
    };
    use jolt_claims::protocols::jolt::{
        InstructionClaimReductionPublic, JoltDerivedId, TraceDimensions,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, Polynomial};
    use jolt_sumcheck::ProveRounds;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;

    use crate::reference::views::eq_table;
    use crate::{NaiveSumcheckProver, ProverInputs, SumcheckKernel};

    use super::OptimizedInstructionClaimReductionKernel;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn challenge(round: usize) -> Fr {
        fr(0xA5A5_1234_DEAD_BEEF ^ (round as u64).wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x33)
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn assert_parity(log_t: usize, seed: u64) {
        let mut state = seed;
        let tables: Vec<Vec<Fr>> = (0..5)
            .map(|_| {
                (0..1usize << log_t)
                    .map(|_| fr(splitmix(&mut state)))
                    .collect()
            })
            .collect();
        let tau_low: Vec<Fr> = (0..log_t).map(|i| fr(500 + 41 * i as u64)).collect();
        let gamma = fr(0xC0FF_EE11);
        let relation =
            InstructionClaimReduction::<Fr>::new(TraceDimensions::new(log_t), tau_low.clone());

        let ids = [
            lookup_output_reduced(),
            left_lookup_operand_reduced(),
            right_lookup_operand_reduced(),
            left_instruction_input_reduced(),
            right_instruction_input_reduced(),
        ];
        let opening_tables: BTreeMap<_, _> = ids
            .into_iter()
            .zip(&tables)
            .map(|(id, values)| (id, Polynomial::new(values.clone())))
            .collect();
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(InstructionClaimReductionPublic::EqSpartan),
            Polynomial::new(eq_table(&tau_low)),
        )]);
        let input_claims = InstructionClaimReductionInputClaims {
            lookup_output: fr(0),
            left_lookup_operand: fr(0),
            right_lookup_operand: fr(0),
            left_instruction_input: fr(0),
            right_instruction_input: fr(0),
        };
        let input_points = InstructionClaimReductionInputClaims {
            lookup_output: Vec::new(),
            left_lookup_operand: Vec::new(),
            right_lookup_operand: Vec::new(),
            left_instruction_input: Vec::new(),
            right_instruction_input: Vec::new(),
        };
        let challenges = InstructionClaimReductionChallenges { gamma };
        let inputs = ProverInputs {
            relation: &relation,
            claims: &input_claims,
            points: &input_points,
            challenges: &challenges,
        };
        let mut reference = NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )
        .unwrap();

        let mut optimized = OptimizedInstructionClaimReductionKernel::new(
            &tau_low,
            tables.iter().map(|t| Polynomial::new(t.clone())).collect(),
            gamma,
        )
        .unwrap();

        // True input claim: the full hypercube sum of the summand.
        let eq = eq_table(&tau_low);
        let gamma_powers = [
            fr(1),
            gamma,
            gamma * gamma,
            gamma * gamma * gamma,
            gamma * gamma * gamma * gamma,
        ];
        let mut claim = fr(0);
        for j in 0..1usize << log_t {
            let mut combo = fr(0);
            for (table, power) in tables.iter().zip(&gamma_powers) {
                combo += *power * table[j];
            }
            claim += eq[j] * combo;
        }

        let rounds = reference.num_rounds();
        assert_eq!(rounds, optimized.num_rounds());
        for round in 0..rounds {
            let bind = round.checked_sub(1).map(challenge);
            let reference_poly = reference.prove_round(bind, round, claim).unwrap();
            let optimized_poly = optimized.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                reference_poly.coefficients(),
                optimized_poly.coefficients(),
                "round {round} polynomial mismatch (log_t={log_t})"
            );
            claim = reference_poly.evaluate(challenge(round));
        }
        reference.finish_rounds(challenge(rounds - 1)).unwrap();
        optimized.finish_rounds(challenge(rounds - 1)).unwrap();

        let reference_outputs = reference.output_claims(&input_claims).unwrap();
        let optimized_outputs = optimized.output_claims(&input_claims).unwrap();
        assert_eq!(reference_outputs, optimized_outputs);

        let sumcheck_point: Vec<Fr> = (0..rounds).map(challenge).collect();
        let output_points = relation
            .derive_opening_points(&sumcheck_point, &input_points)
            .unwrap();
        optimized
            .validate_derived_tables(&relation, &input_points, &output_points, &challenges)
            .unwrap();
    }

    #[test]
    fn parity_even_log_t() {
        assert_parity(4, 7);
    }

    #[test]
    fn parity_odd_log_t() {
        assert_parity(3, 99);
    }
}
