//! Optimized instruction input-virtualization (stage 3) kernel.
//!
//! The summand is
//! `eq(r_product, j) · ((is_rs2·rs2 + is_imm·imm) + γ·(is_rs1·rs1 + is_pc·upc))(j)`
//! — degree 3. The eight operand/flag leaves stay separate tables (their
//! products carry the round polynomial's curvature — the stage-2 lesson noted
//! on the reference kernel), but the eq weight is factored out via
//! `GruenSplitEqPolynomial`: each round emits
//! `s(t) = ℓ(t) · Σ_y E_out·E_in · q(t, y)` with `q` the flag/value quadratic,
//! at the naive prover's `t = 0..=3` sample points through the same
//! `from_evals` constructor — byte-identical round polynomials and output
//! claims, with no `T`-sized eq table materialized or bound.
//!
//! The bound eq factor is pinned to the verifier's scalar path by
//! [`validate_derived_tables`](crate::SumcheckKernel::validate_derived_tables)
//! (Gruen scalar vs `derive_output_term(EqProduct)`).

use jolt_claims::protocols::jolt::geometry::instruction::{
    imm, left_operand_is_pc, left_operand_is_rs1, right_operand_is_imm, right_operand_is_rs2,
    rs1_value, rs2_value, unexpanded_pc,
};
use jolt_claims::protocols::jolt::relations::instruction::InstructionInputOutputClaims;
use jolt_claims::protocols::jolt::{InstructionInputPublic, JoltDerivedId};
use jolt_field::Field;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::outputs::InstructionInput;
use jolt_witness::JoltWitnessPlane;

use crate::reference::views::dense_view;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// The eight operand/flag tables, in output-claim declaration order:
/// `[is_rs1, rs1, is_pc, upc, is_rs2, rs2, is_imm, imm]`.
const NUM_TABLES: usize = 8;

/// Optimized [`PrepareKernel`] implementor for the `instruction_input` slot.
pub struct OptimizedInstructionInput;

impl<F: Field> PrepareKernel<F, InstructionInput<F>> for OptimizedInstructionInput {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionInput<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionInput<F>>>, KernelError<F>> {
        let ids = [
            left_operand_is_rs1(),
            rs1_value(),
            left_operand_is_pc(),
            unexpanded_pc(),
            right_operand_is_rs2(),
            rs2_value(),
            right_operand_is_imm(),
            imm(),
        ];
        let mut tables = Vec::with_capacity(NUM_TABLES);
        for id in ids {
            tables.push(Polynomial::new(dense_view(witness, id)?));
        }
        Ok(Box::new(OptimizedInstructionInputKernel::new(
            inputs.relation.product_remainder_opening_point(),
            tables,
            inputs.challenges.gamma,
        )?))
    }
}

pub struct OptimizedInstructionInputKernel<F: Field> {
    log_t: usize,
    gamma: F,
    /// `[is_rs1, rs1, is_pc, upc, is_rs2, rs2, is_imm, imm]` — output-claim
    /// declaration order.
    tables: Vec<Polynomial<F>>,
    gruen: GruenSplitEqPolynomial<F>,
    bind_scratch: Vec<F>,
    rounds_bound: usize,
}

impl<F: Field> OptimizedInstructionInputKernel<F> {
    pub fn new(
        r_product: &[F],
        tables: Vec<Polynomial<F>>,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        let log_t = r_product.len();
        if tables.len() != NUM_TABLES {
            return Err(KernelError::InvariantViolation {
                reason: "instruction input virtualization expects eight operand tables",
            });
        }
        for table in &tables {
            if table.len() != 1 << log_t {
                return Err(KernelError::TableSizeMismatch {
                    table: "instruction input operand table".to_owned(),
                    expected: 1 << log_t,
                    got: table.len(),
                });
            }
        }
        Ok(Self {
            log_t,
            gamma,
            tables,
            gruen: GruenSplitEqPolynomial::new(r_product, BindingOrder::LowToHigh),
            bind_scratch: Vec::new(),
            rounds_bound: 0,
        })
    }

    /// `s(t) = ℓ(t) · Σ_y E(y) · q(t, y)` at `t = 0..=3`, with
    /// `q = (is_rs2·rs2 + is_imm·imm) + γ·(is_rs1·rs1 + is_pc·upc)`.
    fn message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        const POINTS: usize = 4;
        let q_evals = self.gruen.par_fold_out_in(
            || {
                (
                    [F::zero(); POINTS],
                    [F::zero(); NUM_TABLES],
                    [F::zero(); NUM_TABLES],
                )
            },
            |(acc, evals, steps), row, _x_in, e_in| {
                for ((table, eval), step) in self
                    .tables
                    .iter()
                    .zip(evals.iter_mut())
                    .zip(steps.iter_mut())
                {
                    let table = table.evals();
                    let lo = table[2 * row];
                    *eval = lo;
                    *step = table[2 * row + 1] - lo;
                }
                for value in acc.iter_mut() {
                    let right = evals[4] * evals[5] + evals[6] * evals[7];
                    let left = evals[0] * evals[1] + evals[2] * evals[3];
                    *value += e_in * (right + self.gamma * left);
                    for (eval, step) in evals.iter_mut().zip(steps.iter()) {
                        *eval += *step;
                    }
                }
            },
            |_x_out, e_out, (mut acc, _, _)| {
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

impl<F: Field> ProveRounds<F> for OptimizedInstructionInputKernel<F> {
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

impl<F: Field> SumcheckKernel<F> for OptimizedInstructionInputKernel<F> {
    type Relation = InstructionInput<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<InstructionInputOutputClaims<F>, SumcheckKernelError<F>> {
        if self.rounds_bound != self.log_t {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.log_t - self.rounds_bound,
            });
        }
        Ok(InstructionInputOutputClaims {
            left_operand_is_rs1: self.tables[0].evals()[0],
            rs1_value: self.tables[1].evals()[0],
            left_operand_is_pc: self.tables[2].evals()[0],
            unexpanded_pc: self.tables[3].evals()[0],
            right_operand_is_rs2: self.tables[4].evals()[0],
            rs2_value: self.tables[5].evals()[0],
            right_operand_is_imm: self.tables[6].evals()[0],
            imm: self.tables[7].evals()[0],
        })
    }

    /// Pin the fully-bound Gruen scalar to the verifier's
    /// `derive_output_term(EqProduct)`, exactly as the naive tier's
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
        let id = JoltDerivedId::from(InstructionInputPublic::EqProduct);
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

    use jolt_claims::protocols::jolt::geometry::instruction::{
        imm, left_operand_is_pc, left_operand_is_rs1, right_operand_is_imm, right_operand_is_rs2,
        rs1_value, rs2_value, unexpanded_pc,
    };
    use jolt_claims::protocols::jolt::relations::instruction::{
        InstructionInputChallenges, InstructionInputInputClaims,
    };
    use jolt_claims::protocols::jolt::{InstructionInputPublic, JoltDerivedId, TraceDimensions};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, Polynomial};
    use jolt_sumcheck::ProveRounds;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage3::outputs::InstructionInput;

    use crate::reference::views::eq_table;
    use crate::{NaiveSumcheckProver, ProverInputs, SumcheckKernel};

    use super::OptimizedInstructionInputKernel;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn challenge(round: usize) -> Fr {
        fr(0x1357_9BDF_2468_ACE0 ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x55)
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
        // Flag tables Boolean (as in a real trace), value tables arbitrary.
        let tables: Vec<Vec<Fr>> = (0..8)
            .map(|position| {
                (0..1usize << log_t)
                    .map(|_| {
                        let raw = splitmix(&mut state);
                        if position % 2 == 0 {
                            fr(raw & 1)
                        } else {
                            fr(raw)
                        }
                    })
                    .collect()
            })
            .collect();
        let r_product: Vec<Fr> = (0..log_t).map(|i| fr(900 + 53 * i as u64)).collect();
        let gamma = fr(0xDADA_CAFE);
        let relation = InstructionInput::<Fr>::new(TraceDimensions::new(log_t), r_product.clone());

        let ids = [
            left_operand_is_rs1(),
            rs1_value(),
            left_operand_is_pc(),
            unexpanded_pc(),
            right_operand_is_rs2(),
            rs2_value(),
            right_operand_is_imm(),
            imm(),
        ];
        let opening_tables: BTreeMap<_, _> = ids
            .into_iter()
            .zip(&tables)
            .map(|(id, values)| (id, Polynomial::new(values.clone())))
            .collect();
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(InstructionInputPublic::EqProduct),
            Polynomial::new(eq_table(&r_product)),
        )]);
        let input_claims = InstructionInputInputClaims {
            right_instruction_input: fr(0),
            left_instruction_input: fr(0),
        };
        let input_points = InstructionInputInputClaims {
            right_instruction_input: Vec::new(),
            left_instruction_input: Vec::new(),
        };
        let challenges = InstructionInputChallenges { gamma };
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

        let mut optimized = OptimizedInstructionInputKernel::new(
            &r_product,
            tables.iter().map(|t| Polynomial::new(t.clone())).collect(),
            gamma,
        )
        .unwrap();

        // True input claim: the full hypercube sum of the summand.
        let eq = eq_table(&r_product);
        let mut claim = fr(0);
        for j in 0..1usize << log_t {
            let right = tables[4][j] * tables[5][j] + tables[6][j] * tables[7][j];
            let left = tables[0][j] * tables[1][j] + tables[2][j] * tables[3][j];
            claim += eq[j] * (right + gamma * left);
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
        assert_parity(4, 21);
    }

    #[test]
    fn parity_odd_log_t() {
        assert_parity(3, 8080);
    }
}
