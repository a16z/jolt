//! The stage-5 `FieldRegistersValEvaluation` kernel: a hand-rolled member over
//! the cycle domain.
//!
//! The summand is `LT(j, r_field_rw.cycle) · rd_inc(j) · rd_wa(r_address, j)`
//! — the jolt registers value-evaluation kernel's structure at the FR
//! dimensions: the "field-register value at `(r_address, r_cycle)` is the sum
//! of earlier increments" identity. The `rd_wa` table is the address-bound
//! slice of the FR witness oracle's `(2^4 × T)` one-hot write-address grid (an
//! opening-side fold at the upstream FR read-write address prefix), and
//! `LtCycle` is ONE multilinear: `LtPolynomial::evaluations(r_cycle)`. The
//! FieldInline id family cannot ride the jolt-keyed
//! [`NaiveSumcheckProver`](crate::NaiveSumcheckProver), so the tables and the
//! expression are hand-held (the
//! [`field_registers_claim_reduction`](super::field_registers_claim_reduction)
//! pattern).

#[cfg(feature = "allocative")]
use allocative::{Allocative, Key, Visitor};
use jolt_claims::protocols::field_inline::{
    FieldInlineDerivedId, FieldRegistersValEvaluationPublic, FIELD_REGISTERS_LOG_K,
};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, LtPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage5::field_registers_val_evaluation::FieldRegistersValEvaluation;
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessPlane;

use super::views::eq_table;
use crate::backend::{PrepareKernel, ProofSession};
use crate::kernel::{ProverInputs, SumcheckKernel};
use crate::reference::ReferenceBackend;
use crate::{KernelError, SumcheckKernelError};

impl<F: JoltField> PrepareKernel<F, FieldRegistersValEvaluation<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, FieldRegistersValEvaluation<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = FieldRegistersValEvaluation<F>>>, KernelError<F>>
    {
        use jolt_claims::protocols::field_inline::{
            FieldInlineCommittedPolynomial, FieldInlinePolynomialId, FieldInlineVirtualPolynomial,
        };
        use jolt_witness::WitnessError;

        let relation = inputs.relation;
        let log_t = relation.trace_dimensions().log_t();
        let registers_val_point: &[F] = &inputs.points.registers_val;
        if registers_val_point.len() != FIELD_REGISTERS_LOG_K + log_t {
            return Err(KernelError::InvariantViolation {
                reason: "FR value-evaluation input point has the wrong variable count",
            });
        }
        let (r_address, r_cycle) = registers_val_point.split_at(FIELD_REGISTERS_LOG_K);

        let field_inline =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "field-registers value-evaluation field-inline oracle",
                }))?;
        // The address-bound `rd_wa` slice, folded from the one-hot grid:
        // `wa[j] = Σ_k eq(r_address, k) · grid[k·2^log_t + j]`.
        let wa_grid = field_inline.oracle_table(FieldInlinePolynomialId::Virtual(
            FieldInlineVirtualPolynomial::FieldRdWa,
        ))?;
        let cycles = 1usize << log_t;
        if wa_grid.len() != cycles << FIELD_REGISTERS_LOG_K {
            return Err(KernelError::TableSizeMismatch {
                table: "FieldRdWa".to_owned(),
                expected: cycles << FIELD_REGISTERS_LOG_K,
                got: wa_grid.len(),
            });
        }
        let eq_address = eq_table(r_address);
        let wa_folded: Vec<F> = (0..cycles)
            .map(|j| {
                eq_address
                    .iter()
                    .enumerate()
                    .map(|(k, eq)| *eq * wa_grid[(k << log_t) | j])
                    .sum()
            })
            .collect();
        let rd_inc = field_inline.oracle_table(FieldInlinePolynomialId::Committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
        ))?;

        Ok(Box::new(FieldRegistersValEvaluationKernel {
            relation: relation.clone(),
            lt_cycle: Polynomial::new(LtPolynomial::evaluations(r_cycle)),
            rd_inc: Polynomial::new(rd_inc),
            rd_wa: Polynomial::new(wa_folded),
            rounds_bound: 0,
        }))
    }
}

struct FieldRegistersValEvaluationKernel<F: JoltField> {
    relation: FieldRegistersValEvaluation<F>,
    /// `Lt(·, r_field_rw.cycle)` over the cycle domain (big-endian, like the
    /// jolt registers value-evaluation kernel's `LtCycle` table).
    lt_cycle: Polynomial<F>,
    rd_inc: Polynomial<F>,
    rd_wa: Polynomial<F>,
    rounds_bound: usize,
}

// Size arithmetic rather than a derive, like the sibling kernels.
#[cfg(feature = "allocative")]
impl<F: JoltField> Allocative for FieldRegistersValEvaluationKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        for (key, table) in [
            (Key::new("lt_cycle"), &self.lt_cycle),
            (Key::new("rd_inc"), &self.rd_inc),
            (Key::new("rd_wa"), &self.rd_wa),
        ] {
            visitor.visit_simple(key, table.len() * size_of::<F>());
        }
        visitor.exit();
    }
}

impl<F: JoltField> FieldRegistersValEvaluationKernel<F> {
    fn remaining_rounds(&self) -> usize {
        self.relation.rounds() - self.rounds_bound
    }

    fn bind_tables(&mut self, challenge: F) {
        for table in [&mut self.lt_cycle, &mut self.rd_inc, &mut self.rd_wa] {
            table.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        self.rounds_bound += 1;
    }

    fn require_fully_bound(&self) -> Result<(), SumcheckKernelError<F>> {
        match self.remaining_rounds() {
            0 => Ok(()),
            remaining => Err(SumcheckKernelError::NotFullyBound { remaining }),
        }
    }
}

impl<F: JoltField> ProveRounds<F> for FieldRegistersValEvaluationKernel<F> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind_tables(challenge);
        }
        let half = (1usize << self.remaining_rounds()) / 2;
        let degree = self.relation.degree();
        let order = BindingOrder::LowToHigh;
        let mut evals = Vec::with_capacity(degree + 1);
        for sample in 0..=degree {
            let point = F::from_u64(sample as u64);
            let sum = (0..half)
                .map(|y| {
                    self.lt_cycle
                        .sumcheck_round_eval_with_order(y, point, order)
                        * self.rd_inc.sumcheck_round_eval_with_order(y, point, order)
                        * self.rd_wa.sumcheck_round_eval_with_order(y, point, order)
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
        self.bind_tables(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for FieldRegistersValEvaluationKernel<F> {
    type Relation = FieldRegistersValEvaluation<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, FieldRegistersValEvaluation<F>>,
    ) -> Result<SumcheckOutputClaims<F, FieldRegistersValEvaluation<F>>, SumcheckKernelError<F>>
    {
        use jolt_claims::protocols::field_inline::relations::registers::FieldRegistersValEvaluationOutputClaims;

        self.require_fully_bound()?;
        Ok(FieldRegistersValEvaluationOutputClaims {
            rd_inc: self.rd_inc.evals()[0],
            rd_wa: self.rd_wa.evals()[0],
        })
    }

    /// The `LtCycle` cross-check: the bound Lt table's final value must equal
    /// the verifier's `derive_output_term` at the bound point (the same
    /// tie-down the naive tier performs for jolt-family members).
    fn validate_derived_tables(
        &self,
        relation: &FieldRegistersValEvaluation<F>,
        input_points: &SumcheckInputPoints<F, FieldRegistersValEvaluation<F>>,
        output_points: &SumcheckOutputPoints<F, FieldRegistersValEvaluation<F>>,
        challenges: &ConcreteSumcheckChallenges<F, FieldRegistersValEvaluation<F>>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        let expected = relation.derive_output_term(
            &FieldInlineDerivedId::from(FieldRegistersValEvaluationPublic::LtCycle),
            input_points,
            output_points,
            challenges,
        )?;
        let got = self.lt_cycle.evals()[0];
        if got != expected {
            return Err(SumcheckKernelError::Verifier(
                VerifierError::StageClaimSumcheckFailed {
                    stage: "FieldRegistersValEvaluation".to_string(),
                    reason: format!(
                        "LtCycle table bound to {got:?}, but derive_output_term gives {expected:?}"
                    ),
                },
            ));
        }
        Ok(())
    }
}
