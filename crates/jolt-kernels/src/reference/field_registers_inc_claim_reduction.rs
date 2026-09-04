//! The stage-6b `FieldRegistersIncClaimReduction` kernel: a hand-rolled member
//! over the cycle domain.
//!
//! The summand is `(eq(r_rw, j) + γ·eq(r_val, j)) · FieldRdInc(j)` — the jolt
//! increment claim-reduction kernel's register leg at the FR dimensions:
//! reducing the two upstream `FieldRdInc` openings (stage-4 FR read/write,
//! stage-5 FR val evaluation, folded by the member-drawn gamma) to the single
//! reduced opening the stage-8 joint opening consumes. The increment table is
//! the committed dense trace view; each eq leaf is one multilinear over its
//! upstream FR cycle sub-point. The FieldInline id family cannot ride the
//! jolt-keyed [`NaiveSumcheckProver`](crate::NaiveSumcheckProver), so the
//! tables and the expression are hand-held (the
//! [`field_registers_claim_reduction`](super::field_registers_claim_reduction)
//! pattern).

#[cfg(feature = "allocative")]
use allocative::{Allocative, Key, Visitor};
use jolt_claims::protocols::field_inline::{
    FieldInlineDerivedId, FieldRegistersIncClaimReductionPublic,
};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::field_registers_inc_claim_reduction::FieldRegistersIncClaimReduction;
use jolt_verifier::VerifierError;
use jolt_witness::JoltWitnessPlane;

use super::views::eq_table;
use crate::backend::{PrepareKernel, ProofSession};
use crate::kernel::{ProverInputs, SumcheckKernel};
use crate::reference::ReferenceBackend;
use crate::{KernelError, SumcheckKernelError};

impl<F: JoltField> PrepareKernel<F, FieldRegistersIncClaimReduction<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, FieldRegistersIncClaimReduction<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = FieldRegistersIncClaimReduction<F>>>,
        KernelError<F>,
    > {
        use jolt_claims::protocols::field_inline::{
            FieldInlineChallengeId, FieldInlineCommittedPolynomial, FieldInlinePolynomialId,
            FieldRegistersIncClaimReductionChallenge,
        };
        use jolt_claims::SumcheckChallenges as _;
        use jolt_witness::WitnessError;

        let relation = inputs.relation;
        let [read_write_cycle, val_evaluation_cycle] = relation.cycle_points();
        for point in [read_write_cycle, val_evaluation_cycle] {
            if point.len() != relation.rounds() {
                return Err(KernelError::InvariantViolation {
                    reason: "FR increment reduction cycle point has the wrong variable count",
                });
            }
        }

        let field_inline =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "field-registers increment claim-reduction field-inline oracle",
                }))?;
        let rd_inc = field_inline.oracle_table(FieldInlinePolynomialId::Committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
        ))?;
        let gamma = inputs
            .challenges
            .resolve_challenge(&FieldInlineChallengeId::from(
                FieldRegistersIncClaimReductionChallenge::Gamma,
            ))
            .ok_or(KernelError::InvariantViolation {
                reason: "FR increment claim reduction is missing its gamma challenge",
            })?;

        Ok(Box::new(FieldRegistersIncClaimReductionKernel {
            relation: relation.clone(),
            gamma,
            eq_read_write: Polynomial::new(eq_table(read_write_cycle)),
            eq_val_evaluation: Polynomial::new(eq_table(val_evaluation_cycle)),
            rd_inc: Polynomial::new(rd_inc),
            rounds_bound: 0,
        }))
    }
}

struct FieldRegistersIncClaimReductionKernel<F: JoltField> {
    relation: FieldRegistersIncClaimReduction<F>,
    gamma: F,
    /// `eq(stage-4 FR read/write cycle, ·)` over the cycle domain
    /// (big-endian, like the jolt increment reduction kernel's eq tables).
    eq_read_write: Polynomial<F>,
    /// `eq(stage-5 FR val-evaluation cycle, ·)` over the cycle domain.
    eq_val_evaluation: Polynomial<F>,
    rd_inc: Polynomial<F>,
    rounds_bound: usize,
}

// Size arithmetic rather than a derive, like the sibling kernels.
#[cfg(feature = "allocative")]
impl<F: JoltField> Allocative for FieldRegistersIncClaimReductionKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        for (key, table) in [
            (Key::new("eq_read_write"), &self.eq_read_write),
            (Key::new("eq_val_evaluation"), &self.eq_val_evaluation),
            (Key::new("rd_inc"), &self.rd_inc),
        ] {
            visitor.visit_simple(key, table.len() * size_of::<F>());
        }
        visitor.exit();
    }
}

impl<F: JoltField> FieldRegistersIncClaimReductionKernel<F> {
    fn remaining_rounds(&self) -> usize {
        self.relation.rounds() - self.rounds_bound
    }

    fn bind_tables(&mut self, challenge: F) {
        for table in [
            &mut self.eq_read_write,
            &mut self.eq_val_evaluation,
            &mut self.rd_inc,
        ] {
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

impl<F: JoltField> ProveRounds<F> for FieldRegistersIncClaimReductionKernel<F> {
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
        let gamma = self.gamma;
        let mut evals = Vec::with_capacity(degree + 1);
        for sample in 0..=degree {
            let point = F::from_u64(sample as u64);
            let sum = (0..half)
                .map(|y| {
                    let eq = self
                        .eq_read_write
                        .sumcheck_round_eval_with_order(y, point, order)
                        + gamma
                            * self
                                .eq_val_evaluation
                                .sumcheck_round_eval_with_order(y, point, order);
                    eq * self.rd_inc.sumcheck_round_eval_with_order(y, point, order)
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

impl<F: JoltField> SumcheckKernel<F> for FieldRegistersIncClaimReductionKernel<F> {
    type Relation = FieldRegistersIncClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, FieldRegistersIncClaimReduction<F>>,
    ) -> Result<SumcheckOutputClaims<F, FieldRegistersIncClaimReduction<F>>, SumcheckKernelError<F>>
    {
        use jolt_claims::protocols::field_inline::relations::claim_reductions::increments::FieldRegistersIncClaimReductionOutputClaims;

        self.require_fully_bound()?;
        Ok(FieldRegistersIncClaimReductionOutputClaims {
            rd_inc: self.rd_inc.evals()[0],
        })
    }

    /// The eq-table cross-checks: both bound eq tables' final values must
    /// equal the verifier's `derive_output_term` at the bound point (the same
    /// tie-down the naive tier performs for jolt-family members).
    fn validate_derived_tables(
        &self,
        relation: &FieldRegistersIncClaimReduction<F>,
        input_points: &SumcheckInputPoints<F, FieldRegistersIncClaimReduction<F>>,
        output_points: &SumcheckOutputPoints<F, FieldRegistersIncClaimReduction<F>>,
        challenges: &ConcreteSumcheckChallenges<F, FieldRegistersIncClaimReduction<F>>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        for (public, table) in [
            (
                FieldRegistersIncClaimReductionPublic::EqReadWrite,
                &self.eq_read_write,
            ),
            (
                FieldRegistersIncClaimReductionPublic::EqValEvaluation,
                &self.eq_val_evaluation,
            ),
        ] {
            let expected = relation.derive_output_term(
                &FieldInlineDerivedId::from(public),
                input_points,
                output_points,
                challenges,
            )?;
            let got = table.evals()[0];
            if got != expected {
                return Err(SumcheckKernelError::Verifier(
                    VerifierError::StageClaimSumcheckFailed {
                        stage: "FieldRegistersIncClaimReduction".to_string(),
                        reason: format!(
                            "{public:?} table bound to {got:?}, but derive_output_term gives \
                             {expected:?}"
                        ),
                    },
                ));
            }
        }
        Ok(())
    }
}
