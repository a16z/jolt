//! The stage-2 `FieldRegistersClaimReduction` kernel: a hand-rolled member
//! over the cycle domain.
//!
//! The summand is `eq(τ_low, t) · (rd + γ·rs1 + γ²·rs2)(t)`, degree 2 —
//! structurally the jolt registers claim reduction (same eq-table handling,
//! same LowToHigh binding, same round-poly accumulation), but over the
//! FieldInline id family, which the jolt-keyed [`NaiveSumcheckProver`]
//! (`crate::NaiveSumcheckProver`) cannot serve; the tables and the expression
//! are hand-held instead. `τ_low` is the stage-1 remainder cycle binding the
//! relation carries; `γ` is the member's drawn challenge.

use jolt_claims::protocols::field_inline::{
    FieldInlineDerivedId, FieldRegistersClaimReductionPublic,
};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::field_registers_claim_reduction::FieldRegistersClaimReduction;
use jolt_witness::JoltWitnessPlane;

use crate::backend::{PrepareKernel, ProofSession};
use crate::kernel::{ProverInputs, SumcheckKernel};
use crate::reference::ReferenceBackend;
use crate::{KernelError, SumcheckKernelError};

impl<F: JoltField> PrepareKernel<F, FieldRegistersClaimReduction<F>> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, FieldRegistersClaimReduction<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = FieldRegistersClaimReduction<F>>>,
        KernelError<F>,
    > {
        use jolt_claims::protocols::field_inline::{
            FieldInlinePolynomialId, FieldInlineVirtualPolynomial,
        };
        use jolt_claims::SumcheckChallenges as _;
        use jolt_poly::EqPolynomial;
        use jolt_witness::WitnessError;

        let field_inline =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "field-registers claim-reduction field-inline oracle",
                }))?;
        let table = |polynomial: FieldInlineVirtualPolynomial| {
            field_inline
                .oracle_table(FieldInlinePolynomialId::Virtual(polynomial))
                .map(Polynomial::new)
        };
        let gamma = inputs
            .challenges
            .resolve_challenge(
                &jolt_claims::protocols::field_inline::FieldInlineChallengeId::from(
                    jolt_claims::protocols::field_inline::FieldRegistersClaimReductionChallenge::Gamma,
                ),
            )
            .ok_or(KernelError::InvariantViolation {
                reason: "field-registers claim reduction is missing its gamma challenge",
            })?;
        Ok(Box::new(FieldRegistersClaimReductionKernel {
            relation: inputs.relation.clone(),
            gamma,
            eq_spartan: Polynomial::new(
                EqPolynomial::new(inputs.relation.tau_low().to_vec()).evaluations(),
            ),
            rd_value: table(FieldInlineVirtualPolynomial::FieldRdValue)?,
            rs1_value: table(FieldInlineVirtualPolynomial::FieldRs1Value)?,
            rs2_value: table(FieldInlineVirtualPolynomial::FieldRs2Value)?,
            rounds_bound: 0,
        }))
    }
}

struct FieldRegistersClaimReductionKernel<F: JoltField> {
    relation: FieldRegistersClaimReduction<F>,
    gamma: F,
    /// `eq(τ_low, ·)` over the cycle domain (big-endian, like the jolt
    /// registers claim reduction's `EqSpartan` table).
    eq_spartan: Polynomial<F>,
    rd_value: Polynomial<F>,
    rs1_value: Polynomial<F>,
    rs2_value: Polynomial<F>,
    rounds_bound: usize,
}

// Size arithmetic rather than a derive, like the sibling kernels.
#[cfg(feature = "allocative")]
impl<F: JoltField> allocative::Allocative for FieldRegistersClaimReductionKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        for (key, table) in [
            (allocative::Key::new("eq_spartan"), &self.eq_spartan),
            (allocative::Key::new("rd_value"), &self.rd_value),
            (allocative::Key::new("rs1_value"), &self.rs1_value),
            (allocative::Key::new("rs2_value"), &self.rs2_value),
        ] {
            visitor.visit_simple(key, table.len() * size_of::<F>());
        }
        visitor.exit();
    }
}

impl<F: JoltField> FieldRegistersClaimReductionKernel<F> {
    fn remaining_rounds(&self) -> usize {
        self.relation.rounds() - self.rounds_bound
    }

    fn bind_tables(&mut self, challenge: F) {
        self.eq_spartan
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.rd_value
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.rs1_value
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.rs2_value
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.rounds_bound += 1;
    }

    fn require_fully_bound(&self) -> Result<(), SumcheckKernelError<F>> {
        match self.remaining_rounds() {
            0 => Ok(()),
            remaining => Err(SumcheckKernelError::NotFullyBound { remaining }),
        }
    }
}

impl<F: JoltField> ProveRounds<F> for FieldRegistersClaimReductionKernel<F> {
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
                    let batched = self
                        .rd_value
                        .sumcheck_round_eval_with_order(y, point, order)
                        + gamma
                            * self
                                .rs1_value
                                .sumcheck_round_eval_with_order(y, point, order)
                        + gamma
                            * gamma
                            * self
                                .rs2_value
                                .sumcheck_round_eval_with_order(y, point, order);
                    self.eq_spartan
                        .sumcheck_round_eval_with_order(y, point, order)
                        * batched
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

impl<F: JoltField> SumcheckKernel<F> for FieldRegistersClaimReductionKernel<F> {
    type Relation = FieldRegistersClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, FieldRegistersClaimReduction<F>>,
    ) -> Result<SumcheckOutputClaims<F, FieldRegistersClaimReduction<F>>, SumcheckKernelError<F>>
    {
        use jolt_claims::protocols::field_inline::relations::claim_reductions::registers::FieldRegistersClaimReductionOutputClaims;

        self.require_fully_bound()?;
        Ok(FieldRegistersClaimReductionOutputClaims {
            rd_value: self.rd_value.evals()[0],
            rs1_value: self.rs1_value.evals()[0],
            rs2_value: self.rs2_value.evals()[0],
        })
    }

    /// The `EqSpartan` cross-check: the bound eq table's final value must
    /// equal the verifier's `derive_output_term` at the bound point (the
    /// same tie-down the naive tier performs for jolt-family members).
    fn validate_derived_tables(
        &self,
        relation: &FieldRegistersClaimReduction<F>,
        input_points: &SumcheckInputPoints<F, FieldRegistersClaimReduction<F>>,
        output_points: &SumcheckOutputPoints<F, FieldRegistersClaimReduction<F>>,
        challenges: &ConcreteSumcheckChallenges<F, FieldRegistersClaimReduction<F>>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        let expected = relation.derive_output_term(
            &FieldInlineDerivedId::from(FieldRegistersClaimReductionPublic::EqSpartan),
            input_points,
            output_points,
            challenges,
        )?;
        let got = self.eq_spartan.evals()[0];
        if got != expected {
            return Err(SumcheckKernelError::Verifier(
                jolt_verifier::VerifierError::StageClaimSumcheckFailed {
                    stage: "FieldRegistersClaimReduction".to_string(),
                    reason: format!(
                        "EqSpartan table bound to {got:?}, but derive_output_term gives \
                         {expected:?}"
                    ),
                },
            ));
        }
        Ok(())
    }
}
