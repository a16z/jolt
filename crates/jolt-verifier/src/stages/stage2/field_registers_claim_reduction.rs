//! The stage 2 `FieldRegistersClaimReduction` sumcheck instance — the first
//! FieldInline-family batch member (spec: `field-inline-protocol.md`, "Stage 2
//! Composition").
//!
//! Reduces the three FR Spartan-outer value openings (`FieldRdValue`,
//! `FieldRs1Value`, `FieldRs2Value`, batched by gamma) to the shared stage-2
//! product point `r_prod`: the relation is trace-domain (`log_T` rounds), so it
//! binds the same batch-point suffix as the product remainder and derives the
//! same reversed opening point — the point agreement the spec's `r_prod`
//! sharing is built on (pinned by the batch test
//! `field_registers_claim_reduction_shares_the_product_remainder_point`).
//!
//! Owns the reduced-claim opening-point derivation and the `EqSpartan`
//! public-value computation, mirroring the instruction claim reduction:
//! `EqSpartan = Eq(reduced opening point, tau_low)` where `tau_low` is stage
//! 1's remainder point tail, reversed.

use jolt_claims::protocols::field_inline::relations::claim_reductions::registers;
pub use jolt_claims::protocols::field_inline::relations::claim_reductions::registers::{
    FieldRegistersClaimReductionChallenges, FieldRegistersClaimReductionInputClaims,
    FieldRegistersClaimReductionOutputClaims,
};
use jolt_claims::protocols::field_inline::{
    FieldInlineDerivedId, FieldInlineRelationId, FieldRegistersClaimReductionPublic,
    FieldRegistersTraceDimensions,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// Wire the consumed FR value opening *values* from stage 1's composed outer
/// sumcheck (the FR-local appended segment). Fail-closed: an FR-on proof whose
/// stage-1 carrier lacks the FR payload cannot feed this reduction.
pub fn field_registers_claim_reduction_input_values_from_upstream<F: Field>(
    stage1: &Stage1ClearOutput<F>,
) -> Result<FieldRegistersClaimReductionInputClaims<F>, VerifierError> {
    let outer =
        stage1
            .field_inline_output_values
            .as_ref()
            .ok_or(VerifierError::MissingProofPayload {
                field: "stage1.field_inline_output_values",
            })?;
    Ok(FieldRegistersClaimReductionInputClaims {
        rd_value: outer.rd_value,
        rs1_value: outer.rs1_value,
        rs2_value: outer.rs2_value,
    })
}

#[derive(Clone)]
pub struct FieldRegistersClaimReduction<F: Field> {
    symbolic: registers::ClaimReduction,
    tau_low: Vec<F>,
}

impl<F: Field> FieldRegistersClaimReduction<F> {
    pub fn new(dimensions: FieldRegistersTraceDimensions, tau_low: Vec<F>) -> Self {
        Self {
            symbolic: registers::ClaimReduction::new(dimensions),
            tau_low,
        }
    }

    pub fn tau_low(&self) -> &[F] {
        &self.tau_low
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: format!("{:?}", FieldInlineRelationId::FieldRegistersClaimReduction),
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for FieldRegistersClaimReduction<F> {
    type Symbolic = registers::ClaimReduction;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &FieldRegistersClaimReductionInputClaims<Vec<F>>,
    ) -> Result<FieldRegistersClaimReductionOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(FieldRegistersClaimReductionOutputClaims {
            rd_value: opening_point.clone(),
            rs1_value: opening_point.clone(),
            rs2_value: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &FieldInlineDerivedId,
        _input_points: &FieldRegistersClaimReductionInputClaims<Vec<F>>,
        output_points: &FieldRegistersClaimReductionOutputClaims<Vec<F>>,
        _challenges: &FieldRegistersClaimReductionChallenges<F>,
    ) -> Result<F, VerifierError> {
        let FieldInlineDerivedId::FieldRegistersClaimReduction(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
        };
        match public_id {
            // The reduced openings share one opening point; bind it against the
            // low product remainder challenges (`tau_low`) — the same
            // derivation as the instruction claim reduction's `EqSpartan`.
            FieldRegistersClaimReductionPublic::EqSpartan => {
                try_eq_mle(output_points.rd_value(), &self.tau_low).map_err(public_input_failed)
            }
        }
    }
}
