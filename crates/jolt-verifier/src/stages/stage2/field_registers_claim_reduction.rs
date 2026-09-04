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

use jolt_claims::protocols::field_inline::relations::claim_reductions::registers::ClaimReduction;
pub use jolt_claims::protocols::field_inline::relations::claim_reductions::registers::{
    FieldRegistersClaimReductionChallenges, FieldRegistersClaimReductionInputClaims,
    FieldRegistersClaimReductionOutputClaims,
};
use jolt_claims::protocols::field_inline::{
    FieldInlineDerivedId, FieldRegistersClaimReductionPublic, FieldRegistersTraceDimensions,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;

use crate::stages::derivations;
use crate::stages::relations::{project_public, stage_claim_failed, ConcreteSumcheck};
use crate::VerifierError;

#[derive(Clone)]
pub struct FieldRegistersClaimReduction<F: JoltField> {
    symbolic: ClaimReduction,
    tau_low: Vec<F>,
}

impl<F: JoltField> FieldRegistersClaimReduction<F> {
    pub fn new(dimensions: FieldRegistersTraceDimensions, tau_low: Vec<F>) -> Self {
        Self {
            symbolic: ClaimReduction::new(dimensions),
            tau_low,
        }
    }

    pub fn tau_low(&self) -> &[F] {
        &self.tau_low
    }
}

// Only the point geometry stays hand-written: the symbolic output expression
// references the opening point and `EqSpartan` as opaque `Derived` leaves, so
// their derivations cannot come from it. Everything else (claim evaluation,
// struct fill, id projection) is trait defaults + derive-generated code.
impl<F: JoltField> ConcreteSumcheck<F> for FieldRegistersClaimReduction<F> {
    type Symbolic = ClaimReduction;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &FieldRegistersClaimReductionInputClaims<Vec<F>>,
    ) -> Result<FieldRegistersClaimReductionOutputClaims<Vec<F>>, VerifierError> {
        Ok(FieldRegistersClaimReductionOutputClaims::from_shared_point(
            derivations::reversed(sumcheck_point),
        ))
    }

    fn derive_output_term(
        &self,
        id: &FieldInlineDerivedId,
        _input_points: &FieldRegistersClaimReductionInputClaims<Vec<F>>,
        output_points: &FieldRegistersClaimReductionOutputClaims<Vec<F>>,
        _challenges: &FieldRegistersClaimReductionChallenges<F>,
    ) -> Result<F, VerifierError> {
        match project_public(id)? {
            // The reduced openings share one opening point; bind it against the
            // low product remainder challenges (`tau_low`) — literally the
            // instruction claim reduction's `EqSpartan` derivation.
            FieldRegistersClaimReductionPublic::EqSpartan => {
                derivations::eq_at_point(output_points.rd_value(), &self.tau_low)
                    .map_err(|reason| stage_claim_failed(self.id(), reason))
            }
        }
    }
}
