//! Stage-6b reduction of the standalone fused-increment claim to the cycle
//! point shared with Booleanity.

use jolt_claims::protocols::jolt::lattice::relations::fused_inc_claim_reduction as relation;
use jolt_claims::protocols::jolt::{
    FusedIncClaimReductionPublic, JoltDerivedId, JoltRelationId, TraceDimensions,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::try_eq_mle;
pub use relation::{FusedIncClaimReductionInputClaims, FusedIncClaimReductionOutputClaims};

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage6a::inc_virtualization::IncVirtualizationOutput;
use crate::VerifierError;

pub fn input_values_from_upstream<F: Field>(
    inc_virtualization: &IncVirtualizationOutput<F>,
) -> FusedIncClaimReductionInputClaims<F> {
    FusedIncClaimReductionInputClaims {
        fused_inc: inc_virtualization.output_values.fused_inc,
    }
}

pub fn input_points_from_upstream<F: Field>(
    inc_virtualization: &IncVirtualizationOutput<F>,
) -> FusedIncClaimReductionInputClaims<Vec<F>> {
    FusedIncClaimReductionInputClaims {
        fused_inc: inc_virtualization.output_points.fused_inc().to_vec(),
    }
}

pub struct FusedIncClaimReduction<F: Field> {
    symbolic: relation::FusedIncClaimReduction,
    inc_virtualization_point: Vec<F>,
}

impl<F: Field> FusedIncClaimReduction<F> {
    pub fn new(trace: TraceDimensions, inc_virtualization_point: Vec<F>) -> Self {
        Self {
            symbolic: relation::FusedIncClaimReduction::new(trace),
            inc_virtualization_point,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::FusedIncClaimReduction,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for FusedIncClaimReduction<F> {
    type Symbolic = relation::FusedIncClaimReduction;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &FusedIncClaimReductionInputClaims<Vec<F>>,
    ) -> Result<FusedIncClaimReductionOutputClaims<Vec<F>>, VerifierError> {
        Ok(FusedIncClaimReductionOutputClaims {
            fused_inc: sumcheck_point.iter().rev().copied().collect(),
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &FusedIncClaimReductionInputClaims<Vec<F>>,
        output_points: &FusedIncClaimReductionOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::FusedIncClaimReduction(
            FusedIncClaimReductionPublic::EqIncVirtualization,
        ) = id
        else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        try_eq_mle(output_points.fused_inc(), &self.inc_virtualization_point)
            .map_err(public_input_failed)
    }
}
