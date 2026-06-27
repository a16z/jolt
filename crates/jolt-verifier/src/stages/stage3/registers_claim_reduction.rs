//! The stage 3 `RegistersClaimReduction` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover (while
//! producing the stage 3 batch proof) and the verifier (after checking it). It
//! owns the register-reduction opening-point derivation and the `EqSpartan`
//! public-value computation (against the product uni-skip `tau_low`), so the
//! input/output claim algebra lives here once.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::claim_reductions::registers::{
    RegistersClaimReductionInputClaims, RegistersClaimReductionOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::TraceDimensions, JoltChallengeId, JoltDerivedId, JoltRelationId,
    RegistersClaimReductionChallenge, RegistersClaimReductionPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// Wire the consumed openings from stage 1's outer sumcheck register values.
/// Only the values feed the input claim (the output points come from this
/// relation's own sumcheck point), so the input points are left empty.
/// (Verifier-side constructor for the moved [`RegistersClaimReductionInputClaims`].)
pub fn registers_claim_reduction_inputs_from_upstream<F: Field>(
    stage1: &Stage1ClearOutput<F>,
) -> RegistersClaimReductionInputClaims<OpeningClaim<F>> {
    let value = |value: F| OpeningClaim {
        point: Vec::new(),
        value,
    };
    RegistersClaimReductionInputClaims {
        rd_write_value: value(stage1.outer.rd_write_value),
        rs1_value: value(stage1.outer.rs1_value),
        rs2_value: value(stage1.outer.rs2_value),
    }
}

pub struct RegistersClaimReduction<F: Field> {
    symbolic: relations::claim_reductions::registers::ClaimReduction,
    gamma: F,
    product_uniskip_tau_low: Vec<F>,
}

impl<F: Field> RegistersClaimReduction<F> {
    pub fn new(
        trace_dimensions: TraceDimensions,
        gamma: F,
        product_uniskip_tau_low: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::claim_reductions::registers::ClaimReduction::new(trace_dimensions),
            gamma,
            product_uniskip_tau_low,
        }
    }
}

impl<F: Field> ConcreteSumcheck<F> for RegistersClaimReduction<F> {
    type Symbolic = relations::claim_reductions::registers::ClaimReduction;
    type Inputs<C> = RegistersClaimReductionInputClaims<C>;
    type Outputs<C> = RegistersClaimReductionOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &RegistersClaimReductionInputClaims<C>,
    ) -> Result<RegistersClaimReductionOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(RegistersClaimReductionOutputClaims {
            rd_write_value: opening_point.clone(),
            rs1_value: opening_point.clone(),
            rs2_value: opening_point,
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::RegistersClaimReduction(RegistersClaimReductionChallenge::Gamma) => {
                Ok(self.gamma)
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &RegistersClaimReductionInputClaims<C>,
        outputs: Option<&RegistersClaimReductionOutputClaims<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        let outputs = outputs.ok_or(VerifierError::MissingStageClaimDerived { id: *id })?;
        let JoltDerivedId::RegistersClaimReduction(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        match public_id {
            // Every reduction output shares the one opening point.
            RegistersClaimReductionPublic::EqSpartan => try_eq_mle(
                outputs.rd_write_value.point(),
                &self.product_uniskip_tau_low,
            )
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RegistersClaimReduction,
                reason: error.to_string(),
            }),
        }
    }
}
