//! The stage 3 `RegistersClaimReduction` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover (while
//! producing the stage 3 batch proof) and the verifier (after checking it). It
//! owns the register-reduction opening-point derivation and the `EqSpartan`
//! public-value computation (against the product uni-skip `tau_low`), so the
//! input/output claim algebra lives here once.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::claim_reductions::registers::{
    RegistersClaimReductionChallenges, RegistersClaimReductionInputClaims,
    RegistersClaimReductionOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::TraceDimensions, JoltDerivedId, JoltRelationId,
    RegistersClaimReductionPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// Wire the consumed opening *values* from stage 1's outer sumcheck register
/// values. (Verifier-side constructor for the moved
/// [`RegistersClaimReductionInputClaims`].)
pub fn registers_claim_reduction_input_values_from_upstream<F: Field>(
    stage1: &Stage1ClearOutput<F>,
) -> RegistersClaimReductionInputClaims<F> {
    let outer = &stage1.output_values.outer_remainder;
    RegistersClaimReductionInputClaims {
        rd_write_value: outer.rd_write_value,
        rs1_value: outer.rs1_value,
        rs2_value: outer.rs2_value,
    }
}

/// Wire the consumed opening *points* from stage 1. Only the values feed the input
/// claim (the output points come from this relation's own sumcheck point), so the
/// input points are left empty.
pub fn registers_claim_reduction_input_points_from_upstream<F: Field>(
    _stage1: &Stage1ClearOutput<F>,
) -> RegistersClaimReductionInputClaims<Vec<F>> {
    RegistersClaimReductionInputClaims {
        rd_write_value: Vec::new(),
        rs1_value: Vec::new(),
        rs2_value: Vec::new(),
    }
}

pub struct RegistersClaimReduction<F: Field> {
    symbolic: relations::claim_reductions::registers::ClaimReduction,
    product_uniskip_tau_low: Vec<F>,
}

impl<F: Field> RegistersClaimReduction<F> {
    pub fn new(trace_dimensions: TraceDimensions, product_uniskip_tau_low: Vec<F>) -> Self {
        Self {
            symbolic: relations::claim_reductions::registers::ClaimReduction::new(trace_dimensions),
            product_uniskip_tau_low,
        }
    }
}

impl<F: Field> ConcreteSumcheck<F> for RegistersClaimReduction<F> {
    type Symbolic = relations::claim_reductions::registers::ClaimReduction;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &RegistersClaimReductionInputClaims<Vec<F>>,
    ) -> Result<RegistersClaimReductionOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(RegistersClaimReductionOutputClaims {
            rd_write_value: opening_point.clone(),
            rs1_value: opening_point.clone(),
            rs2_value: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &RegistersClaimReductionInputClaims<Vec<F>>,
        output_points: &RegistersClaimReductionOutputClaims<Vec<F>>,
        _challenges: &RegistersClaimReductionChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::RegistersClaimReduction(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        match public_id {
            // Every reduction output shares the one opening point.
            RegistersClaimReductionPublic::EqSpartan => try_eq_mle(
                output_points.rd_write_value(),
                &self.product_uniskip_tau_low,
            )
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RegistersClaimReduction,
                reason: error.to_string(),
            }),
        }
    }
}
