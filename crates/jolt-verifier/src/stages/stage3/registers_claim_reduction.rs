//! The stage 3 `RegistersClaimReduction` sumcheck instance.
//!
//! Owns the register-reduction opening-point derivation and the `EqSpartan`
//! public-value computation (against the product uni-skip `tau_low`).
//!
//! The reduced `rs1_value`/`rs2_value` outputs alias the instruction-input
//! virtualization's: all stage-3 members bind the same batch-point suffix (equal
//! rounds, default offsets) and derive the same reversed opening point. Declared
//! in `aliased_output_openings` below; the generated drivers absorb each via its
//! instruction-input source and enforce the wire copies equal it.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::claim_reductions::registers::{
    RegistersClaimReductionChallenges, RegistersClaimReductionInputClaims,
    RegistersClaimReductionOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::claim_reductions::registers as registers_geometry,
    geometry::dimensions::TraceDimensions, geometry::instruction, JoltDerivedId, JoltOpeningId,
    JoltRelationId, RegistersClaimReductionPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage1::Stage1BatchOutputClaims;
use crate::VerifierError;

/// Wire the consumed opening *values* from stage 1's outer sumcheck register
/// values. Takes the ZK-agnostic stage-1 output-claims aggregate.
pub fn registers_claim_reduction_input_values_from_upstream<F: Field>(
    stage1: &Stage1BatchOutputClaims<F>,
) -> RegistersClaimReductionInputClaims<F> {
    let outer = &stage1.outer_remainder;
    RegistersClaimReductionInputClaims {
        rd_write_value: outer.rd_write_value,
        rs1_value: outer.rs1_value,
        rs2_value: outer.rs2_value,
    }
}

#[derive(Clone)]
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

    pub fn product_uniskip_tau_low(&self) -> &[F] {
        &self.product_uniskip_tau_low
    }
}

impl<F: Field> ConcreteSumcheck<F> for RegistersClaimReduction<F> {
    type Symbolic = relations::claim_reductions::registers::ClaimReduction;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn aliased_output_openings() -> Vec<(JoltOpeningId, JoltOpeningId)> {
        vec![
            (
                registers_geometry::rs1_value_reduced(),
                instruction::rs1_value(),
            ),
            (
                registers_geometry::rs2_value_reduced(),
                instruction::rs2_value(),
            ),
        ]
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
