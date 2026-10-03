#[cfg(feature = "field-inline")]
use jolt_claims::protocols::composed::ComposedClaims;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::composed::FieldProductUniskipInputs;

use jolt_claims::protocols::composed::geometry::SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::composed::{
    ProductUniskip as SelectedSymbolic, UniskipInputs as SelectedInputs,
    UniskipOutputs as SelectedOutputs,
};
#[cfg(not(feature = "field-inline"))]
use jolt_claims::protocols::jolt::relations::spartan::{
    ProductUniskip as SelectedSymbolic, ProductUniskipInputClaims as SelectedInputs,
    ProductUniskipOutputClaims as SelectedOutputs,
};
pub use jolt_claims::protocols::jolt::relations::spartan::{
    ProductUniskipInputClaims, ProductUniskipOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::spartan::SpartanProductDimensions, JoltDerivedId, JoltRelationId,
    SpartanProductVirtualizationPublic,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::JoltField;
use jolt_poly::lagrange::centered_lagrange_evals;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// Wire the three consumed Spartan-outer opening *values* from the stage 1 outer
/// output. Only the values feed the input claim (the uni-skip's output point comes
/// from its own sumcheck point), so the input points are left empty.
pub fn product_uniskip_input_values_from_stage1<F: JoltField>(
    stage1: &Stage1ClearOutput<F>,
) -> SelectedInputs<F> {
    let outer = &stage1.output_values.outer_remainder;
    let inputs = ProductUniskipInputClaims {
        product: outer.product,
        should_branch: outer.should_branch,
        should_jump: outer.should_jump,
    };
    #[cfg(feature = "field-inline")]
    let inputs = ComposedClaims {
        base: inputs,
        field_inline: FieldProductUniskipInputs {
            product: outer.field_inline.product,
            inv_product: outer.field_inline.inv_product,
        },
    };
    inputs
}

#[derive(Clone)]
pub struct ProductUniskip<F: JoltField> {
    symbolic: SelectedSymbolic,
    tau_high: F,
}

impl<F: JoltField> ProductUniskip<F> {
    pub fn new(dimensions: SpartanProductDimensions, tau_high: F) -> Self {
        Self {
            symbolic: SelectedSymbolic::new(dimensions),
            tau_high,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanProductVirtualization,
        reason: reason.to_string(),
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for ProductUniskip<F> {
    type Symbolic = SelectedSymbolic;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &SelectedInputs<Vec<F>>,
    ) -> Result<SelectedOutputs<Vec<F>>, VerifierError> {
        let output = ProductUniskipOutputClaims {
            uniskip: sumcheck_point.to_vec(),
        };
        #[cfg(feature = "field-inline")]
        let output = output.into();
        Ok(output)
    }

    fn derive_input_term(
        &self,
        id: &JoltDerivedId,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::SpartanProductVirtualization(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
        };
        match public_id {
            SpartanProductVirtualizationPublic::UniskipLagrangeWeight(index) => {
                let weights =
                    centered_lagrange_evals(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, self.tau_high)
                        .map_err(public_input_failed)?;
                weights.get(*index).copied().ok_or_else(|| {
                    public_input_failed(format!(
                        "product uni-skip Lagrange weight index {index} out of range for domain size {SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE}"
                    ))
                })
            }
            SpartanProductVirtualizationPublic::LagrangeWeight(_)
            | SpartanProductVirtualizationPublic::TauKernel => {
                Err(VerifierError::MissingStageClaimDerived { id: (*id).into() })
            }
        }
    }
}

#[cfg(all(test, feature = "field-inline"))]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};

    #[test]
    fn composed_input_claim_matches_five_lane_fold() {
        assert_eq!(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, 5);

        let tau_high = Fr::from_u64(23);
        let relation = ProductUniskip::new(SpartanProductDimensions::new(4), tau_high);
        let inputs = ProductUniskipInputClaims::<Fr> {
            product: Fr::from_u64(3),
            should_branch: Fr::from_u64(5),
            should_jump: Fr::from_u64(7),
        };
        let field_product = Fr::from_u64(11);
        let field_inv_product = Fr::from_u64(13);
        let inputs = ComposedClaims {
            base: inputs,
            field_inline: FieldProductUniskipInputs {
                product: field_product,
                inv_product: field_inv_product,
            },
        };

        let weights =
            centered_lagrange_evals(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, tau_high).unwrap();
        let lane_inputs = [
            inputs.product,
            inputs.should_branch,
            inputs.should_jump,
            field_product,
            field_inv_product,
        ];
        let expected: Fr = weights
            .iter()
            .zip(lane_inputs)
            .map(|(weight, input)| *weight * input)
            .sum();

        let composed = relation
            .input_claim(&inputs, &NoChallenges::default())
            .unwrap();
        assert_eq!(composed, expected);
    }
}
