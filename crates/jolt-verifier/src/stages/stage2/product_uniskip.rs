//! The stage 2 `SpartanProductVirtualization` product uni-skip sumcheck instance.
//!
//! The companion of the [`ProductRemainder`](super::product_remainder) relation:
//! the product uni-skip first round, a standalone centered-integer sumcheck whose
//! reduced opening the remainder consumes. Modelling it as a [`ConcreteSumcheck`]
//! single-sources its input-claim algebra — the Lagrange-weighted sum of the three
//! Spartan-outer openings (`product`, `should_branch`, `should_jump`) — so it stays
//! in lockstep with the BlindFold constraint, which evaluates the same
//! `spartan::product_uniskip` input formula.
//!
//! Unlike the remainder, the uni-skip's first-round binding-point draw (`tau_high`)
//! is still drawn inline in the stage-2 verifier and its Lagrange weights are an
//! *input* derived (resolved before binding), so this relation overrides
//! `derive_input_term` rather than `derive_output_term`.

#[cfg(feature = "field-inline")]
use std::sync::OnceLock;

use jolt_claims::protocols::jolt::relations;
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
use jolt_r1cs::constraints::jolt::SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// Wire the three consumed Spartan-outer opening *values* from the stage 1 outer
/// output. Only the values feed the input claim (the uni-skip's output point comes
/// from its own sumcheck point), so the input points are left empty.
pub fn product_uniskip_input_values_from_stage1<F: JoltField>(
    stage1: &Stage1ClearOutput<F>,
) -> ProductUniskipInputClaims<F> {
    let outer = &stage1.output_values.outer_remainder;
    ProductUniskipInputClaims {
        product: outer.product,
        should_branch: outer.should_branch,
        should_jump: outer.should_jump,
    }
}

#[derive(Clone)]
pub struct ProductUniskip<F: JoltField> {
    symbolic: relations::spartan::ProductUniskip,
    tau_high: F,
    /// The two FR lane input values (`FieldProduct`, `FieldInvProduct`
    /// openings at `FieldRegistersSpartanOuter`), set by the stage-2 verifier
    /// from the stage-1 FR carrier before the input claim is computed. The
    /// composed `input_claim` consumes them at the lane indices following the
    /// ordinary lanes.
    #[cfg(feature = "field-inline")]
    field_inline_inputs: OnceLock<[F; 2]>,
}

impl<F: JoltField> ProductUniskip<F> {
    pub fn new(dimensions: SpartanProductDimensions, tau_high: F) -> Self {
        Self {
            symbolic: relations::spartan::ProductUniskip::new(dimensions),
            tau_high,
            #[cfg(feature = "field-inline")]
            field_inline_inputs: OnceLock::new(),
        }
    }

    /// Supply the FR lane input values (`[FieldProduct, FieldInvProduct]` at
    /// `FieldRegistersSpartanOuter`) from stage 1's FR carrier. Must be called
    /// before `input_claim`; rejects a second set at different values (one
    /// proof per relation instance).
    #[cfg(feature = "field-inline")]
    pub fn set_field_inline_inputs(&self, product: F, inv_product: F) -> Result<(), VerifierError> {
        let values = [product, inv_product];
        let stored = self.field_inline_inputs.get_or_init(|| values);
        if *stored != values {
            return Err(public_input_failed(
                "field-inline product uni-skip inputs already set to different values",
            ));
        }
        Ok(())
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanProductVirtualization,
        reason: reason.to_string(),
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for ProductUniskip<F> {
    type Symbolic = relations::spartan::ProductUniskip;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &ProductUniskipInputClaims<Vec<F>>,
    ) -> Result<ProductUniskipOutputClaims<Vec<F>>, VerifierError> {
        Ok(ProductUniskipOutputClaims {
            uniskip: sumcheck_point.to_vec(),
        })
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
            // The uni-skip first-round Lagrange weights, evaluated at `tau_high`; the
            // input claim reweights the three Spartan-outer openings by
            // `UniskipLagrangeWeight(0..2)` exactly as the formula's
            // `product_uniskip_weight(i)`.
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
            // `LagrangeWeight`/`TauKernel` belong to the product *remainder* relation,
            // not the uni-skip: `product_uniskip` reweights via `product_uniskip_weight`
            // -> `UniskipLagrangeWeight` only. Reject rather than silently aliasing them,
            // so a misrouted public surfaces.
            SpartanProductVirtualizationPublic::LagrangeWeight(_)
            | SpartanProductVirtualizationPublic::TauKernel => {
                Err(VerifierError::MissingStageClaimDerived { id: (*id).into() })
            }
        }
    }

    /// The composed input claim over the feature-aware lane domain: the
    /// ordinary three lanes (the jolt symbolic `input_expression`, whose
    /// `UniskipLagrangeWeight`s already evaluate over the composed domain) plus
    /// the two FR lanes at the following indices — the same Lagrange-weighted
    /// fold, extended per `specs/field-inline-protocol.md` "Stage 2
    /// Composition". The jolt symbolic expression cannot name the FR openings,
    /// so the FR contribution comes from the jolt-claims composed-lane helper
    /// (pinned against the field-constraint product rows in `jolt-r1cs`).
    #[cfg(feature = "field-inline")]
    fn input_claim(
        &self,
        input_values: &ProductUniskipInputClaims<F>,
        challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        use jolt_claims::protocols::field_inline::geometry::product::{
            composed_uniskip_input_contribution, FieldProductLaneInputs,
        };
        use jolt_claims::InputClaims as _;
        use jolt_claims::SumcheckChallenges as _;
        use jolt_r1cs::constraints::jolt::SPARTAN_PRODUCT_BASE_LANES;

        let ordinary = self.symbolic().input_expression::<F>().try_evaluate(
            |id| {
                input_values
                    .resolve_input(id)
                    .ok_or(VerifierError::MissingOpeningClaim { id: (*id).into() })
            },
            |id| {
                challenges
                    .resolve_challenge(id)
                    .ok_or(VerifierError::MissingStageClaimChallenge { id: (*id).into() })
            },
            |id| self.derive_input_term(id, challenges),
        )?;

        let [product, inv_product] = *self.field_inline_inputs.get().ok_or_else(|| {
            public_input_failed(
                "field-inline product uni-skip inputs not set (the stage-2 verifier must \
                 supply them from the stage-1 FR carrier before the input claim)",
            )
        })?;
        let weights = centered_lagrange_evals(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, self.tau_high)
            .map_err(public_input_failed)?;
        let field_inline = composed_uniskip_input_contribution(
            &weights,
            SPARTAN_PRODUCT_BASE_LANES,
            &FieldProductLaneInputs {
                product,
                inv_product,
            },
        )
        .ok_or_else(|| {
            public_input_failed(format!(
                "composed product uni-skip weights do not cover the FR lanes (domain size \
                 {SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE})"
            ))
        })?;
        Ok(ordinary + field_inline)
    }
}

#[cfg(all(test, feature = "field-inline"))]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};

    /// The composed uni-skip input claim over the feature-aware 5-lane domain
    /// equals the ordinary symbolic fold (lanes 0..3, weights over the SAME
    /// composed domain) plus the FR lanes at the following indices — pinned
    /// against a from-scratch Lagrange-weighted sum over all five lane inputs.
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
        relation
            .set_field_inline_inputs(field_product, field_inv_product)
            .unwrap();

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

    /// An FR-on build whose FR inputs were never supplied cannot compute the
    /// composed input claim — the producer path (no FR proving yet) fails
    /// closed here rather than silently proving the 3-lane fold.
    #[test]
    fn composed_input_claim_requires_field_inline_inputs() {
        let relation = ProductUniskip::new(SpartanProductDimensions::new(4), Fr::from_u64(23));
        let inputs = ProductUniskipInputClaims::<Fr> {
            product: Fr::from_u64(3),
            should_branch: Fr::from_u64(5),
            should_jump: Fr::from_u64(7),
        };

        assert!(relation
            .input_claim(&inputs, &NoChallenges::default())
            .is_err());
    }
}
