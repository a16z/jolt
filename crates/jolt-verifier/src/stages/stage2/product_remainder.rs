//! The stage 2 `SpartanProductVirtualization` product-remainder sumcheck instance.
//!
//! Owns the product opening-point derivation and the uni-skip Lagrange-weight /
//! `TauKernel` public-value computation, in lockstep with the BlindFold constraint's
//! `spartan::product_remainder` formula.
//!
//! The companion product *uni-skip* first round is a univariate skip rather than a
//! [`ConcreteSumcheck`], so it stays hand-coded in the stage-2 verifier; this
//! relation consumes that uni-skip's reduced opening as its input claim.

#[cfg(feature = "field-inline")]
use crate::stages::composed::ComposedClaims;
use std::collections::BTreeSet;

#[cfg(feature = "field-inline")]
use crate::stages::composed::{
    ProductInputs as SelectedInputs, ProductOutputs as SelectedOutputs,
    ProductRemainder as SelectedSymbolic,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::relations::product::FieldRegistersProductOutputClaims;
#[cfg(not(feature = "field-inline"))]
use jolt_claims::protocols::jolt::relations::spartan::{
    ProductRemainder as SelectedSymbolic, ProductRemainderInputClaims as SelectedInputs,
    ProductRemainderOutputClaims as SelectedOutputs,
};
pub use jolt_claims::protocols::jolt::relations::spartan::{
    ProductRemainderInputClaims, ProductRemainderOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::spartan, geometry::spartan::SpartanProductDimensions, JoltDerivedId, JoltRelationId,
    SpartanProductVirtualizationPublic,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::JoltField;
use jolt_poly::{
    lagrange::{centered_lagrange_evals, centered_lagrange_kernel},
    try_eq_mle,
};
use jolt_r1cs::constraints::jolt::SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE;

use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

/// Wire the consumed opening *value* from the product uni-skip's reduced output
/// claim (the output point comes from this relation's own sumcheck point).
#[cfg_attr(
    not(feature = "field-inline"),
    expect(
        clippy::useless_conversion,
        reason = "field-inline selects a composed claim or opening id"
    )
)]
pub fn product_remainder_input_values_from_uniskip_output<F: JoltField>(
    product_uniskip_output_claim: F,
) -> SelectedInputs<F> {
    ProductRemainderInputClaims {
        product_uniskip: product_uniskip_output_claim,
    }
    .into()
}

impl<F: JoltField> ProductRemainder<F> {
    pub fn uniskip_challenge(&self) -> F {
        self.uniskip_challenge
    }

    pub fn tau_high(&self) -> F {
        self.tau_high
    }
}

#[derive(Clone)]
pub struct ProductRemainder<F: JoltField> {
    symbolic: SelectedSymbolic,
    uniskip_challenge: F,
    tau_high: F,
    tau_low: Vec<F>,
}

impl<F: JoltField> ProductRemainder<F> {
    pub fn new(
        dimensions: SpartanProductDimensions,
        uniskip_challenge: F,
        tau_high: F,
        tau_low: Vec<F>,
    ) -> Self {
        Self {
            symbolic: SelectedSymbolic::new(dimensions),
            uniskip_challenge,
            tau_high,
            tau_low,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanProductVirtualization,
        reason: reason.to_string(),
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for ProductRemainder<F> {
    type Symbolic = SelectedSymbolic;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    #[cfg_attr(
        not(feature = "field-inline"),
        expect(
            clippy::useless_conversion,
            reason = "field-inline selects a composed claim or opening id"
        )
    )]
    fn wire_output_openings(&self) -> BTreeSet<<SelectedSymbolic as SymbolicSumcheck>::OpeningId> {
        // Two wire openings beyond the output-`Expr`-referenced set:
        // `write_lookup_output_to_rd` and `virtual_instruction` are absorbed here
        // but their constraining fold happens downstream, in stage 6a's bytecode
        // read-RAF input claim.
        let mut openings = self.symbolic().expected_output_openings::<F>();
        openings.extend::<[<SelectedSymbolic as SymbolicSumcheck>::OpeningId; 2]>([
            spartan::write_lookup_output_to_rd_product().into(),
            spartan::virtual_instruction_product().into(),
        ]);
        openings
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &SelectedInputs<Vec<F>>,
    ) -> Result<SelectedOutputs<Vec<F>>, VerifierError> {
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        let output = ProductRemainderOutputClaims {
            left_instruction_input: opening_point.clone(),
            right_instruction_input: opening_point.clone(),
            jump_flag: opening_point.clone(),
            write_lookup_output_to_rd: opening_point.clone(),
            lookup_output: opening_point.clone(),
            branch_flag: opening_point.clone(),
            next_is_noop: opening_point.clone(),
            virtual_instruction: opening_point.clone(),
        };
        #[cfg(feature = "field-inline")]
        let output = ComposedClaims {
            base: output,
            field_inline: FieldRegistersProductOutputClaims {
                rs1_value: opening_point.clone(),
                rs2_value: opening_point.clone(),
                rd_value: opening_point,
            },
        };
        Ok(output)
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &SelectedInputs<Vec<F>>,
        output_points: &SelectedOutputs<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::SpartanProductVirtualization(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
        };
        match public_id {
            // The uni-skip first-round Lagrange weights, evaluated at the product
            // uni-skip challenge; the product remainder reweights its operands by
            // `LagrangeWeight(0..2)` exactly as the formula's `product_weight(i)`.
            SpartanProductVirtualizationPublic::LagrangeWeight(index) => {
                let weights = centered_lagrange_evals(
                    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
                    self.uniskip_challenge,
                )
                .map_err(public_input_failed)?;
                weights
                    .get(*index)
                    .copied()
                    .ok_or_else(|| public_input_failed(format!(
                        "product remainder Lagrange weight index {index} out of range for domain size {SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE}"
                    )))
            }
            // `UniskipLagrangeWeight` belongs to the product uni-skip relation, not the
            // remainder: `product_remainder` reweights via `product_weight` ->
            // `LagrangeWeight` only (plus `TauKernel`). Reject rather than silently
            // aliasing it onto the Lagrange-weight path, so a misrouted public surfaces.
            SpartanProductVirtualizationPublic::UniskipLagrangeWeight(_) => {
                Err(VerifierError::MissingStageClaimDerived { id: (*id).into() })
            }
            // The product opening point binds the uni-skip kernel (against
            // `tau_high`) and the equality of the low remainder challenges
            // (`tau_low`) with the produced product opening point.
            SpartanProductVirtualizationPublic::TauKernel => {
                let product_opening = output_points.left_instruction_input();
                let tau_high_bound = centered_lagrange_kernel(
                    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
                    self.tau_high,
                    self.uniskip_challenge,
                )
                .map_err(public_input_failed)?;
                let tau_low_eq =
                    try_eq_mle(&self.tau_low, product_opening).map_err(public_input_failed)?;
                Ok(tau_high_bound * tau_low_eq)
            }
        }
    }
}

#[cfg(all(test, feature = "field-inline"))]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::super::outputs::FieldRegistersProductOutputClaims;
    use super::*;
    use jolt_field::{Fr, Ring};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn output_values() -> ProductRemainderOutputClaims<Fr> {
        ProductRemainderOutputClaims {
            left_instruction_input: fr(2),
            right_instruction_input: fr(3),
            jump_flag: fr(5),
            write_lookup_output_to_rd: fr(7),
            lookup_output: fr(11),
            branch_flag: fr(13),
            next_is_noop: fr(17),
            virtual_instruction: fr(19),
        }
    }

    fn field_inline_outputs() -> FieldRegistersProductOutputClaims<Fr> {
        FieldRegistersProductOutputClaims {
            rs1_value: fr(23),
            rs2_value: fr(29),
            rd_value: fr(31),
        }
    }

    fn fixture() -> (
        ProductRemainder<Fr>,
        SelectedInputs<Vec<Fr>>,
        SelectedOutputs<Vec<Fr>>,
    ) {
        let log_t = 4usize;
        let uniskip_challenge = fr(37);
        let tau_high = fr(41);
        let tau_low: Vec<Fr> = (50..54).map(fr).collect();
        let relation = ProductRemainder::new(
            SpartanProductDimensions::new(log_t),
            uniskip_challenge,
            tau_high,
            tau_low,
        );

        let sumcheck_point: Vec<Fr> = (60..64).map(fr).collect();
        let input_points = SelectedInputs::<Vec<Fr>>::default();
        let output_points = relation
            .derive_opening_points(&sumcheck_point, &input_points)
            .unwrap();
        (relation, input_points, output_points)
    }

    /// The composed `expected_output` over the feature-aware 5-lane domain
    /// equals the from-scratch factored form: `tau_kernel · (Σ w_i·L_i) ·
    /// (Σ w_i·R_i)` over all five lanes (ordinary lane table, then the FR
    /// lanes' rs1·rs2 and rs1·rd factors), with weights over the composed
    /// domain — the `field-inline-protocol.md` "Stage 2 Composition" algebra.
    #[test]
    fn composed_expected_output_matches_five_lane_factored_form() {
        assert_eq!(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, 5);
        let (relation, input_points, output_points) = fixture();
        let outputs = output_values();
        let field_inline = field_inline_outputs();
        let outputs = ComposedClaims {
            base: outputs,
            field_inline: field_inline.clone(),
        };

        let weights = centered_lagrange_evals(
            SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
            relation.uniskip_challenge(),
        )
        .unwrap();
        let left_lanes = [
            outputs.left_instruction_input,
            outputs.lookup_output,
            outputs.jump_flag,
            field_inline.rs1_value,
            field_inline.rs1_value,
        ];
        let right_lanes = [
            outputs.right_instruction_input,
            outputs.branch_flag,
            Fr::from_u64(1) - outputs.next_is_noop,
            field_inline.rs2_value,
            field_inline.rd_value,
        ];
        let fold = |lanes: [Fr; 5]| {
            weights
                .iter()
                .zip(lanes)
                .map(|(weight, lane)| *weight * lane)
                .sum::<Fr>()
        };
        let tau_kernel = centered_lagrange_kernel(
            SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
            relation.tau_high(),
            relation.uniskip_challenge(),
        )
        .unwrap()
            * try_eq_mle(
                relation.tau_low.as_slice(),
                output_points.left_instruction_input(),
            )
            .unwrap();
        let expected = tau_kernel * fold(left_lanes) * fold(right_lanes);

        let composed = relation
            .expected_output(
                &input_points,
                &outputs,
                &output_points,
                &NoChallenges::default(),
            )
            .unwrap();
        assert_eq!(composed, expected);
    }
}
