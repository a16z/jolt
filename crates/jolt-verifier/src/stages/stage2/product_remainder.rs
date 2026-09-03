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
use std::sync::{Arc, OnceLock};

#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::relations::product::FieldRegistersProductOutputClaims;
use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::spartan::{
    ProductRemainderInputClaims, ProductRemainderOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::spartan, geometry::spartan::SpartanProductDimensions, JoltDerivedId, JoltOpeningId,
    JoltRelationId, SpartanProductVirtualizationPublic,
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
pub fn product_remainder_input_values_from_uniskip_output<F: JoltField>(
    product_uniskip_output_claim: F,
) -> ProductRemainderInputClaims<F> {
    ProductRemainderInputClaims {
        product_uniskip: product_uniskip_output_claim,
    }
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
    symbolic: relations::spartan::ProductRemainder,
    uniskip_challenge: F,
    tau_high: F,
    tau_low: Vec<F>,
    /// The three FR product-row opening values (`FieldRs1Value`,
    /// `FieldRs2Value`, `FieldRdValue` at `FieldRegistersProduct`), set by the
    /// stage-2 verifier from the proof's FR product appendage before the batch
    /// check. The composed `expected_output` folds them into the two factors at
    /// the lane indices following the ordinary lanes.
    /// Behind an `Arc` so relation clones share the cell — the prove-side
    /// composed kernel clones the batch's relation instance and its write
    /// must be visible to the driver's curation and expected-output fold.
    #[cfg(feature = "field-inline")]
    field_inline_outputs: Arc<OnceLock<FieldRegistersProductOutputClaims<F>>>,
}

impl<F: JoltField> ProductRemainder<F> {
    pub fn new(
        dimensions: SpartanProductDimensions,
        uniskip_challenge: F,
        tau_high: F,
        tau_low: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::spartan::ProductRemainder::new(dimensions),
            uniskip_challenge,
            tau_high,
            tau_low,
            #[cfg(feature = "field-inline")]
            field_inline_outputs: Arc::new(OnceLock::new()),
        }
    }

    /// Supply the FR product-row opening values from the proof's FR product
    /// appendage. Must be called before the batch's expected-output check;
    /// rejects a second set at different values (one proof per relation
    /// instance).
    #[cfg(feature = "field-inline")]
    pub fn set_field_inline_outputs(
        &self,
        values: FieldRegistersProductOutputClaims<F>,
    ) -> Result<(), VerifierError> {
        let stored = self.field_inline_outputs.get_or_init(|| values.clone());
        if *stored != values {
            return Err(public_input_failed(
                "field-inline product outputs already set to different values",
            ));
        }
        Ok(())
    }

    /// The FR product-row opening values, once supplied — read by the
    /// prove-side driver's curated absorb and the stage-2 recipe's claim
    /// assembly.
    #[cfg(feature = "field-inline")]
    pub fn field_inline_outputs(&self) -> Option<&FieldRegistersProductOutputClaims<F>> {
        self.field_inline_outputs.get()
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanProductVirtualization,
        reason: reason.to_string(),
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for ProductRemainder<F> {
    type Symbolic = relations::spartan::ProductRemainder;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn wire_output_openings(&self) -> std::collections::BTreeSet<JoltOpeningId> {
        // Two wire openings beyond the output-`Expr`-referenced set:
        // `write_lookup_output_to_rd` and `virtual_instruction` are absorbed here
        // but their constraining fold happens downstream, in stage 6a's bytecode
        // read-RAF input claim.
        let mut openings = self.symbolic().expected_output_openings::<F>();
        openings.extend([
            spartan::write_lookup_output_to_rd_product(),
            spartan::virtual_instruction_product(),
        ]);
        openings
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &ProductRemainderInputClaims<Vec<F>>,
    ) -> Result<ProductRemainderOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(ProductRemainderOutputClaims {
            left_instruction_input: opening_point.clone(),
            right_instruction_input: opening_point.clone(),
            jump_flag: opening_point.clone(),
            write_lookup_output_to_rd: opening_point.clone(),
            lookup_output: opening_point.clone(),
            branch_flag: opening_point.clone(),
            next_is_noop: opening_point.clone(),
            virtual_instruction: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &ProductRemainderInputClaims<Vec<F>>,
        output_points: &ProductRemainderOutputClaims<Vec<F>>,
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

    /// The composed expected output claim over the feature-aware lane domain:
    /// `tau_kernel · (ordinary_left + fr_left) · (ordinary_right + fr_right)`,
    /// per `specs/field-inline-protocol.md` "Stage 2 Composition". The ordinary
    /// factors are the jolt symbolic factor expressions (whose `LagrangeWeight`s
    /// already evaluate over the composed domain via `derive_output_term`); the
    /// FR contributions come from the jolt-claims composed-lane helper (pinned
    /// against the field-constraint product rows in `jolt-r1cs`) over the
    /// values supplied via
    /// [`set_field_inline_outputs`](ProductRemainder::set_field_inline_outputs).
    /// The jolt symbolic `output_expression` cannot name the FR openings, so
    /// the composed form is assembled here from the same factor pieces.
    #[cfg(feature = "field-inline")]
    fn expected_output(
        &self,
        input_points: &ProductRemainderInputClaims<Vec<F>>,
        output_values: &ProductRemainderOutputClaims<F>,
        output_points: &ProductRemainderOutputClaims<Vec<F>>,
        challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        use jolt_claims::protocols::field_inline::geometry::product::{
            composed_remainder_factor_contributions, FieldProductLaneFactors,
        };
        use jolt_claims::protocols::jolt::{JoltDerivedId, JoltExpr};
        use jolt_claims::{OutputClaims as _, SumcheckChallenges as _};
        use jolt_r1cs::constraints::jolt::SPARTAN_PRODUCT_BASE_LANES;

        let evaluate_factor = |expression: JoltExpr<F>| {
            expression.try_evaluate(
                |id| {
                    output_values
                        .resolve_output(id)
                        .ok_or(VerifierError::MissingOpeningClaim { id: (*id).into() })
                },
                |id| {
                    challenges
                        .resolve_challenge(id)
                        .ok_or(VerifierError::MissingStageClaimChallenge { id: (*id).into() })
                },
                |id| self.derive_output_term(id, input_points, output_points, challenges),
            )
        };
        let ordinary_left = evaluate_factor(self.symbolic.left_factor_expression())?;
        let ordinary_right = evaluate_factor(self.symbolic.right_factor_expression())?;
        let tau_kernel = self.derive_output_term(
            &JoltDerivedId::SpartanProductVirtualization(
                SpartanProductVirtualizationPublic::TauKernel,
            ),
            input_points,
            output_points,
            challenges,
        )?;

        let field_inline = self.field_inline_outputs.get().ok_or_else(|| {
            public_input_failed(
                "field-inline product outputs not set (stage2::verify must supply them \
                 before the batch check)",
            )
        })?;
        let weights =
            centered_lagrange_evals(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, self.uniskip_challenge)
                .map_err(public_input_failed)?;
        let (fr_left, fr_right) = composed_remainder_factor_contributions(
            &weights,
            SPARTAN_PRODUCT_BASE_LANES,
            &FieldProductLaneFactors {
                rs1_value: field_inline.rs1_value,
                rs2_value: field_inline.rs2_value,
                rd_value: field_inline.rd_value,
            },
        )
        .ok_or_else(|| {
            public_input_failed(format!(
                "composed product remainder weights do not cover the FR lanes (domain size \
                 {SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE})"
            ))
        })?;

        Ok(tau_kernel * (ordinary_left + fr_left) * (ordinary_right + fr_right))
    }
}

#[cfg(all(test, feature = "field-inline"))]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::super::outputs::FieldRegistersProductOutputClaims;
    use super::*;
    use jolt_field::{Fr, Ring};
    use jolt_r1cs::constraints::jolt::SPARTAN_PRODUCT_BASE_LANES;

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
        ProductRemainderInputClaims<Vec<Fr>>,
        ProductRemainderOutputClaims<Vec<Fr>>,
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
        let input_points = ProductRemainderInputClaims::<Vec<Fr>>::default();
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
        relation
            .set_field_inline_outputs(field_inline.clone())
            .unwrap();

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

    /// With zero FR factor values the composed form reduces to the ordinary
    /// jolt symbolic `output_expression` (evaluated over the same composed
    /// domain weights) — pinning the override's ordinary-factor legs to the
    /// symbolic source of truth rather than a restated lane table.
    #[test]
    fn composed_expected_output_reduces_to_symbolic_form_without_field_lanes() {
        let (relation, input_points, output_points) = fixture();
        let outputs = output_values();
        relation
            .set_field_inline_outputs(FieldRegistersProductOutputClaims {
                rs1_value: Fr::from_u64(0),
                rs2_value: Fr::from_u64(0),
                rd_value: Fr::from_u64(0),
            })
            .unwrap();

        use jolt_claims::OutputClaims as _;
        let symbolic = relation
            .symbolic
            .output_expression::<Fr>()
            .try_evaluate(
                |id| {
                    outputs
                        .resolve_output(id)
                        .ok_or(VerifierError::MissingOpeningClaim { id: (*id).into() })
                },
                |id| Err::<Fr, _>(VerifierError::MissingStageClaimChallenge { id: (*id).into() }),
                |id| {
                    relation.derive_output_term(
                        id,
                        &input_points,
                        &output_points,
                        &NoChallenges::default(),
                    )
                },
            )
            .unwrap();

        let composed = relation
            .expected_output(
                &input_points,
                &outputs,
                &output_points,
                &NoChallenges::default(),
            )
            .unwrap();
        assert_eq!(composed, symbolic);
    }

    /// FR-lane weighting sits at the lane indices following the ordinary
    /// lanes: perturbing only `rd_value` (the `FieldInvProduct` lane's right
    /// factor) changes the composed output by
    /// `tau_kernel · composed_left · w_4·delta`.
    #[test]
    fn composed_expected_output_weights_field_lanes_at_composed_indices() {
        let (relation, input_points, output_points) = fixture();
        let outputs = output_values();
        let field_inline = field_inline_outputs();
        relation
            .set_field_inline_outputs(field_inline.clone())
            .unwrap();
        let base = relation
            .expected_output(
                &input_points,
                &outputs,
                &output_points,
                &NoChallenges::default(),
            )
            .unwrap();

        let (relation_perturbed, input_points, output_points) = fixture();
        let delta = fr(97);
        let mut perturbed = field_inline;
        perturbed.rd_value += delta;
        relation_perturbed
            .set_field_inline_outputs(perturbed)
            .unwrap();
        let shifted = relation_perturbed
            .expected_output(
                &input_points,
                &outputs,
                &output_points,
                &NoChallenges::default(),
            )
            .unwrap();

        let weights = centered_lagrange_evals(
            SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
            relation_perturbed.uniskip_challenge(),
        )
        .unwrap();
        let tau_kernel = relation_perturbed
            .derive_output_term(
                &JoltDerivedId::SpartanProductVirtualization(
                    SpartanProductVirtualizationPublic::TauKernel,
                ),
                &input_points,
                &output_points,
                &NoChallenges::default(),
            )
            .unwrap();
        // The composed left factor is unchanged by an rd_value shift; read it
        // back off the base/shifted difference.
        // The InverseProduct lane's composed index: base + 1 = 4 on the 5-lane domain.
        let w4 = weights.get(SPARTAN_PRODUCT_BASE_LANES.checked_add(1).unwrap());
        let w4 = *w4.unwrap();
        let composed_left = {
            let field_inline = field_inline_outputs();
            let ordinary_weights = &weights;
            let left_lanes = [
                output_values().left_instruction_input,
                output_values().lookup_output,
                output_values().jump_flag,
                field_inline.rs1_value,
                field_inline.rs1_value,
            ];
            ordinary_weights
                .iter()
                .zip(left_lanes)
                .map(|(weight, lane)| *weight * lane)
                .sum::<Fr>()
        };
        assert_eq!(shifted - base, tau_kernel * composed_left * (w4 * delta));
    }
}
