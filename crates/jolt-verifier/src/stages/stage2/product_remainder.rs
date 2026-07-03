//! The stage 2 `SpartanProductVirtualization` product-remainder sumcheck instance.
//!
//! Owns the product opening-point derivation and the uni-skip Lagrange-weight /
//! `TauKernel` public-value computation, in lockstep with the BlindFold constraint's
//! `spartan::product_remainder` formula.
//!
//! The companion product *uni-skip* first round is a univariate skip rather than a
//! [`ConcreteSumcheck`], so it stays hand-coded in the stage-2 verifier; this
//! relation consumes that uni-skip's reduced opening as its input claim.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::spartan::{
    ProductRemainderInputClaims, ProductRemainderOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::spartan, geometry::spartan::SpartanProductDimensions, JoltDerivedId, JoltOpeningId,
    JoltRelationId, SpartanProductVirtualizationPublic,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::{
    lagrange::{centered_lagrange_evals, centered_lagrange_kernel},
    try_eq_mle,
};
use jolt_r1cs::constraints::jolt::SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE;

use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

/// Wire the consumed opening *value* from the product uni-skip's reduced output
/// claim (the output point comes from this relation's own sumcheck point).
pub fn product_remainder_input_values_from_uniskip_output<F: Field>(
    product_uniskip_output_claim: F,
) -> ProductRemainderInputClaims<F> {
    ProductRemainderInputClaims {
        product_uniskip: product_uniskip_output_claim,
    }
}

pub struct ProductRemainder<F: Field> {
    symbolic: relations::spartan::ProductRemainder,
    uniskip_challenge: F,
    tau_high: F,
    tau_low: Vec<F>,
}

impl<F: Field> ProductRemainder<F> {
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
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanProductVirtualization,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for ProductRemainder<F> {
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
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
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
                Err(VerifierError::MissingStageClaimDerived { id: *id })
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
