//! The stage 2 `SpartanProductVirtualization` product-remainder sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover (while
//! producing the stage 2 batch proof) and the verifier (after checking it). It
//! owns the product opening-point derivation and the uni-skip Lagrange-weight /
//! `TauKernel` public-value computation, so the input/output claim algebra lives
//! here once (and stays in lockstep with the BlindFold constraint, which evaluates
//! the same `spartan::product_remainder` formula).
//!
//! The companion product *uni-skip* first round is a univariate skip rather than a
//! [`SumcheckInstance`], so it stays hand-coded in the stage-2 verifier; this
//! relation consumes that uni-skip's reduced opening as its input claim.

use jolt_claims::protocols::jolt::{
    formulas::spartan::{self, SpartanProductDimensions},
    JoltPublicId, JoltRelationClaims, JoltRelationId, SpartanProductVirtualizationPublic,
};
use jolt_field::Field;
use jolt_poly::{
    lagrange::{centered_lagrange_evals, centered_lagrange_kernel},
    try_eq_mle,
};
use jolt_r1cs::constraints::jolt::SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, SumcheckInstance};
use crate::VerifierError;

/// Produced product-remainder openings (the eight virtualized instruction-product
/// operands and flags), all sharing the single product opening point. Generic over
/// the cell (`F` on the wire / serialized proof form, `OpeningClaim<F>` on the
/// clear path). Field order is the canonical Fiat-Shamir order and must match
/// [`spartan::product_remainder_output_openings`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(SpartanProductVirtualization)]
pub struct ProductRemainderOutputClaims<C> {
    #[opening(LeftInstructionInput)]
    pub left_instruction_input: C,
    #[opening(RightInstructionInput)]
    pub right_instruction_input: C,
    #[opening(OpFlags(CircuitFlags::Jump))]
    pub jump_flag: C,
    #[opening(OpFlags(CircuitFlags::WriteLookupOutputToRD))]
    pub write_lookup_output_to_rd: C,
    #[opening(LookupOutput)]
    pub lookup_output: C,
    #[opening(InstructionFlags(InstructionFlags::Branch))]
    pub branch_flag: C,
    #[opening(NextIsNoop)]
    pub next_is_noop: C,
    #[opening(OpFlags(CircuitFlags::VirtualInstruction))]
    pub virtual_instruction: C,
}

/// Consumed product-remainder input: the product uni-skip's reduced opening. The
/// relation reads only this value (its output point comes from its own sumcheck
/// point), so the input point is left empty. Generic over the cell.
#[derive(Clone, Debug, InputClaims)]
pub struct ProductRemainderInputClaims<C> {
    #[opening(UnivariateSkip, from = SpartanProductVirtualization)]
    pub product_uniskip: C,
}

impl<F: Field> ProductRemainderInputClaims<OpeningClaim<F>> {
    /// Wire the consumed opening from the product uni-skip's reduced output claim.
    /// Only the value feeds the input claim (the output point comes from this
    /// relation's own sumcheck point), so the input point is left empty.
    pub fn from_uniskip_output(product_uniskip_output_claim: F) -> Self {
        Self {
            product_uniskip: OpeningClaim {
                point: Vec::new(),
                value: product_uniskip_output_claim,
            },
        }
    }
}

pub struct ProductRemainder<F: Field> {
    claims: JoltRelationClaims<F>,
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
            claims: spartan::product_remainder(dimensions),
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

impl<F: Field> SumcheckInstance<F> for ProductRemainder<F> {
    type Inputs<C> = ProductRemainderInputClaims<C>;
    type Outputs<C> = ProductRemainderOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &ProductRemainderInputClaims<C>,
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

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        _inputs: &ProductRemainderInputClaims<C>,
        outputs: &ProductRemainderOutputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        let JoltPublicId::SpartanProductVirtualization(public_id) = id else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        match public_id {
            // The uni-skip first-round Lagrange weights, evaluated at the product
            // uni-skip challenge; the product remainder reweights its operands by
            // `LagrangeWeight(0..2)` exactly as the formula's `product_weight(i)`.
            SpartanProductVirtualizationPublic::LagrangeWeight(index)
            | SpartanProductVirtualizationPublic::UniskipLagrangeWeight(index) => {
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
            // The product opening point binds the uni-skip kernel (against
            // `tau_high`) and the equality of the low remainder challenges
            // (`tau_low`) with the produced product opening point.
            SpartanProductVirtualizationPublic::TauKernel => {
                let product_opening = outputs.left_instruction_input.point();
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
