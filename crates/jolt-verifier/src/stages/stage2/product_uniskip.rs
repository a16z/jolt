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

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::spartan::{
    ProductUniskipInputClaims, ProductUniskipOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::spartan::SpartanProductDimensions, JoltDerivedId, JoltRelationId,
    SpartanProductVirtualizationPublic,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::lagrange::centered_lagrange_evals;
use jolt_r1cs::constraints::jolt::SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// Wire the three consumed Spartan-outer opening *values* from the stage 1 outer
/// output. Only the values feed the input claim (the uni-skip's output point comes
/// from its own sumcheck point), so the input points are left empty.
pub fn product_uniskip_input_values_from_stage1<F: Field>(
    stage1: &Stage1ClearOutput<F>,
) -> ProductUniskipInputClaims<F> {
    let outer = &stage1.output_values.outer_remainder;
    ProductUniskipInputClaims {
        product: outer.product,
        should_branch: outer.should_branch,
        should_jump: outer.should_jump,
    }
}

/// Wire the three consumed Spartan-outer opening *points* (all empty — these
/// openings carry no point at this stage).
pub fn product_uniskip_input_points_from_stage1<F: Field>(
    _stage1: &Stage1ClearOutput<F>,
) -> ProductUniskipInputClaims<Vec<F>> {
    ProductUniskipInputClaims {
        product: Vec::new(),
        should_branch: Vec::new(),
        should_jump: Vec::new(),
    }
}

pub struct ProductUniskip<F: Field> {
    symbolic: relations::spartan::ProductUniskip,
    tau_high: F,
}

impl<F: Field> ProductUniskip<F> {
    pub fn new(dimensions: SpartanProductDimensions, tau_high: F) -> Self {
        Self {
            symbolic: relations::spartan::ProductUniskip::new(dimensions),
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

impl<F: Field> ConcreteSumcheck<F> for ProductUniskip<F> {
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
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
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
                Err(VerifierError::MissingStageClaimDerived { id: *id })
            }
        }
    }
}
