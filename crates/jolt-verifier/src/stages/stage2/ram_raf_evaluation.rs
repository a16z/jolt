//! The stage 2 `RamRafEvaluation` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover (while
//! producing the stage 2 batch proof) and the verifier (after checking it). It
//! owns the RAM RAF address opening-point derivation and the `UnmapAddress`
//! public-value computation, so the input/output claim algebra lives here once (and
//! stays in lockstep with the BlindFold constraint, which evaluates the same
//! `ram::raf_evaluation` formula). The phase-3 cycle scaling on the input is baked
//! into that formula's constant coefficient.
//!
//! The produced `ram_ra` opening point is `[r_address(log_k) ‖ tau_low(log_t)]`;
//! `UnmapAddress` reads only the address prefix.

use jolt_claims::protocols::jolt::relations;
use jolt_claims::protocols::jolt::{
    geometry::{dimensions::ReadWriteDimensions, ram::RamRafEvaluationDimensions},
    JoltPublicId, JoltRelationId, RamRafEvaluationPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::{IdentityPolynomial, MultilinearEvaluation};
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// The produced RAM RAF `ram_ra` opening, sharing the single RAF opening point.
/// Generic over the cell (`F` on the wire / serialized proof form, `OpeningClaim<F>`
/// on the clear path).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamRafEvaluation)]
pub struct RamRafEvaluationOutputClaims<C> {
    #[opening(RamRa)]
    pub ram_ra: C,
}

/// The consumed RAM address opening from stage 1's outer sumcheck. The relation
/// reads only this value (its output point comes from its own sumcheck point), so
/// the input point is left empty. Generic over the cell.
#[derive(Clone, Debug, InputClaims)]
pub struct RamRafEvaluationInputClaims<C> {
    #[opening(RamAddress, from = SpartanOuter)]
    pub ram_address: C,
}

impl<F: Field> RamRafEvaluationInputClaims<OpeningClaim<F>> {
    pub fn from_upstream(stage1: &Stage1ClearOutput<F>) -> Self {
        Self {
            ram_address: OpeningClaim {
                point: Vec::new(),
                value: stage1.outer.ram_address,
            },
        }
    }
}

pub struct RamRafEvaluation<F: Field> {
    symbolic: relations::ram::RafEvaluation,
    read_write_dimensions: ReadWriteDimensions,
    ram_log_k: usize,
    lowest_address: u64,
    tau_low: Vec<F>,
}

impl<F: Field> RamRafEvaluation<F> {
    pub fn new(
        read_write_dimensions: ReadWriteDimensions,
        raf_dimensions: RamRafEvaluationDimensions,
        ram_log_k: usize,
        lowest_address: u64,
        tau_low: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::ram::RafEvaluation::new(raf_dimensions),
            read_write_dimensions,
            ram_log_k,
            lowest_address,
            tau_low,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamRafEvaluation,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for RamRafEvaluation<F> {
    type Symbolic = relations::ram::RafEvaluation;
    type Inputs<C> = RamRafEvaluationInputClaims<C>;
    type Outputs<C> = RamRafEvaluationOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &RamRafEvaluationInputClaims<C>,
    ) -> Result<RamRafEvaluationOutputClaims<Vec<F>>, VerifierError> {
        let address = self
            .read_write_dimensions
            .address_opening_point(sumcheck_point)
            .map_err(public_input_failed)?;
        if address.len() != self.ram_log_k {
            return Err(public_input_failed(format!(
                "RAM RAF address point length mismatch: expected {}, got {}",
                self.ram_log_k,
                address.len()
            )));
        }
        let opening_point = [address.as_slice(), self.tau_low.as_slice()].concat();
        Ok(RamRafEvaluationOutputClaims {
            ram_ra: opening_point,
        })
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        _inputs: &RamRafEvaluationInputClaims<C>,
        outputs: Option<&RamRafEvaluationOutputClaims<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        let outputs = outputs.ok_or(VerifierError::MissingStageClaimPublic { id: *id })?;
        let JoltPublicId::RamRafEvaluation(public_id) = id else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        match public_id {
            // The produced opening point is `[r_address(log_k) ‖ tau_low]`; the
            // unmap reads only the address prefix and lifts it back to a byte
            // address (`identity(r_address) * 8 + lowest_address`).
            RamRafEvaluationPublic::UnmapAddress => {
                let point = outputs.ram_ra.point();
                if point.len() < self.ram_log_k {
                    return Err(public_input_failed(format!(
                        "RAM RAF opening point is too short: expected at least {}, got {}",
                        self.ram_log_k,
                        point.len()
                    )));
                }
                let address = &point[..self.ram_log_k];
                Ok(
                    IdentityPolynomial::new(self.ram_log_k).evaluate(address) * F::from_u64(8)
                        + F::from_u64(self.lowest_address),
                )
            }
        }
    }
}
