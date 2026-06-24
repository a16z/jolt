//! The stage 2 `RamReadWriteChecking` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover (while
//! producing the stage 2 batch proof) and the verifier (after checking it). It
//! owns the RAM read-write opening-point derivation and the `EqCycle` public-value
//! computation, so the input/output claim algebra lives here once instead of being
//! hand-coded on each side (and stays in lockstep with the BlindFold constraint,
//! which evaluates the same `ram::read_write_checking` formula).

use jolt_claims::protocols::jolt::{
    formulas::{dimensions::ReadWriteDimensions, ram},
    JoltChallengeId, JoltPublicId, JoltRelationClaims, JoltRelationId, RamReadWriteChallenge,
    RamReadWritePublic,
};
use jolt_field::Field;
use jolt_poly::try_eq_mle;
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, ConcreteSumcheck};
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// Produced RAM read-write openings (`val`, `ra`, committed `inc`), all sharing
/// the single read-write opening point. Generic over the cell (`F` on the wire /
/// serialized proof form, `OpeningClaim<F>` on the clear path).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamReadWriteChecking)]
pub struct RamReadWriteOutputClaims<C> {
    #[opening(RamVal)]
    pub val: C,
    #[opening(RamRa)]
    pub ra: C,
    #[opening(committed = RamInc)]
    pub inc: C,
}

/// Consumed RAM read/write value openings from stage 1's outer sumcheck, reduced
/// by the read-write checking sumcheck. The relation reads only these values (its
/// output points come from its own sumcheck point and `product_tau_low`), so the
/// input points are left empty. Generic over the cell.
#[derive(Clone, Debug, InputClaims)]
pub struct RamReadWriteInputClaims<C> {
    #[opening(RamReadValue, from = SpartanOuter)]
    pub ram_read_value: C,
    #[opening(RamWriteValue, from = SpartanOuter)]
    pub ram_write_value: C,
}

impl<F: Field> RamReadWriteInputClaims<OpeningClaim<F>> {
    pub fn from_upstream(stage1: &Stage1ClearOutput<F>) -> Self {
        let value = |value: F| OpeningClaim {
            point: Vec::new(),
            value,
        };
        Self {
            ram_read_value: value(stage1.outer.ram_read_value),
            ram_write_value: value(stage1.outer.ram_write_value),
        }
    }
}

pub struct RamReadWriteChecking<F: Field> {
    claims: JoltRelationClaims<F>,
    dimensions: ReadWriteDimensions,
    ram_log_k: usize,
    gamma: F,
    product_tau_low: Vec<F>,
}

impl<F: Field> RamReadWriteChecking<F> {
    pub fn new(
        dimensions: ReadWriteDimensions,
        ram_log_k: usize,
        gamma: F,
        product_tau_low: Vec<F>,
    ) -> Self {
        Self {
            claims: ram::read_write_checking(dimensions),
            dimensions,
            ram_log_k,
            gamma,
            product_tau_low,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamReadWriteChecking,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for RamReadWriteChecking<F> {
    type Inputs<C> = RamReadWriteInputClaims<C>;
    type Outputs<C> = RamReadWriteOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &RamReadWriteInputClaims<C>,
    ) -> Result<RamReadWriteOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .dimensions
            .read_write_opening_point(sumcheck_point)
            .map_err(public_input_failed)?
            .opening_point;
        Ok(RamReadWriteOutputClaims {
            val: opening_point.clone(),
            ra: opening_point.clone(),
            inc: opening_point,
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma) => Ok(self.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        _inputs: &RamReadWriteInputClaims<C>,
        outputs: &RamReadWriteOutputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        let JoltPublicId::RamReadWrite(public_id) = id else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        match public_id {
            // The opening point is `[r_address(log_k) || r_cycle(log_t)]`, so the
            // cycle sub-point is the suffix past the address bits.
            RamReadWritePublic::EqCycle => {
                let r_cycle = &outputs.val.point()[self.ram_log_k..];
                try_eq_mle(&self.product_tau_low, r_cycle).map_err(public_input_failed)
            }
        }
    }
}
