//! The stage 4 `RegistersReadWriteChecking` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover (while
//! producing the stage 4 batch proof) and the verifier (after checking it). It
//! owns the register read-write point derivation and the `EqCycle` public-value
//! computation, so the input/output claim algebra lives here once instead of
//! being hand-coded on each side.

use jolt_claims::protocols::jolt::{
    formulas::{
        dimensions::{ReadWriteDimensions, REGISTER_ADDRESS_BITS},
        registers,
    },
    JoltChallengeId, JoltPublicId, JoltRelationClaims, JoltRelationId, RegistersReadWriteChallenge,
    RegistersReadWritePublic,
};
use jolt_field::Field;
use jolt_poly::try_eq_mle;
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, ConcreteSumcheck};
use crate::stages::stage3::outputs::Stage3ClearOutput;
use crate::VerifierError;

/// Produced register read-write openings, all sharing the single read-write
/// opening point. Generic over the cell (`F` on the wire, `Vec<F>` for ZK points,
/// `OpeningClaim<F>` on the clear path).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RegistersReadWriteChecking)]
pub struct RegistersReadWriteOutputClaims<C> {
    #[opening(RegistersVal)]
    pub registers_val: C,
    #[opening(Rs1Ra)]
    pub rs1_ra: C,
    #[opening(Rs2Ra)]
    pub rs2_ra: C,
    #[opening(RdWa)]
    pub rd_wa: C,
    #[opening(committed = RdInc)]
    pub rd_inc: C,
}

/// Consumed register openings reduced by the read-write checking sumcheck, wired
/// from the upstream registers claim-reduction relation (stage 3). Generic over
/// the cell.
#[derive(Clone, Debug, InputClaims)]
pub struct RegistersReadWriteInputClaims<C> {
    #[opening(RdWriteValue, from = RegistersClaimReduction)]
    pub rd_write_value: C,
    #[opening(Rs1Value, from = RegistersClaimReduction)]
    pub rs1_value: C,
    #[opening(Rs2Value, from = RegistersClaimReduction)]
    pub rs2_value: C,
}

impl<F: Field> RegistersReadWriteInputClaims<OpeningClaim<F>> {
    /// Wire the consumed openings from stage 3's registers claim-reduction output,
    /// all sharing that relation's opening point.
    pub fn from_upstream(stage3: &Stage3ClearOutput<F>) -> Self {
        // The stage-3 register-reduction openings already carry their shared
        // opening point (point + value), so the consumed claims are just clones.
        let reduction = &stage3.output_claims.registers_claim_reduction;
        Self {
            rd_write_value: reduction.rd_write_value.clone(),
            rs1_value: reduction.rs1_value.clone(),
            rs2_value: reduction.rs2_value.clone(),
        }
    }
}

pub struct RegistersReadWriteChecking<F: Field> {
    claims: JoltRelationClaims<F>,
    register_dimensions: ReadWriteDimensions,
    gamma: F,
}

impl<F: Field> RegistersReadWriteChecking<F> {
    pub fn new(register_dimensions: ReadWriteDimensions, gamma: F) -> Self {
        Self {
            claims: registers::read_write_checking(register_dimensions),
            register_dimensions,
            gamma,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RegistersReadWriteChecking,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for RegistersReadWriteChecking<F> {
    type Inputs<C> = RegistersReadWriteInputClaims<C>;
    type Outputs<C> = RegistersReadWriteOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &RegistersReadWriteInputClaims<C>,
    ) -> Result<RegistersReadWriteOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .register_dimensions
            .read_write_opening_point(sumcheck_point)
            .map_err(public_input_failed)?
            .opening_point;
        Ok(RegistersReadWriteOutputClaims {
            registers_val: opening_point.clone(),
            rs1_ra: opening_point.clone(),
            rs2_ra: opening_point.clone(),
            rd_wa: opening_point.clone(),
            rd_inc: opening_point,
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::RegistersReadWrite(RegistersReadWriteChallenge::Gamma) => {
                Ok(self.gamma)
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        inputs: &RegistersReadWriteInputClaims<C>,
        outputs: Option<&RegistersReadWriteOutputClaims<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        let outputs =
            outputs.ok_or(VerifierError::MissingStageClaimPublic { id: *id })?;
        let JoltPublicId::RegistersReadWrite(public_id) = id else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        match public_id {
            RegistersReadWritePublic::EqCycle => {
                let fixed_cycle = inputs.rd_write_value.point();
                let registers_cycle = &outputs.registers_val.point()[REGISTER_ADDRESS_BITS..];
                try_eq_mle(fixed_cycle, registers_cycle).map_err(public_input_failed)
            }
        }
    }
}
