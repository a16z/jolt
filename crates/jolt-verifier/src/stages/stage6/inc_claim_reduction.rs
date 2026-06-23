//! The stage 6 `IncClaimReduction` cycle-phase sumcheck instance.
//!
//! Reduces the RAM-`Inc` claims from RAM read-write checking and RAM value-check,
//! and the register-`Inc` claims from register read-write checking and register
//! value-evaluation, into the single committed `RamInc` / `RdInc` openings that
//! anchor the stage-8 final batched opening. Its publics are the per-source `Eq`
//! coefficients comparing this sumcheck's cycle to each source's cycle.

use jolt_claims::protocols::jolt::{
    formulas::{claim_reductions::increments, dimensions::TraceDimensions},
    IncClaimReductionChallenge, IncClaimReductionPublic, JoltChallengeId, JoltPublicId,
    JoltRelationClaims, JoltRelationId,
};
use jolt_field::Field;
use jolt_poly::try_eq_mle;
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, SumcheckInstance};
use crate::stages::{
    stage2::Stage2ClearOutput, stage4::Stage4ClearOutput, stage5::Stage5ClearOutput,
};
use crate::VerifierError;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(IncClaimReduction)]
pub struct IncClaimReductionOutputClaims<C> {
    #[opening(committed = RamInc)]
    pub ram_inc: C,
    #[opening(committed = RdInc)]
    pub rd_inc: C,
}

/// The four reduced `Inc` openings consumed from the read-write / value
/// relations of RAM and registers.
#[derive(Clone, Debug, InputClaims)]
pub struct IncClaimReductionInputClaims<C> {
    #[opening(committed = RamInc, from = RamReadWriteChecking)]
    pub ram_inc_read_write: C,
    #[opening(committed = RamInc, from = RamValCheck)]
    pub ram_inc_val_check: C,
    #[opening(committed = RdInc, from = RegistersReadWriteChecking)]
    pub rd_inc_read_write: C,
    #[opening(committed = RdInc, from = RegistersValEvaluation)]
    pub rd_inc_val_evaluation: C,
}

impl<F: Field> IncClaimReductionInputClaims<OpeningClaim<F>> {
    pub fn from_upstream(
        stage2: &Stage2ClearOutput<F>,
        stage4: &Stage4ClearOutput<F>,
        stage5: &Stage5ClearOutput<F>,
    ) -> Self {
        Self {
            ram_inc_read_write: stage2.output_claims.ram_read_write.inc.clone(),
            ram_inc_val_check: stage4.output_claims.ram_val_check.ram_inc.clone(),
            rd_inc_read_write: stage4.output_claims.registers_read_write.rd_inc.clone(),
            rd_inc_val_evaluation: stage5.output_claims.registers_val_evaluation.rd_inc.clone(),
        }
    }
}

pub struct IncClaimReduction<F: Field> {
    claims: JoltRelationClaims<F>,
    gamma: F,
    ram_read_write_cycle: Vec<F>,
    ram_val_check_cycle: Vec<F>,
    registers_read_write_cycle: Vec<F>,
    registers_val_evaluation_cycle: Vec<F>,
}

impl<F: Field> IncClaimReduction<F> {
    pub fn new(
        trace_dimensions: TraceDimensions,
        gamma: F,
        ram_read_write_cycle: Vec<F>,
        ram_val_check_cycle: Vec<F>,
        registers_read_write_cycle: Vec<F>,
        registers_val_evaluation_cycle: Vec<F>,
    ) -> Self {
        Self {
            claims: increments::claim_reduction(trace_dimensions),
            gamma,
            ram_read_write_cycle,
            ram_val_check_cycle,
            registers_read_write_cycle,
            registers_val_evaluation_cycle,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::IncClaimReduction,
        reason: reason.to_string(),
    }
}

impl<F: Field> SumcheckInstance<F> for IncClaimReduction<F> {
    type Inputs<C> = IncClaimReductionInputClaims<C>;
    type Outputs<C> = IncClaimReductionOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &IncClaimReductionInputClaims<C>,
    ) -> Result<IncClaimReductionOutputClaims<Vec<F>>, VerifierError> {
        // Both reduced openings share the cycle opening point (the reversed
        // sumcheck point).
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(IncClaimReductionOutputClaims {
            ram_inc: opening_point.clone(),
            rd_inc: opening_point,
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::IncClaimReduction(IncClaimReductionChallenge::Gamma) => Ok(self.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        _inputs: &IncClaimReductionInputClaims<C>,
        outputs: &IncClaimReductionOutputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        let JoltPublicId::IncClaimReduction(public) = id else {
            return Err(VerifierError::MissingStageClaimPublic { id: *id });
        };
        let opening_point = outputs.ram_inc.point();
        let cycle = match public {
            IncClaimReductionPublic::EqRamReadWrite => &self.ram_read_write_cycle,
            IncClaimReductionPublic::EqRamValCheck => &self.ram_val_check_cycle,
            IncClaimReductionPublic::EqRegistersReadWrite => &self.registers_read_write_cycle,
            IncClaimReductionPublic::EqRegistersValEvaluation => {
                &self.registers_val_evaluation_cycle
            }
        };
        try_eq_mle(opening_point, cycle).map_err(public_input_failed)
    }
}
