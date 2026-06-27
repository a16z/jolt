//! The stage 6 `IncClaimReduction` cycle-phase sumcheck instance.
//!
//! Reduces the RAM-`Inc` claims from RAM read-write checking and RAM value-check,
//! and the register-`Inc` claims from register read-write checking and register
//! value-evaluation, into the single committed `RamInc` / `RdInc` openings that
//! anchor the stage-8 final batched opening. Its publics are the per-source `Eq`
//! coefficients comparing this sumcheck's cycle to each source's cycle.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::claim_reductions::increments::{
    IncClaimReductionChallenges, IncClaimReductionInputClaims, IncClaimReductionOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::TraceDimensions, IncClaimReductionPublic, JoltDerivedId, JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::{
    stage2::Stage2ClearOutput, stage4::Stage4ClearOutput, stage5::Stage5ClearOutput,
};
use crate::VerifierError;

/// Wire the four reduced `Inc` openings from the read-write / value relations of
/// RAM and registers. (Verifier-side constructor for the moved
/// [`IncClaimReductionInputClaims`].)
pub fn inc_claim_reduction_inputs_from_upstream<F: Field>(
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
) -> IncClaimReductionInputClaims<OpeningClaim<F>> {
    IncClaimReductionInputClaims {
        ram_inc_read_write: stage2.output_claims.ram_read_write.inc.clone(),
        ram_inc_val_check: stage4.output_claims.ram_val_check.ram_inc.clone(),
        rd_inc_read_write: stage4.output_claims.registers_read_write.rd_inc.clone(),
        rd_inc_val_evaluation: stage5.output_claims.registers_val_evaluation.rd_inc.clone(),
    }
}

pub struct IncClaimReduction<F: Field> {
    symbolic: relations::claim_reductions::increments::ClaimReduction,
    ram_read_write_cycle: Vec<F>,
    ram_val_check_cycle: Vec<F>,
    registers_read_write_cycle: Vec<F>,
    registers_val_evaluation_cycle: Vec<F>,
}

impl<F: Field> IncClaimReduction<F> {
    pub fn new(
        trace_dimensions: TraceDimensions,
        ram_read_write_cycle: Vec<F>,
        ram_val_check_cycle: Vec<F>,
        registers_read_write_cycle: Vec<F>,
        registers_val_evaluation_cycle: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::claim_reductions::increments::ClaimReduction::new(
                trace_dimensions,
            ),
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

impl<F: Field> ConcreteSumcheck<F> for IncClaimReduction<F> {
    type Symbolic = relations::claim_reductions::increments::ClaimReduction;
    type Inputs<C> = IncClaimReductionInputClaims<C>;
    type Outputs<C> = IncClaimReductionOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
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

    fn derive_output_term<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &IncClaimReductionInputClaims<C>,
        outputs: &IncClaimReductionOutputClaims<OpeningClaim<F>>,
        _challenges: &IncClaimReductionChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::IncClaimReduction(public) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
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
