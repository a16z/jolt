//! The stage 5 `RegistersValEvaluation` sumcheck instance.

use jolt_claims::protocols::jolt::{
    formulas::{
        dimensions::{TraceDimensions, REGISTER_ADDRESS_BITS},
        registers,
    },
    JoltChallengeId, JoltRelationClaims, JoltRelationId, RegistersValEvaluationChallenge,
};
use jolt_field::Field;
use jolt_poly::LtPolynomial;

use crate::stages::relations::{GetPoint, OpeningClaim, SumcheckInstance};
use crate::stages::stage5::inputs::{
    RegistersValEvaluationInputs, RegistersValEvaluationOutputOpeningClaims,
};
use crate::VerifierError;

pub struct RegistersValEvaluationRelation<F: Field> {
    claims: JoltRelationClaims<F>,
    trace_dimensions: TraceDimensions,
}

impl<F: Field> RegistersValEvaluationRelation<F> {
    pub fn new(trace_dimensions: TraceDimensions) -> Self {
        Self {
            claims: registers::val_evaluation(trace_dimensions),
            trace_dimensions,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RegistersValEvaluation,
        reason: reason.to_string(),
    }
}

impl<F: Field> SumcheckInstance<F> for RegistersValEvaluationRelation<F> {
    type Inputs<C> = RegistersValEvaluationInputs<C>;
    type Outputs<C> = RegistersValEvaluationOutputOpeningClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn derive_output_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        inputs: &RegistersValEvaluationInputs<C>,
    ) -> Result<RegistersValEvaluationOutputOpeningClaims<Vec<F>>, VerifierError> {
        let expected_len = REGISTER_ADDRESS_BITS + self.trace_dimensions.log_t();
        let register_point = inputs.registers_val.point();
        if register_point.len() != expected_len {
            return Err(public_input_failed(format!(
                "register read-write opening point has {} variables, expected {expected_len}",
                register_point.len()
            )));
        }
        let address = &register_point[..REGISTER_ADDRESS_BITS];
        let cycle = self
            .trace_dimensions
            .cycle_opening_point(sumcheck_point)
            .map_err(public_input_failed)?;
        let opening_point = [address, cycle.as_slice()].concat();
        // rd_inc and rd_wa are opened at the same point.
        Ok(RegistersValEvaluationOutputOpeningClaims {
            rd_inc: opening_point.clone(),
            rd_wa: opening_point,
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        Err(VerifierError::MissingStageClaimChallenge { id: *id })
    }

    fn output_challenge<C: GetPoint<F>>(
        &self,
        id: &JoltChallengeId,
        inputs: &RegistersValEvaluationInputs<C>,
        outputs: &RegistersValEvaluationOutputOpeningClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::RegistersValEvaluation(RegistersValEvaluationChallenge::LtCycle) => {
                let registers_cycle = &outputs.rd_inc.point()[REGISTER_ADDRESS_BITS..];
                let fixed_cycle = &inputs.registers_val.point()[REGISTER_ADDRESS_BITS..];
                Ok(LtPolynomial::evaluate(registers_cycle, fixed_cycle))
            }
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }
}
