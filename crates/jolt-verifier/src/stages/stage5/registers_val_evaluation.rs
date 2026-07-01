//! The stage 5 `RegistersValEvaluation` sumcheck instance.

use core::marker::PhantomData;

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::registers::{
    RegistersValEvaluationInputClaims, RegistersValEvaluationOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::{TraceDimensions, REGISTER_ADDRESS_BITS},
    JoltDerivedId, JoltRelationId, RegistersValEvaluationPublic,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::LtPolynomial;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage4::{Stage4OutputClaims, Stage4OutputPoints};
use crate::VerifierError;

/// Wire the consumed `RegistersVal` opening *value* from the upstream register
/// read-write checking (stage 4). Takes the ZK-agnostic output-claims aggregate.
pub fn registers_val_evaluation_input_values_from_upstream<F: Field>(
    stage4: &Stage4OutputClaims<F>,
) -> RegistersValEvaluationInputClaims<F> {
    RegistersValEvaluationInputClaims {
        registers_val: stage4.registers_read_write.registers_val,
    }
}

/// Wire the consumed `RegistersVal` opening *point* from the upstream register
/// read-write checking (stage 4).
pub fn registers_val_evaluation_input_points_from_upstream<F: Field>(
    stage4: &Stage4OutputPoints<F>,
) -> RegistersValEvaluationInputClaims<Vec<F>> {
    RegistersValEvaluationInputClaims {
        registers_val: stage4.registers_read_write_point().to_vec(),
    }
}

pub struct RegistersValEvaluation<F: Field> {
    symbolic: relations::registers::ValEvaluation,
    trace_dimensions: TraceDimensions,
    _field: PhantomData<F>,
}

impl<F: Field> RegistersValEvaluation<F> {
    pub fn new(trace_dimensions: TraceDimensions) -> Self {
        Self {
            symbolic: relations::registers::ValEvaluation::new(trace_dimensions),
            trace_dimensions,
            _field: PhantomData,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RegistersValEvaluation,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for RegistersValEvaluation<F> {
    type Symbolic = relations::registers::ValEvaluation;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        input_points: &RegistersValEvaluationInputClaims<Vec<F>>,
    ) -> Result<RegistersValEvaluationOutputClaims<Vec<F>>, VerifierError> {
        let expected_len = REGISTER_ADDRESS_BITS + self.trace_dimensions.log_t();
        let register_point = input_points.registers_val();
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
        Ok(RegistersValEvaluationOutputClaims {
            rd_inc: opening_point.clone(),
            rd_wa: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        input_points: &RegistersValEvaluationInputClaims<Vec<F>>,
        output_points: &RegistersValEvaluationOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        match id {
            JoltDerivedId::RegistersValEvaluation(RegistersValEvaluationPublic::LtCycle) => {
                let registers_cycle = &output_points.rd_inc()[REGISTER_ADDRESS_BITS..];
                let fixed_cycle = &input_points.registers_val()[REGISTER_ADDRESS_BITS..];
                Ok(LtPolynomial::evaluate(registers_cycle, fixed_cycle))
            }
            _ => Err(VerifierError::MissingStageClaimDerived { id: *id }),
        }
    }
}
