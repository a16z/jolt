//! The stage 5 `RegistersValEvaluation` sumcheck instance.

use core::marker::PhantomData;

use jolt_claims::protocols::jolt::relations;
use jolt_claims::protocols::jolt::{
    geometry::dimensions::{TraceDimensions, REGISTER_ADDRESS_BITS},
    JoltPublicId, JoltRelationId, RegistersValEvaluationPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::LtPolynomial;
use jolt_verifier_derive::{InputClaims, OutputClaims};
use serde::{Deserialize, Serialize};

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::stage4::Stage4ClearOutput;
use crate::VerifierError;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RegistersValEvaluation)]
pub struct RegistersValEvaluationOutputClaims<C> {
    #[opening(committed = RdInc)]
    pub rd_inc: C,
    #[opening(RdWa)]
    pub rd_wa: C,
}

/// Consumed register value-evaluation opening, wired from the upstream register
/// read-write checking.
#[derive(Clone, Debug, InputClaims)]
pub struct RegistersValEvaluationInputClaims<C> {
    #[opening(RegistersVal, from = RegistersReadWriteChecking)]
    pub registers_val: C,
}

impl<F: Field> RegistersValEvaluationInputClaims<OpeningClaim<F>> {
    /// Wire the consumed `RegistersVal` opening from the upstream register
    /// read-write checking (stage 4).
    pub fn from_upstream(stage4: &Stage4ClearOutput<F>) -> Self {
        Self {
            registers_val: stage4
                .output_claims
                .registers_read_write
                .registers_val
                .clone(),
        }
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
    type Inputs<C> = RegistersValEvaluationInputClaims<C>;
    type Outputs<C> = RegistersValEvaluationOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        inputs: &RegistersValEvaluationInputClaims<C>,
    ) -> Result<RegistersValEvaluationOutputClaims<Vec<F>>, VerifierError> {
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
        Ok(RegistersValEvaluationOutputClaims {
            rd_inc: opening_point.clone(),
            rd_wa: opening_point,
        })
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltPublicId,
        inputs: &RegistersValEvaluationInputClaims<C>,
        outputs: Option<&RegistersValEvaluationOutputClaims<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        let outputs = outputs.ok_or(VerifierError::MissingStageClaimPublic { id: *id })?;
        match id {
            JoltPublicId::RegistersValEvaluation(RegistersValEvaluationPublic::LtCycle) => {
                let registers_cycle = &outputs.rd_inc.point()[REGISTER_ADDRESS_BITS..];
                let fixed_cycle = &inputs.registers_val.point()[REGISTER_ADDRESS_BITS..];
                Ok(LtPolynomial::evaluate(registers_cycle, fixed_cycle))
            }
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        }
    }
}
