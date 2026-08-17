//! The stage 5 `FieldRegistersValEvaluation` sumcheck instance — the FR Twist
//! val-evaluation member (spec: `field-inline-protocol.md`, "Stage 5
//! Composition").
//!
//! Consumes the `FieldRegistersVal` opening produced by the stage-4 FR
//! read/write checking and opens `FieldRdInc`/`FieldRdWa` at the same FR
//! address and this instance's cycle point, weighted by the `LtCycle` public.
//! Mirrors the ordinary `RegistersValEvaluation`: `LtCycle = Lt(own cycle
//! sub-point, upstream FR read/write cycle sub-point)`.

use core::marker::PhantomData;

use jolt_claims::protocols::field_inline::relations::registers;
pub use jolt_claims::protocols::field_inline::relations::registers::{
    FieldRegistersValEvaluationInputClaims, FieldRegistersValEvaluationOutputClaims,
};
use jolt_claims::protocols::field_inline::{
    FieldInlineDerivedId, FieldInlineRelationId, FieldRegistersTraceDimensions,
    FieldRegistersValEvaluationPublic, FIELD_REGISTERS_LOG_K,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::LtPolynomial;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage4::{Stage4OutputClaims, Stage4OutputPoints};
use crate::VerifierError;

/// Wire the consumed `FieldRegistersVal` opening *value* from the upstream FR
/// read-write checking (stage 4). The upstream cell is a plain (non-optional)
/// field of the FR-on stage-4 claims, so presence is a compile-time fact.
pub fn field_registers_val_evaluation_input_values_from_upstream<F: Field>(
    stage4: &Stage4OutputClaims<F>,
) -> FieldRegistersValEvaluationInputClaims<F> {
    FieldRegistersValEvaluationInputClaims {
        registers_val: stage4.field_registers_read_write.registers_val,
    }
}

/// Wire the consumed `FieldRegistersVal` opening *point* from the upstream FR
/// read-write checking (stage 4).
pub fn field_registers_val_evaluation_input_points_from_upstream<F: Field>(
    stage4: &Stage4OutputPoints<F>,
) -> FieldRegistersValEvaluationInputClaims<Vec<F>> {
    FieldRegistersValEvaluationInputClaims {
        registers_val: stage4.field_registers_read_write_point().to_vec(),
    }
}

#[derive(Clone)]
pub struct FieldRegistersValEvaluation<F: Field> {
    symbolic: registers::ValEvaluation,
    trace_dimensions: FieldRegistersTraceDimensions,
    _field: PhantomData<F>,
}

impl<F: Field> FieldRegistersValEvaluation<F> {
    pub fn new(trace_dimensions: FieldRegistersTraceDimensions) -> Self {
        Self {
            symbolic: registers::ValEvaluation::new(trace_dimensions),
            trace_dimensions,
            _field: PhantomData,
        }
    }

    pub fn trace_dimensions(&self) -> FieldRegistersTraceDimensions {
        self.trace_dimensions
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage: format!("{:?}", FieldInlineRelationId::FieldRegistersValEvaluation),
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for FieldRegistersValEvaluation<F> {
    type Symbolic = registers::ValEvaluation;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        input_points: &FieldRegistersValEvaluationInputClaims<Vec<F>>,
    ) -> Result<FieldRegistersValEvaluationOutputClaims<Vec<F>>, VerifierError> {
        #[expect(
            clippy::arithmetic_side_effects,
            reason = "FIELD_REGISTERS_LOG_K is a small constant and log_t an ilog2 result (< 64); the sum cannot overflow usize"
        )]
        let expected_len = FIELD_REGISTERS_LOG_K + self.trace_dimensions.log_t();
        let register_point = input_points.registers_val();
        if register_point.len() != expected_len {
            return Err(public_input_failed(format!(
                "field-register read-write opening point has {} variables, expected {expected_len}",
                register_point.len()
            )));
        }
        let address = register_point.get(..FIELD_REGISTERS_LOG_K).ok_or_else(|| {
            public_input_failed(
                "field-register read-write opening point address prefix is out of range",
            )
        })?;
        let cycle = self
            .trace_dimensions
            .cycle_opening_point(sumcheck_point)
            .map_err(public_input_failed)?;
        let opening_point = [address, cycle.as_slice()].concat();
        // rd_inc and rd_wa are opened at the same point.
        Ok(FieldRegistersValEvaluationOutputClaims {
            rd_inc: opening_point.clone(),
            rd_wa: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &FieldInlineDerivedId,
        input_points: &FieldRegistersValEvaluationInputClaims<Vec<F>>,
        output_points: &FieldRegistersValEvaluationOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let FieldInlineDerivedId::FieldRegistersValEvaluation(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
        };
        match public_id {
            // Own cycle sub-point first, upstream FR read/write cycle second —
            // the ordinary `RegistersValEvaluation` `LtCycle` argument order
            // (the spec's `Lt(r_field_val.cycle, r_field_rw.cycle)`).
            FieldRegistersValEvaluationPublic::LtCycle => {
                let registers_cycle = output_points
                    .rd_inc()
                    .get(FIELD_REGISTERS_LOG_K..)
                    .ok_or_else(|| {
                        public_input_failed(
                            "rd_inc opening point is shorter than the field-register address \
                             width",
                        )
                    })?;
                let fixed_cycle = input_points
                    .registers_val()
                    .get(FIELD_REGISTERS_LOG_K..)
                    .ok_or_else(|| {
                        public_input_failed(
                            "field-register read-write opening point is shorter than the \
                             field-register address width",
                        )
                    })?;
                Ok(LtPolynomial::evaluate(registers_cycle, fixed_cycle))
            }
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
#[expect(
    clippy::as_conversions,
    reason = "tests use plain arithmetic on fixture data"
)]
mod tests {
    use super::*;

    use jolt_claims::protocols::jolt::geometry::dimensions::{
        TraceDimensions, REGISTER_ADDRESS_BITS,
    };
    use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationInputClaims;
    use jolt_claims::protocols::jolt::{JoltDerivedId, RegistersValEvaluationPublic};
    use jolt_field::{Fr, FromPrimitiveInt};

    use crate::stages::stage5::registers_val_evaluation::RegistersValEvaluation;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// The FR val-evaluation opening point is the upstream FR address prefix
    /// followed by this instance's reversed cycle point, and `LtCycle`
    /// evaluates over exactly the two cycle sub-points (own cycle, upstream FR
    /// read/write cycle).
    #[test]
    fn opening_point_reuses_upstream_address_and_lt_cycle_uses_the_cycle_points() {
        let log_t = 4usize;
        let relation =
            FieldRegistersValEvaluation::<Fr>::new(FieldRegistersTraceDimensions::new(log_t));

        let upstream_address: Vec<Fr> = (0..FIELD_REGISTERS_LOG_K as u64)
            .map(|i| fr(10 + i))
            .collect();
        let upstream_cycle: Vec<Fr> = (0..log_t as u64).map(|i| fr(20 + i)).collect();
        let input_points = FieldRegistersValEvaluationInputClaims::<Vec<Fr>> {
            registers_val: [upstream_address.as_slice(), upstream_cycle.as_slice()].concat(),
        };
        let point: Vec<Fr> = (0..log_t as u64).map(|i| fr(30 + i)).collect();

        let output_points = relation
            .derive_opening_points(&point, &input_points)
            .unwrap();
        let own_cycle: Vec<Fr> = point.iter().rev().copied().collect();
        assert_eq!(
            output_points.rd_inc(),
            [upstream_address.as_slice(), own_cycle.as_slice()].concat()
        );
        assert_eq!(output_points.rd_inc(), output_points.rd_wa());

        let lt = relation
            .derive_output_term(
                &FieldInlineDerivedId::FieldRegistersValEvaluation(
                    FieldRegistersValEvaluationPublic::LtCycle,
                ),
                &input_points,
                &output_points,
                &NoChallenges::default(),
            )
            .unwrap();
        assert_eq!(lt, LtPolynomial::evaluate(&own_cycle, &upstream_cycle));
    }

    /// The FR `LtCycle` mirrors the ordinary registers val-evaluation
    /// derivation exactly: fed the same two cycle sub-points (behind each
    /// family's own address-prefix width), the two publics are equal.
    #[test]
    fn field_registers_lt_cycle_matches_registers_val_evaluation_derivation() {
        let log_t = 4usize;
        let jolt_relation = RegistersValEvaluation::<Fr>::new(TraceDimensions::new(log_t));
        let field_relation =
            FieldRegistersValEvaluation::<Fr>::new(FieldRegistersTraceDimensions::new(log_t));

        let upstream_cycle: Vec<Fr> = (0..log_t as u64).map(|i| fr(50 + i)).collect();
        let point: Vec<Fr> = (0..log_t as u64).map(|i| fr(70 + i)).collect();

        let jolt_input_points = RegistersValEvaluationInputClaims::<Vec<Fr>> {
            registers_val: [
                vec![fr(1); REGISTER_ADDRESS_BITS].as_slice(),
                upstream_cycle.as_slice(),
            ]
            .concat(),
        };
        let field_input_points = FieldRegistersValEvaluationInputClaims::<Vec<Fr>> {
            registers_val: [
                vec![fr(2); FIELD_REGISTERS_LOG_K].as_slice(),
                upstream_cycle.as_slice(),
            ]
            .concat(),
        };

        let jolt_points = jolt_relation
            .derive_opening_points(&point, &jolt_input_points)
            .unwrap();
        let field_points = field_relation
            .derive_opening_points(&point, &field_input_points)
            .unwrap();
        // Both instances bind the same cycle point behind their address prefixes.
        assert_eq!(
            jolt_points.rd_inc().get(REGISTER_ADDRESS_BITS..),
            field_points.rd_inc().get(FIELD_REGISTERS_LOG_K..),
        );

        let jolt_lt = jolt_relation
            .derive_output_term(
                &JoltDerivedId::RegistersValEvaluation(RegistersValEvaluationPublic::LtCycle),
                &jolt_input_points,
                &jolt_points,
                &NoChallenges::default(),
            )
            .unwrap();
        let field_lt = field_relation
            .derive_output_term(
                &FieldInlineDerivedId::FieldRegistersValEvaluation(
                    FieldRegistersValEvaluationPublic::LtCycle,
                ),
                &field_input_points,
                &field_points,
                &NoChallenges::default(),
            )
            .unwrap();

        assert_eq!(field_lt, jolt_lt);
    }
}
