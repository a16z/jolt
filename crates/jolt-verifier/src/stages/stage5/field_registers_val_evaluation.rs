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

use jolt_claims::protocols::field_inline::relations::registers::ValEvaluation;
pub use jolt_claims::protocols::field_inline::relations::registers::{
    FieldRegistersValEvaluationInputClaims, FieldRegistersValEvaluationOutputClaims,
};
use jolt_claims::protocols::field_inline::{
    FieldInlineDerivedId, FieldRegistersTraceDimensions, FieldRegistersValEvaluationPublic,
    FIELD_REGISTERS_LOG_K,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::JoltField;

use crate::stages::derivations;
use crate::stages::relations::{project_public, stage_claim_failed, ConcreteSumcheck};
use crate::VerifierError;

#[derive(Clone)]
pub struct FieldRegistersValEvaluation<F: JoltField> {
    symbolic: ValEvaluation,
    trace_dimensions: FieldRegistersTraceDimensions,
    _field: PhantomData<F>,
}

impl<F: JoltField> FieldRegistersValEvaluation<F> {
    pub fn new(trace_dimensions: FieldRegistersTraceDimensions) -> Self {
        Self {
            symbolic: ValEvaluation::new(trace_dimensions),
            trace_dimensions,
            _field: PhantomData,
        }
    }

    pub fn trace_dimensions(&self) -> FieldRegistersTraceDimensions {
        self.trace_dimensions
    }
}

// Only the point geometry stays hand-written: the symbolic output expression
// references the opening point and `LtCycle` as opaque `Derived` leaves, so
// their derivations cannot come from it. Everything else (claim evaluation,
// struct fill, id projection) is trait defaults + derive-generated code.
impl<F: JoltField> ConcreteSumcheck<F> for FieldRegistersValEvaluation<F> {
    type Symbolic = ValEvaluation;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        input_points: &FieldRegistersValEvaluationInputClaims<Vec<F>>,
    ) -> Result<FieldRegistersValEvaluationOutputClaims<Vec<F>>, VerifierError> {
        let address = derivations::val_evaluation_address(
            input_points.registers_val(),
            FIELD_REGISTERS_LOG_K,
            self.trace_dimensions.log_t(),
            "field-register",
        )
        .map_err(|reason| stage_claim_failed(self.id(), reason))?;
        let cycle = self
            .trace_dimensions
            .cycle_opening_point(sumcheck_point)
            .map_err(|reason| stage_claim_failed(self.id(), reason))?;
        Ok(FieldRegistersValEvaluationOutputClaims::from_shared_point(
            [address, cycle.as_slice()].concat(),
        ))
    }

    fn derive_output_term(
        &self,
        id: &FieldInlineDerivedId,
        input_points: &FieldRegistersValEvaluationInputClaims<Vec<F>>,
        output_points: &FieldRegistersValEvaluationOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        match project_public(id)? {
            // Own cycle sub-point first, upstream FR read/write cycle second —
            // literally the ordinary `RegistersValEvaluation` `LtCycle`
            // derivation at the FR geometry (the spec's
            // `Lt(r_field_val.cycle, r_field_rw.cycle)`).
            FieldRegistersValEvaluationPublic::LtCycle => derivations::lt_at_cycle(
                output_points.rd_inc(),
                input_points.registers_val(),
                FIELD_REGISTERS_LOG_K,
                "field-register",
            )
            .map_err(|reason| stage_claim_failed(self.id(), reason)),
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
    use jolt_poly::LtPolynomial;

    use jolt_field::{Fr, Ring};

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
}
