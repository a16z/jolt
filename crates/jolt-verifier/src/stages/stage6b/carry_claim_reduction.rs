//! The stage 6 `CarryClaimReduction` cycle-phase sumcheck instance
//! (implicit-carry).
//!
//! Reduces the committed `Carry` column's openings from product
//! virtualization and the shift sumcheck, together with the `carry_init`
//! all-zeros public pair (claim 0, enforcing `Carry(0) = 0`), into the single
//! committed `Carry` opening that anchors the stage-8 final batched opening.
//! Its publics are the `Eq` coefficients comparing this sumcheck's cycle to
//! each source's cycle, plus the all-zeros selector.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::claim_reductions::carry::{
    CarryClaimReductionChallenges, CarryClaimReductionInputClaims, CarryClaimReductionOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::TraceDimensions, CarryClaimReductionPublic, JoltDerivedId, JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;
use jolt_poly::{try_eq_mle, EqPolynomial};

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::{
    stage2::{Stage2BatchOutputClaims, Stage2BatchOutputPoints},
    stage3::{Stage3OutputClaims, Stage3OutputPoints},
};
use crate::VerifierError;

/// Wire the two consumed `Carry` opening *values*. Clear-only.
pub fn carry_claim_reduction_input_values_from_upstream<F: JoltField>(
    stage2: &Stage2BatchOutputClaims<F>,
    stage3: &Stage3OutputClaims<F>,
) -> CarryClaimReductionInputClaims<F> {
    CarryClaimReductionInputClaims {
        carry_product: stage2.product_remainder.carry,
        carry_shift: stage3.shift.carry,
    }
}

/// Wire the two consumed `Carry` opening *points*. ZK-agnostic.
pub fn carry_claim_reduction_input_points_from_upstream<F: JoltField>(
    stage2: &Stage2BatchOutputPoints<F>,
    stage3: &Stage3OutputPoints<F>,
) -> CarryClaimReductionInputClaims<Vec<F>> {
    CarryClaimReductionInputClaims {
        carry_product: stage2.product_remainder.carry().to_vec(),
        carry_shift: stage3.shift.carry().to_vec(),
    }
}

#[derive(Clone)]
pub struct CarryClaimReduction<F: JoltField> {
    symbolic: relations::claim_reductions::carry::CarryReduction,
    product_cycle: Vec<F>,
    shift_cycle: Vec<F>,
}

impl<F: JoltField> CarryClaimReduction<F> {
    pub fn new(
        trace_dimensions: TraceDimensions,
        product_cycle: Vec<F>,
        shift_cycle: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::claim_reductions::carry::CarryReduction::new(trace_dimensions),
            product_cycle,
            shift_cycle,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::CarryClaimReduction,
        reason: reason.to_string(),
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for CarryClaimReduction<F> {
    type Symbolic = relations::claim_reductions::carry::CarryReduction;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &CarryClaimReductionInputClaims<Vec<F>>,
    ) -> Result<CarryClaimReductionOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(CarryClaimReductionOutputClaims {
            carry: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &CarryClaimReductionInputClaims<Vec<F>>,
        output_points: &CarryClaimReductionOutputClaims<Vec<F>>,
        _challenges: &CarryClaimReductionChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::CarryClaimReduction(public) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let opening_point = output_points.carry();
        match public {
            CarryClaimReductionPublic::EqCarryProduct => {
                try_eq_mle(opening_point, &self.product_cycle).map_err(public_input_failed)
            }
            CarryClaimReductionPublic::EqCarryShift => {
                try_eq_mle(opening_point, &self.shift_cycle).map_err(public_input_failed)
            }
            CarryClaimReductionPublic::EqZeroSelector => {
                Ok(EqPolynomial::zero_selector(opening_point))
            }
        }
    }
}
