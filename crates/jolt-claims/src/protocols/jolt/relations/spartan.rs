//! Spartan symbolic sumcheck relations.

use jolt_field::RingCore;

use crate::opening;
use crate::protocols::jolt::formulas::spartan::{
    is_first_in_sequence_shift, is_noop_shift, is_virtual_shift, next_is_first_in_sequence_outer,
    next_is_noop_product, next_is_virtual_outer, next_pc_outer, next_unexpanded_pc_outer, pc_shift,
    shift_challenge, shift_public, unexpanded_pc_shift, SHIFT_DEGREE,
};
use crate::protocols::jolt::{
    JoltExpr, JoltRelationId, JoltSumcheckSpec, SpartanShiftChallenge, SpartanShiftPublic,
    TraceDimensions,
};
use crate::SymbolicSumcheck;

/// The Spartan shift sumcheck: relates each `Next*` column from the outer
/// sumcheck (and `next_is_noop` from the product remainder) to the shifted
/// column at the same cycle, folded by `gamma` and weighted by the `EqPlusOne`
/// publics.
pub struct Shift {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for Shift {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::SpartanShift
    }

    fn sumcheck(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck(SHIFT_DEGREE)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = shift_challenge(SpartanShiftChallenge::Gamma);
        opening(next_unexpanded_pc_outer())
            + gamma.clone() * opening(next_pc_outer())
            + gamma.clone().pow(2) * opening(next_is_virtual_outer())
            + gamma.clone().pow(3) * opening(next_is_first_in_sequence_outer())
            + gamma.pow(4) * (JoltExpr::one() - opening(next_is_noop_product()))
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = shift_challenge(SpartanShiftChallenge::Gamma);
        shift_public(SpartanShiftPublic::EqPlusOneOuter)
            * (opening(unexpanded_pc_shift())
                + gamma.clone() * opening(pc_shift())
                + gamma.clone().pow(2) * opening(is_virtual_shift())
                + gamma.clone().pow(3) * opening(is_first_in_sequence_shift()))
            + shift_public(SpartanShiftPublic::EqPlusOneProduct)
                * gamma.pow(4)
                * (JoltExpr::one() - opening(is_noop_shift()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{JoltChallengeId, JoltPublicId};
    use jolt_field::Fr;

    #[test]
    fn shift_symbolic_matches_dependencies() {
        let relation = Shift::new(TraceDimensions::new(5));
        assert_eq!(Shift::id(), JoltRelationId::SpartanShift);
        assert_eq!(
            relation.sumcheck(),
            TraceDimensions::new(5).sumcheck(SHIFT_DEGREE)
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(SpartanShiftChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![
                JoltPublicId::from(SpartanShiftPublic::EqPlusOneOuter),
                JoltPublicId::from(SpartanShiftPublic::EqPlusOneProduct),
            ]
        );
    }
}
