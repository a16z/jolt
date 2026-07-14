//! The cycle phase of the bytecode read-RAF symbolic sumcheck.

use jolt_field::RingCore;

use super::{BytecodeReadRafCycleShape, BytecodeReadRafInputClaims, BytecodeReadRafOutputClaims};
use crate::protocols::jolt::geometry::bytecode::{
    bytecode_read_raf_address_phase_opening, read_raf_cycle_output,
};
use crate::protocols::jolt::{
    BytecodeReadRafChallenge, JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId,
    JoltRelationId,
};
use crate::{opening, SumcheckChallenges, SymbolicSumcheck};

/// Fiat-Shamir challenge drawn by the cycle phase of the bytecode read-RAF
/// sumcheck.
#[derive(Clone, Copy, Debug, SumcheckChallenges)]
pub struct BytecodeReadRafCyclePhaseChallenges<F> {
    #[challenge(BytecodeReadRafChallenge::Gamma)]
    pub gamma: F,
}

/// The cycle phase of the bytecode read-RAF sumcheck: starts from the staged
/// address-phase opening and reduces to the bytecode-table cycle output.
pub struct ReadRafCyclePhase {
    shape: BytecodeReadRafCycleShape,
}

impl SymbolicSumcheck for ReadRafCyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafCycleShape;
    type Challenges<F> = BytecodeReadRafCyclePhaseChallenges<F>;
    type Inputs<C> = BytecodeReadRafInputClaims<C>;
    type Outputs<C> = BytecodeReadRafOutputClaims<C>;

    fn new(shape: BytecodeReadRafCycleShape) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeReadRaf
    }

    fn rounds(&self) -> usize {
        self.shape.0.log_t()
    }

    fn degree(&self) -> usize {
        self.shape.0.num_committed_ra_polys() + 1
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        read_raf_cycle_output(self.shape.0, self.shape.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use crate::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;

    fn dimensions(num_committed_ra_polys: usize) -> BytecodeReadRafDimensions {
        BytecodeReadRafDimensions::new(5, 10, num_committed_ra_polys)
    }

    #[test]
    fn read_raf_cycle_phase_symbolic_matches_dependencies() {
        let relation = ReadRafCyclePhase::new((dimensions(2), NUM_BYTECODE_VAL_STAGES));
        assert_eq!(ReadRafCyclePhase::id(), JoltRelationId::BytecodeReadRaf);
        assert_eq!(relation.rounds(), dimensions(2).log_t());
        assert_eq!(
            relation.degree(),
            dimensions(2).num_committed_ra_polys() + 1
        );
    }
}
