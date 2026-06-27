//! The cycle phase of the bytecode read-RAF symbolic sumcheck.

use jolt_field::RingCore;

use crate::protocols::jolt::geometry::bytecode::{
    bytecode_read_raf_address_phase_opening, read_raf_cycle_output, BytecodeReadRafDimensions,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltExpr, JoltOpeningId, JoltDerivedId, JoltRelationId, JoltSumcheckSpec,
};
use crate::{opening, SymbolicSumcheck};

/// The cycle phase of the bytecode read-RAF sumcheck: starts from the staged
/// address-phase opening and reduces to the bytecode-table cycle output.
pub struct ReadRafCyclePhase {
    shape: BytecodeReadRafDimensions,
}

impl SymbolicSumcheck for ReadRafCyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafDimensions;

    fn new(shape: BytecodeReadRafDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeReadRaf
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.cycle_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        read_raf_cycle_output(self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::BytecodeReadRafChallenge;
    use jolt_field::Fr;

    fn dimensions(num_committed_ra_polys: usize) -> BytecodeReadRafDimensions {
        BytecodeReadRafDimensions::new(5, 10, num_committed_ra_polys)
    }

    #[test]
    fn read_raf_cycle_phase_symbolic_matches_dependencies() {
        let relation = ReadRafCyclePhase::new(dimensions(2));
        assert_eq!(ReadRafCyclePhase::id(), JoltRelationId::BytecodeReadRaf);
        assert_eq!(relation.spec(), dimensions(2).cycle_sumcheck());
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            vec![bytecode_read_raf_address_phase_opening()]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(BytecodeReadRafChallenge::Gamma)]
        );
    }
}
