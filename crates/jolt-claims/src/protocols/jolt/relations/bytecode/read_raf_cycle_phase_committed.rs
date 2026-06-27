//! The committed-program cycle phase of the bytecode read-RAF symbolic sumcheck.

use jolt_field::RingCore;

use crate::protocols::jolt::geometry::bytecode::{
    bytecode_read_raf_address_phase_opening, read_raf_cycle_output_committed,
    BytecodeReadRafDimensions,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, JoltSumcheckSpec,
};
use crate::{opening, SymbolicSumcheck};

/// Committed-program cycle phase: the per-stage Val factors come from the
/// `BytecodeValStage(s)` openings staged at the end of the address phase
/// instead of public bytecode-table evaluations.
pub struct ReadRafCyclePhaseCommitted {
    shape: BytecodeReadRafDimensions,
}

impl SymbolicSumcheck for ReadRafCyclePhaseCommitted {
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
        read_raf_cycle_output_committed(self.shape)
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
    fn read_raf_cycle_phase_committed_symbolic_matches_dependencies() {
        let relation = ReadRafCyclePhaseCommitted::new(dimensions(2));
        assert_eq!(
            ReadRafCyclePhaseCommitted::id(),
            JoltRelationId::BytecodeReadRaf
        );
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
