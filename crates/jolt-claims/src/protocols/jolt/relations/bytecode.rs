//! Bytecode read-RAF symbolic sumcheck relations.

use jolt_field::RingCore;

use crate::protocols::jolt::formulas::bytecode::{
    bytecode_challenge, bytecode_read_raf_address_phase_opening, pc_spartan_outer,
    pc_spartan_shift, read_raf_cycle_output, read_raf_cycle_output_committed, stage1_claim,
    stage2_claim, stage3_claim, stage4_claim, stage5_claim, BytecodeReadRafDimensions,
};
use crate::protocols::jolt::{
    BytecodeReadRafChallenge, JoltChallengeId, JoltExpr, JoltOpeningId, JoltPublicId,
    JoltRelationId, JoltSumcheckSpec,
};
use crate::{opening, SymbolicSumcheck};

/// The full bytecode read-RAF sumcheck: folds the five staged claims plus the
/// Spartan outer/shift PC openings against the bytecode-table cycle output.
pub struct ReadRaf {
    shape: BytecodeReadRafDimensions,
}

impl SymbolicSumcheck for ReadRaf {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafDimensions;

    fn new(shape: BytecodeReadRafDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeReadRaf
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = bytecode_challenge(BytecodeReadRafChallenge::Gamma);

        gamma.clone().pow(7)
            + stage1_claim()
            + gamma.clone() * stage2_claim()
            + gamma.clone().pow(2) * stage3_claim()
            + gamma.clone().pow(3) * stage4_claim()
            + gamma.clone().pow(4) * stage5_claim::<F>()
            + gamma.clone().pow(5) * opening(pc_spartan_outer())
            + gamma.pow(6) * opening(pc_spartan_shift())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        read_raf_cycle_output(self.shape)
    }
}

/// The address phase of the bytecode read-RAF sumcheck: the same folded input
/// claim, reduced to the staged address-phase opening.
pub struct ReadRafAddressPhase {
    shape: BytecodeReadRafDimensions,
}

impl SymbolicSumcheck for ReadRafAddressPhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafDimensions;

    fn new(shape: BytecodeReadRafDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeReadRaf
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.address_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = bytecode_challenge(BytecodeReadRafChallenge::Gamma);

        gamma.clone().pow(7)
            + stage1_claim()
            + gamma.clone() * stage2_claim()
            + gamma.clone().pow(2) * stage3_claim()
            + gamma.clone().pow(3) * stage4_claim()
            + gamma.clone().pow(4) * stage5_claim::<F>()
            + gamma.clone().pow(5) * opening(pc_spartan_outer())
            + gamma.pow(6) * opening(pc_spartan_shift())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }
}

/// The cycle phase of the bytecode read-RAF sumcheck: starts from the staged
/// address-phase opening and reduces to the bytecode-table cycle output.
pub struct ReadRafCyclePhase {
    shape: BytecodeReadRafDimensions,
}

impl SymbolicSumcheck for ReadRafCyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
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

/// Committed-program cycle phase: the per-stage Val factors come from the
/// `BytecodeValStage(s)` openings staged at the end of the address phase
/// instead of public bytecode-table evaluations.
pub struct ReadRafCyclePhaseCommitted {
    shape: BytecodeReadRafDimensions,
}

impl SymbolicSumcheck for ReadRafCyclePhaseCommitted {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
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
    use crate::protocols::jolt::BytecodeReadRafPublic;
    use jolt_field::Fr;

    fn dimensions(num_committed_ra_polys: usize) -> BytecodeReadRafDimensions {
        BytecodeReadRafDimensions::new(5, 10, num_committed_ra_polys)
    }

    fn stage_gammas() -> Vec<JoltChallengeId> {
        vec![
            JoltChallengeId::from(BytecodeReadRafChallenge::Gamma),
            JoltChallengeId::from(BytecodeReadRafChallenge::Stage1Gamma),
            JoltChallengeId::from(BytecodeReadRafChallenge::Stage2Gamma),
            JoltChallengeId::from(BytecodeReadRafChallenge::Stage3Gamma),
            JoltChallengeId::from(BytecodeReadRafChallenge::Stage4Gamma),
            JoltChallengeId::from(BytecodeReadRafChallenge::Stage5Gamma),
        ]
    }

    #[test]
    fn read_raf_symbolic_matches_dependencies() {
        let relation = ReadRaf::new(dimensions(2));
        assert_eq!(ReadRaf::id(), JoltRelationId::BytecodeReadRaf);
        assert_eq!(relation.spec(), dimensions(2).sumcheck());
        assert_eq!(relation.required_challenges::<Fr>(), stage_gammas());
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(0)),
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(1)),
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(2)),
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(3)),
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(4)),
                JoltPublicId::from(BytecodeReadRafPublic::SpartanOuterRaf),
                JoltPublicId::from(BytecodeReadRafPublic::SpartanShiftRaf),
                JoltPublicId::from(BytecodeReadRafPublic::Entry),
            ]
        );
    }

    #[test]
    fn read_raf_address_phase_symbolic_matches_dependencies() {
        let relation = ReadRafAddressPhase::new(dimensions(2));
        assert_eq!(ReadRafAddressPhase::id(), JoltRelationId::BytecodeReadRaf);
        assert_eq!(relation.spec(), dimensions(2).address_sumcheck());
        assert_eq!(relation.required_challenges::<Fr>(), stage_gammas());
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![bytecode_read_raf_address_phase_opening()]
        );
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
