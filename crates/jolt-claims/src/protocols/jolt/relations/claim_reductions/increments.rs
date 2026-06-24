//! Increment claim-reduction symbolic sumcheck relation.

use jolt_field::RingCore;

use crate::protocols::jolt::formulas::claim_reductions::increments::{
    inc_challenge, inc_public, ram_inc_read_write, ram_inc_reduced, ram_inc_val_check,
    rd_inc_read_write, rd_inc_reduced, rd_inc_val_evaluation,
};
use crate::protocols::jolt::{
    IncClaimReductionChallenge, IncClaimReductionPublic, JoltChallengeId, JoltExpr, JoltOpeningId,
    JoltPublicId, JoltRelationId, JoltSumcheckSpec, TraceDimensions,
};
use crate::{opening, SymbolicSumcheck};

/// Batches the RAM/register increment openings (`RamInc` read-write and
/// val-check, `RdInc` read-write and val-evaluation) by `gamma` and reduces
/// them to the increment-claim-reduction openings weighted by the eq publics.
pub struct ClaimReduction {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for ClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
    type ChallengeId = JoltChallengeId;
    type Shape = TraceDimensions;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::IncClaimReduction
    }

    fn sumcheck(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck(2)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = inc_challenge(IncClaimReductionChallenge::Gamma);

        opening(ram_inc_read_write())
            + gamma.clone() * opening(ram_inc_val_check())
            + gamma.clone().pow(2) * opening(rd_inc_read_write())
            + gamma.clone().pow(3) * opening(rd_inc_val_evaluation())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = inc_challenge(IncClaimReductionChallenge::Gamma);

        let ram_output_coeff = inc_public(IncClaimReductionPublic::EqRamReadWrite)
            + gamma.clone() * inc_public(IncClaimReductionPublic::EqRamValCheck);
        let rd_output_coeff = inc_public(IncClaimReductionPublic::EqRegistersReadWrite)
            + gamma.clone() * inc_public(IncClaimReductionPublic::EqRegistersValEvaluation);
        ram_output_coeff * opening(ram_inc_reduced())
            + gamma.pow(2) * rd_output_coeff * opening(rd_inc_reduced())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    fn dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    #[test]
    fn claim_reduction_exposes_expected_dependencies() {
        let relation = ClaimReduction::new(dimensions());

        assert_eq!(ClaimReduction::id(), JoltRelationId::IncClaimReduction);
        assert_eq!(relation.sumcheck(), dimensions().sumcheck(2));
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            vec![
                ram_inc_read_write(),
                ram_inc_val_check(),
                rd_inc_read_write(),
                rd_inc_val_evaluation(),
            ]
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![ram_inc_reduced(), rd_inc_reduced()]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(IncClaimReductionChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![
                JoltPublicId::from(IncClaimReductionPublic::EqRamReadWrite),
                JoltPublicId::from(IncClaimReductionPublic::EqRamValCheck),
                JoltPublicId::from(IncClaimReductionPublic::EqRegistersReadWrite),
                JoltPublicId::from(IncClaimReductionPublic::EqRegistersValEvaluation),
            ]
        );
    }
}
