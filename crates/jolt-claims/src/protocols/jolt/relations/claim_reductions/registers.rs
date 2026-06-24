//! Registers claim-reduction symbolic sumcheck relation.

use jolt_field::RingCore;

use crate::protocols::jolt::formulas::claim_reductions::registers::{
    rd_write_value_reduced, rd_write_value_spartan, reduction_challenge, reduction_public,
    rs1_value_reduced, rs1_value_spartan, rs2_value_reduced, rs2_value_spartan,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltExpr, JoltOpeningId, JoltPublicId, JoltRelationId, JoltSumcheckSpec,
    RegistersClaimReductionChallenge, RegistersClaimReductionPublic, TraceDimensions,
};
use crate::{opening, SymbolicSumcheck};

/// Batches the Spartan-outer register openings (`RdWriteValue`, `Rs1Value`,
/// `Rs2Value`) by `gamma` and reduces them to the registers-claim-reduction
/// openings weighted by the `EqSpartan` public.
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
        JoltRelationId::RegistersClaimReduction
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck(2)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = reduction_challenge(RegistersClaimReductionChallenge::Gamma);

        opening(rd_write_value_spartan())
            + gamma.clone() * opening(rs1_value_spartan())
            + gamma.clone().pow(2) * opening(rs2_value_spartan())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = reduction_challenge(RegistersClaimReductionChallenge::Gamma);
        let eq_spartan = reduction_public(RegistersClaimReductionPublic::EqSpartan);

        eq_spartan.clone() * opening(rd_write_value_reduced())
            + eq_spartan.clone() * gamma.clone() * opening(rs1_value_reduced())
            + eq_spartan * gamma.pow(2) * opening(rs2_value_reduced())
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

        assert_eq!(
            ClaimReduction::id(),
            JoltRelationId::RegistersClaimReduction
        );
        assert_eq!(relation.spec(), dimensions().sumcheck(2));
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            vec![
                rd_write_value_spartan(),
                rs1_value_spartan(),
                rs2_value_spartan(),
            ]
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![
                rd_write_value_reduced(),
                rs1_value_reduced(),
                rs2_value_reduced(),
            ]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(
                RegistersClaimReductionChallenge::Gamma
            )]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(RegistersClaimReductionPublic::EqSpartan)]
        );
    }
}
