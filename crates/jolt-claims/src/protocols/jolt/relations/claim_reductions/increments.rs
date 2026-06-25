//! Increment claim-reduction symbolic sumcheck relation.

use jolt_field::RingCore;

use crate::protocols::jolt::formulas::claim_reductions::increments::{
    ram_inc_read_write, ram_inc_reduced, ram_inc_val_check, rd_inc_read_write, rd_inc_reduced,
    rd_inc_val_evaluation,
};
use crate::protocols::jolt::{
    IncClaimReductionChallenge, IncClaimReductionPublic, JoltChallengeId, JoltExpr, JoltOpeningId,
    JoltPublicId, JoltRelationId, JoltSumcheckSpec, TraceDimensions,
};
use crate::{challenge, opening, public, SymbolicSumcheck};

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

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck(2)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(IncClaimReductionChallenge::Gamma);

        opening(ram_inc_read_write())
            + gamma.clone() * opening(ram_inc_val_check())
            + gamma.clone().pow(2) * opening(rd_inc_read_write())
            + gamma.clone().pow(3) * opening(rd_inc_val_evaluation())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(IncClaimReductionChallenge::Gamma);

        let ram_output_coeff = public(IncClaimReductionPublic::EqRamReadWrite)
            + gamma.clone() * public(IncClaimReductionPublic::EqRamValCheck);
        let rd_output_coeff = public(IncClaimReductionPublic::EqRegistersReadWrite)
            + gamma.clone() * public(IncClaimReductionPublic::EqRegistersValEvaluation);
        ram_output_coeff * opening(ram_inc_reduced())
            + gamma.pow(2) * rd_output_coeff * opening(rd_inc_reduced())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    #[test]
    fn claim_reduction_evaluates_like_core_formula() {
        let relation = ClaimReduction::new(dimensions());

        let ram_rw = Fr::from_u64(3);
        let ram_val = Fr::from_u64(5);
        let rd_rw = Fr::from_u64(7);
        let rd_val = Fr::from_u64(11);
        let ram_reduced = Fr::from_u64(13);
        let rd_reduced = Fr::from_u64(17);
        let eq_ram_rw = Fr::from_u64(19);
        let eq_ram_val = Fr::from_u64(23);
        let eq_rd_rw = Fr::from_u64(29);
        let eq_rd_val = Fr::from_u64(31);
        let gamma = Fr::from_u64(37);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_inc_read_write() => ram_rw,
                id if id == ram_inc_val_check() => ram_val,
                id if id == rd_inc_read_write() => rd_rw,
                id if id == rd_inc_val_evaluation() => rd_val,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::IncClaimReduction(IncClaimReductionChallenge::Gamma) => gamma,
                _ => zero,
            },
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_inc_reduced() => ram_reduced,
                id if id == rd_inc_reduced() => rd_reduced,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::IncClaimReduction(IncClaimReductionChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRamReadWrite) => {
                    eq_ram_rw
                }
                JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRamValCheck) => {
                    eq_ram_val
                }
                JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRegistersReadWrite) => {
                    eq_rd_rw
                }
                JoltPublicId::IncClaimReduction(
                    IncClaimReductionPublic::EqRegistersValEvaluation,
                ) => eq_rd_val,
                _ => zero,
            },
        );

        let gamma_2 = gamma * gamma;
        assert_eq!(
            input,
            ram_rw + gamma * ram_val + gamma_2 * rd_rw + gamma_2 * gamma * rd_val
        );
        assert_eq!(
            output,
            ram_reduced * (eq_ram_rw + gamma * eq_ram_val)
                + gamma_2 * rd_reduced * (eq_rd_rw + gamma * eq_rd_val)
        );
    }

    #[test]
    fn claim_reduction_exposes_expected_dependencies() {
        let relation = ClaimReduction::new(dimensions());

        assert_eq!(ClaimReduction::id(), JoltRelationId::IncClaimReduction);
        assert_eq!(relation.spec(), dimensions().sumcheck(2));
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
