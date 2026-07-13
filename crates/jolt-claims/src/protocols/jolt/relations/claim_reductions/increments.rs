//! Increment claim-reduction symbolic sumcheck relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::increments::{
    inc_consumers_input, ram_inc_reduced, rd_inc_reduced,
};
use crate::protocols::jolt::{
    IncClaimReductionChallenge, IncClaimReductionPublic, JoltChallengeId, JoltDerivedId, JoltExpr,
    JoltOpeningId, JoltRelationId, TraceDimensions,
};
use crate::{
    challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(IncClaimReduction)]
pub struct IncClaimReductionOutputClaims<C> {
    #[opening(committed = RamInc)]
    pub ram_inc: C,
    #[opening(committed = RdInc)]
    pub rd_inc: C,
}

/// The four reduced `Inc` openings consumed from the read-write / value
/// relations of RAM and registers.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct IncClaimReductionInputClaims<C> {
    #[opening(committed = RamInc, from = RamReadWriteChecking)]
    pub ram_inc_read_write: C,
    #[opening(committed = RamInc, from = RamValCheck)]
    pub ram_inc_val_check: C,
    #[opening(committed = RdInc, from = RegistersReadWriteChecking)]
    pub rd_inc_read_write: C,
    #[opening(committed = RdInc, from = RegistersValEvaluation)]
    pub rd_inc_val_evaluation: C,
}

/// Fiat-Shamir challenge drawn by the increment claim-reduction sumcheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
pub struct IncClaimReductionChallenges<F> {
    #[challenge(IncClaimReductionChallenge::Gamma)]
    pub gamma: F,
}

/// Batches the RAM/register increment openings (`RamInc` read-write and
/// val-check, `RdInc` read-write and val-evaluation) by `gamma` and reduces
/// them to the increment-claim-reduction openings weighted by the eq publics.
pub struct ClaimReduction {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for ClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = IncClaimReductionChallenges<F>;
    type Inputs<C> = IncClaimReductionInputClaims<C>;
    type Outputs<C> = IncClaimReductionOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::IncClaimReduction
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        inc_consumers_input(challenge(IncClaimReductionChallenge::Gamma))
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(IncClaimReductionChallenge::Gamma);

        let ram_output_coeff = derived(IncClaimReductionPublic::EqRamReadWrite)
            + gamma.clone() * derived(IncClaimReductionPublic::EqRamValCheck);
        let rd_output_coeff = derived(IncClaimReductionPublic::EqRegistersReadWrite)
            + gamma.clone() * derived(IncClaimReductionPublic::EqRegistersValEvaluation);
        ram_output_coeff * opening(ram_inc_reduced())
            + gamma.pow(2) * rd_output_coeff * opening(rd_inc_reduced())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ram::{ram_inc, ram_inc_val_check};
    use crate::protocols::jolt::geometry::registers::{rd_inc_read_write, rd_inc_val_evaluation};
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
                id if id == ram_inc() => ram_rw,
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
                JoltDerivedId::IncClaimReduction(IncClaimReductionPublic::EqRamReadWrite) => {
                    eq_ram_rw
                }
                JoltDerivedId::IncClaimReduction(IncClaimReductionPublic::EqRamValCheck) => {
                    eq_ram_val
                }
                JoltDerivedId::IncClaimReduction(IncClaimReductionPublic::EqRegistersReadWrite) => {
                    eq_rd_rw
                }
                JoltDerivedId::IncClaimReduction(
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
        assert_eq!(relation.rounds(), dimensions().log_t());
        assert_eq!(relation.degree(), 2);
    }
}
