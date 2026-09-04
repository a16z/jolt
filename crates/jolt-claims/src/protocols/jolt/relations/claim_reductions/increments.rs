//! Increment claim-reduction symbolic sumcheck relation.

use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::increments::{
    ram_inc_reduced, rd_inc_reduced,
};
use crate::protocols::jolt::geometry::ram::{ram_inc, ram_inc_val_check};
use crate::protocols::jolt::geometry::registers::{rd_inc_read_write, rd_inc_val_evaluation};
use crate::protocols::jolt::{
    IncClaimReductionChallenge, IncClaimReductionPublic, JoltChallengeId, JoltDerivedId,
    JoltOpeningId, JoltRelationId, TraceDimensions,
};
use crate::twist::claim_reductions as twist;
use crate::{InputClaims, OutputClaims, SumcheckChallenges};

#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
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
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct IncClaimReductionChallenges<F> {
    #[challenge(IncClaimReductionChallenge::Gamma)]
    pub gamma: F,
}

/// Batches the RAM/register increment openings (`RamInc` read-write and
/// val-check, `RdInc` read-write and val-evaluation) by `gamma` and reduces
/// them to the increment-claim-reduction openings weighted by the eq publics.
#[derive(Clone)]
pub struct ClaimReduction {
    shape: TraceDimensions,
}

// RAM group first, registers group second — the relation's γ offset order.
// The lattice `read_raf` fold consumes the same four openings in this order at
// its own gamma powers.
twist::instantiate_increment_reduction! {
    relation = ClaimReduction,
    id = JoltRelationId::IncClaimReduction,
    ids = (JoltRelationId, JoltOpeningId, JoltDerivedId, JoltChallengeId),
    dimensions = TraceDimensions,
    challenges = IncClaimReductionChallenges,
    inputs = IncClaimReductionInputClaims,
    outputs = IncClaimReductionOutputClaims,
    groups = vec![
        twist::IncrementReductionGroup {
            consumed: [ram_inc(), ram_inc_val_check()],
            eq_publics: [
                IncClaimReductionPublic::EqRamReadWrite.into(),
                IncClaimReductionPublic::EqRamValCheck.into(),
            ],
            reduced: ram_inc_reduced(),
        },
        twist::IncrementReductionGroup {
            consumed: [rd_inc_read_write(), rd_inc_val_evaluation()],
            eq_publics: [
                IncClaimReductionPublic::EqRegistersReadWrite.into(),
                IncClaimReductionPublic::EqRegistersValEvaluation.into(),
            ],
            reduced: rd_inc_reduced(),
        },
    ],
    gamma = IncClaimReductionChallenge::Gamma,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ram::{ram_inc, ram_inc_val_check};
    use crate::protocols::jolt::geometry::registers::{rd_inc_read_write, rd_inc_val_evaluation};
    use crate::SymbolicSumcheck;
    use jolt_field::{Fr, Ring};

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
