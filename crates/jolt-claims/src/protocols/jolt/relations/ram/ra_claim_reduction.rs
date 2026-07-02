//! RAM `ra` claim-reduction symbolic sumcheck relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::ram::{
    ram_ra, ram_ra_claim_reduction, ram_ra_raf_evaluation, ram_ra_val_check,
};
use crate::protocols::jolt::{
    JoltExpr, JoltRelationId, RamRaClaimReductionChallenge, RamRaClaimReductionPublic,
    TraceDimensions,
};
use crate::SymbolicSumcheck;
use crate::{challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges};

/// Produced RAM-RA reduced opening, generic over the cell (`F` on the wire,
/// `Vec<F>` for ZK points, `OpeningClaim<F>` (point + value) on the clear path).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamRaClaimReduction)]
pub struct RamRaClaimReductionOutputClaims<C> {
    #[opening(RamRa)]
    pub ram_ra: C,
}

/// Consumed RAM-RA openings reduced by the `RamRaClaimReduction` sumcheck, wired
/// from the upstream RAF-evaluation, read-write-checking, and val-check
/// relations. Generic over the cell (`OpeningClaim<F>` on the clear path,
/// `Vec<F>` for ZK points).
#[derive(Clone, Debug, InputClaims)]
pub struct RamRaClaimReductionInputClaims<C> {
    #[opening(RamRa, from = RamRafEvaluation)]
    pub raf: C,
    #[opening(RamRa, from = RamReadWriteChecking)]
    pub read_write: C,
    #[opening(RamRa, from = RamValCheck)]
    pub val_check: C,
}

/// Fiat-Shamir challenge drawn by the RAM `ra` claim-reduction sumcheck.
#[derive(Clone, Copy, Debug, SumcheckChallenges)]
pub struct RamRaClaimReductionChallenges<F> {
    #[challenge(RamRaClaimReductionChallenge::Gamma)]
    pub gamma: F,
}

/// The RAM `ra` claim-reduction sumcheck: folds the three `ra` openings (RAF,
/// read/write, val-check) by `gamma` on the input side, and matches the reduced
/// `ra` opening weighted by the matching cycle-`eq` publics on the output side.
pub struct RaClaimReduction {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for RaClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = RamRaClaimReductionChallenges<F>;
    type Inputs<C> = RamRaClaimReductionInputClaims<C>;
    type Outputs<C> = RamRaClaimReductionOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamRaClaimReduction
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(RamRaClaimReductionChallenge::Gamma);
        opening(ram_ra_raf_evaluation())
            + gamma.clone() * opening(ram_ra())
            + gamma.clone().pow(2) * opening(ram_ra_val_check())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(RamRaClaimReductionChallenge::Gamma);
        (derived(RamRaClaimReductionPublic::EqCycleRaf)
            + gamma.clone() * derived(RamRaClaimReductionPublic::EqCycleReadWrite)
            + gamma.pow(2) * derived(RamRaClaimReductionPublic::EqCycleValCheck))
            * opening(ram_ra_claim_reduction())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ram::RamRaClaimReductionPublicValues;
    use crate::protocols::jolt::{JoltChallengeId, JoltDerivedId};
    use jolt_field::{Fr, FromPrimitiveInt};

    fn trace_dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    #[test]
    fn ra_claim_reduction_evaluates_like_core_formula() {
        let relation = RaClaimReduction::new(trace_dimensions());

        let raf = Fr::from_u64(3);
        let rw = Fr::from_u64(5);
        let val = Fr::from_u64(7);
        let gamma = Fr::from_u64(11);
        let reduced = Fr::from_u64(13);
        let eq_raf = Fr::from_u64(17);
        let eq_rw = Fr::from_u64(19);
        let eq_val = Fr::from_u64(23);
        let public_values = RamRaClaimReductionPublicValues {
            eq_cycle_raf: eq_raf,
            eq_cycle_read_write: eq_rw,
            eq_cycle_val_check: eq_val,
        };
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_ra_raf_evaluation() => raf,
                id if id == ram_ra() => rw,
                id if id == ram_ra_val_check() => val,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamRaClaimReduction(RamRaClaimReductionChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_)
                | JoltChallengeId::IncVirtualization(_)
                | JoltChallengeId::UnsignedIncChunkReconstruction(_)
                | JoltChallengeId::AdviceBytesValidity(_) => zero,
            },
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_ra_claim_reduction() => reduced,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamRaClaimReduction(RamRaClaimReductionChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_)
                | JoltChallengeId::IncVirtualization(_)
                | JoltChallengeId::UnsignedIncChunkReconstruction(_)
                | JoltChallengeId::AdviceBytesValidity(_) => zero,
            },
            |id| match *id {
                JoltDerivedId::RamRaClaimReduction(id) => public_values.value(id),
                _ => zero,
            },
        );

        assert_eq!(input, raf + gamma * rw + gamma * gamma * val);
        assert_eq!(
            output,
            (eq_raf + gamma * eq_rw + gamma * gamma * eq_val) * reduced
        );
    }

    #[test]
    fn ra_claim_reduction_symbolic_matches_dependencies() {
        let relation = RaClaimReduction::new(trace_dimensions());

        assert_eq!(RaClaimReduction::id(), JoltRelationId::RamRaClaimReduction);
        assert_eq!(relation.rounds(), trace_dimensions().log_t());
        assert_eq!(relation.degree(), 2);
        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![
                ram_ra_raf_evaluation(),
                ram_ra(),
                ram_ra_val_check(),
                ram_ra_claim_reduction(),
            ]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(RamRaClaimReductionChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![
                JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleRaf),
                JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleReadWrite),
                JoltDerivedId::from(RamRaClaimReductionPublic::EqCycleValCheck),
            ]
        );
    }
}
