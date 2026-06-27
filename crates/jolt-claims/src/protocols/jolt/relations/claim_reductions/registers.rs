//! Registers claim-reduction symbolic sumcheck relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::registers::{
    rd_write_value_reduced, rd_write_value_spartan, rs1_value_reduced, rs1_value_spartan,
    rs2_value_reduced, rs2_value_spartan,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId, JoltSumcheckSpec,
    RegistersClaimReductionChallenge, RegistersClaimReductionPublic, TraceDimensions,
};
use crate::{
    challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck,
};

/// Produced register claim-reduction openings (`rd` write value, `rs1`/`rs2`
/// values reduced to the Spartan point), all sharing the single reduction opening
/// point. Generic over the cell.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RegistersClaimReduction)]
pub struct RegistersClaimReductionOutputClaims<C> {
    #[opening(RdWriteValue)]
    pub rd_write_value: C,
    #[opening(Rs1Value)]
    pub rs1_value: C,
    #[opening(Rs2Value)]
    pub rs2_value: C,
}

/// Consumed register openings reduced by this sumcheck, wired from stage 1's outer
/// sumcheck. The relation reads only these values, so the input points are left
/// empty. Generic over the cell.
#[derive(Clone, Debug, InputClaims)]
pub struct RegistersClaimReductionInputClaims<C> {
    #[opening(RdWriteValue, from = SpartanOuter)]
    pub rd_write_value: C,
    #[opening(Rs1Value, from = SpartanOuter)]
    pub rs1_value: C,
    #[opening(Rs2Value, from = SpartanOuter)]
    pub rs2_value: C,
}

/// Fiat-Shamir challenge drawn by the registers claim-reduction sumcheck.
#[derive(Clone, Copy, Debug, SumcheckChallenges)]
pub struct RegistersClaimReductionChallenges<F> {
    #[challenge(RegistersClaimReductionChallenge::Gamma)]
    pub gamma: F,
}

/// Batches the Spartan-outer register openings (`RdWriteValue`, `Rs1Value`,
/// `Rs2Value`) by `gamma` and reduces them to the registers-claim-reduction
/// openings weighted by the `EqSpartan` public.
pub struct ClaimReduction {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for ClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = RegistersClaimReductionChallenges<F>;
    type Inputs<C> = RegistersClaimReductionInputClaims<C>;
    type Outputs<C> = RegistersClaimReductionOutputClaims<C>;

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
        let gamma = challenge(RegistersClaimReductionChallenge::Gamma);

        opening(rd_write_value_spartan())
            + gamma.clone() * opening(rs1_value_spartan())
            + gamma.clone().pow(2) * opening(rs2_value_spartan())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(RegistersClaimReductionChallenge::Gamma);
        let eq_spartan = derived(RegistersClaimReductionPublic::EqSpartan);

        eq_spartan.clone() * opening(rd_write_value_reduced())
            + eq_spartan.clone() * gamma.clone() * opening(rs1_value_reduced())
            + eq_spartan * gamma.pow(2) * opening(rs2_value_reduced())
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

        let rd_spartan = Fr::from_u64(3);
        let rs1_spartan = Fr::from_u64(5);
        let rs2_spartan = Fr::from_u64(7);
        let rd_reduced = Fr::from_u64(11);
        let rs1_reduced = Fr::from_u64(13);
        let rs2_reduced = Fr::from_u64(17);
        let gamma = Fr::from_u64(19);
        let eq_spartan = Fr::from_u64(23);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == rd_write_value_spartan() => rd_spartan,
                id if id == rs1_value_spartan() => rs1_spartan,
                id if id == rs2_value_spartan() => rs2_spartan,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RegistersClaimReduction(
                    RegistersClaimReductionChallenge::Gamma,
                ) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == rd_write_value_reduced() => rd_reduced,
                id if id == rs1_value_reduced() => rs1_reduced,
                id if id == rs2_value_reduced() => rs2_reduced,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RegistersClaimReduction(
                    RegistersClaimReductionChallenge::Gamma,
                ) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |id| match *id {
                JoltDerivedId::RegistersClaimReduction(
                    RegistersClaimReductionPublic::EqSpartan,
                ) => eq_spartan,
                _ => zero,
            },
        );

        assert_eq!(
            input,
            rd_spartan + gamma * rs1_spartan + gamma * gamma * rs2_spartan
        );
        assert_eq!(
            output,
            eq_spartan * (rd_reduced + gamma * rs1_reduced + gamma * gamma * rs2_reduced)
        );
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
            relation.required_deriveds::<Fr>(),
            vec![JoltDerivedId::from(
                RegistersClaimReductionPublic::EqSpartan
            )]
        );
    }
}
