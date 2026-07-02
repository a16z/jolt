//! RAM read/write-checking symbolic sumcheck relation.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::ram::{
    ram_inc, ram_ra, ram_read_value, ram_val, ram_write_value,
};
use crate::protocols::jolt::{
    JoltExpr, JoltRelationId, RamReadWriteChallenge, RamReadWritePublic, ReadWriteDimensions,
};
use crate::SymbolicSumcheck;
use crate::{challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges};

/// Produced RAM read-write openings (`val`, `ra`, committed `inc`), all sharing
/// the single read-write opening point. Generic over the opening cell (`F` for the
/// serialized wire value, `Vec<F>` for the derived opening point).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamReadWriteChecking)]
pub struct RamReadWriteOutputClaims<C> {
    #[opening(RamVal)]
    pub val: C,
    #[opening(RamRa)]
    pub ra: C,
    #[opening(committed = RamInc)]
    pub inc: C,
}

/// Consumed RAM read/write value openings from stage 1's outer sumcheck, reduced
/// by the read-write checking sumcheck. The relation reads only these values (its
/// output points come from its own sumcheck point and `product_tau_low`), so the
/// input points are left empty. Generic over the cell.
#[derive(Clone, Debug, PartialEq, Eq, InputClaims)]
pub struct RamReadWriteInputClaims<C> {
    #[opening(RamReadValue, from = SpartanOuter)]
    pub ram_read_value: C,
    #[opening(RamWriteValue, from = SpartanOuter)]
    pub ram_write_value: C,
}

/// Fiat-Shamir challenge drawn by the RAM read/write-checking sumcheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
pub struct RamReadWriteChallenges<F> {
    #[challenge(RamReadWriteChallenge::Gamma)]
    pub gamma: F,
}

/// The RAM read/write-checking sumcheck: folds the read and write values by
/// `gamma` on the input side, and reconstructs them from `ra`, `val`, and `inc`
/// weighted by the cycle-`eq` public on the output side.
pub struct ReadWriteChecking {
    shape: ReadWriteDimensions,
}

impl SymbolicSumcheck for ReadWriteChecking {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = ReadWriteDimensions;
    type Challenges<F> = RamReadWriteChallenges<F>;
    type Inputs<C> = RamReadWriteInputClaims<C>;
    type Outputs<C> = RamReadWriteOutputClaims<C>;

    fn new(shape: ReadWriteDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamReadWriteChecking
    }

    fn rounds(&self) -> usize {
        self.shape.read_write_rounds()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(ram_read_value())
            + challenge(RamReadWriteChallenge::Gamma) * opening(ram_write_value())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        derived(RamReadWritePublic::EqCycle) * opening(ram_ra()) * opening(ram_val())
            + derived(RamReadWritePublic::EqCycle)
                * challenge(RamReadWriteChallenge::Gamma)
                * opening(ram_ra())
                * opening(ram_val())
            + derived(RamReadWritePublic::EqCycle)
                * challenge(RamReadWriteChallenge::Gamma)
                * opening(ram_ra())
                * opening(ram_inc())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{BooleanityChallenge, JoltChallengeId, JoltDerivedId};
    use jolt_field::{Fr, FromPrimitiveInt};

    fn read_write_dimensions() -> ReadWriteDimensions {
        ReadWriteDimensions::new(5, 4, 2, 1)
    }

    #[test]
    fn read_write_claims_evaluate_like_core_formula() {
        let relation = ReadWriteChecking::new(read_write_dimensions());

        let read = Fr::from_u64(3);
        let write = Fr::from_u64(5);
        let ra = Fr::from_u64(7);
        let val = Fr::from_u64(11);
        let inc = Fr::from_u64(13);
        let gamma = Fr::from_u64(17);
        let eq = Fr::from_u64(19);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_read_value() => read,
                id if id == ram_write_value() => write,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma) => gamma,
                JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
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
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_ra() => ra,
                id if id == ram_val() => val,
                id if id == ram_inc() => inc,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma) => gamma,
                JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
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
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |id| match *id {
                JoltDerivedId::RamReadWrite(RamReadWritePublic::EqCycle) => eq,
                _ => zero,
            },
        );

        assert_eq!(input, read + gamma * write);
        assert_eq!(output, eq * ra * (val + gamma * (val + inc)));
    }

    #[test]
    fn read_write_symbolic_matches_dependencies() {
        let relation = ReadWriteChecking::new(read_write_dimensions());

        assert_eq!(
            ReadWriteChecking::id(),
            JoltRelationId::RamReadWriteChecking
        );
        assert_eq!(
            relation.rounds(),
            read_write_dimensions().read_write_rounds()
        );
        assert_eq!(relation.degree(), 3);
    }

    #[test]
    fn challenges_resolve_gamma_and_miss_others() {
        let challenges = RamReadWriteChallenges {
            gamma: Fr::from_u64(7),
        };

        assert_eq!(
            challenges.resolve_challenge(&JoltChallengeId::from(RamReadWriteChallenge::Gamma)),
            Some(Fr::from_u64(7)),
        );
        assert_eq!(
            challenges.resolve_challenge(&JoltChallengeId::from(BooleanityChallenge::Gamma)),
            None,
        );
    }
}
