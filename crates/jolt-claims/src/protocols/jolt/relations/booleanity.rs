//! Booleanity symbolic sumcheck relations.

use jolt_field::RingCore;

use crate::opening;
use crate::protocols::jolt::formulas::booleanity::{
    booleanity_address_phase_opening, booleanity_cycle_output, BooleanityDimensions,
};
use crate::protocols::jolt::{JoltExpr, JoltRelationId, JoltSumcheckSpec};
use crate::SymbolicSumcheck;

/// The full booleanity sumcheck over both the address and cycle variables:
/// asserts every one-hot `ra` opening is boolean (`ra^2 - ra == 0`), folded
/// across the RA family by `gamma` and weighted by the `EqAddressCycle` public.
/// Its input claim is zero — the boolean constraint sums to zero rather than
/// reducing a prior opening.
pub struct Booleanity {
    shape: BooleanityDimensions,
}

impl SymbolicSumcheck for Booleanity {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = BooleanityDimensions;

    fn new(shape: BooleanityDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::Booleanity
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        booleanity_cycle_output(self.shape)
    }
}

/// The address-phase split of the booleanity sumcheck: binds the address
/// variables and reduces to the intermediate `BooleanityAddrClaim` opening.
pub struct BooleanityAddressPhase {
    shape: BooleanityDimensions,
}

impl SymbolicSumcheck for BooleanityAddressPhase {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = BooleanityDimensions;

    fn new(shape: BooleanityDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::Booleanity
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.address_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(booleanity_address_phase_opening())
    }
}

/// The cycle-phase split of the booleanity sumcheck: takes the
/// `BooleanityAddrClaim` opening as input and reduces to the boolean-constraint
/// output over the cycle variables.
pub struct BooleanityCyclePhase {
    shape: BooleanityDimensions,
}

impl SymbolicSumcheck for BooleanityCyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = BooleanityDimensions;

    fn new(shape: BooleanityDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::Booleanity
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.cycle_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(booleanity_address_phase_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        booleanity_cycle_output(self.shape)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::formulas::ra::JoltRaPolynomialLayout;
    use crate::protocols::jolt::{
        BooleanityChallenge, BooleanityPublic, JoltChallengeId, JoltCommittedPolynomial,
        JoltOpeningId, JoltPublicId, JoltSumcheckSpec,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions(instruction: usize, bytecode: usize, ram: usize) -> BooleanityDimensions {
        let layout = JoltRaPolynomialLayout::new(instruction, bytecode, ram).unwrap();
        BooleanityDimensions::new(layout, 5, 8)
    }

    #[test]
    fn booleanity_evaluates_like_core_formula() {
        let relation = Booleanity::new(dimensions(1, 1, 1));

        let instruction_ra = Fr::from_u64(3);
        let bytecode_ra = Fr::from_u64(5);
        let ram_ra = Fr::from_u64(7);
        let gamma = Fr::from_u64(11);
        let eq_address_cycle = Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id
                    == JoltOpeningId::committed(
                        JoltCommittedPolynomial::InstructionRa(0),
                        JoltRelationId::Booleanity,
                    ) =>
                {
                    instruction_ra
                }
                id if id
                    == JoltOpeningId::committed(
                        JoltCommittedPolynomial::BytecodeRa(0),
                        JoltRelationId::Booleanity,
                    ) =>
                {
                    bytecode_ra
                }
                id if id
                    == JoltOpeningId::committed(
                        JoltCommittedPolynomial::RamRa(0),
                        JoltRelationId::Booleanity,
                    ) =>
                {
                    ram_ra
                }
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::Booleanity(BooleanityChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltPublicId::Booleanity(BooleanityPublic::EqAddressCycle) => eq_address_cycle,
                _ => zero,
            },
        );

        let gamma_2 = gamma * gamma;
        let gamma_4 = gamma_2 * gamma_2;
        assert_eq!(
            output,
            eq_address_cycle
                * ((instruction_ra * instruction_ra - instruction_ra)
                    + gamma_2 * (bytecode_ra * bytecode_ra - bytecode_ra)
                    + gamma_4 * (ram_ra * ram_ra - ram_ra))
        );
    }

    #[test]
    fn booleanity_symbolic_matches_dependencies() {
        let relation = Booleanity::new(dimensions(1, 1, 1));
        assert_eq!(Booleanity::id(), JoltRelationId::Booleanity);
        assert_eq!(relation.spec(), JoltSumcheckSpec::boolean(13, 3));
        assert!(relation
            .input_expression::<Fr>()
            .required_openings()
            .is_empty());
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(BooleanityChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(BooleanityPublic::EqAddressCycle)]
        );
    }

    #[test]
    fn booleanity_address_phase_symbolic_matches_dependencies() {
        let relation = BooleanityAddressPhase::new(dimensions(1, 1, 1));
        assert_eq!(BooleanityAddressPhase::id(), JoltRelationId::Booleanity);
        assert_eq!(relation.spec(), JoltSumcheckSpec::boolean(8, 3));
        assert!(relation
            .input_expression::<Fr>()
            .required_openings()
            .is_empty());
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![booleanity_address_phase_opening()]
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert!(relation.required_publics::<Fr>().is_empty());
    }

    #[test]
    fn booleanity_cycle_phase_symbolic_matches_dependencies() {
        let relation = BooleanityCyclePhase::new(dimensions(1, 1, 1));
        assert_eq!(BooleanityCyclePhase::id(), JoltRelationId::Booleanity);
        assert_eq!(relation.spec(), JoltSumcheckSpec::boolean(5, 3));
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            vec![booleanity_address_phase_opening()]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(BooleanityChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(BooleanityPublic::EqAddressCycle)]
        );
    }
}
