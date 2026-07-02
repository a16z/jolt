//! Instruction read-RAF symbolic sumcheck relation.

use jolt_field::RingCore;
use jolt_lookup_tables::{LookupTableKind, XLEN};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::instruction::{
    eq_table_value, instruction_ra_product, instruction_raf_flag, left_lookup_operand_reduced,
    lookup_output_reduced, lookup_table_flag, right_lookup_operand_reduced,
    InstructionReadRafDimensions, READ_RAF_BASE_DEGREE,
};
use crate::protocols::jolt::{
    InstructionReadRafChallenge, InstructionReadRafPublic, JoltExpr, JoltRelationId,
};
use crate::SymbolicSumcheck;
use crate::{challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(InstructionReadRaf)]
pub struct InstructionReadRafOutputClaims<C> {
    #[opening(LookupTableFlag)]
    pub lookup_table_flags: Vec<C>,
    #[opening(InstructionRa)]
    pub instruction_ra: Vec<C>,
    #[opening(InstructionRafFlag)]
    pub instruction_raf_flag: C,
}

/// Consumed instruction-lookup openings (the reduced lookup output + left/right
/// operands), wired from the upstream instruction claim-reduction.
#[derive(Clone, Debug, PartialEq, Eq, InputClaims)]
pub struct InstructionReadRafInputClaims<C> {
    #[opening(LookupOutput, from = InstructionClaimReduction)]
    pub lookup_output: C,
    #[opening(LeftLookupOperand, from = InstructionClaimReduction)]
    pub left_lookup_operand: C,
    #[opening(RightLookupOperand, from = InstructionClaimReduction)]
    pub right_lookup_operand: C,
}

/// Fiat-Shamir challenge drawn by the instruction read-RAF sumcheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
pub struct InstructionReadRafChallenges<F> {
    #[challenge(InstructionReadRafChallenge::Gamma)]
    pub gamma: F,
}

/// The instruction read-RAF sumcheck: relates the reduced lookup
/// output/operands to the per-table flag products (weighted by `EqTableValue`
/// publics) and the read-address-flag terms, all folded by `gamma`.
pub struct ReadRaf {
    shape: InstructionReadRafDimensions,
}

impl SymbolicSumcheck for ReadRaf {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = InstructionReadRafDimensions;
    type Challenges<F> = InstructionReadRafChallenges<F>;
    type Inputs<C> = InstructionReadRafInputClaims<C>;
    type Outputs<C> = InstructionReadRafOutputClaims<C>;

    fn new(shape: InstructionReadRafDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::InstructionReadRaf
    }

    fn rounds(&self) -> usize {
        self.shape.sumcheck_rounds()
    }

    fn degree(&self) -> usize {
        self.shape.num_virtual_ra_polys() + READ_RAF_BASE_DEGREE
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(InstructionReadRafChallenge::Gamma);
        opening(lookup_output_reduced())
            + gamma.clone() * opening(left_lookup_operand_reduced())
            + gamma.pow(2) * opening(right_lookup_operand_reduced())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let ra_product = instruction_ra_product(self.shape);
        let mut output = JoltExpr::zero();

        for table in LookupTableKind::<XLEN>::iter() {
            output = output
                + derived(eq_table_value(table))
                    * ra_product.clone()
                    * opening(lookup_table_flag(table));
        }

        output = output
            + derived(InstructionReadRafPublic::EqRafConstant) * ra_product.clone()
            + derived(InstructionReadRafPublic::EqRafFlag)
                * ra_product
                * opening(instruction_raf_flag());

        output
    }
}

#[cfg(test)]
#[expect(clippy::panic)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::instruction::instruction_ra;
    use crate::protocols::jolt::{
        JoltChallengeId, JoltDerivedId, JoltOpeningId, JoltPolynomialId, JoltVirtualPolynomial,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

    fn read_raf_dimensions(num_virtual_ra_polys: usize) -> InstructionReadRafDimensions {
        InstructionReadRafDimensions::try_from((5, 128, num_virtual_ra_polys))
            .unwrap_or_else(|err| panic!("test read-RAF dimensions should be nonzero: {err}"))
    }

    #[test]
    fn read_raf_evaluates_like_core_formula() {
        let dimensions = read_raf_dimensions(2);
        let relation = ReadRaf::new(dimensions);

        let lookup_output = Fr::from_u64(3);
        let left_lookup_operand = Fr::from_u64(5);
        let right_lookup_operand = Fr::from_u64(7);
        let gamma = Fr::from_u64(11);
        let ra_0 = Fr::from_u64(2);
        let ra_1 = Fr::from_u64(3);
        let table_flags: Vec<_> = (0..LookupTableKind::<XLEN>::COUNT)
            .map(|i| Fr::from_u64(i as u64 + 5))
            .collect();
        let table_values: Vec<_> = (0..LookupTableKind::<XLEN>::COUNT)
            .map(|i| Fr::from_u64(2 * i as u64 + 13))
            .collect();
        let raf_constant = Fr::from_u64(23);
        let raf_flag_coeff = Fr::from_u64(29);
        let raf_flag = Fr::from_u64(31);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == lookup_output_reduced() => lookup_output,
                id if id == left_lookup_operand_reduced() => left_lookup_operand,
                id if id == right_lookup_operand_reduced() => right_lookup_operand,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionReadRaf(InstructionReadRafChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
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
                id if id == instruction_ra(0) => ra_0,
                id if id == instruction_ra(1) => ra_1,
                JoltOpeningId::Polynomial {
                    polynomial:
                        JoltPolynomialId::Virtual(JoltVirtualPolynomial::LookupTableFlag(index)),
                    relation: JoltRelationId::InstructionReadRaf,
                } => table_flags[index],
                id if id == instruction_raf_flag() => raf_flag,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionReadRaf(InstructionReadRafChallenge::Gamma)
                | JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |id| match *id {
                JoltDerivedId::InstructionReadRaf(InstructionReadRafPublic::EqTableValue(
                    index,
                )) => table_values[index],
                JoltDerivedId::InstructionReadRaf(InstructionReadRafPublic::EqRafConstant) => {
                    raf_constant
                }
                JoltDerivedId::InstructionReadRaf(InstructionReadRafPublic::EqRafFlag) => {
                    raf_flag_coeff
                }
                _ => zero,
            },
        );

        assert_eq!(
            input,
            lookup_output + gamma * left_lookup_operand + gamma * gamma * right_lookup_operand
        );
        let table_sum = table_values
            .iter()
            .zip(table_flags.iter())
            .fold(zero, |sum, (value, flag)| sum + *value * *flag);
        assert_eq!(
            output,
            ra_0 * ra_1 * (table_sum + raf_constant + raf_flag_coeff * raf_flag)
        );
    }

    #[test]
    fn read_raf_symbolic_matches_dependencies() {
        let dimensions = read_raf_dimensions(2);
        let relation = ReadRaf::new(dimensions);
        assert_eq!(ReadRaf::id(), JoltRelationId::InstructionReadRaf);
        assert_eq!(relation.rounds(), dimensions.sumcheck_rounds());
        assert_eq!(
            relation.degree(),
            dimensions.num_virtual_ra_polys() + READ_RAF_BASE_DEGREE
        );
    }
}
