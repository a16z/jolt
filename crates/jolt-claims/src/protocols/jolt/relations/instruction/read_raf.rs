use jolt_algebra::Ring;
use jolt_lookup_tables::{LookupTableKind, XLEN};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::instruction::{
    left_lookup_operand_reduced, lookup_output_reduced, right_lookup_operand_reduced,
};
use crate::protocols::jolt::geometry::instruction::{
    eq_table_value, instruction_ra_factors, instruction_raf_flag, lookup_table_flag,
    InstructionReadRafDimensions, READ_RAF_ADDRESS_ROUND_DEGREE, READ_RAF_BASE_DEGREE,
};
use crate::protocols::jolt::{
    InstructionReadRafChallenge, InstructionReadRafPublic, JoltExpr, JoltFactoredExpr,
    JoltRelationId,
};
use crate::{challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges};
use crate::{FactoredExpr, SymbolicSumcheck};

#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
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

#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct InstructionReadRafInputClaims<C> {
    #[opening(LookupOutput, from = InstructionClaimReduction)]
    pub lookup_output: C,
    #[opening(LeftLookupOperand, from = InstructionClaimReduction)]
    pub left_lookup_operand: C,
    #[opening(RightLookupOperand, from = InstructionClaimReduction)]
    pub right_lookup_operand: C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct InstructionReadRafChallenges<F> {
    #[challenge(InstructionReadRafChallenge::Gamma)]
    pub gamma: F,
}

/// The instruction read-RAF sumcheck: relates the reduced lookup
/// output/operands to the per-table flag products (weighted by `EqTableValue`
/// publics) and the read-address-flag terms, all folded by `gamma`.
#[derive(Clone)]
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

    /// The address rounds come first (see
    /// [`InstructionReadRafDimensions::opening_point`]) at degree
    /// [`READ_RAF_ADDRESS_ROUND_DEGREE`]; the cycle rounds multiply in the
    /// `num_virtual_ra_polys` `ra` chunks.
    fn round_degree(&self, round: usize) -> usize {
        if round < self.shape.instruction_address_bits() {
            READ_RAF_ADDRESS_ROUND_DEGREE
        } else {
            self.degree()
        }
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        let gamma = challenge(InstructionReadRafChallenge::Gamma);
        opening(lookup_output_reduced())
            + gamma.clone() * opening(left_lookup_operand_reduced())
            + gamma.pow(2) * opening(right_lookup_operand_reduced())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        self.output_factored().expand()
    }

    /// `∏_i ra_i · (Σ_t eq_t · flag_t + eq_raf_constant + eq_raf_flag · raf_flag)`.
    fn output_factored<F: Ring>(&self) -> JoltFactoredExpr<F> {
        let mut table_sum = JoltExpr::zero();
        for table in LookupTableKind::<XLEN>::iter() {
            table_sum =
                table_sum + derived(eq_table_value(table)) * opening(lookup_table_flag(table));
        }
        table_sum = table_sum
            + derived(InstructionReadRafPublic::EqRafConstant)
            + derived(InstructionReadRafPublic::EqRafFlag) * opening(instruction_raf_flag());

        let mut factors = instruction_ra_factors(self.shape);
        factors.push(table_sum);
        FactoredExpr::product(factors)
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
    use jolt_algebra::{Fr, Ring};

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
                _ => zero,
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
            |_| zero,
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
}
