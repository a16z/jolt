//! Instruction read-RAF symbolic sumcheck relation.

use jolt_field::Ring;
use jolt_lookup_tables::{LookupTableKind, XLEN};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::instruction::{
    left_lookup_operand_reduced, lookup_output_reduced, right_lookup_operand_reduced,
};
use crate::protocols::jolt::geometry::instruction::{
    eq_table_value, instruction_ra_product, instruction_raf_flag, lookup_table_flag,
    InstructionReadRafDimensions, READ_RAF_BASE_DEGREE,
};
use crate::protocols::jolt::{
    InstructionReadRafChallenge, InstructionReadRafPublic, JoltExpr, JoltRelationId,
};
use crate::SymbolicSumcheck;
use crate::{challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges};

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

/// Consumed instruction-lookup openings (the reduced lookup output + left/right
/// operands), wired from the upstream instruction claim-reduction.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
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

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        let gamma = challenge(InstructionReadRafChallenge::Gamma);
        opening(lookup_output_reduced())
            + gamma.clone() * opening(left_lookup_operand_reduced())
            + gamma.pow(2) * opening(right_lookup_operand_reduced())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
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
