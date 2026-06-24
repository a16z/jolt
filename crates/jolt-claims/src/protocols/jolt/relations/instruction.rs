//! Instruction symbolic sumcheck relations.

use jolt_field::RingCore;
use jolt_lookup_tables::{LookupTableKind, XLEN};

use crate::opening;
use crate::protocols::jolt::formulas::instruction::{
    committed_instruction_ra_product, eq_table_value, imm, input_challenge, input_public,
    instruction_ra_product, instruction_raf_flag, left_instruction_input_product,
    left_lookup_operand_reduced, left_operand_is_pc, left_operand_is_rs1, lookup_output_reduced,
    lookup_table_flag, ra_virtualization_challenge, ra_virtualization_public, read_raf_challenge,
    read_raf_public, right_instruction_input_product, right_lookup_operand_reduced,
    right_operand_is_imm, right_operand_is_rs2, rs1_value, rs2_value, unexpanded_pc,
    weighted_instruction_ra_sum, InstructionRaVirtualizationDimensions,
    InstructionReadRafDimensions, INPUT_VIRTUALIZATION_DEGREE,
};
use crate::protocols::jolt::{
    InstructionInputChallenge, InstructionInputPublic, InstructionRaVirtualizationChallenge,
    InstructionRaVirtualizationPublic, InstructionReadRafChallenge, InstructionReadRafPublic,
    JoltExpr, JoltRelationId, JoltSumcheckSpec, TraceDimensions,
};
use crate::SymbolicSumcheck;

/// The instruction input-virtualization sumcheck: relates the left/right
/// instruction-input products from the product sumcheck to the per-operand
/// flag/value openings, folded by `gamma` and weighted by the `EqProduct` public.
pub struct InputVirtualization {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for InputVirtualization {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::InstructionInputVirtualization
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck(INPUT_VIRTUALIZATION_DEGREE)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(right_instruction_input_product())
            + input_challenge(InstructionInputChallenge::Gamma)
                * opening(left_instruction_input_product())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        input_public(InstructionInputPublic::EqProduct)
            * opening(right_operand_is_rs2())
            * opening(rs2_value())
            + input_public(InstructionInputPublic::EqProduct)
                * opening(right_operand_is_imm())
                * opening(imm())
            + input_public(InstructionInputPublic::EqProduct)
                * input_challenge(InstructionInputChallenge::Gamma)
                * opening(left_operand_is_rs1())
                * opening(rs1_value())
            + input_public(InstructionInputPublic::EqProduct)
                * input_challenge(InstructionInputChallenge::Gamma)
                * opening(left_operand_is_pc())
                * opening(unexpanded_pc())
    }
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
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = InstructionReadRafDimensions;

    fn new(shape: InstructionReadRafDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::InstructionReadRaf
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = read_raf_challenge(InstructionReadRafChallenge::Gamma);
        opening(lookup_output_reduced())
            + gamma.clone() * opening(left_lookup_operand_reduced())
            + gamma.pow(2) * opening(right_lookup_operand_reduced())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let ra_product = instruction_ra_product(self.shape);
        let mut output = JoltExpr::zero();

        for table in LookupTableKind::<XLEN>::iter() {
            output = output
                + read_raf_public(eq_table_value(table))
                    * ra_product.clone()
                    * opening(lookup_table_flag(table));
        }

        output = output
            + read_raf_public(InstructionReadRafPublic::EqRafConstant) * ra_product.clone()
            + read_raf_public(InstructionReadRafPublic::EqRafFlag)
                * ra_product
                * opening(instruction_raf_flag());

        output
    }
}

/// The instruction RA-virtualization sumcheck: relates the virtual
/// instruction-RA openings (folded by `gamma`) to the per-virtual products of
/// committed instruction-RA openings, weighted by the `EqCycle` public.
pub struct RaVirtualization {
    shape: InstructionRaVirtualizationDimensions,
}

impl SymbolicSumcheck for RaVirtualization {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = InstructionRaVirtualizationDimensions;

    fn new(shape: InstructionRaVirtualizationDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::InstructionRaVirtualization
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = ra_virtualization_challenge(InstructionRaVirtualizationChallenge::Gamma);
        weighted_instruction_ra_sum(self.shape, gamma)
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = ra_virtualization_challenge(InstructionRaVirtualizationChallenge::Gamma);
        let eq_cycle = ra_virtualization_public(InstructionRaVirtualizationPublic::EqCycle);
        let mut output = JoltExpr::zero();
        for virtual_index in 0..self.shape.num_virtual_ra_polys() {
            output = output
                + eq_cycle.clone()
                    * gamma.clone().pow(virtual_index)
                    * committed_instruction_ra_product(self.shape, virtual_index);
        }
        output
    }
}

#[cfg(test)]
#[expect(clippy::panic)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{JoltChallengeId, JoltPublicId};
    use jolt_field::Fr;

    fn read_raf_dimensions(num_virtual_ra_polys: usize) -> InstructionReadRafDimensions {
        InstructionReadRafDimensions::try_from((5, 128, num_virtual_ra_polys))
            .unwrap_or_else(|err| panic!("test read-RAF dimensions should be nonzero: {err}"))
    }

    fn ra_virtualization_dimensions(
        num_virtual_ra_polys: usize,
        num_committed_per_virtual: usize,
    ) -> InstructionRaVirtualizationDimensions {
        InstructionRaVirtualizationDimensions::try_from((
            5,
            num_virtual_ra_polys,
            num_committed_per_virtual,
        ))
        .unwrap_or_else(|err| panic!("test RA virtualization dimensions should be valid: {err}"))
    }

    fn eq_table_value_publics() -> Vec<JoltPublicId> {
        LookupTableKind::<XLEN>::iter()
            .map(|table| JoltPublicId::from(eq_table_value(table)))
            .collect()
    }

    #[test]
    fn input_virtualization_symbolic_matches_dependencies() {
        let relation = InputVirtualization::new(TraceDimensions::new(5));
        assert_eq!(
            InputVirtualization::id(),
            JoltRelationId::InstructionInputVirtualization
        );
        assert_eq!(
            relation.spec(),
            TraceDimensions::new(5).sumcheck(INPUT_VIRTUALIZATION_DEGREE)
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(InstructionInputChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(InstructionInputPublic::EqProduct)]
        );
    }

    #[test]
    fn read_raf_symbolic_matches_dependencies() {
        let dimensions = read_raf_dimensions(2);
        let relation = ReadRaf::new(dimensions);
        assert_eq!(ReadRaf::id(), JoltRelationId::InstructionReadRaf);
        assert_eq!(relation.spec(), dimensions.sumcheck());
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(InstructionReadRafChallenge::Gamma)]
        );
        let mut expected_publics = eq_table_value_publics();
        expected_publics.extend([
            JoltPublicId::from(InstructionReadRafPublic::EqRafConstant),
            JoltPublicId::from(InstructionReadRafPublic::EqRafFlag),
        ]);
        assert_eq!(relation.required_publics::<Fr>(), expected_publics);
    }

    #[test]
    fn ra_virtualization_symbolic_matches_dependencies() {
        let dimensions = ra_virtualization_dimensions(3, 2);
        let relation = RaVirtualization::new(dimensions);
        assert_eq!(
            RaVirtualization::id(),
            JoltRelationId::InstructionRaVirtualization
        );
        assert_eq!(relation.spec(), dimensions.sumcheck());
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(
                InstructionRaVirtualizationChallenge::Gamma
            )]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(
                InstructionRaVirtualizationPublic::EqCycle
            )]
        );
    }
}
