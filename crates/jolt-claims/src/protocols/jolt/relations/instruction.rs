//! Instruction symbolic sumcheck relations.

use jolt_field::RingCore;
use jolt_lookup_tables::{LookupTableKind, XLEN};

use crate::protocols::jolt::geometry::instruction::{
    committed_instruction_ra_product, eq_table_value, imm, instruction_ra_product,
    instruction_raf_flag, left_instruction_input_product, left_lookup_operand_reduced,
    left_operand_is_pc, left_operand_is_rs1, lookup_output_reduced, lookup_table_flag,
    right_instruction_input_product, right_lookup_operand_reduced, right_operand_is_imm,
    right_operand_is_rs2, rs1_value, rs2_value, unexpanded_pc, weighted_instruction_ra_sum,
    InstructionRaVirtualizationDimensions, InstructionReadRafDimensions,
    INPUT_VIRTUALIZATION_DEGREE,
};
use crate::protocols::jolt::{
    InstructionInputChallenge, InstructionInputPublic, InstructionRaVirtualizationChallenge,
    InstructionRaVirtualizationPublic, InstructionReadRafChallenge, InstructionReadRafPublic,
    JoltExpr, JoltRelationId, JoltSumcheckSpec, TraceDimensions,
};
use crate::SymbolicSumcheck;
use crate::{challenge, opening, public};

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
            + challenge(InstructionInputChallenge::Gamma)
                * opening(left_instruction_input_product())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        public(InstructionInputPublic::EqProduct)
            * opening(right_operand_is_rs2())
            * opening(rs2_value())
            + public(InstructionInputPublic::EqProduct)
                * opening(right_operand_is_imm())
                * opening(imm())
            + public(InstructionInputPublic::EqProduct)
                * challenge(InstructionInputChallenge::Gamma)
                * opening(left_operand_is_rs1())
                * opening(rs1_value())
            + public(InstructionInputPublic::EqProduct)
                * challenge(InstructionInputChallenge::Gamma)
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
                + public(eq_table_value(table))
                    * ra_product.clone()
                    * opening(lookup_table_flag(table));
        }

        output = output
            + public(InstructionReadRafPublic::EqRafConstant) * ra_product.clone()
            + public(InstructionReadRafPublic::EqRafFlag)
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
        let gamma = challenge(InstructionRaVirtualizationChallenge::Gamma);
        weighted_instruction_ra_sum(self.shape, gamma)
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(InstructionRaVirtualizationChallenge::Gamma);
        let eq_cycle = public(InstructionRaVirtualizationPublic::EqCycle);
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
    use crate::protocols::jolt::geometry::instruction::instruction_ra;
    use crate::protocols::jolt::{
        JoltChallengeId, JoltCommittedPolynomial, JoltOpeningId, JoltPolynomialId, JoltPublicId,
        JoltVirtualPolynomial,
    };
    use jolt_field::{Fr, FromPrimitiveInt};

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

    fn trace_dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    fn eq_table_value_publics() -> Vec<JoltPublicId> {
        LookupTableKind::<XLEN>::iter()
            .map(|table| JoltPublicId::from(eq_table_value(table)))
            .collect()
    }

    #[test]
    fn input_virtualization_evaluates_like_core_formula() {
        let relation = InputVirtualization::new(trace_dimensions());

        let right_input = Fr::from_u64(3);
        let left_input = Fr::from_u64(5);
        let right_is_rs2 = Fr::from_u64(7);
        let rs2 = Fr::from_u64(11);
        let right_is_imm = Fr::from_u64(13);
        let imm_value = Fr::from_u64(17);
        let left_is_rs1 = Fr::from_u64(19);
        let rs1 = Fr::from_u64(23);
        let left_is_pc = Fr::from_u64(29);
        let pc = Fr::from_u64(31);
        let gamma = Fr::from_u64(37);
        let eq_product = Fr::from_u64(41);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == right_instruction_input_product() => right_input,
                id if id == left_instruction_input_product() => left_input,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionInput(InstructionInputChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
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
                id if id == right_operand_is_rs2() => right_is_rs2,
                id if id == rs2_value() => rs2,
                id if id == right_operand_is_imm() => right_is_imm,
                id if id == imm() => imm_value,
                id if id == left_operand_is_rs1() => left_is_rs1,
                id if id == rs1_value() => rs1,
                id if id == left_operand_is_pc() => left_is_pc,
                id if id == unexpanded_pc() => pc,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionInput(InstructionInputChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
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
                JoltPublicId::InstructionInput(InstructionInputPublic::EqProduct) => eq_product,
                _ => zero,
            },
        );

        assert_eq!(input, right_input + gamma * left_input);
        assert_eq!(
            output,
            eq_product
                * (right_is_rs2 * rs2
                    + right_is_imm * imm_value
                    + gamma * left_is_rs1 * rs1
                    + gamma * left_is_pc * pc)
        );
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
                JoltPublicId::InstructionReadRaf(InstructionReadRafPublic::EqTableValue(index)) => {
                    table_values[index]
                }
                JoltPublicId::InstructionReadRaf(InstructionReadRafPublic::EqRafConstant) => {
                    raf_constant
                }
                JoltPublicId::InstructionReadRaf(InstructionReadRafPublic::EqRafFlag) => {
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
    fn ra_virtualization_evaluates_like_core_formula() {
        let dimensions = ra_virtualization_dimensions(3, 2);
        let relation = RaVirtualization::new(dimensions);

        let virtual_ra = [Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)];
        let committed_ra = [
            Fr::from_u64(11),
            Fr::from_u64(13),
            Fr::from_u64(17),
            Fr::from_u64(19),
            Fr::from_u64(23),
            Fr::from_u64(29),
        ];
        let gamma = Fr::from_u64(31);
        let eq_cycle = Fr::from_u64(37);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                JoltOpeningId::Polynomial {
                    polynomial: JoltPolynomialId::Virtual(JoltVirtualPolynomial::InstructionRa(i)),
                    relation: JoltRelationId::InstructionReadRaf,
                } => virtual_ra[i],
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionRaVirtualization(
                    InstructionRaVirtualizationChallenge::Gamma,
                ) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
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
                JoltOpeningId::Polynomial {
                    polynomial:
                        JoltPolynomialId::Committed(JoltCommittedPolynomial::InstructionRa(i)),
                    relation: JoltRelationId::InstructionRaVirtualization,
                } => committed_ra[i],
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::InstructionRaVirtualization(
                    InstructionRaVirtualizationChallenge::Gamma,
                ) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |id| match *id {
                JoltPublicId::InstructionRaVirtualization(
                    InstructionRaVirtualizationPublic::EqCycle,
                ) => eq_cycle,
                _ => zero,
            },
        );

        assert_eq!(
            input,
            virtual_ra[0] + gamma * virtual_ra[1] + gamma * gamma * virtual_ra[2]
        );
        assert_eq!(
            output,
            eq_cycle
                * (committed_ra[0] * committed_ra[1]
                    + gamma * committed_ra[2] * committed_ra[3]
                    + gamma * gamma * committed_ra[4] * committed_ra[5])
        );
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
