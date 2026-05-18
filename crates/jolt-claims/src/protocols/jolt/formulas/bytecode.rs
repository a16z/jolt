use jolt_field::RingCore;
use jolt_lookup_tables::LookupTableKind;
use jolt_riscv::{CircuitFlags, InstructionFlags, NUM_CIRCUIT_FLAGS};

use crate::{challenge, opening, public, SameEvaluationAs};

use super::super::{
    BytecodeReadRafChallenge, BytecodeReadRafPublic, JoltChallengeId, JoltCommittedPolynomial,
    JoltExpr, JoltOpeningId, JoltPublicId, JoltStageClaims, JoltStageId, JoltVirtualPolynomial,
};
use super::dimensions::JoltSumcheckSpec;

const CIRCUIT_FLAGS: [CircuitFlags; NUM_CIRCUIT_FLAGS] = [
    CircuitFlags::AddOperands,
    CircuitFlags::SubtractOperands,
    CircuitFlags::MultiplyOperands,
    CircuitFlags::Load,
    CircuitFlags::Store,
    CircuitFlags::Jump,
    CircuitFlags::WriteLookupOutputToRD,
    CircuitFlags::VirtualInstruction,
    CircuitFlags::Assert,
    CircuitFlags::DoNotUpdateUnexpandedPC,
    CircuitFlags::Advice,
    CircuitFlags::IsCompressed,
    CircuitFlags::IsFirstInSequence,
    CircuitFlags::IsLastInSequence,
];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafDimensions {
    log_t: usize,
    log_k: usize,
    committed_ra_polys: usize,
}

impl BytecodeReadRafDimensions {
    pub const fn new(log_t: usize, log_k: usize, committed_ra_polys: usize) -> Self {
        Self {
            log_t,
            log_k,
            committed_ra_polys,
        }
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn log_k(self) -> usize {
        self.log_k
    }

    pub const fn num_committed_ra_polys(self) -> usize {
        self.committed_ra_polys
    }

    pub const fn sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t + self.log_k, self.committed_ra_polys + 1)
    }
}

impl From<(usize, usize, usize)> for BytecodeReadRafDimensions {
    fn from((log_t, log_k, committed_ra_polys): (usize, usize, usize)) -> Self {
        Self::new(log_t, log_k, committed_ra_polys)
    }
}

pub fn read_raf<const XLEN: usize, F>(dimensions: BytecodeReadRafDimensions) -> JoltStageClaims<F>
where
    F: RingCore,
{
    let gamma = bytecode_challenge(BytecodeReadRafChallenge::Gamma);

    let input = gamma.clone().pow(7)
        + stage1_claim()
        + gamma.clone() * stage2_claim()
        + gamma.clone().pow(2) * stage3_claim()
        + gamma.clone().pow(3) * stage4_claim()
        + gamma.clone().pow(4) * stage5_claim::<XLEN, F>()
        + gamma.clone().pow(5) * opening(pc_spartan_outer())
        + gamma.pow(6) * opening(pc_spartan_shift());

    let gamma = bytecode_challenge(BytecodeReadRafChallenge::Gamma);
    let output_coeff = bytecode_public(BytecodeReadRafPublic::StageValue(0))
        + gamma.clone() * bytecode_public(BytecodeReadRafPublic::StageValue(1))
        + gamma.clone().pow(2) * bytecode_public(BytecodeReadRafPublic::StageValue(2))
        + gamma.clone().pow(3) * bytecode_public(BytecodeReadRafPublic::StageValue(3))
        + gamma.clone().pow(4) * bytecode_public(BytecodeReadRafPublic::StageValue(4))
        + gamma.clone().pow(5) * bytecode_public(BytecodeReadRafPublic::SpartanOuterRaf)
        + gamma.clone().pow(6) * bytecode_public(BytecodeReadRafPublic::SpartanShiftRaf)
        + gamma.pow(7) * bytecode_public(BytecodeReadRafPublic::Entry);

    JoltStageClaims::new(
        JoltStageId::BytecodeReadRaf,
        dimensions.sumcheck(),
        input,
        output_coeff * bytecode_ra_product(dimensions),
    )
    .with_input_challenges([
        JoltChallengeId::from(BytecodeReadRafChallenge::Gamma),
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage1Gamma),
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage2Gamma),
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage3Gamma),
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage4Gamma),
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage5Gamma),
    ])
    .with_consistency([
        unexpanded_pc_spartan_shift().same_evaluation_as(unexpanded_pc_instruction_input())
    ])
}

fn stage1_claim<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage1Gamma);
    let mut claim =
        opening(unexpanded_pc_spartan_outer()) + beta.clone() * opening(imm_spartan_outer());

    for (i, flag) in CIRCUIT_FLAGS.into_iter().enumerate() {
        claim = claim + beta.clone().pow(i + 2) * opening(op_flag_spartan_outer(flag));
    }

    claim
}

fn stage2_claim<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage2Gamma);

    opening(op_flag_product(CircuitFlags::Jump))
        + beta.clone() * opening(instruction_flag_product(InstructionFlags::Branch))
        + beta.clone().pow(2) * opening(op_flag_product(CircuitFlags::WriteLookupOutputToRD))
        + beta.pow(3) * opening(op_flag_product(CircuitFlags::VirtualInstruction))
}

fn stage3_claim<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage3Gamma);

    opening(imm_instruction_input())
        + beta.clone() * opening(unexpanded_pc_spartan_shift())
        + beta.clone().pow(2)
            * opening(instruction_flag_input(
                InstructionFlags::LeftOperandIsRs1Value,
            ))
        + beta.clone().pow(3) * opening(instruction_flag_input(InstructionFlags::LeftOperandIsPC))
        + beta.clone().pow(4)
            * opening(instruction_flag_input(
                InstructionFlags::RightOperandIsRs2Value,
            ))
        + beta.clone().pow(5) * opening(instruction_flag_input(InstructionFlags::RightOperandIsImm))
        + beta.clone().pow(6) * opening(instruction_flag_shift(InstructionFlags::IsNoop))
        + beta.clone().pow(7) * opening(op_flag_shift(CircuitFlags::VirtualInstruction))
        + beta.pow(8) * opening(op_flag_shift(CircuitFlags::IsFirstInSequence))
}

fn stage4_claim<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage4Gamma);

    opening(rd_wa_read_write())
        + beta.clone() * opening(rs1_ra_read_write())
        + beta.pow(2) * opening(rs2_ra_read_write())
}

fn stage5_claim<const XLEN: usize, F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage5Gamma);
    let mut claim =
        opening(rd_wa_val_evaluation()) + beta.clone() * opening(instruction_raf_flag());

    for (i, table) in LookupTableKind::<XLEN>::iter().enumerate() {
        claim = claim + beta.clone().pow(i + 2) * opening(lookup_table_flag(table));
    }

    claim
}

fn bytecode_challenge<F>(id: BytecodeReadRafChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn bytecode_public<F>(id: BytecodeReadRafPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

fn bytecode_ra_product<F>(dimensions: BytecodeReadRafDimensions) -> JoltExpr<F>
where
    F: RingCore,
{
    let mut product = JoltExpr::one();
    for i in 0..dimensions.num_committed_ra_polys() {
        product = product * opening(bytecode_ra(i));
    }
    product
}

fn unexpanded_pc_spartan_outer() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnexpandedPC,
        JoltStageId::SpartanOuter,
    )
}

fn imm_spartan_outer() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::Imm, JoltStageId::SpartanOuter)
}

fn op_flag_spartan_outer(flag: CircuitFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(flag),
        JoltStageId::SpartanOuter,
    )
}

fn op_flag_product(flag: CircuitFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(flag),
        JoltStageId::SpartanProductVirtualization,
    )
}

fn instruction_flag_product(flag: InstructionFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(flag),
        JoltStageId::SpartanProductVirtualization,
    )
}

fn imm_instruction_input() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Imm,
        JoltStageId::InstructionInputVirtualization,
    )
}

fn unexpanded_pc_instruction_input() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnexpandedPC,
        JoltStageId::InstructionInputVirtualization,
    )
}

fn unexpanded_pc_spartan_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnexpandedPC,
        JoltStageId::SpartanShift,
    )
}

fn instruction_flag_input(flag: InstructionFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(flag),
        JoltStageId::InstructionInputVirtualization,
    )
}

fn instruction_flag_shift(flag: InstructionFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(flag),
        JoltStageId::SpartanShift,
    )
}

fn op_flag_shift(flag: CircuitFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(flag),
        JoltStageId::SpartanShift,
    )
}

fn rd_wa_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWa,
        JoltStageId::RegistersReadWriteChecking,
    )
}

fn rs1_ra_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Ra,
        JoltStageId::RegistersReadWriteChecking,
    )
}

fn rs2_ra_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Ra,
        JoltStageId::RegistersReadWriteChecking,
    )
}

fn rd_wa_val_evaluation() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWa,
        JoltStageId::RegistersValEvaluation,
    )
}

fn instruction_raf_flag() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionRafFlag,
        JoltStageId::InstructionReadRaf,
    )
}

fn lookup_table_flag<const XLEN: usize>(table: LookupTableKind<XLEN>) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LookupTableFlag(table.index()),
        JoltStageId::InstructionReadRaf,
    )
}

fn pc_spartan_outer() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::PC, JoltStageId::SpartanOuter)
}

fn pc_spartan_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::PC, JoltStageId::SpartanShift)
}

fn bytecode_ra(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeRa(index),
        JoltStageId::BytecodeReadRaf,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{JoltCommittedPolynomial, JoltConsistencyClaim, JoltPolynomialId};
    use jolt_field::{Fr, FromPrimitiveInt};

    const TEST_XLEN: usize = jolt_lookup_tables::XLEN;

    fn dimensions(num_committed_ra_polys: usize) -> BytecodeReadRafDimensions {
        (5, 10, num_committed_ra_polys).into()
    }

    fn gamma_power(gamma: Fr, exponent: usize) -> Fr {
        let mut value = Fr::from_u64(1);
        for _ in 0..exponent {
            value *= gamma;
        }
        value
    }

    fn stage1_openings() -> Vec<JoltOpeningId> {
        let mut openings = vec![unexpanded_pc_spartan_outer(), imm_spartan_outer()];
        openings.extend(CIRCUIT_FLAGS.into_iter().map(op_flag_spartan_outer));
        openings
    }

    fn stage5_lookup_flags() -> Vec<JoltOpeningId> {
        LookupTableKind::<TEST_XLEN>::iter()
            .map(lookup_table_flag)
            .collect()
    }

    #[test]
    fn read_raf_supports_empty_ra_product() {
        let claims = read_raf::<TEST_XLEN, Fr>(dimensions(0));

        assert!(!claims.output.required_openings.iter().any(|opening_id| {
            matches!(
                opening_id,
                JoltOpeningId::Polynomial {
                    polynomial: JoltPolynomialId::Committed(JoltCommittedPolynomial::BytecodeRa(_)),
                    ..
                }
            )
        }));
    }

    #[test]
    fn read_raf_exposes_expected_dependencies() {
        let dimensions = dimensions(2);
        let claims = read_raf::<TEST_XLEN, Fr>(dimensions);

        let mut expected_input = stage1_openings();
        expected_input.extend([
            op_flag_product(CircuitFlags::Jump),
            instruction_flag_product(InstructionFlags::Branch),
            op_flag_product(CircuitFlags::WriteLookupOutputToRD),
            op_flag_product(CircuitFlags::VirtualInstruction),
            imm_instruction_input(),
            unexpanded_pc_spartan_shift(),
            instruction_flag_input(InstructionFlags::LeftOperandIsRs1Value),
            instruction_flag_input(InstructionFlags::LeftOperandIsPC),
            instruction_flag_input(InstructionFlags::RightOperandIsRs2Value),
            instruction_flag_input(InstructionFlags::RightOperandIsImm),
            instruction_flag_shift(InstructionFlags::IsNoop),
            op_flag_shift(CircuitFlags::VirtualInstruction),
            op_flag_shift(CircuitFlags::IsFirstInSequence),
            rd_wa_read_write(),
            rs1_ra_read_write(),
            rs2_ra_read_write(),
            rd_wa_val_evaluation(),
            instruction_raf_flag(),
        ]);
        expected_input.extend(stage5_lookup_flags());
        expected_input.extend([pc_spartan_outer(), pc_spartan_shift()]);

        assert_eq!(claims.id, JoltStageId::BytecodeReadRaf);
        assert_eq!(claims.sumcheck, JoltSumcheckSpec::boolean(15, 3));
        assert_eq!(claims.input.required_openings, expected_input);
        assert_eq!(
            claims.output.required_openings,
            vec![bytecode_ra(0), bytecode_ra(1)]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![
                JoltChallengeId::from(BytecodeReadRafChallenge::Gamma),
                JoltChallengeId::from(BytecodeReadRafChallenge::Stage1Gamma),
                JoltChallengeId::from(BytecodeReadRafChallenge::Stage2Gamma),
                JoltChallengeId::from(BytecodeReadRafChallenge::Stage3Gamma),
                JoltChallengeId::from(BytecodeReadRafChallenge::Stage4Gamma),
                JoltChallengeId::from(BytecodeReadRafChallenge::Stage5Gamma),
            ]
        );
        assert_eq!(
            claims.required_publics(),
            vec![
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(0)),
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(1)),
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(2)),
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(3)),
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(4)),
                JoltPublicId::from(BytecodeReadRafPublic::SpartanOuterRaf),
                JoltPublicId::from(BytecodeReadRafPublic::SpartanShiftRaf),
                JoltPublicId::from(BytecodeReadRafPublic::Entry),
            ]
        );
        assert_eq!(
            claims.consistency,
            vec![JoltConsistencyClaim::same_evaluation(
                unexpanded_pc_spartan_shift(),
                unexpanded_pc_instruction_input(),
            )]
        );
        assert_eq!(claims.num_challenges(), 6);
    }

    #[test]
    fn read_raf_evaluates_like_core_formula() {
        let dimensions = dimensions(2);
        let claims = read_raf::<TEST_XLEN, Fr>(dimensions);

        let gamma = Fr::from_u64(3);
        let stage1_gamma = Fr::from_u64(5);
        let stage2_gamma = Fr::from_u64(7);
        let stage3_gamma = Fr::from_u64(11);
        let stage4_gamma = Fr::from_u64(13);
        let stage5_gamma = Fr::from_u64(17);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression.evaluate(
            |id| match *id {
                id if id == unexpanded_pc_spartan_outer() => Fr::from_u64(19),
                id if id == imm_spartan_outer() => Fr::from_u64(23),
                id if id == op_flag_product(CircuitFlags::Jump) => Fr::from_u64(29),
                id if id == instruction_flag_product(InstructionFlags::Branch) => Fr::from_u64(31),
                id if id == op_flag_product(CircuitFlags::WriteLookupOutputToRD) => {
                    Fr::from_u64(37)
                }
                id if id == op_flag_product(CircuitFlags::VirtualInstruction) => Fr::from_u64(41),
                id if id == imm_instruction_input() => Fr::from_u64(43),
                id if id == unexpanded_pc_spartan_shift() => Fr::from_u64(47),
                id if id == instruction_flag_input(InstructionFlags::LeftOperandIsRs1Value) => {
                    Fr::from_u64(53)
                }
                id if id == instruction_flag_input(InstructionFlags::LeftOperandIsPC) => {
                    Fr::from_u64(59)
                }
                id if id == instruction_flag_input(InstructionFlags::RightOperandIsRs2Value) => {
                    Fr::from_u64(61)
                }
                id if id == instruction_flag_input(InstructionFlags::RightOperandIsImm) => {
                    Fr::from_u64(67)
                }
                id if id == instruction_flag_shift(InstructionFlags::IsNoop) => Fr::from_u64(71),
                id if id == op_flag_shift(CircuitFlags::VirtualInstruction) => Fr::from_u64(73),
                id if id == op_flag_shift(CircuitFlags::IsFirstInSequence) => Fr::from_u64(79),
                id if id == rd_wa_read_write() => Fr::from_u64(83),
                id if id == rs1_ra_read_write() => Fr::from_u64(89),
                id if id == rs2_ra_read_write() => Fr::from_u64(97),
                id if id == rd_wa_val_evaluation() => Fr::from_u64(101),
                id if id == instruction_raf_flag() => Fr::from_u64(103),
                id if id == pc_spartan_outer() => Fr::from_u64(107),
                id if id == pc_spartan_shift() => Fr::from_u64(109),
                JoltOpeningId::Polynomial {
                    polynomial: JoltPolynomialId::Virtual(JoltVirtualPolynomial::OpFlags(flag)),
                    stage: JoltStageId::SpartanOuter,
                } => Fr::from_u64(200 + u64::from(flag as u8)),
                JoltOpeningId::Polynomial {
                    polynomial:
                        JoltPolynomialId::Virtual(JoltVirtualPolynomial::LookupTableFlag(index)),
                    stage: JoltStageId::InstructionReadRaf,
                } => Fr::from_u64(300 + index as u64),
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => gamma,
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage1Gamma) => {
                    stage1_gamma
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage2Gamma) => {
                    stage2_gamma
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage3Gamma) => {
                    stage3_gamma
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage4Gamma) => {
                    stage4_gamma
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage5Gamma) => {
                    stage5_gamma
                }
                _ => zero,
            },
            |_| zero,
        );

        let mut stage1 = Fr::from_u64(19) + stage1_gamma * Fr::from_u64(23);
        for flag in CIRCUIT_FLAGS {
            stage1 += gamma_power(stage1_gamma, usize::from(flag as u8) + 2)
                * Fr::from_u64(200 + u64::from(flag as u8));
        }
        let stage2 = Fr::from_u64(29)
            + stage2_gamma * Fr::from_u64(31)
            + gamma_power(stage2_gamma, 2) * Fr::from_u64(37)
            + gamma_power(stage2_gamma, 3) * Fr::from_u64(41);
        let stage3 = Fr::from_u64(43)
            + stage3_gamma * Fr::from_u64(47)
            + gamma_power(stage3_gamma, 2) * Fr::from_u64(53)
            + gamma_power(stage3_gamma, 3) * Fr::from_u64(59)
            + gamma_power(stage3_gamma, 4) * Fr::from_u64(61)
            + gamma_power(stage3_gamma, 5) * Fr::from_u64(67)
            + gamma_power(stage3_gamma, 6) * Fr::from_u64(71)
            + gamma_power(stage3_gamma, 7) * Fr::from_u64(73)
            + gamma_power(stage3_gamma, 8) * Fr::from_u64(79);
        let stage4 = Fr::from_u64(83)
            + stage4_gamma * Fr::from_u64(89)
            + gamma_power(stage4_gamma, 2) * Fr::from_u64(97);
        let mut stage5 = Fr::from_u64(101) + stage5_gamma * Fr::from_u64(103);
        for table in LookupTableKind::<TEST_XLEN>::iter() {
            stage5 += gamma_power(stage5_gamma, table.index() + 2)
                * Fr::from_u64(300 + table.index() as u64);
        }

        assert_eq!(
            input,
            gamma_power(gamma, 7)
                + stage1
                + gamma * stage2
                + gamma_power(gamma, 2) * stage3
                + gamma_power(gamma, 3) * stage4
                + gamma_power(gamma, 4) * stage5
                + gamma_power(gamma, 5) * Fr::from_u64(107)
                + gamma_power(gamma, 6) * Fr::from_u64(109)
        );

        let stage_values = [
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
            Fr::from_u64(11),
        ];
        let spartan_outer_raf = Fr::from_u64(13);
        let spartan_shift_raf = Fr::from_u64(17);
        let entry = Fr::from_u64(19);
        let bytecode_ra_0 = Fr::from_u64(23);
        let bytecode_ra_1 = Fr::from_u64(29);

        let output = claims.output.expression.evaluate(
            |id| match *id {
                id if id == bytecode_ra(0) => bytecode_ra_0,
                id if id == bytecode_ra(1) => bytecode_ra_1,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltPublicId::BytecodeReadRaf(BytecodeReadRafPublic::StageValue(index)) => {
                    stage_values[index]
                }
                JoltPublicId::BytecodeReadRaf(BytecodeReadRafPublic::SpartanOuterRaf) => {
                    spartan_outer_raf
                }
                JoltPublicId::BytecodeReadRaf(BytecodeReadRafPublic::SpartanShiftRaf) => {
                    spartan_shift_raf
                }
                JoltPublicId::BytecodeReadRaf(BytecodeReadRafPublic::Entry) => entry,
                _ => zero,
            },
        );

        assert_eq!(
            output,
            (stage_values[0]
                + gamma * stage_values[1]
                + gamma_power(gamma, 2) * stage_values[2]
                + gamma_power(gamma, 3) * stage_values[3]
                + gamma_power(gamma, 4) * stage_values[4]
                + gamma_power(gamma, 5) * spartan_outer_raf
                + gamma_power(gamma, 6) * spartan_shift_raf
                + gamma_power(gamma, 7) * entry)
                * bytecode_ra_0
                * bytecode_ra_1
        );
    }
}
