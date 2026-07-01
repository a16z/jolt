//! The full bytecode read-RAF symbolic sumcheck (monolith).

use jolt_field::RingCore;

use crate::protocols::jolt::geometry::bytecode::{
    pc_spartan_outer, pc_spartan_shift, read_raf_cycle_output, stage1_claim, stage2_claim,
    stage3_claim, stage4_claim, stage5_claim, BytecodeReadRafDimensions,
};
use crate::protocols::jolt::{
    BytecodeReadRafChallenge, JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId,
    JoltRelationId,
};
use crate::{challenge, opening, SumcheckChallenges, SymbolicSumcheck};

/// Fiat-Shamir challenges drawn by the full bytecode read-RAF sumcheck: the
/// batching `gamma` plus the five per-stage gammas folding the staged claims.
#[derive(Clone, Copy, Debug, SumcheckChallenges)]
pub struct BytecodeReadRafChallenges<F> {
    #[challenge(BytecodeReadRafChallenge::Gamma)]
    pub gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage1Gamma)]
    pub stage1_gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage2Gamma)]
    pub stage2_gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage3Gamma)]
    pub stage3_gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage4Gamma)]
    pub stage4_gamma: F,
    #[challenge(BytecodeReadRafChallenge::Stage5Gamma)]
    pub stage5_gamma: F,
}

/// The full bytecode read-RAF sumcheck: folds the five staged claims plus the
/// Spartan outer/shift PC openings against the bytecode-table cycle output.
pub struct ReadRaf {
    shape: BytecodeReadRafDimensions,
}

impl SymbolicSumcheck for ReadRaf {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafDimensions;
    type Challenges<F> = BytecodeReadRafChallenges<F>;
    type Inputs<C> = crate::NoInputs<C>;
    type Outputs<C> = crate::NoOutputs<C>;

    fn new(shape: BytecodeReadRafDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeReadRaf
    }

    fn rounds(&self) -> usize {
        self.shape.sumcheck_rounds()
    }

    fn degree(&self) -> usize {
        self.shape.num_committed_ra_polys() + 1
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(BytecodeReadRafChallenge::Gamma);

        gamma.clone().pow(7)
            + stage1_claim()
            + gamma.clone() * stage2_claim()
            + gamma.clone().pow(2) * stage3_claim()
            + gamma.clone().pow(3) * stage4_claim()
            + gamma.clone().pow(4) * stage5_claim::<F>()
            + gamma.clone().pow(5) * opening(pc_spartan_outer())
            + gamma.pow(6) * opening(pc_spartan_shift())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        read_raf_cycle_output(self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::bytecode::{
        bytecode_ra, imm_instruction_input, imm_spartan_outer, instruction_flag_input,
        instruction_flag_product, instruction_flag_shift, instruction_raf_flag, op_flag_product,
        op_flag_shift, rd_wa_read_write, rd_wa_val_evaluation, rs1_ra_read_write,
        rs2_ra_read_write, unexpanded_pc_spartan_outer, unexpanded_pc_spartan_shift,
    };
    use crate::protocols::jolt::{BytecodeReadRafPublic, JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_lookup_tables::{LookupTableKind, XLEN};
    use jolt_riscv::{CircuitFlags, InstructionFlags, CIRCUIT_FLAGS};

    fn dimensions(num_committed_ra_polys: usize) -> BytecodeReadRafDimensions {
        BytecodeReadRafDimensions::new(5, 10, num_committed_ra_polys)
    }

    fn gamma_power(gamma: Fr, exponent: usize) -> Fr {
        let mut value = Fr::from_u64(1);
        for _ in 0..exponent {
            value *= gamma;
        }
        value
    }

    fn stage_gammas() -> Vec<JoltChallengeId> {
        vec![
            JoltChallengeId::from(BytecodeReadRafChallenge::Gamma),
            JoltChallengeId::from(BytecodeReadRafChallenge::Stage1Gamma),
            JoltChallengeId::from(BytecodeReadRafChallenge::Stage2Gamma),
            JoltChallengeId::from(BytecodeReadRafChallenge::Stage3Gamma),
            JoltChallengeId::from(BytecodeReadRafChallenge::Stage4Gamma),
            JoltChallengeId::from(BytecodeReadRafChallenge::Stage5Gamma),
        ]
    }

    #[test]
    fn read_raf_evaluates_like_core_formula() {
        let dimensions = dimensions(2);
        let relation = ReadRaf::new(dimensions);

        let gamma = Fr::from_u64(3);
        let stage1_gamma = Fr::from_u64(5);
        let stage2_gamma = Fr::from_u64(7);
        let stage3_gamma = Fr::from_u64(11);
        let stage4_gamma = Fr::from_u64(13);
        let stage5_gamma = Fr::from_u64(17);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
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
                    relation: JoltRelationId::SpartanOuter,
                } => Fr::from_u64(200 + u64::from(flag as u8)),
                JoltOpeningId::Polynomial {
                    polynomial:
                        JoltPolynomialId::Virtual(JoltVirtualPolynomial::LookupTableFlag(index)),
                    relation: JoltRelationId::InstructionReadRaf,
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
        for table in LookupTableKind::<XLEN>::iter() {
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

        let output = relation.output_expression::<Fr>().evaluate(
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
                JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::StageValue(index)) => {
                    stage_values[index]
                }
                JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::SpartanOuterRaf) => {
                    spartan_outer_raf
                }
                JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::SpartanShiftRaf) => {
                    spartan_shift_raf
                }
                JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::Entry) => entry,
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

    #[test]
    fn read_raf_symbolic_matches_dependencies() {
        let relation = ReadRaf::new(dimensions(2));
        assert_eq!(ReadRaf::id(), JoltRelationId::BytecodeReadRaf);
        assert_eq!(relation.rounds(), dimensions(2).sumcheck_rounds());
        assert_eq!(
            relation.degree(),
            dimensions(2).num_committed_ra_polys() + 1
        );
        assert_eq!(relation.required_challenges::<Fr>(), stage_gammas());
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![
                JoltDerivedId::from(BytecodeReadRafPublic::StageValue(0)),
                JoltDerivedId::from(BytecodeReadRafPublic::StageValue(1)),
                JoltDerivedId::from(BytecodeReadRafPublic::StageValue(2)),
                JoltDerivedId::from(BytecodeReadRafPublic::StageValue(3)),
                JoltDerivedId::from(BytecodeReadRafPublic::StageValue(4)),
                JoltDerivedId::from(BytecodeReadRafPublic::SpartanOuterRaf),
                JoltDerivedId::from(BytecodeReadRafPublic::SpartanShiftRaf),
                JoltDerivedId::from(BytecodeReadRafPublic::Entry),
            ]
        );
    }

    /// Every staged gamma the relation draws (`stage_gammas`) must resolve to a
    /// distinct field of the `Challenges` struct. A missing or mismatched
    /// `#[challenge(..)]` among the six fields would surface here as a `None` or a
    /// wrong value.
    #[test]
    fn challenges_resolve_every_stage_gamma() {
        let challenges = BytecodeReadRafChallenges {
            gamma: Fr::from_u64(2),
            stage1_gamma: Fr::from_u64(3),
            stage2_gamma: Fr::from_u64(5),
            stage3_gamma: Fr::from_u64(7),
            stage4_gamma: Fr::from_u64(11),
            stage5_gamma: Fr::from_u64(13),
        };
        let expected = [
            (BytecodeReadRafChallenge::Gamma, Fr::from_u64(2)),
            (BytecodeReadRafChallenge::Stage1Gamma, Fr::from_u64(3)),
            (BytecodeReadRafChallenge::Stage2Gamma, Fr::from_u64(5)),
            (BytecodeReadRafChallenge::Stage3Gamma, Fr::from_u64(7)),
            (BytecodeReadRafChallenge::Stage4Gamma, Fr::from_u64(11)),
            (BytecodeReadRafChallenge::Stage5Gamma, Fr::from_u64(13)),
        ];
        for (id, value) in expected {
            assert_eq!(
                challenges.resolve_challenge(&JoltChallengeId::from(id)),
                Some(value),
            );
        }
        // Each id the relation declares it draws resolves under the struct.
        for id in stage_gammas() {
            assert!(challenges.resolve_challenge(&id).is_some());
        }
    }
}
