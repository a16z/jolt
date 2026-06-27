//! The address phase of the bytecode read-RAF symbolic sumcheck.

use jolt_field::RingCore;
use jolt_riscv::{CircuitFlags, InstructionFlags, CIRCUIT_FLAGS};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::bytecode::{
    bytecode_read_raf_address_phase_opening, pc_spartan_outer, pc_spartan_shift, stage1_claim,
    stage2_claim, stage3_claim, stage4_claim, stage5_claim, BytecodeReadRafDimensions,
};
use crate::protocols::jolt::{
    BytecodeReadRafChallenge, JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId,
    JoltRelationId, JoltSumcheckSpec,
};
use crate::{challenge, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck};

/// The address-phase produced openings: the `BytecodeReadRafAddrClaim`
/// intermediate, plus (committed-program mode only) the staged `BytecodeValStage`
/// openings. In full-program mode `val_stages` is empty.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeReadRaf)]
pub struct BytecodeReadRafAddressPhaseOutputClaims<C> {
    #[opening(BytecodeReadRafAddrClaim)]
    pub intermediate: C,
    #[opening(BytecodeValStage)]
    pub val_stages: Vec<C>,
}

/// The prior-proof openings the address-phase input claim binds: every stage-1..5
/// opening the `read_raf_address_phase` input `Expr` folds (plus the two PC
/// claims). The generic `input_claim` evaluates the bind from these via that
/// `Expr`, so the gamma-folding formula lives in one place rather than a
/// hand-written 25-opening resolver. The `op_flags` / `lookup_table_flags`
/// families are indexed openings (`OpFlags(CIRCUIT_FLAGS[i])` /
/// `LookupTableFlag(i)`).
#[derive(Clone, Debug, InputClaims)]
pub struct BytecodeReadRafAddressPhaseInputClaims<C> {
    #[opening(UnexpandedPC, from = SpartanOuter)]
    pub outer_unexpanded_pc: C,
    #[opening(Imm, from = SpartanOuter)]
    pub outer_imm: C,
    #[opening(OpFlags(CIRCUIT_FLAGS), from = SpartanOuter)]
    pub outer_op_flags: Vec<C>,
    #[opening(PC, from = SpartanOuter)]
    pub outer_pc: C,
    #[opening(OpFlags(CircuitFlags::Jump), from = SpartanProductVirtualization)]
    pub product_jump: C,
    #[opening(InstructionFlags(InstructionFlags::Branch), from = SpartanProductVirtualization)]
    pub product_branch: C,
    #[opening(OpFlags(CircuitFlags::WriteLookupOutputToRD), from = SpartanProductVirtualization)]
    pub product_write_lookup_output_to_rd: C,
    #[opening(OpFlags(CircuitFlags::VirtualInstruction), from = SpartanProductVirtualization)]
    pub product_virtual_instruction: C,
    #[opening(Imm, from = InstructionInputVirtualization)]
    pub instruction_input_imm: C,
    #[opening(UnexpandedPC, from = SpartanShift)]
    pub shift_unexpanded_pc: C,
    #[opening(InstructionFlags(InstructionFlags::LeftOperandIsRs1Value), from = InstructionInputVirtualization)]
    pub left_operand_is_rs1_value: C,
    #[opening(InstructionFlags(InstructionFlags::LeftOperandIsPC), from = InstructionInputVirtualization)]
    pub left_operand_is_pc: C,
    #[opening(InstructionFlags(InstructionFlags::RightOperandIsRs2Value), from = InstructionInputVirtualization)]
    pub right_operand_is_rs2_value: C,
    #[opening(InstructionFlags(InstructionFlags::RightOperandIsImm), from = InstructionInputVirtualization)]
    pub right_operand_is_imm: C,
    #[opening(InstructionFlags(InstructionFlags::IsNoop), from = SpartanShift)]
    pub is_noop: C,
    #[opening(OpFlags(CircuitFlags::VirtualInstruction), from = SpartanShift)]
    pub shift_virtual_instruction: C,
    #[opening(OpFlags(CircuitFlags::IsFirstInSequence), from = SpartanShift)]
    pub shift_is_first_in_sequence: C,
    #[opening(PC, from = SpartanShift)]
    pub shift_pc: C,
    #[opening(RdWa, from = RegistersReadWriteChecking)]
    pub rd_wa_read_write: C,
    #[opening(Rs1Ra, from = RegistersReadWriteChecking)]
    pub rs1_ra: C,
    #[opening(Rs2Ra, from = RegistersReadWriteChecking)]
    pub rs2_ra: C,
    #[opening(RdWa, from = RegistersValEvaluation)]
    pub rd_wa_val_evaluation: C,
    #[opening(InstructionRafFlag, from = InstructionReadRaf)]
    pub instruction_raf_flag: C,
    #[opening(LookupTableFlag, from = InstructionReadRaf)]
    pub lookup_table_flags: Vec<C>,
}

/// Fiat-Shamir challenges drawn by the address phase of the bytecode read-RAF
/// sumcheck: the batching `gamma` plus the five per-stage gammas (the same set
/// the full monolith folds).
#[derive(Clone, Copy, Debug, SumcheckChallenges)]
pub struct BytecodeReadRafAddressPhaseChallenges<F> {
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

/// The address phase of the bytecode read-RAF sumcheck: the same folded input
/// claim, reduced to the staged address-phase opening.
pub struct ReadRafAddressPhase {
    shape: BytecodeReadRafDimensions,
}

impl SymbolicSumcheck for ReadRafAddressPhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafDimensions;
    type Challenges<F> = BytecodeReadRafAddressPhaseChallenges<F>;

    fn new(shape: BytecodeReadRafDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeReadRaf
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.address_sumcheck()
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
        opening(bytecode_read_raf_address_phase_opening())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    fn dimensions(num_committed_ra_polys: usize) -> BytecodeReadRafDimensions {
        BytecodeReadRafDimensions::new(5, 10, num_committed_ra_polys)
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
    fn read_raf_address_phase_symbolic_matches_dependencies() {
        let relation = ReadRafAddressPhase::new(dimensions(2));
        assert_eq!(ReadRafAddressPhase::id(), JoltRelationId::BytecodeReadRaf);
        assert_eq!(relation.spec(), dimensions(2).address_sumcheck());
        assert_eq!(relation.required_challenges::<Fr>(), stage_gammas());
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![bytecode_read_raf_address_phase_opening()]
        );
    }
}
