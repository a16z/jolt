//! Typed inputs consumed by stage 3.

use jolt_claims::protocols::jolt::{
    formulas::{bytecode, claim_reductions::registers as registers_claim_reduction, instruction},
    JoltRelationId,
};
use jolt_field::Field;
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::stages::relations::OpeningClaim;
use crate::stages::{
    stage1::{Stage1ClearOutput, Stage1Output, Stage1ZkOutput},
    stage2::{Stage2ClearOutput, Stage2Output, Stage2ZkOutput},
};
use crate::VerifierError;

pub use super::instruction_input::InstructionInputOutputClaims;
pub use super::registers_claim_reduction::RegistersClaimReductionOutputClaims;
pub use super::spartan_shift::SpartanShiftOutputClaims;

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear {
        stage1: &'a Stage1ClearOutput<F>,
        stage2: &'a Stage2ClearOutput<F>,
    },
    Zk {
        stage1: &'a Stage1ZkOutput<F, C>,
        stage2: &'a Stage2ZkOutput<F, C>,
    },
}

pub fn deps<'a, F: Field, C>(
    stage1: &'a Stage1Output<F, C>,
    stage2: &'a Stage2Output<F, C>,
) -> Result<Deps<'a, F, C>, crate::VerifierError> {
    match (stage1, stage2) {
        (Stage1Output::Clear(stage1), Stage2Output::Clear(stage2)) => {
            Ok(Deps::Clear { stage1, stage2 })
        }
        (Stage1Output::Zk(stage1), Stage2Output::Zk(stage2)) => Ok(Deps::Zk { stage1, stage2 }),
        (Stage1Output::Clear(_), Stage2Output::Zk(_)) => {
            Err(crate::VerifierError::ExpectedClearProof { field: "stage2" })
        }
        (Stage1Output::Zk(_), Stage2Output::Clear(_)) => {
            Err(crate::VerifierError::ExpectedCommittedProof { field: "stage2" })
        }
    }
}

/// The stage 3 produced opening claims: the Spartan shift, instruction-input
/// virtualization, and register claim-reduction relations' outputs. Generic over
/// the cell (`F` on the wire / serialized proof form, `OpeningClaim<F>` on the
/// clear path once the opening points are derived).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct Stage3OutputClaims<C> {
    pub shift: SpartanShiftOutputClaims<C>,
    pub instruction_input: InstructionInputOutputClaims<C>,
    pub registers_claim_reduction: RegistersClaimReductionOutputClaims<C>,
}

impl<F: Field> Stage3OutputClaims<F> {
    /// The produced opening claims absorbed into the transcript, in canonical
    /// (Fiat-Shamir) order. Three of the sixteen produced openings are aliases —
    /// `instruction_input.unexpanded_pc` equals `shift.unexpanded_pc`, and the
    /// register-reduction `rs1`/`rs2` values equal the instruction-input ones — so
    /// they are absorbed once via their canonical source and skipped here.
    /// [`validate`](Self::validate) enforces those equalities. Single-sources
    /// [`append_to_transcript`](Self::append_to_transcript) and the prover's
    /// output-claim values.
    pub fn opening_values(&self) -> Vec<F> {
        vec![
            self.shift.unexpanded_pc,
            self.shift.pc,
            self.shift.is_virtual,
            self.shift.is_first_in_sequence,
            self.shift.is_noop,
            self.instruction_input.left_operand_is_rs1,
            self.instruction_input.rs1_value,
            self.instruction_input.left_operand_is_pc,
            self.instruction_input.right_operand_is_rs2,
            self.instruction_input.rs2_value,
            self.instruction_input.right_operand_is_imm,
            self.instruction_input.imm,
            self.registers_claim_reduction.rd_write_value,
        ]
    }

    /// Append every absorbed opening to the transcript in canonical order, each
    /// under the `b"opening_claim"` label, matching the prover's commitment order.
    pub fn append_to_transcript<T: Transcript<Challenge = F>>(&self, transcript: &mut T) {
        for value in self.opening_values() {
            transcript.append_labeled(b"opening_claim", &value);
        }
    }

    /// Enforce the three cross-relation opening aliases that let the canonical
    /// order absorb each shared value once: the shift and instruction-input
    /// `unexpanded_pc`, and the register-reduction and instruction-input `rs1`/`rs2`
    /// values. Run by the stage-3 verifier after the per-relation output checks
    /// (which catch any single-claim offset first); this guards the cross-relation
    /// consistency the downstream stages relied on. The errors preserve the
    /// opening ids and stages those downstream checks used.
    pub fn validate(&self) -> Result<(), VerifierError> {
        let [(shift_unexpanded_pc, instruction_unexpanded_pc)] =
            bytecode::read_raf_consistency_openings();
        if self.shift.unexpanded_pc != self.instruction_input.unexpanded_pc {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: JoltRelationId::BytecodeReadRaf,
                left: shift_unexpanded_pc,
                right: instruction_unexpanded_pc,
            });
        }

        let [_, rs2_value_instruction, _, _, _, rs1_value_instruction, _, _] =
            instruction::input_virtualization_output_openings();
        let [_, rs1_value_reduced, rs2_value_reduced] =
            registers_claim_reduction::claim_reduction_output_openings();
        if self.registers_claim_reduction.rs1_value != self.instruction_input.rs1_value {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: JoltRelationId::RegistersReadWriteChecking,
                left: rs1_value_reduced,
                right: rs1_value_instruction,
            });
        }
        if self.registers_claim_reduction.rs2_value != self.instruction_input.rs2_value {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: JoltRelationId::RegistersReadWriteChecking,
                left: rs2_value_reduced,
                right: rs2_value_instruction,
            });
        }
        Ok(())
    }
}

impl<F: Field> Stage3OutputClaims<OpeningClaim<F>> {
    /// The shift relation's shared opening point (every shift output carries it).
    pub fn shift_opening_point(&self) -> &[F] {
        &self.shift.unexpanded_pc.point
    }

    /// The instruction-input relation's shared opening point.
    pub fn instruction_opening_point(&self) -> &[F] {
        &self.instruction_input.unexpanded_pc.point
    }

    /// The register claim-reduction relation's shared opening point.
    pub fn registers_opening_point(&self) -> &[F] {
        &self.registers_claim_reduction.rd_write_value.point
    }
}
