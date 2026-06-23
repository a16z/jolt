//! Typed inputs consumed and outputs produced by stage 3 verification.

use jolt_claims::protocols::jolt::{
    formulas::{bytecode, claim_reductions::registers as registers_claim_reduction, instruction},
    JoltRelationId,
};
use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::stages::relations::{OpeningClaim, OutputClaims};
use crate::stages::zk::outputs::CommittedOutputClaimOutput;
use crate::VerifierError;

pub use super::instruction_input::InstructionInputOutputClaims;
pub use super::registers_claim_reduction::RegistersClaimReductionOutputClaims;
pub use super::spartan_shift::SpartanShiftOutputClaims;

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
        // `shift` delegates to its derived `opening_values()` so its per-field order
        // is single-sourced from the `OutputClaims` derive. The instruction-input
        // and register-reduction openings are listed explicitly because three of
        // them alias canonical sources and are absorbed once (see `validate`):
        // `instruction_input.unexpanded_pc` (= `shift.unexpanded_pc`) and the
        // register-reduction `rs1`/`rs2` values are skipped here.
        self.shift
            .opening_values()
            .into_iter()
            .chain([
                self.instruction_input.left_operand_is_rs1,
                self.instruction_input.rs1_value,
                self.instruction_input.left_operand_is_pc,
                self.instruction_input.right_operand_is_rs2,
                self.instruction_input.rs2_value,
                self.instruction_input.right_operand_is_imm,
                self.instruction_input.imm,
                self.registers_claim_reduction.rd_write_value,
            ])
            .collect()
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

    /// The register claim-reduction relation's shared opening point.
    pub fn registers_opening_point(&self) -> &[F] {
        &self.registers_claim_reduction.rd_write_value.point
    }
}

/// The Fiat-Shamir challenges the verifier draws during stage 3: the three
/// per-relation batching gammas. (The batch's own sumcheck point and batching
/// coefficients are stage-local verification artifacts and are not propagated to
/// later stages.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3Challenges<F: Field> {
    pub shift_gamma: F,
    pub instruction_gamma: F,
    pub registers_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3ClearOutput<F: Field> {
    pub challenges: Stage3Challenges<F>,
    /// The produced stage-3 openings paired with their points (point + value) via
    /// the `OpeningClaim` cell. The opening points are derived from each relation's
    /// sumcheck point; pairing them with the values here lets later stages consume a
    /// ready `OpeningClaim` instead of re-joining a value with a separately-tracked
    /// point.
    pub output_claims: Stage3OutputClaims<OpeningClaim<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3ZkOutput<F: Field, C> {
    pub challenges: Stage3Challenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
}

// The clear variant carries the located opening claims (point + value) that
// stages 4 and 6 read on the hot path; the ZK variant carries only committed
// consistency. Boxing the common clear variant to shrink the rarer ZK one would
// add indirection to every clear-path access.
#[expect(
    clippy::large_enum_variant,
    reason = "clear variant holds the located opening claims read on the hot path; boxing it would penalize the common case"
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage3Output<F: Field, C> {
    Clear(Stage3ClearOutput<F>),
    Zk(Stage3ZkOutput<F, C>),
}

impl<F: Field, C> Stage3Output<F, C> {
    pub fn clear(&self) -> Result<&Stage3ClearOutput<F>, crate::VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(crate::VerifierError::ExpectedClearProof { field: "stage3" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage3ZkOutput<F, C>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage3" }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// Locks the stage-3 Fiat-Shamir append order against silent drift. `shift`
    /// single-sources its order via the `OutputClaims` derive; the instruction-input
    /// and register-reduction relations are curated (three openings alias canonical
    /// sources and are absorbed once via those sources). The aliased openings carry
    /// distinct sentinels here to prove they are skipped.
    #[test]
    fn opening_values_follow_canonical_order() {
        let claims = Stage3OutputClaims {
            shift: SpartanShiftOutputClaims {
                unexpanded_pc: fr(1),
                pc: fr(2),
                is_virtual: fr(3),
                is_first_in_sequence: fr(4),
                is_noop: fr(5),
            },
            instruction_input: InstructionInputOutputClaims {
                left_operand_is_rs1: fr(6),
                rs1_value: fr(7),
                left_operand_is_pc: fr(8),
                unexpanded_pc: fr(101),
                right_operand_is_rs2: fr(9),
                rs2_value: fr(10),
                right_operand_is_imm: fr(11),
                imm: fr(12),
            },
            registers_claim_reduction: RegistersClaimReductionOutputClaims {
                rd_write_value: fr(13),
                rs1_value: fr(102),
                rs2_value: fr(103),
            },
        };

        assert_eq!(claims.opening_values(), (1..=13).map(fr).collect::<Vec<_>>());
    }
}
