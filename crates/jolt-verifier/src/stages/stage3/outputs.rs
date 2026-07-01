//! Typed inputs consumed and outputs produced by stage 3 verification.

use jolt_claims::protocols::jolt::{
    geometry::{bytecode, claim_reductions::registers as registers_claim_reduction, instruction},
    JoltRelationId,
};
use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;
use jolt_transcript::Transcript;

use crate::stages::relations::{OutputClaims, SumcheckBatch};
use crate::stages::zk::outputs::CommittedOutputClaimOutput;
use crate::VerifierError;

pub use super::instruction_input::{InstructionInput, InstructionInputOutputClaims};
pub use super::registers_claim_reduction::{
    RegistersClaimReduction, RegistersClaimReductionOutputClaims,
};
pub use super::spartan_shift::{SpartanShift, SpartanShiftOutputClaims};

/// Source-of-truth for stage 3's sumcheck batch: the three instances in
/// Fiat-Shamir batch order (Spartan shift, instruction-input virtualization,
/// register claim-reduction). `#[derive(SumcheckBatch)]` generates the
/// `Stage3InputClaims<F>`, `Stage3InputPoints<F>`, `Stage3OutputClaims<F>`,
/// `Stage3OutputPoints<F>`, and `Stage3Challenges<F>` aggregates — one field per
/// instance, in this declaration order.
///
/// The opt-out `#[sumcheck_batch(custom_opening_values)]` suppresses the generated
/// `opening_values` / `append_to_transcript`: stage 3 has three cross-relation
/// aliased openings (`instruction_input.unexpanded_pc` = `shift.unexpanded_pc`, and
/// the register-reduction `rs1`/`rs2` = the instruction-input ones) that are
/// absorbed once via their canonical source, so the canonical order is curated by
/// hand below (and enforced by `validate`).
#[derive(SumcheckBatch)]
#[sumcheck_batch(custom_opening_values)]
pub struct Stage3Sumchecks<F: Field> {
    pub shift: SpartanShift<F>,
    pub instruction_input: InstructionInput<F>,
    pub registers_claim_reduction: RegistersClaimReduction<F>,
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

        if self.registers_claim_reduction.rs1_value != self.instruction_input.rs1_value {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: JoltRelationId::RegistersReadWriteChecking,
                left: registers_claim_reduction::rs1_value_reduced(),
                right: instruction::rs1_value(),
            });
        }
        if self.registers_claim_reduction.rs2_value != self.instruction_input.rs2_value {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: JoltRelationId::RegistersReadWriteChecking,
                left: registers_claim_reduction::rs2_value_reduced(),
                right: instruction::rs2_value(),
            });
        }
        Ok(())
    }
}

impl<F: Field> Stage3OutputPoints<F> {
    /// The shift relation's shared opening point (every shift output carries it).
    pub fn shift_opening_point(&self) -> &[F] {
        self.shift.unexpanded_pc()
    }

    /// The register claim-reduction relation's shared opening point.
    pub fn registers_opening_point(&self) -> &[F] {
        self.registers_claim_reduction.rd_write_value()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3ClearOutput<F: Field> {
    pub challenges: Stage3Challenges<F>,
    /// The produced stage-3 opening *values* (wire form); read by later stages and
    /// the Fiat-Shamir opening-claim encoder.
    pub output_values: Stage3OutputClaims<F>,
    /// The produced stage-3 opening *points*, paired field-for-field with
    /// `output_values`. Later stages read each opening's point off these cells.
    pub output_points: Stage3OutputPoints<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3ZkOutput<F: Field, C> {
    pub challenges: Stage3Challenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
}

// The clear variant carries the produced opening values + points read on the hot
// path; the ZK variant carries committed consistency. Boxing the common clear
// variant to shrink the rarer ZK one would add indirection to every clear-path
// access.
#[expect(
    clippy::large_enum_variant,
    reason = "clear variant holds the opening values + points read on the hot path; boxing it would penalize the common case"
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
        let claims = Stage3OutputClaims::<Fr> {
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

        assert_eq!(
            claims.opening_values(),
            (1..=13).map(fr).collect::<Vec<_>>()
        );
    }

    /// A stage-3 output with the three cross-relation aliases satisfied: shift and
    /// instruction-input `unexpanded_pc` equal, and register-reduction `rs1`/`rs2`
    /// equal the instruction-input ones. `validate` accepts it; the tests below
    /// perturb one alias each to assert rejection.
    fn consistent() -> Stage3OutputClaims<Fr> {
        Stage3OutputClaims::<Fr> {
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
                unexpanded_pc: fr(1),
                right_operand_is_rs2: fr(9),
                rs2_value: fr(10),
                right_operand_is_imm: fr(11),
                imm: fr(12),
            },
            registers_claim_reduction: RegistersClaimReductionOutputClaims {
                rd_write_value: fr(13),
                rs1_value: fr(7),
                rs2_value: fr(10),
            },
        }
    }

    #[test]
    fn validate_accepts_consistent_aliases() {
        assert!(consistent().validate().is_ok());
    }

    #[test]
    fn validate_rejects_unexpanded_pc_mismatch() {
        let mut claims = consistent();
        claims.instruction_input.unexpanded_pc = fr(99);
        assert!(claims.validate().is_err());
    }

    #[test]
    fn validate_rejects_rs1_value_mismatch() {
        let mut claims = consistent();
        claims.registers_claim_reduction.rs1_value = fr(99);
        assert!(claims.validate().is_err());
    }

    #[test]
    fn validate_rejects_rs2_value_mismatch() {
        let mut claims = consistent();
        claims.registers_claim_reduction.rs2_value = fr(99);
        assert!(claims.validate().is_err());
    }
}
