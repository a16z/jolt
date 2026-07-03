//! Typed inputs consumed and outputs produced by stage 3 verification.

use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::SumcheckBatch;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

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
/// instance, in this declaration order — plus the batched-verify drivers and the
/// absorb plumbing.
///
/// The instruction-input virtualization declares one cross-relation opening alias
/// (`unexpanded_pc` = the shift's) and the register claim-reduction declares two
/// (`rs1_value`/`rs2_value` = the instruction-input ones), so the generated
/// `append_output_claims` absorbs 13 of the members' 16 expression-referenced
/// openings (each alias once, via its canonical source), the generated
/// `output_shape` count/validator use the same wire sets, and the generated
/// `validate_aliases` (run by `expected_final_claim`) enforces the aliased wire
/// copies equal their sources.
#[derive(SumcheckBatch)]
#[sumcheck_batch(empty_input_points, output_shape)]
pub struct Stage3Sumchecks<F: Field> {
    pub shift: SpartanShift<F>,
    pub instruction_input: InstructionInput<F>,
    pub registers_claim_reduction: RegistersClaimReduction<F>,
}

impl<F: Field> Stage3OutputPoints<F> {
    /// The shift relation's shared opening point (every shift output carries it).
    pub fn shift_opening_point(&self) -> &[F] {
        self.shift.unexpanded_pc()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3ClearOutput<F: Field> {
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
    /// The produced opening points, the ZK counterpart of the clear path's
    /// `output_points`. Read through the same `*_point()` accessors.
    pub output_points: Stage3OutputPoints<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage3Output<F: Field, C> {
    Clear(Stage3ClearOutput<F>),
    Zk(Stage3ZkOutput<F, C>),
}

impl<F: Field, C> Stage3Output<F, C> {
    /// The produced opening points, available regardless of proving mode.
    pub fn output_points(&self) -> &Stage3OutputPoints<F> {
        match self {
            Self::Clear(output) => &output.output_points,
            Self::Zk(output) => &output.output_points,
        }
    }

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
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stages::relations::ConcreteSumcheck;
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn sumchecks() -> Stage3Sumchecks<Fr> {
        let dimensions = TraceDimensions::new(4);
        Stage3Sumchecks {
            shift: SpartanShift::new(dimensions, Vec::new(), Vec::new()),
            instruction_input: InstructionInput::new(dimensions, Vec::new()),
            registers_claim_reduction: RegistersClaimReduction::new(dimensions, Vec::new()),
        }
    }

    /// A stage-3 output with the three cross-relation aliases satisfied: shift and
    /// instruction-input `unexpanded_pc` equal, and register-reduction `rs1`/`rs2`
    /// equal the instruction-input ones. `validate_aliases` accepts it; the tests
    /// below perturb one alias each to assert rejection. The absorb test overrides
    /// the aliased cells with sentinels to prove they are skipped.
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

    /// Locks the stage-3 Fiat-Shamir append order against silent drift: the
    /// generated absorb follows member declaration order and each member's
    /// `canonical_order`, skipping the three aliased openings (absorbed once via
    /// their canonical sources). The aliased cells carry distinct sentinels here
    /// to prove the skip is id-driven, not value-driven.
    #[test]
    fn opening_values_follow_canonical_order() {
        let mut claims = consistent();
        claims.instruction_input.unexpanded_pc = fr(101);
        claims.registers_claim_reduction.rs1_value = fr(102);
        claims.registers_claim_reduction.rs2_value = fr(103);

        assert_eq!(
            sumchecks().opening_values(&claims),
            (1..=13).map(fr).collect::<Vec<_>>()
        );
    }

    /// The generated `output_claim_count` sums the members' wire sets: the 16
    /// expression-referenced openings minus the 3 aliases.
    #[test]
    fn output_claim_count_matches_absorbed_openings() {
        let sumchecks = sumchecks();
        assert_eq!(sumchecks.output_claim_count(), 13);
        assert_eq!(
            sumchecks.opening_values(&consistent()).len(),
            sumchecks.output_claim_count(),
        );
    }

    /// Pins the stage's alias declarations: each aliased id is distinct and
    /// referenced by its declaring member's own output `Expr` (so the batch fold
    /// constrains the wire cell), and each canonical source is absorbed by its
    /// source member (so the value the copy is checked against is
    /// Fiat-Shamir-bound). The point-slice identity the value-only check relies
    /// on is pinned by `aliased_members_derive_identical_opening_points`.
    #[test]
    fn alias_declarations_are_valid() {
        use jolt_claims::SymbolicSumcheck as _;
        use std::collections::BTreeSet;

        let sumchecks = sumchecks();
        let shift_wire = sumchecks.shift.wire_output_openings();
        let instruction_wire = sumchecks.instruction_input.wire_output_openings();

        let instruction_pairs = InstructionInput::<Fr>::aliased_output_openings();
        assert_eq!(instruction_pairs.len(), 1);
        let instruction_expression = sumchecks
            .instruction_input
            .symbolic()
            .expected_output_openings::<Fr>();
        for (aliased, source) in &instruction_pairs {
            assert!(
                instruction_expression.contains(aliased),
                "aliased opening {aliased:?} is not referenced by the instruction-input Expr",
            );
            assert!(
                shift_wire.contains(source),
                "source {source:?} is not absorbed by the shift",
            );
        }

        let register_pairs = RegistersClaimReduction::<Fr>::aliased_output_openings();
        assert_eq!(register_pairs.len(), 2);
        let register_expression = sumchecks
            .registers_claim_reduction
            .symbolic()
            .expected_output_openings::<Fr>();
        for (aliased, source) in &register_pairs {
            assert!(
                register_expression.contains(aliased),
                "aliased opening {aliased:?} is not referenced by the register-reduction Expr",
            );
            assert!(
                instruction_wire.contains(source),
                "source {source:?} is not absorbed by the instruction-input virtualization",
            );
        }

        let mut seen = BTreeSet::new();
        for (aliased, _) in instruction_pairs.iter().chain(&register_pairs) {
            assert!(
                seen.insert(*aliased),
                "duplicate aliased opening {aliased:?}"
            );
        }
    }

    #[test]
    fn validate_aliases_accepts_consistent_aliases() {
        assert!(sumchecks().validate_aliases(&consistent()).is_ok());
    }

    #[test]
    fn validate_aliases_rejects_unexpanded_pc_mismatch() {
        let mut claims = consistent();
        claims.instruction_input.unexpanded_pc = fr(99);
        assert!(sumchecks().validate_aliases(&claims).is_err());
    }

    #[test]
    fn validate_aliases_rejects_rs1_value_mismatch() {
        let mut claims = consistent();
        claims.registers_claim_reduction.rs1_value = fr(99);
        assert!(sumchecks().validate_aliases(&claims).is_err());
    }

    #[test]
    fn validate_aliases_rejects_rs2_value_mismatch() {
        let mut claims = consistent();
        claims.registers_claim_reduction.rs2_value = fr(99);
        assert!(sumchecks().validate_aliases(&claims).is_err());
    }

    /// Pins the structural invariant the alias declarations rely on:
    /// `validate_aliases` checks values only, which is sound because all three
    /// stage-3 members bind the same batch-point slice (equal rounds, default
    /// offsets) and derive the same opening point — each aliased pair is the same
    /// polynomial at the same point by construction, never by proof content.
    #[test]
    fn aliased_members_derive_identical_opening_points() {
        let sumchecks = sumchecks();
        let shift = &sumchecks.shift;
        let instruction_input = &sumchecks.instruction_input;
        let registers = &sumchecks.registers_claim_reduction;
        assert_eq!(shift.rounds(), instruction_input.rounds());
        assert_eq!(shift.rounds(), registers.rounds());
        let batch_num_vars = shift.rounds() + 2;
        let offset = shift.instance_point_offset(batch_num_vars).unwrap();
        assert_eq!(
            instruction_input
                .instance_point_offset(batch_num_vars)
                .unwrap(),
            offset,
        );
        assert_eq!(
            registers.instance_point_offset(batch_num_vars).unwrap(),
            offset,
        );

        let point: Vec<Fr> = (0..shift.rounds() as u64).map(|i| fr(20 + i)).collect();
        let input_points = sumchecks.empty_input_points();
        let shift_points = shift
            .derive_opening_points(&point, &input_points.shift)
            .unwrap();
        let instruction_points = instruction_input
            .derive_opening_points(&point, &input_points.instruction_input)
            .unwrap();
        let register_points = registers
            .derive_opening_points(&point, &input_points.registers_claim_reduction)
            .unwrap();
        assert_eq!(shift_points.unexpanded_pc, instruction_points.unexpanded_pc);
        assert_eq!(instruction_points.rs1_value, register_points.rs1_value);
        assert_eq!(instruction_points.rs2_value, register_points.rs2_value);
    }
}
