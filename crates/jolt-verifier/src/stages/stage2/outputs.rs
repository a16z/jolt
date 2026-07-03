//! Typed inputs consumed and outputs produced by stage 2 verification.

use jolt_field::Field;
use jolt_sumcheck::{BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency};
use serde::{Deserialize, Serialize};

use crate::stages::relations::SumcheckBatch;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

pub use super::instruction_claim_reduction::{
    InstructionClaimReduction, InstructionClaimReductionOutputClaims,
};
pub use super::product_remainder::{ProductRemainder, ProductRemainderOutputClaims};
pub use super::ram_output_check::{RamOutputCheck, RamOutputCheckOutputClaims};
pub use super::ram_raf_evaluation::{RamRafEvaluation, RamRafEvaluationOutputClaims};
pub use super::ram_read_write_checking::{RamReadWriteChecking, RamReadWriteOutputClaims};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: for<'a> Deserialize<'a>"))]
pub struct Stage2OutputClaims<F: Field> {
    pub product_uniskip_output_claim: F,
    pub batch_outputs: Stage2BatchOutputClaims<F>,
}

/// Source-of-truth for stage 2's five-instance sumcheck batch, in Fiat-Shamir
/// batch order (RAM read-write, product remainder, instruction claim-reduction,
/// RAM RAF evaluation, RAM output check). `#[derive(SumcheckBatch)]` generates the
/// `Stage2Batch{Input,Output}{Claims,Points}<F>` and `Stage2BatchChallenges<F>`
/// aggregates — one field per instance, in this declaration order — plus the
/// batched-verify drivers and the absorb plumbing. The product uni-skip is a
/// separate sub-sumcheck, not part of this batch.
///
/// The instruction claim-reduction declares three cross-relation opening aliases
/// (`lookup_output`, `left`/`right_instruction_input` = the product-remainder
/// openings; see its `aliased_output_openings`), and the product remainder
/// declares two staged openings (`write_lookup_output_to_rd` /
/// `virtual_instruction`, absorbed here but folded downstream by stage 6a). The
/// batch therefore absorbs 15 openings — 16 expression-referenced, minus the 3
/// aliases (each absorbed once via its product-remainder source), plus the 2
/// staged — and the generated absorb, `output_shape` count/validator, and
/// `validate_aliases` (run by `expected_final_claim`, enforcing the aliased wire
/// copies equal their sources) all derive from those per-member declarations.
/// The two RAM relations slice their point at the phase-1 `instance_point_offset`.
#[derive(SumcheckBatch)]
#[sumcheck_batch(output_shape)]
pub struct Stage2BatchSumchecks<F: Field> {
    pub ram_read_write: RamReadWriteChecking<F>,
    pub product_remainder: ProductRemainder<F>,
    pub instruction_claim_reduction: InstructionClaimReduction<F>,
    pub ram_raf_evaluation: RamRafEvaluation<F>,
    pub ram_output_check: RamOutputCheck<F>,
}

/// The shared per-relation opening-point accessors over the point-only stage-2
/// batch aggregate.
impl<F: Field> Stage2BatchOutputPoints<F> {
    /// The RAM read-write opening point (shared by `val`/`ra`/`inc`).
    pub fn ram_read_write_point(&self) -> &[F] {
        self.ram_read_write.val()
    }

    /// The product-remainder opening point (shared by all eight openings).
    pub fn product_remainder_point(&self) -> &[F] {
        self.product_remainder.left_instruction_input()
    }

    /// The reduced instruction-claim opening point (shared by all five openings).
    pub fn instruction_claim_reduction_point(&self) -> &[F] {
        self.instruction_claim_reduction.left_lookup_operand()
    }

    /// The RAM RAF opening point (`[r_address ‖ tau_low]`).
    pub fn ram_raf_evaluation_point(&self) -> &[F] {
        self.ram_raf_evaluation.ram_ra()
    }

    /// The RAM output-check opening point (`r_address`).
    pub fn ram_output_check_point(&self) -> &[F] {
        self.ram_output_check.val_final()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ClearOutput<F: Field> {
    /// The produced batch opening *values* (wire form); later stages read each
    /// opening's value directly off these fields.
    pub output_values: Stage2BatchOutputClaims<F>,
    /// The produced batch opening *points*, paired field-for-field with
    /// `output_values`. Later stages read the points through the `*_point()`
    /// accessors.
    pub output_points: Stage2BatchOutputPoints<F>,
    /// The product uni-skip `tau_low` (stage 1's remainder point low half,
    /// reversed), read mode-agnostically via [`Stage2Output::product_tau_low`].
    pub product_tau_low: Vec<F>,
}

/// Stage 2's ZK output, carrying the Fiat-Shamir values BlindFold sources via
/// `input.stage2.<field>`. The two batch gammas are the generated
/// [`Stage2BatchChallenges`] member structs (`challenges.ram_read_write.gamma`,
/// `challenges.instruction_claim_reduction.gamma`; the other three batch relations
/// draw nothing — `NoChallenges`). The remaining three are non-batch draws — the
/// product uni-skip reduction challenge and its freshly-drawn `product_tau_high`
/// scalar (a separate sub-sumcheck), and the RAM output-check address reference
/// point (folded in like stage 6's booleanity reference points) — so they are not
/// part of the per-instance aggregate. `product_tau_low` is opening-derived (stage
/// 1's remainder sumcheck point low half), stored so downstream stage-3 relation
/// construction can read it mode-agnostically via
/// [`Stage2Output::product_tau_low`]; BlindFold independently recomputes it from
/// `stage1.remainder_consistency`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ZkOutput<F: Field, C> {
    pub challenges: Stage2BatchChallenges<F>,
    pub product_uniskip_challenge: F,
    pub product_tau_low: Vec<F>,
    pub product_tau_high: F,
    pub output_address_challenges: Vec<F>,
    pub product_uniskip_consistency: CommittedSumcheckConsistency<F, C>,
    pub product_uniskip_output_claims: CommittedOutputClaimOutput<C>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    /// The produced batch opening points, the ZK counterpart of the clear path's
    /// `output_points`. Later stages read them through the same `*_point()` accessors.
    pub output_points: Stage2BatchOutputPoints<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage2Output<F: Field, C> {
    Clear(Stage2ClearOutput<F>),
    Zk(Stage2ZkOutput<F, C>),
}

impl<F: Field, C> Stage2Output<F, C> {
    /// The product uni-skip `tau_low` (stage 1's remainder point low half,
    /// reversed), available regardless of proving mode. Stage 3's relation
    /// construction evaluates its `EqPlusOne`/`EqSpartan` publics against it.
    pub fn product_tau_low(&self) -> &[F] {
        match self {
            Self::Clear(output) => &output.product_tau_low,
            Self::Zk(output) => &output.product_tau_low,
        }
    }

    /// The produced batch opening points, available regardless of proving mode.
    pub fn batch_output_points(&self) -> &Stage2BatchOutputPoints<F> {
        match self {
            Self::Clear(output) => &output.output_points,
            Self::Zk(output) => &output.output_points,
        }
    }

    pub fn clear(&self) -> Result<&Stage2ClearOutput<F>, crate::VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(crate::VerifierError::ExpectedClearProof { field: "stage2" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage2ZkOutput<F, C>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage2" }),
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stages::relations::draw_recording::{record, DrawEvent};
    use crate::stages::relations::ConcreteSumcheck;
    use common::jolt_device::{JoltDevice, MemoryConfig};
    use jolt_claims::protocols::jolt::geometry::{
        dimensions::{ReadWriteDimensions, TraceDimensions},
        ram::RamRafEvaluationDimensions,
        spartan::SpartanProductDimensions,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_program::preprocess::PublicIoMemory;
    use jolt_transcript::Transcript;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn sumchecks() -> Stage2BatchSumchecks<Fr> {
        let log_t = 4usize;
        let log_k = 3usize;
        let dimensions = ReadWriteDimensions::new(log_t, log_k, 2, 1);
        let raf_dimensions = RamRafEvaluationDimensions::try_from(dimensions).unwrap();
        let public_memory = PublicIoMemory::new(&JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        }))
        .unwrap();
        Stage2BatchSumchecks::<Fr> {
            ram_read_write: RamReadWriteChecking::new(dimensions, log_k, Vec::new()),
            product_remainder: ProductRemainder::new(
                SpartanProductDimensions::new(log_t),
                fr(1),
                fr(2),
                Vec::new(),
            ),
            instruction_claim_reduction: InstructionClaimReduction::new(
                TraceDimensions::new(log_t),
                Vec::new(),
            ),
            ram_raf_evaluation: RamRafEvaluation::new(
                dimensions,
                raf_dimensions,
                log_k,
                0,
                Vec::new(),
            ),
            ram_output_check: RamOutputCheck::new(dimensions, Vec::new(), public_memory),
        }
    }

    /// Pins the batch's `draw_challenges` to the pre-port inline draw: exactly two
    /// squeezes, the RAM read-write gamma then the instruction claim-reduction
    /// gamma (the other three members are `NoChallenges` and draw nothing).
    #[test]
    fn draw_challenges_matches_inline_two_gamma_sequence() {
        let sumchecks = sumchecks();
        let (inline_events, (inline_ram_gamma, inline_instruction_gamma)) =
            record(|t| (t.challenge_scalar(), t.challenge_scalar()));
        let (draw_events, challenges) = record(|t| sumchecks.draw_challenges(t).unwrap());

        assert_eq!(draw_events, inline_events);
        assert_eq!(
            draw_events,
            vec![DrawEvent::Squeeze(1), DrawEvent::Squeeze(2)]
        );
        assert_eq!(challenges.ram_read_write.gamma, inline_ram_gamma);
        assert_eq!(
            challenges.instruction_claim_reduction.gamma,
            inline_instruction_gamma
        );
    }

    /// A stage-2 batch output whose reduced instruction openings equal the
    /// product-remainder ones they alias (`lookup_output`,
    /// `left`/`right_instruction_input`). `validate_aliases` accepts it; the tests
    /// below perturb one alias each to assert rejection. The aliased cells carry
    /// the product values; the absorb test overrides them with sentinels to prove
    /// they are skipped.
    fn consistent_values() -> Stage2BatchOutputClaims<Fr> {
        Stage2BatchOutputClaims::<Fr> {
            ram_read_write: RamReadWriteOutputClaims {
                val: fr(1),
                ra: fr(2),
                inc: fr(3),
            },
            product_remainder: ProductRemainderOutputClaims {
                left_instruction_input: fr(4),
                right_instruction_input: fr(5),
                jump_flag: fr(6),
                write_lookup_output_to_rd: fr(7),
                lookup_output: fr(8),
                branch_flag: fr(9),
                next_is_noop: fr(10),
                virtual_instruction: fr(11),
            },
            instruction_claim_reduction: InstructionClaimReductionOutputClaims {
                lookup_output: fr(8),
                left_lookup_operand: fr(12),
                right_lookup_operand: fr(13),
                left_instruction_input: fr(4),
                right_instruction_input: fr(5),
            },
            ram_raf_evaluation: RamRafEvaluationOutputClaims { ram_ra: fr(14) },
            ram_output_check: RamOutputCheckOutputClaims { val_final: fr(15) },
        }
    }

    /// Locks the stage-2 batch Fiat-Shamir append order against silent drift: the
    /// generated absorb follows member declaration order and each member's
    /// `canonical_order`, skipping the reduction's three aliased openings
    /// (absorbed once via their product-remainder source). The aliased cells carry
    /// distinct sentinels here to prove the skip is id-driven, not value-driven.
    #[test]
    fn opening_values_follow_canonical_order() {
        let mut claims = consistent_values();
        claims.instruction_claim_reduction.lookup_output = fr(101);
        claims.instruction_claim_reduction.left_instruction_input = fr(102);
        claims.instruction_claim_reduction.right_instruction_input = fr(103);

        assert_eq!(
            sumchecks().opening_values(&claims),
            (1..=15).map(fr).collect::<Vec<_>>()
        );
    }

    /// The generated `output_claim_count` sums the members' wire sets: 16
    /// expression-referenced openings, minus the reduction's 3 aliases, plus the
    /// product remainder's 2 staged openings.
    #[test]
    fn output_claim_count_matches_absorbed_openings() {
        let sumchecks = sumchecks();
        assert_eq!(sumchecks.output_claim_count(), 15);
        assert_eq!(
            sumchecks.opening_values(&consistent_values()).len(),
            sumchecks.output_claim_count(),
        );
    }

    /// Pins the reduction's alias declarations: each aliased id is distinct and
    /// referenced by the reduction's own output `Expr` (so the batch fold
    /// constrains the wire cell), and each canonical source is absorbed by the
    /// product remainder (so the value the copy is checked against is
    /// Fiat-Shamir-bound). The point-slice identity the value-only check relies
    /// on is pinned by `aliased_members_derive_identical_opening_points`.
    #[test]
    fn alias_declarations_are_valid() {
        use jolt_claims::SymbolicSumcheck as _;
        use std::collections::BTreeSet;

        let sumchecks = sumchecks();
        let expression_openings = sumchecks
            .instruction_claim_reduction
            .symbolic()
            .expected_output_openings::<Fr>();
        let source_wire_openings = sumchecks.product_remainder.wire_output_openings();

        let pairs = InstructionClaimReduction::<Fr>::aliased_output_openings();
        assert_eq!(pairs.len(), 3);
        let mut seen = BTreeSet::new();
        for (aliased, source) in pairs {
            assert!(
                seen.insert(aliased),
                "duplicate aliased opening {aliased:?}"
            );
            assert!(
                expression_openings.contains(&aliased),
                "aliased opening {aliased:?} is not referenced by the reduction's output Expr",
            );
            assert!(
                source_wire_openings.contains(&source),
                "source {source:?} is not absorbed by the product remainder",
            );
        }
    }

    #[test]
    fn validate_aliases_accepts_consistent_reduction() {
        assert!(sumchecks().validate_aliases(&consistent_values()).is_ok());
    }

    #[test]
    fn validate_aliases_rejects_lookup_output_mismatch() {
        let mut values = consistent_values();
        values.instruction_claim_reduction.lookup_output = fr(99);
        assert!(sumchecks().validate_aliases(&values).is_err());
    }

    #[test]
    fn validate_aliases_rejects_left_instruction_input_mismatch() {
        let mut values = consistent_values();
        values.instruction_claim_reduction.left_instruction_input = fr(99);
        assert!(sumchecks().validate_aliases(&values).is_err());
    }

    #[test]
    fn validate_aliases_rejects_right_instruction_input_mismatch() {
        let mut values = consistent_values();
        values.instruction_claim_reduction.right_instruction_input = fr(99);
        assert!(sumchecks().validate_aliases(&values).is_err());
    }

    /// Pins the structural invariant the alias declaration relies on:
    /// `validate_aliases` checks values only, which is sound because the product
    /// remainder and the instruction claim-reduction bind the same batch-point
    /// slice (equal rounds, default offsets) and derive the same opening point —
    /// the aliased pairs are the same polynomial at the same point by
    /// construction, never by proof content.
    #[test]
    fn aliased_members_derive_identical_opening_points() {
        let sumchecks = sumchecks();
        let product = &sumchecks.product_remainder;
        let reduction = &sumchecks.instruction_claim_reduction;
        assert_eq!(product.rounds(), reduction.rounds());
        let batch_num_vars = product.rounds() + 2;
        assert_eq!(
            product.instance_point_offset(batch_num_vars).unwrap(),
            reduction.instance_point_offset(batch_num_vars).unwrap(),
        );

        let point: Vec<Fr> = (0..product.rounds() as u64).map(|i| fr(20 + i)).collect();
        let input_points = sumchecks.empty_input_points();
        let product_points = product
            .derive_opening_points(&point, &input_points.product_remainder)
            .unwrap();
        let reduction_points = reduction
            .derive_opening_points(&point, &input_points.instruction_claim_reduction)
            .unwrap();
        assert_eq!(product_points.lookup_output, reduction_points.lookup_output);
        assert_eq!(
            product_points.left_instruction_input,
            reduction_points.left_instruction_input,
        );
        assert_eq!(
            product_points.right_instruction_input,
            reduction_points.right_instruction_input,
        );
    }
}
