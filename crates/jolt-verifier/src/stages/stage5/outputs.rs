//! Typed inputs consumed and outputs produced by stage 5 verification.

use jolt_field::JoltField;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::SumcheckBatch;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

#[cfg(feature = "field-inline")]
pub use super::field_registers_val_evaluation::{
    FieldRegistersValEvaluation, FieldRegistersValEvaluationOutputClaims,
};
use super::instruction_read_raf::{
    reconstruct_r_address, InstructionReadRaf, InstructionReadRafOutputClaims,
};
use super::ram_ra_claim_reduction::{RamRaClaimReduction, RamRaClaimReductionOutputClaims};
use super::registers_val_evaluation::{RegistersValEvaluation, RegistersValEvaluationOutputClaims};

/// Source-of-truth for stage 5's sumcheck batch, in Fiat-Shamir batch order
/// (instruction read-RAF, RAM-RA reduction, register value-evaluation, the
/// field-inline FR value-evaluation when composed). `#[derive(SumcheckBatch)]`
/// generates the `Stage5{Input,Output}{Claims,Points}<F>` and
/// `Stage5Challenges<F>` aggregates — one field per instance, in this
/// declaration order — plus the Fiat-Shamir absorb plumbing (`opening_values` /
/// `append_output_claims` on this struct). The field order is load-bearing: it
/// fixes the canonical opening order absorbed into the transcript, which must
/// match the prover's commitment order.
#[derive(SumcheckBatch)]
#[sumcheck_batch(crate = "crate")]
pub struct Stage5Sumchecks<F: JoltField> {
    pub instruction_read_raf: InstructionReadRaf<F>,
    pub ram_ra_claim_reduction: RamRaClaimReduction<F>,
    pub registers_val_evaluation: RegistersValEvaluation<F>,
    /// The FR Twist val-evaluation instance. Declaration position (last) is
    /// the spec's stage-5 batch order (`specs/field-inline-protocol.md`,
    /// "Stage 5 Composition"): its two openings absorb after the ordinary
    /// register value-evaluation ones, and it draws no instance challenge
    /// (`NoChallenges`), so the stage's gamma draw order is unchanged.
    #[cfg(feature = "field-inline")]
    pub field_registers_val_evaluation: FieldRegistersValEvaluation<F>,
}

impl<F: JoltField> Stage5OutputClaims<F> {
    /// Construct the ordinary stage-5 claims. Producers without field-inline
    /// semantics use this regardless of the build's feature set — the FR
    /// val-evaluation slot defaults to all-zero claims, inert because such
    /// producers' proofs never declare the FR axis.
    pub fn new(
        instruction_read_raf: InstructionReadRafOutputClaims<F>,
        ram_ra_claim_reduction: RamRaClaimReductionOutputClaims<F>,
        registers_val_evaluation: RegistersValEvaluationOutputClaims<F>,
    ) -> Self {
        Self {
            instruction_read_raf,
            ram_ra_claim_reduction,
            registers_val_evaluation,
            #[cfg(feature = "field-inline")]
            field_registers_val_evaluation: Default::default(),
        }
    }
}

/// The shared opening-point accessors over the point-only stage-5 aggregate.
impl<F: JoltField> Stage5OutputPoints<F> {
    /// The instruction read-RAF cycle point (shared by the lookup-table-flag
    /// and RAF-flag openings).
    pub fn instruction_r_cycle(&self) -> &[F] {
        self.instruction_read_raf.instruction_raf_flag()
    }

    /// The contiguous instruction address point, reconstructed from the
    /// virtual-RA opening points (each is `chunk ++ r_cycle`).
    pub fn instruction_r_address(&self) -> Vec<F> {
        reconstruct_r_address(&self.instruction_read_raf, self.instruction_r_cycle().len())
    }

    /// The reduced RAM-RA opening point (`address ++ cycle`).
    pub fn ram_reduced_opening_point(&self) -> &[F] {
        self.ram_ra_claim_reduction.ram_ra()
    }

    /// The register value-evaluation opening point (shared by `rd_inc`/`rd_wa`).
    pub fn registers_opening_point(&self) -> &[F] {
        self.registers_val_evaluation.rd_inc()
    }

    /// The FR val-evaluation opening point (shared by the FR `rd_inc`/`rd_wa`).
    #[cfg(feature = "field-inline")]
    pub fn field_registers_val_evaluation_point(&self) -> &[F] {
        self.field_registers_val_evaluation.rd_inc()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct Stage5ClearOutput<F: JoltField> {
    pub challenges: Stage5Challenges<F>,
    /// The produced stage-5 opening *values* (wire form); read by later stages and
    /// the Fiat-Shamir opening-claim encoder.
    pub output_values: Stage5OutputClaims<F>,
    /// The produced stage-5 opening *points*, paired field-for-field with
    /// `output_values`. Later stages read each opening's point off these cells.
    pub output_points: Stage5OutputPoints<F>,
    /// The instruction read-RAF address point, materialized contiguously from the
    /// virtual-RA opening points (which tile it as `chunk ++ r_cycle`). Stored
    /// because stage 6 re-chunks it by the committed-chunk width — a different
    /// split than the virtual-RA cells carry — so it needs a contiguous copy that
    /// downstream code can borrow.
    pub instruction_r_address: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5ZkOutput<F: JoltField, C> {
    pub challenges: Stage5Challenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    /// The produced opening points, the ZK counterpart of the clear path's
    /// `output_points`. Read through the same `*_point()` accessors.
    pub output_points: Stage5OutputPoints<F>,
    /// The contiguous instruction address point, stored (rather than reconstructed
    /// from `output_points` on demand) so stage 6 can borrow it — the per-chunk
    /// virtual-RA cells don't hold it contiguously. Mirrors `Stage5ClearOutput`.
    pub instruction_r_address: Vec<F>,
}

// The clear variant carries the located opening claims (point + value) that
// later stages read on the hot path; the ZK variant carries the committed
// consistency and output-claim commitments. Boxing the common clear variant to
// shrink the rarer ZK one would add indirection to every clear-path access.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage5Output<F: JoltField, C> {
    Clear(Stage5ClearOutput<F>),
    Zk(Stage5ZkOutput<F, C>),
}

impl<F: JoltField, C> Stage5Output<F, C> {
    /// The produced opening points, available regardless of proving mode.
    pub fn output_points(&self) -> &Stage5OutputPoints<F> {
        match self {
            Self::Clear(output) => &output.output_points,
            Self::Zk(output) => &output.output_points,
        }
    }

    /// The contiguous stage-5 instruction address point, stored on both output
    /// variants because the per-chunk virtual-RA cells don't hold it contiguously.
    pub fn instruction_r_address(&self) -> &[F] {
        match self {
            Self::Clear(output) => &output.instruction_r_address,
            Self::Zk(output) => &output.instruction_r_address,
        }
    }

    pub fn clear(&self) -> Result<&Stage5ClearOutput<F>, crate::VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(crate::VerifierError::ExpectedClearProof { field: "stage5" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage5ZkOutput<F, C>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage5" }),
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stages::relations::draw_recording::{record, DrawEvent};
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::geometry::instruction::InstructionReadRafDimensions;
    use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafOutputClaims;
    use jolt_claims::protocols::jolt::relations::ram::RamRaClaimReductionOutputClaims;
    use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationOutputClaims;
    use jolt_field::{Fr, Ring};
    use jolt_transcript::Transcript;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn sumchecks() -> Stage5Sumchecks<Fr> {
        let trace_dimensions = TraceDimensions::new(4);
        Stage5Sumchecks::<Fr> {
            instruction_read_raf: InstructionReadRaf::new(
                InstructionReadRafDimensions::try_from((5, 128, 3)).unwrap(),
            ),
            ram_ra_claim_reduction: RamRaClaimReduction::new(trace_dimensions, 3),
            registers_val_evaluation: RegistersValEvaluation::new(trace_dimensions),
            #[cfg(feature = "field-inline")]
            field_registers_val_evaluation: FieldRegistersValEvaluation::new(
                jolt_claims::protocols::field_inline::FieldRegistersTraceDimensions::new(4),
            ),
        }
    }

    fn claims() -> Stage5OutputClaims<Fr> {
        Stage5OutputClaims::<Fr> {
            instruction_read_raf: InstructionReadRafOutputClaims {
                lookup_table_flags: vec![fr(1), fr(2)],
                instruction_ra: vec![fr(3), fr(4)],
                instruction_raf_flag: fr(5),
            },
            ram_ra_claim_reduction: RamRaClaimReductionOutputClaims { ram_ra: fr(6) },
            registers_val_evaluation: RegistersValEvaluationOutputClaims {
                rd_inc: fr(7),
                rd_wa: fr(8),
            },
            #[cfg(feature = "field-inline")]
            field_registers_val_evaluation: FieldRegistersValEvaluationOutputClaims {
                rd_inc: fr(9),
                rd_wa: fr(10),
            },
        }
    }

    /// Locks the stage-5 Fiat-Shamir append order against silent drift: the
    /// instruction read-RAF openings, then the RAM-RA reduced opening, then the
    /// register value-evaluation openings, under `field-inline` the FR
    /// value-evaluation openings last (the spec's committed row order:
    /// `FieldRdInc`, `FieldRdWa`), each member single-sourcing its own
    /// per-field order from its `OutputClaims` derive. A wrong batch order here
    /// silently breaks soundness, so it is pinned with distinct sentinels.
    #[test]
    fn opening_values_follow_canonical_order() {
        #[cfg(not(feature = "field-inline"))]
        let expected = (1..=8).map(fr).collect::<Vec<_>>();
        #[cfg(feature = "field-inline")]
        let expected = (1..=10).map(fr).collect::<Vec<_>>();
        assert_eq!(sumchecks().opening_values(&claims()), expected);
    }

    /// Pins the batch's `draw_challenges` to the inline draw: the instruction
    /// gamma, then the RAM-RA gamma. The register value-evaluation member draws
    /// nothing, and so does the `field-inline` FR value-evaluation member
    /// (`NoChallenges`) — composing it changes no stage-5 draw.
    #[test]
    fn draw_challenges_matches_inline_draw_sequence() {
        let sumchecks = sumchecks();
        let (inline_events, inline_gammas) =
            record(|t| (0..2).map(|_| t.challenge_scalar()).collect::<Vec<Fr>>());
        let (draw_events, challenges) = record(|t| sumchecks.draw_challenges(t).unwrap());

        assert_eq!(draw_events, inline_events);
        assert_eq!(
            draw_events,
            vec![DrawEvent::Squeeze(1), DrawEvent::Squeeze(2)]
        );
        assert_eq!(
            vec![
                challenges.instruction_read_raf.gamma,
                challenges.ram_ra_claim_reduction.gamma,
            ],
            inline_gammas
        );
    }

    /// The FR val-evaluation member's wire set is exactly the two spec outputs
    /// (`FieldRdInc`, `FieldRdWa` at `FieldRegistersValEvaluation`), so
    /// composing it grows the stage-5 absorbed/committed opening count by two.
    #[cfg(feature = "field-inline")]
    #[test]
    fn field_registers_val_evaluation_wire_set_is_the_two_spec_outputs() {
        use crate::stages::relations::ConcreteSumcheck as _;
        use jolt_claims::protocols::field_inline::geometry::registers::val_evaluation_output_openings;

        let sumchecks = sumchecks();
        let wire = sumchecks
            .field_registers_val_evaluation
            .wire_output_openings();
        assert_eq!(wire, val_evaluation_output_openings().into_iter().collect());

        let others = sumchecks.instruction_read_raf.wire_output_openings().len()
            + sumchecks
                .ram_ra_claim_reduction
                .wire_output_openings()
                .len()
            + sumchecks
                .registers_val_evaluation
                .wire_output_openings()
                .len();
        assert_eq!(sumchecks.output_claim_count(), others + 2);
    }
}
