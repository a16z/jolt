//! Typed inputs consumed and outputs produced by stage 7 verification.

use jolt_claims::protocols::jolt::{JoltAdviceKind, JoltCommittedPolynomial};
use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::SumcheckBatch;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::advice_address_phase::AdviceAddressPhase;
use super::committed_reduction_address_phase::{
    BytecodeReductionAddressPhase, ProgramImageReductionAddressPhase,
};
use super::hamming_weight_claim_reduction::HammingWeightClaimReduction;

/// Source-of-truth for stage 7's sumcheck batch: the instances in Fiat-Shamir
/// batch order (hamming-weight reduction, trusted then untrusted advice address
/// phase, then the committed-program-only bytecode and program-image address
/// phases — each address phase present only when its reduction runs one).
/// `#[derive(SumcheckBatch)]` generates the `Stage7InputClaims<F>`,
/// `Stage7InputPoints<F>`, `Stage7OutputClaims<F>`, `Stage7OutputPoints<F>`, and
/// `Stage7Challenges<F>` aggregates — one field per instance, in this declaration
/// order — plus the `Stage7OutputClaims` Fiat-Shamir opening plumbing
/// (`opening_values` / `append_to_transcript`). The field order is load-bearing:
/// it fixes the canonical opening order absorbed into the transcript, which must
/// match the prover's commitment order. The four `Option` members contribute only
/// when their phase ran.
///
/// The trusted / untrusted advice reductions are two batch members (each absorbs
/// its own claimed sum and draws its own batching coefficient), so they are split
/// into two `Option` fields rather than one — matching the stage-6 cycle-phase
/// `trusted_advice` / `untrusted_advice` idiom. Each member's produced claims are
/// the shared `AdviceAddressPhaseOutputClaims` with only its own kind's slot filled.
#[derive(SumcheckBatch)]
#[sumcheck_batch(
    verify_clear,
    verify_zk,
    derive_opening_points,
    expected_final_claim,
    output_shape
)]
pub struct Stage7Sumchecks<F: Field> {
    pub hamming_weight_claim_reduction: HammingWeightClaimReduction<F>,
    /// Final `TrustedAdvice` claim from the trusted advice reduction's address
    /// phase; present only when that phase runs.
    pub trusted_advice: Option<AdviceAddressPhase<F>>,
    /// Final `UntrustedAdvice` claim from the untrusted advice reduction's address
    /// phase; present only when that phase runs.
    pub untrusted_advice: Option<AdviceAddressPhase<F>>,
    /// Final `BytecodeChunk(i)` claims from the committed-bytecode reduction's
    /// address phase; present only when that phase runs.
    pub bytecode_address_phase: Option<BytecodeReductionAddressPhase<F>>,
    /// Final `ProgramImageInit` claim from the program-image reduction's address
    /// phase; present only when that phase runs.
    pub program_image_address_phase: Option<ProgramImageReductionAddressPhase<F>>,
}

/// The shared opening-point accessors over the point-only stage-7 aggregate.
/// Stages 7/8 read each produced opening's point off these cells.
impl<F: Field> Stage7OutputPoints<F> {
    /// The hamming-weight reduction's shared opening point (the own point of the
    /// one-hot `Ra` polynomials): the first non-empty per-family RA cell. `None`
    /// only if the reduction produced no openings (never in practice — at least one
    /// RA family is always present).
    pub fn hamming_weight_opening_point(&self) -> Option<&[F]> {
        self.hamming_weight_claim_reduction
            .instruction_ra
            .first()
            .or_else(|| self.hamming_weight_claim_reduction.bytecode_ra.first())
            .or_else(|| self.hamming_weight_claim_reduction.ram_ra.first())
            .map(Vec::as_slice)
    }

    /// The advice address-phase final opening point for `kind`, present only when
    /// that kind's address phase ran.
    pub fn advice_point(&self, kind: JoltAdviceKind) -> Option<&[F]> {
        match kind {
            JoltAdviceKind::Trusted => self.trusted_advice.as_ref().and_then(|c| c.trusted()),
            JoltAdviceKind::Untrusted => self.untrusted_advice.as_ref().and_then(|c| c.untrusted()),
        }
    }

    /// The committed-bytecode address-phase final opening point (shared by every
    /// chunk), present only when that address phase ran.
    pub fn bytecode_point(&self) -> Option<&[F]> {
        self.bytecode_address_phase
            .as_ref()
            .and_then(|points| points.chunks().first().map(Vec::as_slice))
    }

    /// The program-image address-phase final opening point, present only when that
    /// address phase ran.
    pub fn program_image_point(&self) -> Option<&[F]> {
        self.program_image_address_phase
            .as_ref()
            .map(|points| points.program_image())
    }
}

/// Final opening of a precommitted polynomial, resolved from whichever stage
/// completed its claim reduction (stage 6b cycle phase or stage 7 address
/// phase). Stage 8 consumes these as anchors and batch members of the final
/// PCS opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrecommittedFinalOpening<F: Field> {
    pub polynomial: JoltCommittedPolynomial,
    pub point: Vec<F>,
    /// `None` in ZK mode, where opening claims stay committed.
    pub opening_claim: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ClearOutput<F: Field> {
    /// The produced stage-7 opening *values* (wire form); read by later stages and
    /// the Fiat-Shamir opening-claim encoder.
    pub output_values: Stage7OutputClaims<F>,
    /// The produced stage-7 opening *points*, paired field-for-field with
    /// `output_values`. Later stages read each opening's point off these cells.
    pub output_points: Stage7OutputPoints<F>,
    /// The hamming-weight reduction's opening point — the own point of the one-hot
    /// `Ra` polynomials, shared by all reduced RA openings. Stored contiguously so
    /// stage 8 can borrow it directly (the per-family RA opening cells can be empty
    /// for a missing family, so it cannot always be read off a cell).
    pub hamming_weight_opening_point: Vec<F>,
    pub precommitted_final_openings: Vec<PrecommittedFinalOpening<F>>,
}

/// ZK counterpart of [`Stage7ClearOutput`]. The produced opening *values* stay
/// committed (in `batch_output_claims`); BlindFold recomputes every per-relation
/// sumcheck point and public it needs from `batch_consistency`, so only the data
/// stage 8 consumes is carried in the clear: the shared hamming-weight opening
/// point and the precommitted final openings (point-only, claims committed).
///
/// The path-agnostically drawn stage-7 challenges are carried so BlindFold can
/// source the hamming-weight batching gamma from
/// `challenges.hamming_weight_claim_reduction.gamma`, matching the
/// `input.stageN.challenges.<relation>.<field>` idiom used by stages 3–5.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ZkOutput<F: Field, C> {
    pub challenges: Stage7Challenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    /// The produced opening points, the ZK counterpart of the clear path's
    /// `Stage7ClearOutput::output_points`. Computed as a byproduct of the unified
    /// `derive_opening_points`; the stored `hamming_weight_opening_point` and the
    /// precommitted final openings are read off it (the per-family RA cells can be
    /// empty, so the shared opening point is also stored contiguously below).
    pub output_points: Stage7OutputPoints<F>,
    pub hamming_weight_opening_point: Vec<F>,
    pub precommitted_final_openings: Vec<PrecommittedFinalOpening<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage7Output<F: Field, C> {
    Clear(Stage7ClearOutput<F>),
    Zk(Stage7ZkOutput<F, C>),
}

impl<F: Field, C> Stage7Output<F, C> {
    pub fn clear(&self) -> Result<&Stage7ClearOutput<F>, crate::VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(crate::VerifierError::ExpectedClearProof { field: "stage7" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage7ZkOutput<F, C>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage7" }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_claims::protocols::jolt::relations::claim_reductions::advice::AdviceAddressPhaseOutputClaims;
    use jolt_claims::protocols::jolt::relations::claim_reductions::bytecode::BytecodeReductionAddressPhaseOutputClaims;
    use jolt_claims::protocols::jolt::relations::claim_reductions::hamming_weight::HammingWeightClaimReductionOutputClaims;
    use jolt_claims::protocols::jolt::relations::claim_reductions::program_image::ProgramImageReductionAddressPhaseOutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// Locks the stage-7 Fiat-Shamir append order against silent drift: the
    /// hamming-weight reduced openings, then the trusted then untrusted advice
    /// address-phase openings, then (when present) the committed-bytecode and
    /// program-image address-phase openings, each member single-sourcing its own
    /// per-field order from its `OutputClaims` derive. A wrong batch order here
    /// silently breaks soundness, so it is pinned with distinct sentinels; the
    /// absent `Option` members drop out of the stream entirely, and each advice
    /// member carries only its own kind's slot.
    #[test]
    fn opening_values_follow_canonical_order() {
        let hamming = HammingWeightClaimReductionOutputClaims {
            instruction_ra: vec![fr(1), fr(2)],
            bytecode_ra: vec![fr(3)],
            ram_ra: vec![fr(4)],
        };
        let trusted_advice = AdviceAddressPhaseOutputClaims {
            trusted: Some(fr(5)),
            untrusted: None,
        };
        let untrusted_advice = AdviceAddressPhaseOutputClaims {
            trusted: None,
            untrusted: Some(fr(6)),
        };

        let without_committed = Stage7OutputClaims::<Fr> {
            hamming_weight_claim_reduction: hamming.clone(),
            trusted_advice: Some(trusted_advice.clone()),
            untrusted_advice: Some(untrusted_advice.clone()),
            bytecode_address_phase: None,
            program_image_address_phase: None,
        };
        assert_eq!(
            without_committed.opening_values(),
            (1..=6).map(fr).collect::<Vec<_>>()
        );

        let with_committed = Stage7OutputClaims::<Fr> {
            hamming_weight_claim_reduction: hamming,
            trusted_advice: Some(trusted_advice),
            untrusted_advice: Some(untrusted_advice),
            bytecode_address_phase: Some(BytecodeReductionAddressPhaseOutputClaims {
                chunks: vec![fr(7), fr(8)],
            }),
            program_image_address_phase: Some(ProgramImageReductionAddressPhaseOutputClaims {
                program_image: fr(9),
            }),
        };
        assert_eq!(
            with_committed.opening_values(),
            (1..=9).map(fr).collect::<Vec<_>>()
        );
    }

    /// Locks the `output_shape` commitment count against Expr/geometry drift: the
    /// per-member `expected_output_openings` counts (which the generated
    /// `output_claim_count` sums) must match the hand-derived opening counts each
    /// configuration commits. Guards the ZK commitment count switching from the
    /// old hand count to the Expr-derived sums.
    #[test]
    #[expect(clippy::unwrap_used)]
    fn output_shape_member_counts_match_hand_derived_openings() {
        use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::{
            claim_reduction_output_openings, HammingWeightClaimReductionDimensions,
        };
        use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
        use jolt_claims::protocols::jolt::relations::claim_reductions;
        use jolt_claims::protocols::jolt::{JoltAdviceKind, PrecommittedReductionDimensions};
        use jolt_claims::SymbolicSumcheck;

        let ra_layout = JoltRaPolynomialLayout::new(2, 1, 1).unwrap();
        let hamming_dimensions = HammingWeightClaimReductionDimensions::new(ra_layout, 4);
        let hamming = claim_reductions::hamming_weight::ClaimReduction::new(hamming_dimensions);
        assert_eq!(
            hamming.expected_output_openings::<Fr>().len(),
            claim_reduction_output_openings(hamming_dimensions)
                .all()
                .len(),
        );

        let advice_dimensions = PrecommittedReductionDimensions::new(4, 3, true);
        for kind in [JoltAdviceKind::Trusted, JoltAdviceKind::Untrusted] {
            let advice = claim_reductions::advice::AddressPhase::new((kind, advice_dimensions));
            assert_eq!(advice.expected_output_openings::<Fr>().len(), 1);
        }

        let chunk_count = 4;
        let bytecode =
            claim_reductions::bytecode::AddressPhase::new((advice_dimensions, chunk_count));
        assert_eq!(bytecode.expected_output_openings::<Fr>().len(), chunk_count);

        let program_image = claim_reductions::program_image::AddressPhase::new(advice_dimensions);
        assert_eq!(program_image.expected_output_openings::<Fr>().len(), 1);
    }
}
