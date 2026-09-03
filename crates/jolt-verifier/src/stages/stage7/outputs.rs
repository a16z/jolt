//! Typed inputs consumed and outputs produced by stage 7 verification.

#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::JoltAdviceKind;
use jolt_field::JoltField;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::SumcheckBatch;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

#[cfg(not(feature = "akita"))]
use super::advice_address_phase::{TrustedAdviceAddressPhase, UntrustedAdviceAddressPhase};
use super::committed_reduction_address_phase::{
    BytecodeReductionAddressPhase, ProgramImageReductionAddressPhase,
};
use super::hamming_weight_claim_reduction::HammingWeightClaimReduction;

/// Source-of-truth for stage 7's sumcheck batch: the instances in Fiat-Shamir
/// batch order (hamming-weight reduction, then the committed-program-only
/// bytecode and program-image address phases on Akita; Dory additionally runs
/// trusted then untrusted advice address phases before committed program — each
/// address phase present only when its reduction runs one).
/// `#[derive(SumcheckBatch)]` generates the `Stage7InputClaims<F>`,
/// `Stage7InputPoints<F>`, `Stage7OutputClaims<F>`, `Stage7OutputPoints<F>`, and
/// `Stage7Challenges<F>` aggregates — one field per instance, in this declaration
/// order — plus the Fiat-Shamir absorb plumbing (`opening_values` /
/// `append_output_claims` on this struct). The field order is load-bearing: it
/// fixes the canonical opening order absorbed into the transcript, which must
/// match the prover's commitment order. The optional members contribute only
/// when their phase ran.
///
/// The trusted / untrusted advice reductions are two batch members (each absorbs
/// its own claimed sum and draws its own batching coefficient), so they are split
/// into two `Option` fields rather than one — matching the stage-6 cycle-phase
/// `trusted_advice` / `untrusted_advice` idiom. Each member is its own per-kind
/// relation type whose produced claims carry a single non-`Option` slot.
#[derive(SumcheckBatch)]
#[sumcheck_batch(crate = "crate")]
pub struct Stage7Sumchecks<F: JoltField> {
    pub hamming_weight_claim_reduction: HammingWeightClaimReduction<F>,
    /// Final `TrustedAdvice` claim from the trusted advice reduction's address
    /// phase; present only when that phase runs. On the prove side the kernel
    /// is the stage-6b cycle-phase object, reclaimed from its `ProofSession`
    /// carry and phase-transitioned inside `prepare`.
    #[cfg(not(feature = "akita"))]
    pub trusted_advice: Option<TrustedAdviceAddressPhase<F>>,
    /// Final `UntrustedAdvice` claim from the untrusted advice reduction's address
    /// phase; present only when that phase runs.
    #[cfg(not(feature = "akita"))]
    pub untrusted_advice: Option<UntrustedAdviceAddressPhase<F>>,
    /// Final `BytecodeChunk(i)` claims from the committed-bytecode reduction's
    /// address phase; present only when that phase runs.
    pub bytecode_address_phase: Option<BytecodeReductionAddressPhase<F>>,
    /// Final `ProgramImageInit` claim from the program-image reduction's address
    /// phase; present only when that phase runs.
    pub program_image_address_phase: Option<ProgramImageReductionAddressPhase<F>>,
}

/// The shared opening-point accessors over the point-only stage-7 aggregate.
/// Stages 7/8 read each produced opening's point off these cells.
impl<F: JoltField> Stage7OutputPoints<F> {
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
    #[cfg(not(feature = "akita"))]
    pub fn advice_point(&self, kind: JoltAdviceKind) -> Option<&[F]> {
        match kind {
            JoltAdviceKind::Trusted => self.trusted_advice.as_ref().map(|c| c.trusted()),
            JoltAdviceKind::Untrusted => self.untrusted_advice.as_ref().map(|c| c.untrusted()),
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

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct Stage7ClearOutput<F: JoltField> {
    /// The produced stage-7 opening *values* (wire form); read by later stages and
    /// the Fiat-Shamir opening-claim encoder.
    pub output_values: Stage7OutputClaims<F>,
    /// The produced stage-7 opening *points*, paired field-for-field with
    /// `output_values`. Stage 8 reads each opening's point off these cells, and
    /// derives the hamming-weight opening point and the precommitted final openings
    /// from them.
    pub output_points: Stage7OutputPoints<F>,
}

/// ZK counterpart of [`Stage7ClearOutput`]. The produced opening *values* stay
/// committed (in `batch_output_claims`); BlindFold recomputes every per-relation
/// sumcheck point and public it needs from `batch_consistency`, so the only data
/// stage 8 consumes in the clear is the produced opening points, off which it reads
/// the hamming-weight opening point and resolves the precommitted final openings.
///
/// The path-agnostically drawn stage-7 challenges are carried so BlindFold can
/// source the hamming-weight batching gamma from
/// `challenges.hamming_weight_claim_reduction.gamma`, matching the
/// `input.stageN.challenges.<relation>.<field>` idiom used by stages 3–5.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ZkOutput<F: JoltField, C> {
    pub challenges: Stage7Challenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    /// The produced opening points, the ZK counterpart of the clear path's
    /// `Stage7ClearOutput::output_points`, computed as a byproduct of the unified
    /// `derive_opening_points`. Stage 8 reads the hamming-weight opening point and
    /// resolves the precommitted final openings off these cells.
    pub output_points: Stage7OutputPoints<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage7Output<F: JoltField, C> {
    Clear(Stage7ClearOutput<F>),
    Zk(Stage7ZkOutput<F, C>),
}

impl<F: JoltField, C> Stage7Output<F, C> {
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
    #[cfg(not(feature = "akita"))]
    use super::*;
    #[cfg(not(feature = "akita"))]
    use crate::stages::stage7::hamming_weight_claim_reduction::{
        hamming_weight_claim_reduction_dimensions, HammingWeightClaimReductionOutputClaims,
    };
    #[cfg(not(feature = "akita"))]
    use jolt_claims::protocols::jolt::relations::claim_reductions::advice::{
        TrustedAdviceAddressPhaseOutputClaims, UntrustedAdviceAddressPhaseOutputClaims,
    };
    #[cfg(not(feature = "akita"))]
    use jolt_claims::protocols::jolt::relations::claim_reductions::bytecode::BytecodeReductionAddressPhaseOutputClaims;
    #[cfg(not(feature = "akita"))]
    use jolt_claims::protocols::jolt::relations::claim_reductions::program_image::ProgramImageReductionAddressPhaseOutputClaims;
    use jolt_field::Fr;
    #[cfg(not(feature = "akita"))]
    use jolt_field::Ring;

    #[cfg(not(feature = "akita"))]
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
    #[cfg(not(feature = "akita"))]
    #[test]
    #[expect(clippy::unwrap_used)]
    fn opening_values_follow_canonical_order() {
        use crate::stages::{CommittedProgramSchedule, PrecommittedSchedule};
        use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
        use jolt_claims::protocols::jolt::TracePolynomialOrder;

        let schedule = PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            4,
            2,
            Some(64),
            Some(64),
            Some(CommittedProgramSchedule {
                bytecode_len: 8,
                bytecode_chunk_count: 2,
                program_image_len_words: 8,
                program_image_start_index: 0,
            }),
        )
        .unwrap();
        let hamming_instance = || {
            let dimensions = hamming_weight_claim_reduction_dimensions(
                JoltRaPolynomialLayout::new(2, 1, 1).unwrap(),
                4,
            )
            .unwrap();
            HammingWeightClaimReduction::new(dimensions, Vec::new(), Vec::new(), Vec::new())
        };
        let trusted_instance = || {
            TrustedAdviceAddressPhase::new(
                schedule.trusted_advice.as_ref().unwrap(),
                None,
                Vec::new(),
            )
        };
        let untrusted_instance = || {
            UntrustedAdviceAddressPhase::new(
                schedule.untrusted_advice.as_ref().unwrap(),
                None,
                Vec::new(),
            )
        };

        // Sentinels are sequential in canonical append order.
        let (trusted, untrusted, chunk1, chunk2, image, plain_last, committed_last) =
            (5, 6, 7, 8, 9, 6, 9);
        let hamming = HammingWeightClaimReductionOutputClaims {
            instruction_ra: vec![fr(1), fr(2)],
            bytecode_ra: vec![fr(3)],
            ram_ra: vec![fr(4)],
        };
        let trusted_advice = TrustedAdviceAddressPhaseOutputClaims {
            trusted: fr(trusted),
        };
        let untrusted_advice = UntrustedAdviceAddressPhaseOutputClaims {
            untrusted: fr(untrusted),
        };

        let without_committed_sumchecks = Stage7Sumchecks::<Fr> {
            hamming_weight_claim_reduction: hamming_instance(),
            trusted_advice: Some(trusted_instance()),
            untrusted_advice: Some(untrusted_instance()),
            bytecode_address_phase: None,
            program_image_address_phase: None,
        };
        let without_committed = Stage7OutputClaims::<Fr> {
            hamming_weight_claim_reduction: hamming.clone(),
            trusted_advice: Some(trusted_advice.clone()),
            untrusted_advice: Some(untrusted_advice.clone()),
            bytecode_address_phase: None,
            program_image_address_phase: None,
        };
        assert_eq!(
            without_committed_sumchecks.opening_values(&without_committed),
            (1..=plain_last).map(fr).collect::<Vec<_>>()
        );

        let with_committed_sumchecks = Stage7Sumchecks::<Fr> {
            hamming_weight_claim_reduction: hamming_instance(),
            trusted_advice: Some(trusted_instance()),
            untrusted_advice: Some(untrusted_instance()),
            bytecode_address_phase: Some(BytecodeReductionAddressPhase::new(
                schedule.bytecode.as_ref().unwrap(),
                None,
                Vec::new(),
            )),
            program_image_address_phase: Some(ProgramImageReductionAddressPhase::new(
                schedule.program_image.as_ref().unwrap(),
                None,
                Vec::new(),
            )),
        };
        let with_committed = Stage7OutputClaims::<Fr> {
            hamming_weight_claim_reduction: hamming,
            trusted_advice: Some(trusted_advice),
            untrusted_advice: Some(untrusted_advice),
            bytecode_address_phase: Some(BytecodeReductionAddressPhaseOutputClaims {
                chunks: vec![fr(chunk1), fr(chunk2)],
            }),
            program_image_address_phase: Some(ProgramImageReductionAddressPhaseOutputClaims {
                program_image: fr(image),
            }),
        };
        assert_eq!(
            with_committed_sumchecks.opening_values(&with_committed),
            (1..=committed_last).map(fr).collect::<Vec<_>>()
        );
    }

    /// Locks the `output_shape` commitment count against Expr/geometry drift: the
    /// per-member `expected_output_openings` counts (which the generated
    /// `output_claim_count` sums) must match the hand-derived opening counts each
    /// configuration commits. Guards the ZK commitment count switching from the
    /// old hand count to the Expr-derived sums.
    #[test]
    #[expect(clippy::unwrap_used)]
    fn output_shape_column_counts_match_hand_derived_openings() {
        use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::{
            claim_reduction_output_openings, HammingWeightClaimReductionDimensions,
        };
        use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
        use jolt_claims::protocols::jolt::relations::claim_reductions;
        use jolt_claims::protocols::jolt::PrecommittedReductionDimensions;
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

        let reduction_dimensions = PrecommittedReductionDimensions::new(4, 3, true);
        #[cfg(not(feature = "akita"))]
        {
            let trusted_advice =
                claim_reductions::advice::TrustedAddressPhase::new(reduction_dimensions);
            assert_eq!(trusted_advice.expected_output_openings::<Fr>().len(), 1);
            let untrusted_advice =
                claim_reductions::advice::UntrustedAddressPhase::new(reduction_dimensions);
            assert_eq!(untrusted_advice.expected_output_openings::<Fr>().len(), 1);
        }

        let chunk_count = 4;
        let bytecode =
            claim_reductions::bytecode::AddressPhase::new((reduction_dimensions, chunk_count));
        assert_eq!(bytecode.expected_output_openings::<Fr>().len(), chunk_count);

        let program_image =
            claim_reductions::program_image::AddressPhase::new(reduction_dimensions);
        assert_eq!(program_image.expected_output_openings::<Fr>().len(), 1);
    }
}
