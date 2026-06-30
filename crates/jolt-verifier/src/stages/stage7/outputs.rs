//! Typed inputs consumed and outputs produced by stage 7 verification.

use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::{OpeningClaim, SumcheckBatch};
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::advice_address_phase::AdviceAddressPhase;
use super::committed_reduction_address_phase::{
    BytecodeReductionAddressPhase, ProgramImageReductionAddressPhase,
};
use super::hamming_weight_claim_reduction::HammingWeightClaimReduction;

/// Source-of-truth for stage 7's sumcheck batch: the instances in Fiat-Shamir
/// batch order (hamming-weight reduction, advice address phase, then the
/// committed-program-only bytecode and program-image address phases — each
/// present only when its phase runs). `#[derive(SumcheckBatch)]` generates the
/// `Stage7InputClaims<F, C>`, `Stage7OutputClaims<F, C>`, and `Stage7Challenges<F>`
/// aggregates — one field per instance, in this declaration order — plus the
/// `Stage7OutputClaims` Fiat-Shamir opening plumbing (`opening_values` /
/// `append_to_transcript`). The field order is load-bearing: it fixes the canonical
/// opening order absorbed into the transcript, which must match the prover's
/// commitment order. The two `Option` members contribute only when their phase ran.
#[derive(SumcheckBatch)]
pub struct Stage7Sumchecks<F: Field> {
    pub hamming_weight_claim_reduction: HammingWeightClaimReduction<F>,
    pub advice_address_phase: AdviceAddressPhase<F>,
    /// Final `BytecodeChunk(i)` claims from the committed-bytecode reduction's
    /// address phase; present only when that phase runs.
    pub bytecode_address_phase: Option<BytecodeReductionAddressPhase<F>>,
    /// Final `ProgramImageInit` claim from the program-image reduction's address
    /// phase; present only when that phase runs.
    pub program_image_address_phase: Option<ProgramImageReductionAddressPhase<F>>,
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
    /// The produced stage-7 openings paired with their points (point + value) via
    /// the `OpeningClaim` cell.
    pub output_claims: Stage7OutputClaims<F, OpeningClaim<F>>,
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
    /// hamming-weight reduced openings, then the advice address-phase openings,
    /// then (when present) the committed-bytecode and program-image address-phase
    /// openings, each member single-sourcing its own per-field order from its
    /// `OutputClaims` derive. A wrong batch order here silently breaks soundness,
    /// so it is pinned with distinct sentinels; the absent `Option` members drop
    /// out of the stream entirely.
    #[test]
    fn opening_values_follow_canonical_order() {
        let hamming = HammingWeightClaimReductionOutputClaims {
            instruction_ra: vec![fr(1), fr(2)],
            bytecode_ra: vec![fr(3)],
            ram_ra: vec![fr(4)],
        };
        let advice = AdviceAddressPhaseOutputClaims {
            trusted: Some(fr(5)),
            untrusted: Some(fr(6)),
        };

        let without_committed = Stage7OutputClaims::<Fr, Fr> {
            hamming_weight_claim_reduction: hamming.clone(),
            advice_address_phase: advice.clone(),
            bytecode_address_phase: None,
            program_image_address_phase: None,
        };
        assert_eq!(
            without_committed.opening_values(),
            (1..=6).map(fr).collect::<Vec<_>>()
        );

        let with_committed = Stage7OutputClaims::<Fr, Fr> {
            hamming_weight_claim_reduction: hamming,
            advice_address_phase: advice,
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
}
