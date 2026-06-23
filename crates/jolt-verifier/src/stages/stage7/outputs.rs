//! Typed inputs consumed and outputs produced by stage 7 verification.

use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::stages::relations::{OpeningClaim, OutputClaims};
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::advice_address_phase::AdviceAddressPhaseOutputClaims;
use super::committed_reduction_address_phase::{
    BytecodeReductionAddressPhaseOutputClaims, ProgramImageReductionAddressPhaseOutputClaims,
};
use super::hamming_weight_claim_reduction::HammingWeightClaimReductionOutputClaims;

/// The stage 7 produced opening claims, declared in canonical (Fiat-Shamir)
/// order: the hamming-weight reduced RA openings (instruction, bytecode, RAM),
/// the trusted/untrusted advice address-phase openings, the committed bytecode
/// chunk openings, then the program-image opening — each present only when its
/// phase ran. [`opening_values`](Self::opening_values) and
/// [`append_to_transcript`](Self::append_to_transcript) single-source the append
/// order from the per-relation declaration orders. Generic over the cell (`F` on
/// the wire).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct Stage7OutputClaims<C> {
    pub hamming_weight_claim_reduction: HammingWeightClaimReductionOutputClaims<C>,
    pub advice_address_phase: AdviceAddressPhaseOutputClaims<C>,
    /// Final `BytecodeChunk(i)` claims from the committed-bytecode reduction's
    /// address phase; present only when that phase runs.
    pub bytecode_address_phase: Option<BytecodeReductionAddressPhaseOutputClaims<C>>,
    /// Final `ProgramImageInit` claim from the program-image reduction's address
    /// phase; present only when that phase runs.
    pub program_image_address_phase: Option<ProgramImageReductionAddressPhaseOutputClaims<C>>,
}

impl<F: Field> Stage7OutputClaims<F> {
    /// The produced opening claims in canonical (Fiat-Shamir) order, single-sourced
    /// from the per-relation declaration orders.
    pub fn opening_values(&self) -> Vec<F> {
        let mut values = self.hamming_weight_claim_reduction.opening_values();
        values.extend(self.advice_address_phase.opening_values());
        if let Some(claims) = &self.bytecode_address_phase {
            values.extend(claims.opening_values());
        }
        if let Some(claims) = &self.program_image_address_phase {
            values.extend(claims.opening_values());
        }
        values
    }

    /// Append every produced opening to the transcript in canonical order, each
    /// under the `b"opening_claim"` label, matching the prover's commitment order.
    pub fn append_to_transcript<T: Transcript<Challenge = F>>(&self, transcript: &mut T) {
        for value in self.opening_values() {
            transcript.append_labeled(b"opening_claim", &value);
        }
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
    /// The produced stage-7 openings paired with their points (point + value) via
    /// the `OpeningClaim` cell.
    pub output_claims: Stage7OutputClaims<OpeningClaim<F>>,
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ZkOutput<F: Field, C> {
    pub public: Stage7PublicOutput<F>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7PublicOutput<F: Field> {
    pub hamming_gamma: F,
}
