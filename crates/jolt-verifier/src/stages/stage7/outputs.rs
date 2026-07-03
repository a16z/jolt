//! Typed inputs consumed and outputs produced by stage 7 verification.

use jolt_claims::protocols::jolt::{BaseJolt, JoltCommitmentMode, JoltCommittedPolynomial};
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
    serialize = "C: serde::Serialize, M::ChunkReconstructionOutputs<C>: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>, M::ChunkReconstructionOutputs<C>: serde::Deserialize<'de>"
))]
pub struct Stage7OutputClaims<C, M = BaseJolt>
where
    C: Clone + core::fmt::Debug + PartialEq + Eq + Send + Sync,
    M: JoltCommitmentMode,
{
    pub hamming_weight_claim_reduction: HammingWeightClaimReductionOutputClaims<C>,
    /// Unsigned-inc chunk claims from the lattice reconstruction; empty in
    /// the homomorphic mode.
    pub chunk_reconstruction: M::ChunkReconstructionOutputs<C>,
    pub advice_address_phase: AdviceAddressPhaseOutputClaims<C>,
    /// Final `BytecodeChunk(i)` claims from the committed-bytecode reduction's
    /// address phase; present only when that phase runs.
    pub bytecode_address_phase: Option<BytecodeReductionAddressPhaseOutputClaims<C>>,
    /// Final `ProgramImageInit` claim from the program-image reduction's address
    /// phase; present only when that phase runs.
    pub program_image_address_phase: Option<ProgramImageReductionAddressPhaseOutputClaims<C>>,
}

impl<F: Field> Stage7OutputClaims<F, BaseJolt> {
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

/// The Fiat-Shamir challenge the verifier draws during stage 7: the hamming-weight
/// claim reduction's batching gamma. Drawn path-agnostically before the ZK/clear
/// branch; carried in [`Stage7ZkOutput`] for BlindFold (the clear path threads the
/// per-relation challenges struct directly into the hamming relation's claims).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7Challenges<F: Field> {
    pub hamming_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ClearOutput<F: Field, M: JoltCommitmentMode = BaseJolt> {
    /// The produced stage-7 openings paired with their points (point + value) via
    /// the `OpeningClaim` cell.
    pub output_claims: Stage7OutputClaims<OpeningClaim<F>, M>,
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
/// source the hamming-weight batching gamma from `challenges.hamming_gamma`,
/// matching the `input.stageN.challenges.<field>` idiom used by stages 3–5.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7ZkOutput<F: Field, C> {
    pub challenges: Stage7Challenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    pub hamming_weight_opening_point: Vec<F>,
    pub precommitted_final_openings: Vec<PrecommittedFinalOpening<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage7Output<F: Field, C, M: JoltCommitmentMode = BaseJolt> {
    Clear(Stage7ClearOutput<F, M>),
    /// BlindFold rides the homomorphic mode only (zk x packed is rejected
    /// fail-closed), so the zk arm stays [`BaseJolt`]-shaped.
    Zk(Stage7ZkOutput<F, C>),
}

impl<F: Field, C, M: JoltCommitmentMode> Stage7Output<F, C, M> {
    pub fn clear(&self) -> Result<&Stage7ClearOutput<F, M>, crate::VerifierError> {
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
