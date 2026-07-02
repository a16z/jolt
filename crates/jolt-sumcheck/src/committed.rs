//! Committed sumcheck round messages.

use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};
use serde::{Deserialize, Serialize};

use crate::error::SumcheckError;
use crate::round_proof::RoundMessage;

const SUMCHECK_COMMITMENT_LABEL: &[u8] = b"sumcheck_commitment";
const OUTPUT_CLAIMS_LABEL: &[u8] = b"output_claims_coms";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedRound<C> {
    pub commitment: C,
    pub degree: usize,
}

impl<C: AppendToTranscript> RoundMessage for CommittedRound<C> {
    fn degree(&self) -> usize {
        self.degree
    }

    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(SUMCHECK_COMMITMENT_LABEL));
        self.commitment.append_to_transcript(transcript);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedOutputClaims<C> {
    pub commitments: Vec<C>,
}

impl<C: AppendToTranscript> AppendToTranscript for CommittedOutputClaims<C> {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&LabelWithCount(
            OUTPUT_CLAIMS_LABEL,
            self.commitments.len() as u64,
        ));
        for commitment in &self.commitments {
            commitment.append_to_transcript(transcript);
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedSumcheckProof<C> {
    pub rounds: Vec<CommittedRound<C>>,
    pub output_claims: CommittedOutputClaims<C>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedCommittedRound<F, C> {
    pub commitment: C,
    pub degree: usize,
    pub challenge: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedSumcheckConsistency<F, C> {
    pub rounds: Vec<VerifiedCommittedRound<F, C>>,
}

impl<F: Copy, C: Clone> CommittedSumcheckConsistency<F, C> {
    pub fn challenges(&self) -> Vec<F> {
        self.rounds.iter().map(|round| round.challenge).collect()
    }

    pub fn round_degrees(&self) -> Vec<usize> {
        self.rounds.iter().map(|round| round.degree).collect()
    }

    pub fn round_commitments(&self) -> Vec<C> {
        self.rounds
            .iter()
            .map(|round| round.commitment.clone())
            .collect()
    }
}

/// A [`CommittedSumcheckConsistency`] paired with the batching data the ZK
/// verify driver folds it against.
///
/// Produced by `jolt-verifier`'s generated per-stage `verify_zk` driver and
/// read back by BlindFold. Committed proofs expose only transcript challenges
/// and commitments, not scalar evaluation claims, so this type carries no claim
/// values — only the per-instance batching coefficients and the combined
/// `(max_num_vars, max_degree)` dimensions.
///
/// Instances are laid out with front-loaded dummy rounds: a shorter instance
/// with `num_vars` rounds is active only in the last `num_vars` rounds, so its
/// challenge suffix begins at `max_num_vars - num_vars`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedCommittedSumcheckConsistency<F: Field, C> {
    pub consistency: CommittedSumcheckConsistency<F, C>,
    pub batching_coefficients: Vec<F>,
    pub max_num_vars: usize,
    pub max_degree: usize,
}

impl<F: Field, C> BatchedCommittedSumcheckConsistency<F, C> {
    /// Returns the front-padding offset for an instance with `num_vars`.
    ///
    /// Batched committed verification front-loads dummy rounds for smaller
    /// instances, so an instance with `num_vars` is evaluated on the suffix
    /// beginning at `max_num_vars - num_vars`.
    pub fn try_round_offset(&self, num_vars: usize) -> Result<usize, SumcheckError<F>> {
        self.max_num_vars
            .checked_sub(num_vars)
            .ok_or(SumcheckError::BatchedPointOutOfRange {
                offset: 0,
                num_vars,
                total: self.consistency.rounds.len(),
            })
    }

    pub fn challenges(&self) -> Vec<F>
    where
        F: Copy,
    {
        self.consistency
            .rounds
            .iter()
            .map(|round| round.challenge)
            .collect()
    }

    /// Returns the suffix challenge vector for an instance with `num_vars`.
    pub fn try_instance_point(&self, num_vars: usize) -> Result<Vec<F>, SumcheckError<F>>
    where
        F: Copy,
    {
        self.try_instance_point_at(self.try_round_offset(num_vars)?, num_vars)
    }

    /// Returns a challenge vector starting at `offset`.
    ///
    /// This is useful for protocols whose instance point is embedded inside the
    /// batched challenge vector but not necessarily at the canonical suffix
    /// offset.
    pub fn try_instance_point_at(
        &self,
        offset: usize,
        num_vars: usize,
    ) -> Result<Vec<F>, SumcheckError<F>>
    where
        F: Copy,
    {
        let end = offset
            .checked_add(num_vars)
            .ok_or(SumcheckError::BatchedPointRangeOverflow { offset, num_vars })?;
        self.consistency
            .rounds
            .get(offset..end)
            .ok_or(SumcheckError::BatchedPointOutOfRange {
                offset,
                num_vars,
                total: self.consistency.rounds.len(),
            })
            .map(|rounds| rounds.iter().map(|round| round.challenge).collect())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedRoundWitness<F> {
    pub coefficients: Vec<F>,
    pub blinding: F,
}

impl<F: Field> CommittedRoundWitness<F> {
    pub fn commit<VC>(
        &self,
        setup: &VC::Setup,
    ) -> Result<CommittedRound<VC::Output>, SumcheckError<F>>
    where
        VC: VectorCommitment<Field = F>,
    {
        if self.coefficients.is_empty() {
            return Err(SumcheckError::EmptyRoundCoefficients);
        }

        Ok(CommittedRound {
            commitment: VC::commit(setup, &self.coefficients, &self.blinding),
            degree: self.coefficients.len() - 1,
        })
    }
}
