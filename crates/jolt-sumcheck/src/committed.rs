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
pub struct CommittedSumcheckCheck<F, C> {
    pub rounds: Vec<VerifiedCommittedRound<F, C>>,
}

impl<F: Copy, C: Clone> CommittedSumcheckCheck<F, C> {
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
