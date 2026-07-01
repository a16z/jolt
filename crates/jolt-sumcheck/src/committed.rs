//! Committed sumcheck round messages.

use ark_serialize::CanonicalSerialize;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_transcript::{FsAbsorb, FsTranscript};
use serde::{Deserialize, Serialize};

use crate::error::SumcheckError;
use crate::round_proof::{RoundDegree, RoundMessage};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedRound<C> {
    pub commitment: C,
    pub degree: usize,
}

impl<C> RoundDegree for CommittedRound<C> {
    fn degree(&self) -> usize {
        self.degree
    }
}

impl<C: CanonicalSerialize, F: Field> RoundMessage<F> for CommittedRound<C> {
    fn append_to_transcript<T: FsTranscript<F>>(&self, transcript: &mut T) {
        transcript.absorb(&self.commitment);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedOutputClaims<C> {
    pub commitments: Vec<C>,
}

impl<C: CanonicalSerialize> CommittedOutputClaims<C> {
    /// Absorbs the committed output-claim commitments into `transcript` as a single
    /// message, like jolt-core (`absorb(&output_claims_commitments)`).
    pub fn append_to_transcript<T: FsAbsorb>(&self, transcript: &mut T) {
        transcript.absorb(&self.commitments);
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
    pub output_claims: CommittedOutputClaims<C>,
}

impl<F: Copy, C> CommittedSumcheckConsistency<F, C> {
    pub fn challenges(&self) -> Vec<F> {
        self.rounds.iter().map(|round| round.challenge).collect()
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
