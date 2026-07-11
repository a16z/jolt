//! Committed sumcheck round messages.

use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};
use rand_core::OsRng;
use serde::{Deserialize, Serialize};

use crate::error::SumcheckError;
use crate::proof::SumcheckProof;
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
/// A shorter instance is active only inside its window. Tail-aligned
/// instances (the default — dummy rounds front-loaded) have their challenge
/// suffix begin at `max_num_vars - num_vars`; head-aligned instances (the
/// precommitted claim-reduction phases) bind the leading challenges and need
/// their offset supplied explicitly via [`Self::try_instance_point_at`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedCommittedSumcheckConsistency<F: Field, C> {
    pub consistency: CommittedSumcheckConsistency<F, C>,
    pub batching_coefficients: Vec<F>,
    pub max_num_vars: usize,
    pub max_degree: usize,
}

impl<F: Field, C> BatchedCommittedSumcheckConsistency<F, C> {
    /// Returns the tail-aligned default offset (`max_num_vars - num_vars`)
    /// for an instance with `num_vars` — the suffix start when the instance's
    /// dummy rounds are front-loaded. Head-aligned instances must not use
    /// this; supply their offset to [`Self::try_instance_point_at`] directly.
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

/// The prover-retained openings of one committed sumcheck: the round
/// polynomials' coefficients and blindings, and the output-claim rows (values
/// chunked to the vector-commitment capacity) and their blindings. This is the
/// BlindFold witness material for the sumcheck — everything needed to open the
/// commitments the proof carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedSumcheckWitness<F> {
    pub round_coefficients: Vec<Vec<F>>,
    pub round_blindings: Vec<F>,
    pub output_claim_rows: Vec<Vec<F>>,
    pub output_claim_blindings: Vec<F>,
}

impl<F> CommittedSumcheckWitness<F> {
    fn new() -> Self {
        Self {
            round_coefficients: Vec::new(),
            round_blindings: Vec::new(),
            output_claim_rows: Vec::new(),
            output_claim_blindings: Vec::new(),
        }
    }
}

/// Incrementally assembles a [`CommittedSumcheckProof`]: per round, commit the
/// round polynomial's coefficients with a fresh blinding, absorb the
/// commitment, and squeeze the round challenge; at the end, row-commit the
/// flattened output-claim values and absorb those commitments. Blindings are
/// drawn from [`OsRng`] and retained in the witness.
pub struct CommittedSumcheckBuilder<'a, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    setup: &'a VC::Setup,
    rng: OsRng,
    rounds: Vec<CommittedRound<VC::Output>>,
    witness: CommittedSumcheckWitness<F>,
}

impl<'a, F, VC> CommittedSumcheckBuilder<'a, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub fn new(setup: &'a VC::Setup) -> Result<Self, SumcheckError<F>> {
        if VC::capacity(setup) == 0 {
            return Err(SumcheckError::ZeroCommitmentCapacity);
        }
        Ok(Self {
            setup,
            rng: OsRng,
            rounds: Vec::new(),
            witness: CommittedSumcheckWitness::new(),
        })
    }

    /// Commit one round polynomial, absorb the commitment, and squeeze the
    /// round challenge.
    pub fn commit_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        transcript: &mut T,
    ) -> Result<F, SumcheckError<F>>
    where
        T: Transcript<Challenge = F>,
    {
        let coefficients = round_poly.coefficients().to_vec();
        if coefficients.len() > VC::capacity(self.setup) {
            return Err(SumcheckError::RoundExceedsCommitmentCapacity {
                coefficients: coefficients.len(),
                capacity: VC::capacity(self.setup),
            });
        }

        let blinding = F::random(&mut self.rng);
        let witness = CommittedRoundWitness {
            coefficients: coefficients.clone(),
            blinding,
        };
        let round = witness.commit::<VC>(self.setup)?;
        round.append_to_transcript(transcript);
        let challenge = transcript.challenge();

        self.rounds.push(round);
        self.witness.round_coefficients.push(coefficients);
        self.witness.round_blindings.push(blinding);
        Ok(challenge)
    }

    /// Row-commit the flattened output-claim values (chunked to the setup's
    /// capacity), absorb the commitments, and assemble the proof, returning it
    /// with the prover-retained witness that opens it.
    #[expect(
        clippy::type_complexity,
        reason = "a proof paired with the witness that opens it, not worth a named pair type"
    )]
    pub fn finish<T>(
        mut self,
        output_claim_values: &[F],
        transcript: &mut T,
    ) -> Result<(SumcheckProof<F, VC::Output>, CommittedSumcheckWitness<F>), SumcheckError<F>>
    where
        T: Transcript<Challenge = F>,
    {
        let capacity = VC::capacity(self.setup);
        let mut commitments = Vec::with_capacity(output_claim_values.len().div_ceil(capacity));
        for row in output_claim_values.chunks(capacity) {
            let blinding = F::random(&mut self.rng);
            let commitment = VC::commit(self.setup, row, &blinding);
            commitments.push(commitment);
            self.witness.output_claim_rows.push(row.to_vec());
            self.witness.output_claim_blindings.push(blinding);
        }

        let output_claims = CommittedOutputClaims { commitments };
        output_claims.append_to_transcript(transcript);
        Ok((
            SumcheckProof::Committed(CommittedSumcheckProof {
                rounds: self.rounds,
                output_claims,
            }),
            self.witness,
        ))
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
