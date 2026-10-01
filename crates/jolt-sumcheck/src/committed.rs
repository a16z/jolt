//! Committed sumcheck round messages.
//!
//! A committed round sends a vector commitment to the round polynomial's
//! `degree + 1` coefficients (padded to the round's public degree bound, which
//! fixes BlindFold's coefficient layout) instead of the coefficients. After the
//! rounds, the output-claim values are row-committed in chunks of the setup's
//! capacity.

#[cfg(feature = "committed")]
use jolt_crypto::VectorCommitment;
#[cfg(feature = "committed")]
use jolt_field::CanonicalEncoding;
use jolt_field::Field;
#[cfg(feature = "committed")]
use jolt_field::JoltField;
#[cfg(feature = "committed")]
use jolt_poly::UnivariatePoly;
#[cfg(feature = "committed")]
use jolt_transcript::{Channel, ProverTranscript, Sponge};
#[cfg(feature = "committed")]
use rand_core::RngCore;
use serde::{Deserialize, Serialize};

use crate::error::SumcheckError;
#[cfg(feature = "committed")]
use crate::round_proof::padded_coefficients;

/// Row commitments to a committed sumcheck's flattened output-claim values.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedOutputClaims<C> {
    pub commitments: Vec<C>,
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
    #[cfg(feature = "committed")]
    fn new() -> Self {
        Self {
            round_coefficients: Vec::new(),
            round_blindings: Vec::new(),
            output_claim_rows: Vec::new(),
            output_claim_blindings: Vec::new(),
        }
    }
}

/// Records a committed sumcheck into the prover transcript: per round, commit
/// the padded round polynomial with a fresh blinding, send the commitment, and
/// draw the round challenge; at the end, row-commit the flattened output-claim
/// values and send those commitments. Blindings are drawn from the
/// caller-supplied `rng` and retained in the witness, so a fixed seed
/// reproduces the proof.
#[cfg(feature = "committed")]
pub struct CommittedSumcheckBuilder<'a, F, VC, R>
where
    F: JoltField,
    VC: VectorCommitment<Field = F>,
    R: RngCore,
{
    setup: &'a VC::Setup,
    rng: R,
    witness: CommittedSumcheckWitness<F>,
}

#[cfg(feature = "committed")]
impl<'a, F, VC, R> CommittedSumcheckBuilder<'a, F, VC, R>
where
    F: JoltField + CanonicalEncoding,
    VC: VectorCommitment<Field = F>,
    R: RngCore,
{
    pub fn new(setup: &'a VC::Setup, rng: R) -> Result<Self, SumcheckError<F>> {
        if VC::capacity(setup) == 0 {
            return Err(SumcheckError::ZeroCommitmentCapacity);
        }
        Ok(Self {
            setup,
            rng,
            witness: CommittedSumcheckWitness::new(),
        })
    }

    /// Commit one round polynomial of degree bound `degree`, send the
    /// commitment, and draw the round challenge.
    pub fn commit_round<H: Sponge>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        degree: usize,
        transcript: &mut ProverTranscript<H>,
    ) -> Result<F, SumcheckError<F>> {
        let coefficients = padded_coefficients(round_poly, degree)?;
        if coefficients.len() > VC::capacity(self.setup) {
            return Err(SumcheckError::RoundExceedsCommitmentCapacity {
                coefficients: coefficients.len(),
                capacity: VC::capacity(self.setup),
            });
        }

        let blinding = F::random(&mut self.rng);
        let commitment = VC::commit(self.setup, &coefficients, &blinding);
        transcript.send(&commitment);
        let challenge = transcript.challenge_small();

        self.witness.round_coefficients.push(coefficients);
        self.witness.round_blindings.push(blinding);
        Ok(challenge)
    }

    /// Row-commit the flattened output-claim values (chunked to the setup's
    /// capacity), send the commitments, and return the prover-retained
    /// witness that opens every commitment sent.
    pub fn finish<H: Sponge>(
        mut self,
        output_claim_values: &[F],
        transcript: &mut ProverTranscript<H>,
    ) -> Result<CommittedSumcheckWitness<F>, SumcheckError<F>> {
        let capacity = VC::capacity(self.setup);
        for row in output_claim_values.chunks(capacity) {
            let blinding = F::random(&mut self.rng);
            transcript.send(&VC::commit(self.setup, row, &blinding));
            self.witness.output_claim_rows.push(row.to_vec());
            self.witness.output_claim_blindings.push(blinding);
        }
        Ok(self.witness)
    }
}
