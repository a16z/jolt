//! Sumcheck verifier: checks round polynomials against the claimed sum.

use jolt_transcript::{AppendToTranscript, Transcript};

use crate::claim::{EvaluationClaim, SumcheckClaim, SumcheckShape};
use crate::committed::{CommittedRound, CommittedSumcheckCheck, VerifiedCommittedRound};
use crate::error::SumcheckError;
use crate::round_proof::{ClearRound, RoundMessage};
use crate::scalar::SumcheckScalar;

/// Stateless sumcheck verifier engine.
pub struct SumcheckVerifier;

impl SumcheckVerifier {
    /// Verifies a sumcheck proof.
    ///
    /// For each round $i = 0, \ldots, n-1$:
    /// 1. The degree bound is enforced against `claim.degree`.
    /// 2. The round sum is checked against the running sum.
    /// 3. The round message is absorbed into the transcript.
    /// 4. A challenge $r_i$ is squeezed from the transcript.
    /// 5. The running sum is updated to the round polynomial at $r_i$.
    ///
    /// On success, returns an [`EvaluationClaim`] `{ point: r, value: v }`
    /// where `v` is the final evaluation and `r = (r_1, ..., r_n)` is the
    /// challenge vector.
    ///
    /// # Errors
    ///
    /// Returns [`SumcheckError`] if any round check fails, a degree bound
    /// is exceeded, or the proof has the wrong number of rounds.
    ///
    /// # Soundness
    ///
    /// When `claim.num_vars == 0`, this function performs no transcript
    /// interaction and no checks: it returns
    /// `EvaluationClaim { point: vec![], value: claim.claimed_sum }`.
    /// Sumcheck trivially reduces to a single oracle query at that point,
    /// so the caller MUST verify `claim.claimed_sum` against the
    /// commitment/oracle layer to retain soundness.
    #[tracing::instrument(skip_all, name = "SumcheckVerifier::verify")]
    pub fn verify<F, T, R>(
        claim: &SumcheckClaim<F>,
        round_proofs: &[R],
        transcript: &mut T,
    ) -> Result<EvaluationClaim<F>, SumcheckError<F>>
    where
        F: SumcheckScalar,
        T: Transcript<Challenge = F>,
        R: ClearRound<F>,
    {
        if round_proofs.len() != claim.num_vars {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: claim.num_vars,
                got: round_proofs.len(),
            });
        }

        let mut running_sum = claim.claimed_sum;
        let mut challenges = Vec::with_capacity(claim.num_vars);

        for (round, round_proof) in round_proofs.iter().enumerate() {
            if round_proof.degree() > claim.degree {
                return Err(SumcheckError::DegreeBoundExceeded {
                    got: round_proof.degree(),
                    max: claim.degree,
                });
            }
            round_proof.check_round_sum(running_sum, round)?;
            round_proof.append_to_transcript(transcript);
            let r: F = transcript.challenge();
            running_sum = round_proof.evaluate(r);
            challenges.push(r);
        }

        Ok(EvaluationClaim {
            point: challenges,
            value: running_sum,
        })
    }

    /// Replays committed sumcheck rounds and returns the transcript-derived data.
    #[tracing::instrument(skip_all, name = "SumcheckVerifier::verify_committed_rounds")]
    pub fn verify_committed_rounds<F, T, C>(
        shape: SumcheckShape,
        round_proofs: &[CommittedRound<C>],
        transcript: &mut T,
    ) -> Result<CommittedSumcheckCheck<F, C>, SumcheckError<F>>
    where
        F: SumcheckScalar,
        T: Transcript<Challenge = F>,
        C: Clone + AppendToTranscript,
    {
        if round_proofs.len() != shape.num_vars {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: shape.num_vars,
                got: round_proofs.len(),
            });
        }

        let mut rounds = Vec::with_capacity(shape.num_vars);
        for round_proof in round_proofs {
            if round_proof.degree() > shape.degree {
                return Err(SumcheckError::DegreeBoundExceeded {
                    got: round_proof.degree(),
                    max: shape.degree,
                });
            }

            round_proof.append_to_transcript(transcript);
            rounds.push(VerifiedCommittedRound {
                commitment: round_proof.commitment.clone(),
                degree: round_proof.degree,
                challenge: transcript.challenge(),
            });
        }

        Ok(CommittedSumcheckCheck { rounds })
    }
}
