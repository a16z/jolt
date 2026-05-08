//! Sumcheck verifier: checks round polynomials against the claimed sum.

use jolt_field::Field;
use jolt_transcript::Transcript;

use crate::claim::{EvaluationClaim, SumcheckClaim};
use crate::error::SumcheckError;
use crate::round_proof::RoundProof;

/// Stateless sumcheck verifier engine.
///
/// Replays the Fiat-Shamir transcript and checks each round against
/// the running sum, ultimately producing the final evaluation point
/// and expected value for an oracle query.
pub struct SumcheckVerifier;

impl SumcheckVerifier {
    /// Verifies a sumcheck proof.
    ///
    /// For each round $i = 0, \ldots, n-1$:
    /// 1. The degree bound is enforced against `claim.degree`.
    /// 2. The round proof's [`RoundProof::check_sum`] is invoked against the
    ///    running sum (clear-mode impls verify `s_i(0) + s_i(1) == running_sum`;
    ///    committed-mode impls defer to BlindFold).
    /// 3. The round proof absorbs its payload into the transcript.
    /// 4. A challenge $r_i$ is squeezed from the transcript.
    /// 5. The running sum is updated to [`RoundProof::evaluate`] at $r_i$.
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
    pub fn verify<F, T, P>(
        claim: &SumcheckClaim<F>,
        round_proofs: &[P],
        transcript: &mut T,
    ) -> Result<EvaluationClaim<F>, SumcheckError<F>>
    where
        F: Field,
        T: Transcript<Challenge = F>,
        P: RoundProof<F>,
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
            round_proof.check_sum(running_sum, round)?;
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
}
