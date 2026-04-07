//! Sumcheck verifier: checks round polynomials against the claimed sum.

use jolt_field::Field;
use jolt_transcript::Transcript;

use crate::claim::SumcheckClaim;
use crate::error::SumcheckError;
use crate::round::RoundVerifier;

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
    /// 1. The round verifier absorbs proof data into the transcript and
    ///    checks consistency (clear mode verifies `s_i(0) + s_i(1) == running_sum`;
    ///    committed mode defers to BlindFold).
    /// 2. A challenge $r_i$ is squeezed from the transcript.
    /// 3. The running sum is updated to $s_i(r_i)$.
    ///
    /// On success, returns `(v, r)` where `v` is the final evaluation
    /// and `r = (r_1, ..., r_n)` is the challenge vector.
    ///
    /// # Errors
    ///
    /// Returns [`SumcheckError`] if any round check fails, a degree bound
    /// is exceeded, or the proof has the wrong number of rounds.
    #[tracing::instrument(skip_all, name = "SumcheckVerifier::verify")]
    pub fn verify<F, T, V>(
        claim: &SumcheckClaim<F>,
        round_proofs: &[V::RoundProof],
        transcript: &mut T,
        verifier: &V,
    ) -> Result<(F, Vec<F>), SumcheckError>
    where
        F: Field,
        T: Transcript<Challenge = F>,
        V: RoundVerifier<F>,
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
            verifier.absorb_and_check(round_proof, running_sum, claim.degree, round, transcript)?;
            let r: F = transcript.challenge();
            running_sum = verifier.next_running_sum(round_proof, r);
            challenges.push(r);
        }

        Ok((running_sum, challenges))
    }
}
