//! Sumcheck verifier: checks round polynomials against the claimed sum.

use jolt_field::Field;
use jolt_transcript::Transcript;

use crate::claim::SumcheckClaim;
use crate::error::SumcheckError;
use crate::handler::{ClearRoundVerifier, RoundVerifier};
use crate::proof::SumcheckProof;

/// Stateless sumcheck verifier engine.
///
/// Replays the Fiat-Shamir transcript and checks each round against
/// the running sum, ultimately producing the final evaluation point
/// and expected value for an oracle query.
pub struct SumcheckVerifier;

impl SumcheckVerifier {
    /// Verifies a sumcheck proof using a pluggable round verifier.
    ///
    /// The verifier handler controls how per-round proof data is absorbed
    /// into the transcript and whether consistency checks are performed.
    /// Use [`ClearRoundVerifier`] for standard proofs or a committed
    /// verifier (from `jolt-blindfold`) for ZK proofs.
    ///
    /// On success, returns `(v, r)` where `v` is the final evaluation
    /// and `r = (r_1, ..., r_n)` is the challenge vector.
    ///
    /// # Errors
    ///
    /// Returns [`SumcheckError`] if the handler's consistency checks fail
    /// or the proof has the wrong number of rounds.
    #[tracing::instrument(skip_all, name = "SumcheckVerifier::verify")]
    pub fn verify_with_handler<F, T, V>(
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

    /// Verifies a cleartext sumcheck proof.
    ///
    /// Convenience wrapper around [`verify_with_handler`](Self::verify_with_handler)
    /// using [`ClearRoundVerifier`].
    ///
    /// For each round $i = 0, \ldots, n-1$:
    /// 1. Checks that $\deg(s_i) \le d$ (the claim's degree bound).
    /// 2. Checks that $s_i(0) + s_i(1)$ equals the running sum.
    /// 3. Absorbs $s_i$ into the transcript and squeezes challenge $r_i$.
    /// 4. Sets the running sum to $s_i(r_i)$.
    ///
    /// On success, returns `(v, \mathbf{r})$ where $v = s_n(r_n)$ is the
    /// final evaluation and $\mathbf{r} = (r_1, \ldots, r_n)$ is the
    /// challenge vector.
    ///
    /// # Errors
    ///
    /// Returns [`SumcheckError`] if any round check fails, a degree bound
    /// is exceeded, or the proof has the wrong number of rounds.
    pub fn verify<F, T>(
        claim: &SumcheckClaim<F>,
        proof: &SumcheckProof<F>,
        transcript: &mut T,
    ) -> Result<(F, Vec<F>), SumcheckError>
    where
        F: Field,
        T: Transcript<Challenge = F>,
    {
        Self::verify_with_handler(
            claim,
            &proof.round_polynomials,
            transcript,
            &ClearRoundVerifier,
        )
    }
}
