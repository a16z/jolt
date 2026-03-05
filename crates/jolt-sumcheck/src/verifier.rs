//! Sumcheck verifier: checks round polynomials against the claimed sum.

use jolt_field::Field;
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::claim::SumcheckClaim;
use crate::error::SumcheckError;
use crate::proof::SumcheckProof;

/// Stateless sumcheck verifier engine.
///
/// Replays the Fiat-Shamir transcript and checks each round polynomial
/// against the running sum, ultimately producing the final evaluation
/// point and expected value for an oracle query.
pub struct SumcheckVerifier;

impl SumcheckVerifier {
    /// Verifies a sumcheck proof against the given claim.
    ///
    /// For each round $i = 0, \ldots, n-1$:
    /// 1. Checks that $\deg(s_i) \le d$ (the claim's degree bound).
    /// 2. Checks that $s_i(0) + s_i(1)$ equals the running sum
    ///    (initialized to `claim.claimed_sum`).
    /// 3. Absorbs $s_i$ into the transcript and squeezes challenge $r_i$.
    /// 4. Sets the running sum to $s_i(r_i)$.
    ///
    /// On success, returns `(v, \mathbf{r})$ where $v = s_n(r_n)$ is the
    /// final evaluation and $\mathbf{r} = (r_1, \ldots, r_n)$ is the
    /// challenge vector. The caller must verify that the underlying
    /// polynomial evaluates to $v$ at $\mathbf{r}$.
    ///
    /// # Errors
    ///
    /// Returns [`SumcheckError`] if any round check fails, a degree bound
    /// is exceeded, or the proof has the wrong number of rounds.
    pub fn verify<F, T>(
        claim: &SumcheckClaim<F>,
        proof: &SumcheckProof<F>,
        transcript: &mut T,
        challenge_fn: impl Fn(T::Challenge) -> F,
    ) -> Result<(F, Vec<F>), SumcheckError>
    where
        F: Field,
        T: Transcript,
    {
        if proof.round_polynomials.len() != claim.num_vars {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: claim.num_vars,
                got: proof.round_polynomials.len(),
            });
        }

        let mut expected_sum = claim.claimed_sum;
        let mut challenges = Vec::with_capacity(claim.num_vars);

        for (round, round_poly) in proof.round_polynomials.iter().enumerate() {
            if round_poly.degree() > claim.degree {
                return Err(SumcheckError::DegreeBoundExceeded {
                    got: round_poly.degree(),
                    max: claim.degree,
                });
            }

            let sum = round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one());
            if sum != expected_sum {
                return Err(SumcheckError::RoundCheckFailed {
                    round,
                    expected: format!("{expected_sum}"),
                    actual: format!("{sum}"),
                });
            }

            append_poly_to_transcript(round_poly, transcript);
            let r = challenge_fn(transcript.challenge());
            expected_sum = round_poly.evaluate(r);
            challenges.push(r);
        }

        Ok((expected_sum, challenges))
    }
}

/// Serializes each coefficient of a univariate polynomial and absorbs the
/// bytes into the transcript.
#[inline]
fn append_poly_to_transcript<F: Field, T: Transcript>(
    poly: &UnivariatePoly<F>,
    transcript: &mut T,
) {
    for coeff in poly.coefficients() {
        coeff.append_to_transcript(transcript);
    }
}
