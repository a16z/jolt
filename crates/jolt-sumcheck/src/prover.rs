//! Sumcheck prover: generates round polynomials and binds witnesses.

use jolt_field::WithChallenge;
use jolt_poly::UnivariatePoly;
use jolt_transcript::Transcript;

use crate::claim::SumcheckClaim;
use crate::handler::{ClearRoundHandler, RoundHandler};
use crate::proof::SumcheckProof;

/// Trait that a concrete witness must implement to participate in the
/// sumcheck protocol.
///
/// The prover engine calls [`round_polynomial`](SumcheckCompute::round_polynomial)
/// to obtain the univariate restriction of the summed polynomial in the
/// current round, then calls [`bind`](SumcheckCompute::bind) with the
/// Fiat-Shamir challenge to fix that variable and advance to the next round.
///
/// # Challenge type
///
/// `bind` accepts `F::Challenge` (e.g., `MontU128Challenge<F>`) rather than a
/// full field element `F`. This enables ~1.6× faster polynomial binding: the
/// 125-bit challenge uses 4×2 Montgomery multiplication instead of 4×4.
///
/// # Implementor contract
///
/// * `round_polynomial` must return a polynomial $s(X)$ of degree at most
///   the claim's `degree` bound such that $s(0) + s(1)$ equals the partial
///   sum over the remaining Boolean hypercube.
/// * `bind(r)` must fix the current leading variable to $r$ in place,
///   reducing `num_vars` by one.
pub trait SumcheckCompute<F: WithChallenge>: Send + Sync {
    /// Computes the round polynomial $s_i(X)$ for the current round.
    ///
    /// The returned univariate polynomial satisfies
    /// $s_i(0) + s_i(1) = \sum_{x' \in \{0,1\}^{n-i}} g(r_1,\ldots,r_{i-1}, X, x')$
    /// summed over $X \in \{0, 1\}$.
    fn round_polynomial(&self) -> UnivariatePoly<F>;

    /// Fixes the current leading variable to `challenge`, reducing the
    /// witness to one fewer variable.
    ///
    /// Receives `F::Challenge` for faster 125-bit × 254-bit Montgomery
    /// multiplication. Convert with `challenge.into()` when full `F` is needed.
    fn bind(&mut self, challenge: F::Challenge);

    /// Provides the running sumcheck claim before each round.
    ///
    /// Called by the prover before [`round_polynomial`](Self::round_polynomial)
    /// with the current running sum: `claimed_sum` for round 0, then
    /// `round_poly.evaluate(challenge)` for subsequent rounds.
    ///
    /// Implementations that derive evaluation points from the claim
    /// (e.g., `P(1) = claim - P(0)` to skip one kernel evaluation)
    /// should override this. The default is a no-op.
    fn set_claim(&mut self, _claim: F) {}
}

/// Stateless sumcheck prover engine.
///
/// Orchestrates the interaction between a [`SumcheckCompute`] and a
/// [`Transcript`], producing a proof artifact determined by the
/// [`RoundHandler`] strategy.
pub struct SumcheckProver;

impl SumcheckProver {
    /// Executes the sumcheck prover protocol with a pluggable round handler.
    ///
    /// The handler controls how round polynomials are absorbed into the
    /// transcript and what proof artifact is produced. Use
    /// [`ClearRoundHandler`] for standard (non-ZK) proofs or a committed
    /// handler (from `jolt-blindfold`) for ZK proofs.
    ///
    /// For each of the `claim.num_vars` rounds:
    /// 1. Queries the witness for the round polynomial $s_i(X)$.
    /// 2. Delegates to `handler.absorb_round_poly()` for transcript binding.
    /// 3. Squeezes a challenge $r_i$ from the transcript.
    /// 4. Binds the witness at $r_i$ using `F::Challenge` for fast 125-bit multiply.
    #[tracing::instrument(skip_all, name = "SumcheckProver::prove")]
    pub fn prove_with_handler<F, T, H>(
        claim: &SumcheckClaim<F>,
        witness: &mut impl SumcheckCompute<F>,
        transcript: &mut T,
        mut handler: H,
    ) -> H::Proof
    where
        F: WithChallenge,
        F::Challenge: From<T::Challenge>,
        T: Transcript,
        H: RoundHandler<F>,
    {
        let mut running_claim = claim.claimed_sum;
        for _round in 0..claim.num_vars {
            witness.set_claim(running_claim);
            let round_poly = witness.round_polynomial();
            handler.absorb_round_poly(&round_poly, transcript);
            let challenge = F::Challenge::from(transcript.challenge());
            let challenge_f: F = challenge.into();
            handler.on_challenge(challenge_f);
            running_claim = round_poly.evaluate(challenge_f);
            witness.bind(challenge);
        }
        handler.finalize()
    }

    /// Executes the sumcheck prover with cleartext round handling.
    ///
    /// Convenience wrapper around [`prove_with_handler`](Self::prove_with_handler)
    /// using [`ClearRoundHandler`]. Polynomial coefficients are appended
    /// directly to the transcript.
    pub fn prove<F, T>(
        claim: &SumcheckClaim<F>,
        witness: &mut impl SumcheckCompute<F>,
        transcript: &mut T,
    ) -> SumcheckProof<F>
    where
        F: WithChallenge,
        F::Challenge: From<T::Challenge>,
        T: Transcript,
    {
        Self::prove_with_handler(
            claim,
            witness,
            transcript,
            ClearRoundHandler::with_capacity(claim.num_vars),
        )
    }
}
