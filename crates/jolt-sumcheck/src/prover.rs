//! Sumcheck prover: generates round polynomials and binds witnesses.

use jolt_field::Field;
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
/// # Implementor contract
///
/// * `round_polynomial` must return a polynomial $s(X)$ of degree at most
///   the claim's `degree` bound such that $s(0) + s(1)$ equals the partial
///   sum over the remaining Boolean hypercube.
/// * `bind(r)` must fix the current leading variable to $r$ in place,
///   reducing `num_vars` by one.
pub trait SumcheckCompute<F: Field>: Send + Sync {
    /// Computes the round polynomial $s_i(X)$ for the current round.
    ///
    /// The returned univariate polynomial satisfies
    /// $s_i(0) + s_i(1) = \sum_{x' \in \{0,1\}^{n-i}} g(r_1,\ldots,r_{i-1}, X, x')$
    /// summed over $X \in \{0, 1\}$.
    fn round_polynomial(&self) -> UnivariatePoly<F>;

    /// Fixes the current leading variable to `challenge`, reducing the
    /// witness to one fewer variable.
    fn bind(&mut self, challenge: F);
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
    /// 4. Binds the witness at $r_i$.
    pub fn prove_with_handler<F, T, H>(
        claim: &SumcheckClaim<F>,
        witness: &mut impl SumcheckCompute<F>,
        transcript: &mut T,
        challenge_fn: impl Fn(T::Challenge) -> F,
        mut handler: H,
    ) -> H::Proof
    where
        F: Field,
        T: Transcript,
        H: RoundHandler<F>,
    {
        for _round in 0..claim.num_vars {
            let round_poly = witness.round_polynomial();
            handler.absorb_round_poly(&round_poly, transcript);
            let challenge = challenge_fn(transcript.challenge());
            handler.on_challenge(challenge);
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
        challenge_fn: impl Fn(T::Challenge) -> F,
    ) -> SumcheckProof<F>
    where
        F: Field,
        T: Transcript,
    {
        Self::prove_with_handler(
            claim,
            witness,
            transcript,
            challenge_fn,
            ClearRoundHandler::with_capacity(claim.num_vars),
        )
    }
}
