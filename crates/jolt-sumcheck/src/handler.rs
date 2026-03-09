//! Strategy traits for sumcheck round polynomial handling.
//!
//! [`RoundHandler`] (prover) and [`RoundVerifier`] (verifier) define how
//! round polynomials are absorbed into the Fiat-Shamir transcript:
//!
//! - **Clear mode** ([`ClearRoundHandler`], [`ClearRoundVerifier`]): polynomial
//!   coefficients are appended directly — no hiding.
//! - **Committed mode** (in `jolt-blindfold`): polynomial coefficients are
//!   committed via a `JoltCommitment` scheme
//!   and only the commitment is appended — hiding for ZK.
//!
//! The sumcheck prover and verifier engines are generic over these traits,
//! so ZK and non-ZK paths share the same proving/verification logic.

use jolt_field::Field;
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::error::SumcheckError;
use crate::proof::SumcheckProof;

/// Strategy for how sumcheck round polynomials are absorbed into the
/// Fiat-Shamir transcript and collected into a proof.
///
/// Implementations decide what data is appended to the transcript at each
/// round (coefficients vs. commitment) and what proof artifact is produced.
pub trait RoundHandler<F: Field> {
    /// The proof artifact produced after all rounds complete.
    type Proof;

    /// Process a single round polynomial.
    ///
    /// Must append sufficient data to the transcript for Fiat-Shamir binding.
    fn absorb_round_poly(&mut self, poly: &UnivariatePoly<F>, transcript: &mut impl Transcript);

    /// Called after each Fiat-Shamir challenge is derived.
    ///
    /// Committed handlers override this to store the challenge alongside
    /// polynomial coefficients and blinding factors for BlindFold witness
    /// construction. The default implementation is a no-op.
    fn on_challenge(&mut self, _challenge: F) {}

    /// Finalize after all rounds, producing the proof.
    fn finalize(self) -> Self::Proof;
}

/// Cleartext handler: appends polynomial coefficients directly.
///
/// Produces a [`SumcheckProof`] containing all round polynomials in the clear.
pub struct ClearRoundHandler<F: Field> {
    round_polynomials: Vec<UnivariatePoly<F>>,
}

impl<F: Field> ClearRoundHandler<F> {
    pub fn new() -> Self {
        Self {
            round_polynomials: Vec::new(),
        }
    }

    /// Creates a handler pre-allocated for `capacity` rounds.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            round_polynomials: Vec::with_capacity(capacity),
        }
    }
}

impl<F: Field> Default for ClearRoundHandler<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field> RoundHandler<F> for ClearRoundHandler<F> {
    type Proof = SumcheckProof<F>;

    fn absorb_round_poly(&mut self, poly: &UnivariatePoly<F>, transcript: &mut impl Transcript) {
        for coeff in poly.coefficients() {
            coeff.append_to_transcript(transcript);
        }
        self.round_polynomials.push(poly.clone());
    }

    fn finalize(self) -> SumcheckProof<F> {
        SumcheckProof {
            round_polynomials: self.round_polynomials,
        }
    }
}

/// Strategy for how the verifier processes per-round proof data.
///
/// Verification proceeds in two phases per round:
/// 1. [`absorb_and_check`](Self::absorb_and_check) — absorb round data into
///    the transcript, optionally verify consistency (clear mode checks
///    `poly(0) + poly(1) == running_sum`; committed mode skips this).
/// 2. [`next_running_sum`](Self::next_running_sum) — compute the next running
///    sum using the derived challenge (clear mode evaluates the polynomial;
///    committed mode returns zero since BlindFold verifies later).
pub trait RoundVerifier<F: Field> {
    /// Per-round proof data (`UnivariatePoly<F>` for clear, commitment for ZK).
    type RoundProof;

    /// Absorb round data into the transcript and verify consistency.
    ///
    /// Called BEFORE challenge derivation. The implementation must append
    /// the same bytes to the transcript as the prover's [`RoundHandler`]
    /// to maintain Fiat-Shamir synchronization.
    fn absorb_and_check(
        &self,
        proof: &Self::RoundProof,
        running_sum: F,
        degree_bound: usize,
        round: usize,
        transcript: &mut impl Transcript,
    ) -> Result<(), SumcheckError>;

    /// Compute the next running sum given the Fiat-Shamir challenge.
    ///
    /// Called AFTER challenge derivation.
    fn next_running_sum(&self, proof: &Self::RoundProof, challenge: F) -> F;
}

/// Cleartext verifier: checks polynomial consistency and evaluates.
///
/// Pairs with [`ClearRoundHandler`] on the prover side.
pub struct ClearRoundVerifier;

impl<F: Field> RoundVerifier<F> for ClearRoundVerifier {
    type RoundProof = UnivariatePoly<F>;

    fn absorb_and_check(
        &self,
        proof: &UnivariatePoly<F>,
        running_sum: F,
        degree_bound: usize,
        round: usize,
        transcript: &mut impl Transcript,
    ) -> Result<(), SumcheckError> {
        if proof.degree() > degree_bound {
            return Err(SumcheckError::DegreeBoundExceeded {
                got: proof.degree(),
                max: degree_bound,
            });
        }

        let sum = proof.evaluate(F::zero()) + proof.evaluate(F::one());
        if sum != running_sum {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: format!("{running_sum}"),
                actual: format!("{sum}"),
            });
        }

        for coeff in proof.coefficients() {
            coeff.append_to_transcript(transcript);
        }

        Ok(())
    }

    fn next_running_sum(&self, proof: &UnivariatePoly<F>, challenge: F) -> F {
        proof.evaluate(challenge)
    }
}
