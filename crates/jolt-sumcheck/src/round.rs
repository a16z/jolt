//! Strategy trait for sumcheck round polynomial verification.
//!
//! [`RoundVerifier`] defines how round polynomials are verified against the
//! Fiat-Shamir transcript:
//!
//! - **Clear mode** ([`ClearRoundVerifier`]): polynomial coefficients are
//!   checked directly — `poly(0) + poly(1) == running_sum`.
//! - **Committed mode** (future, in `jolt-blindfold`): polynomial commitments
//!   are absorbed into the transcript; consistency is verified via BlindFold.

use jolt_field::Field;
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

use crate::error::SumcheckError;

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
    /// the same bytes to the transcript as the prover appended, to maintain
    /// Fiat-Shamir synchronization.
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
/// When `label` is `Some`, a [`LabelWithCount`] word is absorbed before
/// each round's coefficients.
#[derive(Default)]
pub struct ClearRoundVerifier {
    label: Option<&'static [u8]>,
}

impl ClearRoundVerifier {
    /// Verifier with no domain-separation labels (raw coefficient absorption).
    pub fn new() -> Self {
        Self::default()
    }

    /// Verifier that prepends a `LabelWithCount` word before each round's coefficients.
    pub fn with_label(label: &'static [u8]) -> Self {
        Self { label: Some(label) }
    }
}

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

        if let Some(label) = self.label {
            let coeffs = proof.coefficients();
            transcript.append(&LabelWithCount(label, coeffs.len() as u64));
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
