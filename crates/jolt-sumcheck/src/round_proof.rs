//! Per-round proof types and the `RoundProof` trait.
//!
//! [`RoundProof`] unifies the four operations the sumcheck verifier performs
//! on each round: degree bound, sum-check consistency, transcript absorption,
//! and evaluation at the Fiat-Shamir challenge. Concrete implementations
//! encode transcript format (raw vs labeled, full vs compressed) and mode
//! (clear vs committed — future).

use jolt_field::Field;
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

use crate::error::SumcheckError;

/// Per-round proof operations used by the sumcheck verifier.
///
/// Implementations encode one concrete wire format by pairing a round
/// polynomial (or commitment, in future committed-mode impls) with
/// transcript labelling and compression choices. The single verifier loop
/// in [`crate::SumcheckVerifier::verify`] drives any impl uniformly; future
/// ZK support is a new impl, not a new strategy trait.
pub trait RoundProof<F: Field> {
    /// Degree of this round polynomial (for the degree-bound check).
    fn degree(&self) -> usize;

    /// Verify round consistency: `poly(0) + poly(1) == running_sum`.
    ///
    /// Clear-mode impls enforce this; a future committed-mode impl returns
    /// `Ok(())` and defers to BlindFold.
    fn check_sum(&self, running_sum: F, round: usize) -> Result<(), SumcheckError>;

    /// Evaluate at the Fiat-Shamir challenge to compute the next running sum.
    ///
    /// Committed-mode impls may return `F::zero()` since BlindFold verifies
    /// the reduction separately.
    fn evaluate(&self, challenge: F) -> F;

    /// Absorb this round's payload into the transcript. Must match the
    /// bytes the prover appended, including any label prefix.
    fn append_to_transcript(&self, transcript: &mut impl Transcript);
}

impl<F: Field> RoundProof<F> for UnivariatePoly<F> {
    fn degree(&self) -> usize {
        UnivariatePolynomial::degree(self)
    }

    fn check_sum(&self, running_sum: F, round: usize) -> Result<(), SumcheckError> {
        let sum =
            UnivariatePoly::evaluate(self, F::zero()) + UnivariatePoly::evaluate(self, F::one());
        if sum != running_sum {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: format!("{running_sum}"),
                actual: format!("{sum}"),
            });
        }
        Ok(())
    }

    fn evaluate(&self, challenge: F) -> F {
        UnivariatePoly::evaluate(self, challenge)
    }

    fn append_to_transcript(&self, transcript: &mut impl Transcript) {
        for coeff in self.coefficients() {
            coeff.append_to_transcript(transcript);
        }
    }
}

/// Round polynomial paired with a Fiat-Shamir domain-separation label.
///
/// On absorb: prepends `LabelWithCount(label, coeffs.len())` then the
/// coefficients. Degree, sum check, and evaluation delegate to the inner
/// [`UnivariatePoly`].
pub struct LabeledRoundPoly<'a, F: Field> {
    poly: &'a UnivariatePoly<F>,
    label: &'static [u8],
}

impl<'a, F: Field> LabeledRoundPoly<'a, F> {
    pub fn new(poly: &'a UnivariatePoly<F>, label: &'static [u8]) -> Self {
        Self { poly, label }
    }
}

impl<F: Field> RoundProof<F> for LabeledRoundPoly<'_, F> {
    fn degree(&self) -> usize {
        <UnivariatePoly<F> as RoundProof<F>>::degree(self.poly)
    }

    fn check_sum(&self, running_sum: F, round: usize) -> Result<(), SumcheckError> {
        <UnivariatePoly<F> as RoundProof<F>>::check_sum(self.poly, running_sum, round)
    }

    fn evaluate(&self, challenge: F) -> F {
        <UnivariatePoly<F> as RoundProof<F>>::evaluate(self.poly, challenge)
    }

    fn append_to_transcript(&self, transcript: &mut impl Transcript) {
        let coeffs = self.poly.coefficients();
        transcript.append(&LabelWithCount(self.label, coeffs.len() as u64));
        for coeff in coeffs {
            coeff.append_to_transcript(transcript);
        }
    }
}

/// Compressed round polynomial with label. Wire format omits the linear
/// coefficient `c_1`; the verifier recovers it from the sum-check invariant
/// `running_sum = s(0) + s(1) = 2·c_0 + c_1 + c_2 + … + c_d`.
///
/// Construct over an already-full [`UnivariatePoly`]; compression affects
/// only transcript absorption.
///
/// The length check (`>= 2` coefficients) lives in [`Self::check_sum`],
/// which the verifier calls before [`Self::append_to_transcript`].
pub struct CompressedLabeledRoundPoly<'a, F: Field> {
    poly: &'a UnivariatePoly<F>,
    label: &'static [u8],
}

impl<'a, F: Field> CompressedLabeledRoundPoly<'a, F> {
    pub fn new(poly: &'a UnivariatePoly<F>, label: &'static [u8]) -> Self {
        Self { poly, label }
    }
}

impl<F: Field> RoundProof<F> for CompressedLabeledRoundPoly<'_, F> {
    fn degree(&self) -> usize {
        <UnivariatePoly<F> as RoundProof<F>>::degree(self.poly)
    }

    fn check_sum(&self, running_sum: F, round: usize) -> Result<(), SumcheckError> {
        let coeffs = self.poly.coefficients();
        if coeffs.len() < 2 {
            return Err(SumcheckError::CompressedPolynomialTooShort {
                round,
                got: coeffs.len(),
            });
        }
        <UnivariatePoly<F> as RoundProof<F>>::check_sum(self.poly, running_sum, round)
    }

    fn evaluate(&self, challenge: F) -> F {
        <UnivariatePoly<F> as RoundProof<F>>::evaluate(self.poly, challenge)
    }

    fn append_to_transcript(&self, transcript: &mut impl Transcript) {
        let coeffs = self.poly.coefficients();
        transcript.append(&LabelWithCount(self.label, (coeffs.len() - 1) as u64));
        coeffs[0].append_to_transcript(transcript);
        for c in coeffs.iter().skip(2) {
            c.append_to_transcript(transcript);
        }
    }
}
