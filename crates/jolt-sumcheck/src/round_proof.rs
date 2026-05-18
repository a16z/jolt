//! Per-round sumcheck messages.

use jolt_field::Field;
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

use crate::error::SumcheckError;
use crate::scalar::SumcheckScalar;

/// Common interface for one sumcheck round message.
pub trait RoundMessage {
    fn degree(&self) -> usize;

    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T);
}

/// A round message whose polynomial is available to the verifier.
pub trait ClearRound<F: SumcheckScalar>: RoundMessage {
    fn evaluate(&self, challenge: F) -> F;

    fn check_round_well_formed(&self, _round: usize) -> Result<(), SumcheckError<F>> {
        Ok(())
    }
}

impl<F: Field> RoundMessage for UnivariatePoly<F> {
    fn degree(&self) -> usize {
        UnivariatePolynomial::degree(self)
    }

    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        for coeff in self.coefficients() {
            coeff.append_to_transcript(transcript);
        }
    }
}

impl<F: Field> ClearRound<F> for UnivariatePoly<F> {
    fn evaluate(&self, challenge: F) -> F {
        UnivariatePoly::evaluate(self, challenge)
    }
}

/// Round polynomial paired with a Fiat-Shamir domain-separation label.
pub struct LabeledRoundPoly<'a, F: Field> {
    poly: &'a UnivariatePoly<F>,
    label: &'static [u8],
}

impl<'a, F: Field> LabeledRoundPoly<'a, F> {
    pub fn new(poly: &'a UnivariatePoly<F>, label: &'static [u8]) -> Self {
        Self { poly, label }
    }
}

impl<F: Field> RoundMessage for LabeledRoundPoly<'_, F> {
    fn degree(&self) -> usize {
        <UnivariatePoly<F> as RoundMessage>::degree(self.poly)
    }

    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let coeffs = self.poly.coefficients();
        transcript.append(&LabelWithCount(self.label, coeffs.len() as u64));
        for coeff in coeffs {
            coeff.append_to_transcript(transcript);
        }
    }
}

impl<F: Field> ClearRound<F> for LabeledRoundPoly<'_, F> {
    fn evaluate(&self, challenge: F) -> F {
        <UnivariatePoly<F> as ClearRound<F>>::evaluate(self.poly, challenge)
    }
}

/// Compressed round polynomial with label. Wire format omits the linear
/// coefficient `c_1`; the verifier recovers it from the sum-check invariant
/// `running_sum = s(0) + s(1) = 2·c_0 + c_1 + c_2 + … + c_d`.
pub struct CompressedLabeledRoundPoly<'a, F: Field> {
    poly: &'a UnivariatePoly<F>,
    label: &'static [u8],
}

impl<'a, F: Field> CompressedLabeledRoundPoly<'a, F> {
    pub fn new(poly: &'a UnivariatePoly<F>, label: &'static [u8]) -> Self {
        Self { poly, label }
    }
}

impl<F: Field> RoundMessage for CompressedLabeledRoundPoly<'_, F> {
    fn degree(&self) -> usize {
        <UnivariatePoly<F> as RoundMessage>::degree(self.poly)
    }

    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let coeffs = self.poly.coefficients();
        transcript.append(&LabelWithCount(self.label, (coeffs.len() - 1) as u64));
        coeffs[0].append_to_transcript(transcript);
        for c in coeffs.iter().skip(2) {
            c.append_to_transcript(transcript);
        }
    }
}

impl<F: Field> ClearRound<F> for CompressedLabeledRoundPoly<'_, F> {
    fn evaluate(&self, challenge: F) -> F {
        <UnivariatePoly<F> as ClearRound<F>>::evaluate(self.poly, challenge)
    }

    fn check_round_well_formed(&self, round: usize) -> Result<(), SumcheckError<F>> {
        let coeffs = self.poly.coefficients();
        if coeffs.len() < 2 {
            return Err(SumcheckError::CompressedPolynomialTooShort {
                round,
                got: coeffs.len(),
            });
        }
        Ok(())
    }
}
