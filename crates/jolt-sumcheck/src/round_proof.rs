//! Per-round sumcheck messages.

use jolt_field::Field;
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use jolt_transcript::FsTranscript;

use crate::error::SumcheckError;

/// Degree of a sumcheck round message.
///
/// Field-agnostic supertrait of [`RoundMessage`]: commitment-backed round
/// messages report a degree without pinning a challenge field, which keeps
/// `degree()` unambiguous when the message type implements `RoundMessage<F>`
/// for more than one `F`.
pub trait RoundDegree {
    fn degree(&self) -> usize;
}

/// Common interface for one sumcheck round message that absorbs into a
/// Fiat-Shamir transcript over challenge field `F`.
pub trait RoundMessage<F: Field>: RoundDegree {
    fn append_to_transcript<T: FsTranscript<F>>(&self, transcript: &mut T);
}

/// A round message whose polynomial is available to the verifier.
pub trait ClearRound<F: Field>: RoundMessage<F> {
    fn evaluate(&self, challenge: F) -> F;

    fn coefficient_linear_combination(&self, coefficients: &[F]) -> F;

    fn check_round_well_formed(&self, _round: usize) -> Result<(), SumcheckError<F>> {
        Ok(())
    }
}

impl<F: Field> RoundDegree for UnivariatePoly<F> {
    fn degree(&self) -> usize {
        UnivariatePolynomial::degree(self)
    }
}

impl<F: Field> RoundMessage<F> for UnivariatePoly<F> {
    fn append_to_transcript<T: FsTranscript<F>>(&self, transcript: &mut T) {
        transcript.absorb_field_slice(self.coefficients());
    }
}

impl<F: Field> ClearRound<F> for UnivariatePoly<F> {
    fn evaluate(&self, challenge: F) -> F {
        UnivariatePoly::evaluate(self, challenge)
    }

    fn coefficient_linear_combination(&self, coefficients: &[F]) -> F {
        self.coefficients()
            .iter()
            .zip(coefficients)
            .map(|(&coefficient, &scale)| coefficient * scale)
            .sum()
    }
}

/// Borrowed round-polynomial wrapper. Rounds are domain-separated positionally
/// (and by the transcript's one-time `DomainSeparator`/instance), matching
/// jolt-core — no per-round label is absorbed.
pub struct LabeledRoundPoly<'a, F: Field> {
    poly: &'a UnivariatePoly<F>,
}

impl<'a, F: Field> LabeledRoundPoly<'a, F> {
    pub fn new(poly: &'a UnivariatePoly<F>) -> Self {
        Self { poly }
    }
}

impl<F: Field> RoundDegree for LabeledRoundPoly<'_, F> {
    fn degree(&self) -> usize {
        <UnivariatePoly<F> as RoundDegree>::degree(self.poly)
    }
}

impl<F: Field> RoundMessage<F> for LabeledRoundPoly<'_, F> {
    fn append_to_transcript<T: FsTranscript<F>>(&self, transcript: &mut T) {
        transcript.absorb_field_slice(self.poly.coefficients());
    }
}

impl<F: Field> ClearRound<F> for LabeledRoundPoly<'_, F> {
    fn evaluate(&self, challenge: F) -> F {
        <UnivariatePoly<F> as ClearRound<F>>::evaluate(self.poly, challenge)
    }

    fn coefficient_linear_combination(&self, coefficients: &[F]) -> F {
        <UnivariatePoly<F> as ClearRound<F>>::coefficient_linear_combination(
            self.poly,
            coefficients,
        )
    }
}

/// Compressed round polynomial with label. Wire format omits the linear
/// coefficient `c_1`; the verifier recovers it from the sum-check invariant
/// `running_sum = s(0) + s(1) = 2·c_0 + c_1 + c_2 + … + c_d`.
pub struct CompressedLabeledRoundPoly<'a, F: Field> {
    poly: &'a UnivariatePoly<F>,
}

impl<'a, F: Field> CompressedLabeledRoundPoly<'a, F> {
    pub fn new(poly: &'a UnivariatePoly<F>) -> Self {
        Self { poly }
    }
}

impl<F: Field> RoundDegree for CompressedLabeledRoundPoly<'_, F> {
    fn degree(&self) -> usize {
        <UnivariatePoly<F> as RoundDegree>::degree(self.poly)
    }
}

impl<F: Field> RoundMessage<F> for CompressedLabeledRoundPoly<'_, F> {
    fn append_to_transcript<T: FsTranscript<F>>(&self, transcript: &mut T) {
        // Absorb the compressed coefficients (linear term c1 omitted) as ONE message,
        // matching the verifier's `absorb_field_slice(coeffs_except_linear_term)`.
        let coeffs = self.poly.coefficients();
        let mut compressed = Vec::with_capacity(coeffs.len().saturating_sub(1));
        compressed.push(coeffs[0]);
        compressed.extend_from_slice(&coeffs[2..]);
        transcript.absorb_field_slice(&compressed);
    }
}

impl<F: Field> ClearRound<F> for CompressedLabeledRoundPoly<'_, F> {
    fn evaluate(&self, challenge: F) -> F {
        <UnivariatePoly<F> as ClearRound<F>>::evaluate(self.poly, challenge)
    }

    fn coefficient_linear_combination(&self, coefficients: &[F]) -> F {
        <UnivariatePoly<F> as ClearRound<F>>::coefficient_linear_combination(
            self.poly,
            coefficients,
        )
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
