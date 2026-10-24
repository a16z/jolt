use jolt_field::Field;
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use jolt_transcript::{Encoding, Transcript};

use crate::error::SumcheckError;

pub trait RoundMessage {
    fn degree(&self) -> usize;

    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T);
}

/// A round message whose polynomial is available to the verifier.
pub trait ClearRound<F: Field>: RoundMessage {
    fn evaluate(&self, challenge: F) -> F;

    fn coefficient_linear_combination(&self, coefficients: &[F]) -> F;

    fn check_round_well_formed(&self, _round: usize) -> Result<(), SumcheckError<F>> {
        Ok(())
    }
}

/// A round polynomial absorbs as one `u64` count word, then its
/// coefficients. The prover chooses the degree up to the bound, so the count
/// word keeps a longer polynomial from absorbing like a shorter one. Uni-skip
/// and standard rounds absorb alike; their position tells them apart.
impl<F: Field + Encoding> RoundMessage for UnivariatePoly<F> {
    fn degree(&self) -> usize {
        UnivariatePolynomial::degree(self)
    }

    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let coeffs = self.coefficients();
        transcript.append(&(coeffs.len() as u64));
        transcript.append_values(coeffs);
    }
}

impl<F: Field + Encoding> ClearRound<F> for UnivariatePoly<F> {
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

/// Compressed round polynomial. Wire format omits the linear coefficient
/// `c_1`; the verifier recovers it from the sum-check invariant
/// `running_sum = s(0) + s(1) = 2·c_0 + c_1 + c_2 + … + c_d`. It absorbs one
/// `u64` count word for the sent coefficients, then those coefficients.
pub struct CompressedRoundPoly<'a, F: Field> {
    poly: &'a UnivariatePoly<F>,
}

impl<'a, F: Field> CompressedRoundPoly<'a, F> {
    pub fn new(poly: &'a UnivariatePoly<F>) -> Self {
        Self { poly }
    }
}

impl<F: Field + Encoding> RoundMessage for CompressedRoundPoly<'_, F> {
    fn degree(&self) -> usize {
        <UnivariatePoly<F> as RoundMessage>::degree(self.poly)
    }

    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let coeffs = self.poly.coefficients();
        // An empty round polynomial would absorb nothing (not even the
        // count word) and silently desynchronize the prover and verifier
        // transcripts; every construction path produces >= 2 coefficients.
        debug_assert!(!coeffs.is_empty(), "round polynomial has no coefficients");
        let Some((constant, rest)) = coeffs.split_first() else {
            return;
        };
        transcript.append(&((coeffs.len() - 1) as u64));
        transcript.append(constant);
        for c in rest.iter().skip(1) {
            transcript.append(c);
        }
    }
}

impl<F: Field + Encoding> ClearRound<F> for CompressedRoundPoly<'_, F> {
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
