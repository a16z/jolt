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

/// Absorbs a compressed round's sent coefficients `[c0, c2, ..., cd]`: one
/// `u64` count word, then the coefficients as one message zero-padded to a
/// word. The prover's recorder, the verifier's compressed-proof loop and the
/// BlindFold prover all call this, so the encoding has one owner.
pub fn append_compressed_round<F: Encoding, T: Transcript>(transcript: &mut T, sent: &[F]) {
    transcript.append(&(sent.len() as u64));
    transcript.append_values(sent);
}
