use jolt_field::Field;
use jolt_poly::{UnivariatePoly, UnivariatePolynomial};
use jolt_transcript::{Encoding, PrivateChannel, Transcript};

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

/// A round polynomial absorbs as one public `u64` count word, then its
/// coefficients as one private message. The prover chooses the degree up to
/// the bound, so the count word keeps a longer polynomial from absorbing like
/// a shorter one. Uni-skip and standard rounds absorb alike; their position
/// tells them apart.
impl<F: Field + Encoding> RoundMessage for UnivariatePoly<F> {
    fn degree(&self) -> usize {
        UnivariatePolynomial::degree(self)
    }

    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        let coeffs = self.coefficients();
        transcript.append(&(coeffs.len() as u64));
        transcript.append_private(coeffs);
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
/// public `u64` count word, then the coefficients as one private message.
/// The prover's recorder, the verifier's compressed-proof loop and the
/// BlindFold prover all call this, so the encoding has one owner.
pub fn append_compressed_round<F: Field + Encoding, T: Transcript>(transcript: &mut T, sent: &[F]) {
    transcript.append(&(sent.len() as u64));
    transcript.append_private(sent);
}

/// `poly`'s coefficients padded with zeros to the `degree + 1` of its degree
/// bound: the fixed-width form a masked round sends, since a masked message's
/// length must not depend on the witness.
///
/// # Errors
///
/// [`SumcheckError::DegreeOverflow`] if `degree + 1` overflows, and
/// [`SumcheckError::DegreeBoundExceeded`] if `poly` has more coefficients.
pub fn padded_coefficients<F: Field>(
    poly: &UnivariatePoly<F>,
    degree: usize,
) -> Result<Vec<F>, SumcheckError<F>> {
    let width = degree
        .checked_add(1)
        .ok_or(SumcheckError::DegreeOverflow { degree })?;
    let coefficients = poly.coefficients();
    if coefficients.len() > width {
        return Err(SumcheckError::DegreeBoundExceeded {
            got: UnivariatePolynomial::degree(poly),
            max: degree,
        });
    }
    let mut padded = coefficients.to_vec();
    padded.resize(width, F::zero());
    Ok(padded)
}

/// The fixed-width compressed form of a round of degree bound `degree`:
/// `[c0, c2, …, c_degree]`, the linear coefficient omitted and nothing
/// trimmed.
///
/// # Errors
///
/// As [`padded_coefficients`].
pub fn compressed_wire<F: Field>(
    poly: &UnivariatePoly<F>,
    degree: usize,
) -> Result<Vec<F>, SumcheckError<F>> {
    let padded = padded_coefficients(poly, degree.max(1))?;
    Ok(UnivariatePoly::new(padded)
        .compress()
        .coeffs_except_linear_term()
        .to_vec())
}
