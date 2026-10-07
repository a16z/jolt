//! Round polynomials on the wire.
//!
//! A round polynomial of degree bound `d` always travels at full width, so its
//! shape is fixed by public parameters: the full form sends the `d + 1`
//! coefficients, and the compressed Boolean-hypercube form sends the `d`
//! coefficients `[c0, c2, ..., cd]`, from which the verifier recovers `c1`
//! through `s(0) + s(1) = running_sum`. A polynomial of lower degree is padded
//! with zero coefficients.

use jolt_field::{CanonicalEncoding, Field};
use jolt_poly::{CompressedPoly, UnivariatePoly, UnivariatePolynomial};
use jolt_transcript::{ProverTranscript, Sponge, VerifierTranscript};

use crate::error::SumcheckError;

/// `poly`'s coefficients padded with zeros to `degree + 1`.
///
/// # Errors
///
/// [`SumcheckError::DegreeBoundExceeded`] if `poly` exceeds `degree`.
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

/// Sends `poly` as a full round message of degree bound `degree`.
///
/// # Errors
///
/// [`SumcheckError::DegreeBoundExceeded`] if `poly` exceeds `degree`.
pub fn send_full_round<F, H>(
    poly: &UnivariatePoly<F>,
    degree: usize,
    transcript: &mut ProverTranscript<H>,
) -> Result<(), SumcheckError<F>>
where
    F: Field + CanonicalEncoding,
    H: Sponge,
{
    transcript.send_all(&padded_coefficients(poly, degree)?);
    Ok(())
}

/// Sends `poly` as a compressed Boolean-hypercube round message of degree
/// bound `degree >= 1`, omitting the linear coefficient.
///
/// # Errors
///
/// [`SumcheckError::DegreeBoundExceeded`] if `poly` exceeds `degree`.
pub fn send_compressed_round<F, H>(
    poly: &UnivariatePoly<F>,
    degree: usize,
    transcript: &mut ProverTranscript<H>,
) -> Result<(), SumcheckError<F>>
where
    F: Field + CanonicalEncoding,
    H: Sponge,
{
    let padded = padded_coefficients(poly, degree.max(1))?;
    let compressed = UnivariatePoly::new(padded).compress();
    transcript.send_all(compressed.coeffs_except_linear_term());
    Ok(())
}

/// Receives a full round message of degree bound `degree`.
///
/// # Errors
///
/// [`SumcheckError::Transcript`] if the proof does not hold `degree + 1`
/// canonical coefficients.
pub fn receive_full_round<F, H>(
    degree: usize,
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<UnivariatePoly<F>, SumcheckError<F>>
where
    F: Field + CanonicalEncoding,
    H: Sponge,
{
    let width = degree
        .checked_add(1)
        .ok_or(SumcheckError::DegreeOverflow { degree })?;
    Ok(UnivariatePoly::new(transcript.receive_n(width)?))
}

/// Receives a compressed Boolean-hypercube round message of degree bound
/// `degree >= 1`.
///
/// # Errors
///
/// [`SumcheckError::Transcript`] if the proof does not hold `degree` canonical
/// coefficients.
pub fn receive_compressed_round<F, H>(
    degree: usize,
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<CompressedPoly<F>, SumcheckError<F>>
where
    F: Field + CanonicalEncoding,
    H: Sponge,
{
    Ok(CompressedPoly::new(transcript.receive_n(degree.max(1))?))
}
