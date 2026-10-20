//! Round polynomials on the wire.
//!
//! A round polynomial of degree bound `d` always travels at full width, so its
//! shape is fixed by public parameters: the full form sends the `d + 1`
//! coefficients, and the compressed Boolean-hypercube form sends the `d`
//! coefficients `[c0, c2, ..., cd]`, from which the verifier recovers `c1`
//! through `s(0) + s(1) = running_sum`. A polynomial of lower degree is padded
//! with zero coefficients.

use jolt_algebra::{CanonicalEncoding, Field};
use jolt_poly::{CompressedPoly, UnivariatePoly, UnivariatePolynomial};
use jolt_transcript::{
    PadSlots, PrivateChannel, ProverTranscript, Sponge, TranscriptError, VerifierTranscript,
};

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
    let _: PadSlots = transcript.exchange_private_base(&mut padded_coefficients(poly, degree)?)?;
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
    let mut wire = UnivariatePoly::new(padded)
        .compress()
        .coeffs_except_linear_term()
        .to_vec();
    let _: PadSlots = transcript.exchange_private_base(&mut wire)?;
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
    Ok(UnivariatePoly::new(receive_private(width, transcript)?))
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
    Ok(CompressedPoly::new(receive_private(
        degree.max(1),
        transcript,
    )?))
}

/// Receives `count` private field messages; `count` comes from the public
/// degree bound.
fn receive_private<F, H>(
    count: usize,
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<Vec<F>, SumcheckError<F>>
where
    F: Field + CanonicalEncoding,
    H: Sponge,
{
    let mut values = vec![F::zero(); count];
    let _: PadSlots = transcript.exchange_private_base(&mut values)?;
    Ok(values)
}

/// Sends one private field message, such as a claimed evaluation.
///
/// # Errors
///
/// As [`PrivateChannel::exchange_private_base`]; clear mode never fails.
pub fn send_private_value<F, H>(
    value: F,
    transcript: &mut ProverTranscript<H>,
) -> Result<PadSlots, TranscriptError>
where
    F: Field + CanonicalEncoding,
    H: Sponge,
{
    transcript.exchange_private_base(&mut [value])
}

/// Receives one private field message and the pad slots it used.
///
/// # Errors
///
/// [`TranscriptError`] if the proof does not hold a canonical value.
pub fn receive_private_value<F, H>(
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<(F, PadSlots), TranscriptError>
where
    F: Field + CanonicalEncoding,
    H: Sponge,
{
    let mut value = [F::zero()];
    let pads = transcript.exchange_private_base(&mut value)?;
    let [value] = value;
    Ok((value, pads))
}
