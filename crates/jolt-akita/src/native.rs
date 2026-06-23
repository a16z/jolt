use std::io::Cursor;

use akita_pcs::{AkitaDeserialize, AkitaSerialize};
use jolt_field::{CanonicalBytes, FixedByteSize};
use jolt_openings::OpeningsError;
use jolt_poly::{MultilinearPoly, Polynomial};

use crate::types::{jolt_to_akita_index, AkitaField, NativeDensePoly};

pub(crate) fn dense_polynomials(
    polynomials: &[Polynomial<AkitaField>],
) -> Result<Vec<NativeDensePoly>, OpeningsError> {
    polynomials
        .iter()
        .map(|poly| {
            let evals = jolt_to_akita_evals(poly.num_vars(), poly.evals())?;
            NativeDensePoly::from_field_evals(poly.num_vars(), &evals).map_err(akita_error)
        })
        .collect()
}

pub(crate) fn jolt_to_akita_evals(
    num_vars: usize,
    jolt_evals: &[AkitaField],
) -> Result<Vec<AkitaField>, OpeningsError> {
    let Some(expected) = 1usize.checked_shl(num_vars as u32) else {
        return Err(invalid_batch(format!(
            "Akita polynomial dimension {num_vars} exceeds usize bit width"
        )));
    };
    if jolt_evals.len() != expected {
        return Err(invalid_batch(format!(
            "Akita polynomial has {} evaluations but dimension {num_vars} requires {expected}",
            jolt_evals.len()
        )));
    }
    if num_vars == 0 {
        return Ok(jolt_evals.to_vec());
    }
    let mut akita_evals = vec![AkitaField::zero(); jolt_evals.len()];
    for (jolt_index, &eval) in jolt_evals.iter().enumerate() {
        let akita_index = jolt_to_akita_index(num_vars, jolt_index);
        akita_evals[akita_index] = eval;
    }
    Ok(akita_evals)
}

pub(crate) fn polynomial_evaluations<P>(polynomial: &P) -> Vec<AkitaField>
where
    P: MultilinearPoly<AkitaField> + ?Sized,
{
    let capacity = if polynomial.num_vars() < usize::BITS as usize {
        1usize << polynomial.num_vars()
    } else {
        0
    };
    let mut evals = Vec::with_capacity(capacity);
    polynomial.for_each_row(polynomial.num_vars(), &mut |_, row| {
        evals.extend_from_slice(row);
    });
    evals
}

pub(crate) fn serialize_akita<T>(value: &T) -> Result<Vec<u8>, OpeningsError>
where
    T: AkitaSerialize,
{
    let mut bytes = Vec::with_capacity(value.compressed_size());
    value
        .serialize_compressed(&mut bytes)
        .map_err(akita_error)?;
    Ok(bytes)
}

pub(crate) fn deserialize_akita<T>(bytes: &[u8], ctx: &T::Context) -> Result<T, OpeningsError>
where
    T: AkitaDeserialize,
{
    T::deserialize_compressed(Cursor::new(bytes), ctx).map_err(akita_error)
}

pub(crate) fn field_bytes(value: AkitaField) -> Vec<u8> {
    let mut bytes = vec![0u8; AkitaField::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    bytes
}

pub(crate) fn invalid_batch(message: impl Into<String>) -> OpeningsError {
    OpeningsError::InvalidBatch(message.into())
}

pub(crate) fn akita_error(error: impl ToString) -> OpeningsError {
    OpeningsError::InvalidBatch(error.to_string())
}

pub(crate) fn transparent_zk_error() -> OpeningsError {
    OpeningsError::InvalidBatch(
        "Akita native adapter is transparent-only and does not support ZK openings yet".to_owned(),
    )
}
