use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Validate,
};
use std::io::{Read, Write};

pub const MAX_JOLT_COMMITMENTS: usize = 1 << 12;
pub const MAX_OPENING_CLAIMS: usize = 1 << 16;
pub const MAX_SUMCHECK_ROUNDS: usize = 1 << 16;
pub const MAX_UNIPOLY_COEFFS: usize = 1 << 16;
pub const MAX_BLINDFOLD_VECTOR_LEN: usize = 1 << 20;

pub fn serialize_vec_with_len<T: CanonicalSerialize, W: Write>(
    values: &[T],
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    values.len().serialize_with_mode(&mut writer, compress)?;
    for value in values {
        value.serialize_with_mode(&mut writer, compress)?;
    }
    Ok(())
}

pub fn serialized_vec_with_len_size<T: CanonicalSerialize>(
    values: &[T],
    compress: Compress,
) -> usize {
    values.len().serialized_size(compress)
        + values
            .iter()
            .map(|value| value.serialized_size(compress))
            .sum::<usize>()
}

pub fn deserialize_bounded_len<R: Read>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
    max_len: usize,
) -> Result<usize, SerializationError> {
    let len = usize::deserialize_with_mode(&mut reader, compress, validate)?;
    if len > max_len {
        return Err(SerializationError::InvalidData);
    }
    Ok(len)
}

pub fn deserialize_bounded_vec<T: CanonicalDeserialize, R: Read>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
    max_len: usize,
) -> Result<Vec<T>, SerializationError> {
    let len = deserialize_bounded_len(&mut reader, compress, validate, max_len)?;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(T::deserialize_with_mode(&mut reader, compress, validate)?);
    }
    Ok(values)
}

pub fn deserialize_bounded_u32_len<R: Read>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
    max_len: usize,
) -> Result<usize, SerializationError> {
    let len = u32::deserialize_with_mode(&mut reader, compress, validate)? as usize;
    if len > max_len {
        return Err(SerializationError::InvalidData);
    }
    Ok(len)
}
