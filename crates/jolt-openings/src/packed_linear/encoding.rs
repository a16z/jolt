use jolt_field::Field;

use crate::OpeningsError;

pub(super) fn field_bytes<F>(value: F) -> Vec<u8>
where
    F: Field,
{
    value.to_bytes_le_vec()
}

pub(super) fn field_from_bytes<F>(bytes: &[u8]) -> Result<F, OpeningsError>
where
    F: Field,
{
    if bytes.len() != F::NUM_BYTES {
        return Err(OpeningsError::InvalidBatch(format!(
            "packed linear proof field encoding has {} bytes but expected {}",
            bytes.len(),
            F::NUM_BYTES
        )));
    }
    let value = F::from_le_bytes_mod_order(bytes);
    if value.to_bytes_le_vec() != bytes {
        return Err(OpeningsError::InvalidBatch(
            "packed linear proof field encoding is not canonical".to_string(),
        ));
    }
    Ok(value)
}

pub(super) fn encode_round<F>(round: [F; 3]) -> [Vec<u8>; 3]
where
    F: Field,
{
    [
        field_bytes(round[0]),
        field_bytes(round[1]),
        field_bytes(round[2]),
    ]
}

pub(super) fn decode_round<F>(round: &[Vec<u8>; 3]) -> Result<[F; 3], OpeningsError>
where
    F: Field,
{
    Ok([
        field_from_bytes(&round[0])?,
        field_from_bytes(&round[1])?,
        field_from_bytes(&round[2])?,
    ])
}
