use crate::CryptoError;
use jolt_field::{CanonicalEncoding, Field};

/// Exact field modulus, padded to 32-byte big-endian form.
///
/// Uses the identity that the canonical form of `-1` in `Z_q` is `q - 1`.
///
/// # Errors
///
/// Returns [`CryptoError::InvalidSetup`] if the modulus exceeds 256 bits.
pub fn field_modulus_be_bytes<F: Field + CanonicalEncoding>() -> Result<[u8; 32], CryptoError> {
    let too_wide =
        || CryptoError::InvalidSetup("field modulus exceeds the 256-bit descriptor bound".into());
    if F::NUM_BYTES == 0 {
        return Err(too_wide());
    }
    let mut little_endian = [0u8; 32];
    let canonical = little_endian.get_mut(..F::NUM_BYTES).ok_or_else(too_wide)?;
    (-F::one()).to_bytes_le(canonical);
    let mut carry = 1u16;
    for byte in canonical {
        let sum = u16::from(*byte) + carry;
        *byte = sum as u8;
        carry = sum >> 8;
        if carry == 0 {
            break;
        }
    }
    let modulus_bytes = if carry == 0 {
        F::NUM_BYTES
    } else {
        *little_endian.get_mut(F::NUM_BYTES).ok_or_else(too_wide)? = carry as u8;
        F::NUM_BYTES + 1
    };
    let mut output = [0u8; 32];
    let start = output.len() - modulus_bytes;
    for (destination, source) in output
        .iter_mut()
        .skip(start)
        .zip(little_endian.iter().take(modulus_bytes).rev())
    {
        *destination = *source;
    }
    Ok(output)
}
