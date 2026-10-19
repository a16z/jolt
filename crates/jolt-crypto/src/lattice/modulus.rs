/// Exact field modulus, padded to 32-byte big-endian form.
///
/// Uses the identity that the canonical form of `-1` in `Z_q` is `q - 1`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] if the modulus exceeds 256 bits.
pub fn field_modulus_be_bytes<F: Field + CanonicalEncoding>() -> Result<[u8; 32], AkitaError> {
    if F::NUM_BYTES == 0 || F::NUM_BYTES > 32 {
        return Err(AkitaError::InvalidSetup(
            "Akita field modulus exceeds the 256-bit descriptor bound".into(),
        ));
    }
    let mut little_endian = [0u8; 32];
    (-F::one()).to_bytes_le(&mut little_endian[..F::NUM_BYTES]);
    let mut carry = 1u16;
    for byte in &mut little_endian[..F::NUM_BYTES] {
        let sum = u16::from(*byte) + carry;
        *byte = sum as u8;
        carry = sum >> 8;
        if carry == 0 {
            break;
        }
    }
    let modulus_bytes = if carry == 0 {
        F::NUM_BYTES
    } else if F::NUM_BYTES < 32 {
        little_endian[F::NUM_BYTES] = carry as u8;
        F::NUM_BYTES + 1
    } else {
        return Err(AkitaError::InvalidSetup(
            "Akita field modulus exceeds the 256-bit descriptor bound".into(),
        ));
    };
    let mut output = [0u8; 32];
    let start = output.len() - modulus_bytes;
    for (destination, source) in output[start..]
        .iter_mut()
        .zip(little_endian[..modulus_bytes].iter().rev())
    {
        *destination = *source;
    }
    Ok(output)
}
