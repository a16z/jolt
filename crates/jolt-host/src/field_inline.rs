//! Host encodings for field-inline guests.

use jolt_field::JoltField;
use postcard::Error;

/// The canonical little-endian `u64` limbs of `value`, zero-extended to `N`
/// limbs: the form `jolt::field_from_limbs!` imports and
/// `jolt::field_to_limbs!` reads out. `None` when the field's canonical
/// encoding is wider than `N` limbs.
pub fn canonical_limbs<F: JoltField, const N: usize>(value: F) -> Option<[u64; N]> {
    let bytes = value.to_bytes_le_vec();
    if bytes.len() > 8 * N {
        return None;
    }
    let mut limbs = [0u64; N];
    for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks(8)) {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        *limb = u64::from_le_bytes(word);
    }
    Some(limbs)
}

/// Encode four `(r_i, x_i)` pairs and their native-field equality-polynomial
/// evaluation as four canonical little-endian u64 limbs. The guest checks this
/// value independently using field instructions. Fields wider than 256 bits
/// cannot be represented by the guest's input contract.
pub fn eqpoly_inputs<F: JoltField>(pairs: [[u64; 2]; 4]) -> Result<Vec<u8>, Error> {
    let value = pairs.iter().fold(F::one(), |acc, [r, x]| {
        let r = F::from_u64(*r);
        let x = F::from_u64(*x);
        acc * (r * x + (F::one() - r) * (F::one() - x))
    });
    let limbs = canonical_limbs::<F, 4>(value).ok_or(Error::SerializeBufferFull)?;
    let mut inputs = postcard::to_stdvec(&pairs)?;
    inputs.extend(postcard::to_stdvec(&limbs)?);
    Ok(inputs)
}
