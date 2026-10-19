//! Host encodings for field-inline guests.

use jolt_field::JoltField;
use postcard::Error;

/// The canonical little-endian `u64` limbs of `value`, zero-extended to `N`
/// limbs: the form `jolt::field_from_limbs!` imports and
/// `jolt::field_to_limbs!` reads out. `None` when the value does not fit in
/// `N` limbs, or `N` is zero.
pub fn canonical_limbs<F: JoltField, const N: usize>(value: F) -> Option<[u64; N]> {
    let bytes = value.to_bytes_le_vec();
    if N == 0 || bytes.chunks(8).skip(N).flatten().any(|&byte| byte != 0) {
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
/// value independently using field instructions. Values wider than 256 bits
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

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};

    #[test]
    fn limb_width_depends_on_the_value() {
        assert_eq!(canonical_limbs::<Fr, 0>(Fr::from_u64(0)), None);
        assert_eq!(canonical_limbs::<Fr, 1>(Fr::from_u64(0)), Some([0]));
        assert_eq!(canonical_limbs::<Fr, 1>(Fr::from_u64(42)), Some([42]));
        assert_eq!(
            canonical_limbs::<Fr, 1>(Fr::from_u64(u64::MAX)),
            Some([u64::MAX])
        );
        let radix = Fr::from_u128(1u128 << 64);
        assert_eq!(canonical_limbs::<Fr, 1>(radix), None);
        assert_eq!(canonical_limbs::<Fr, 2>(radix), Some([0, 1]));
        let high = Fr::from_u128(1u128 << 127);
        assert_eq!(canonical_limbs::<Fr, 1>(high), None);
        assert_eq!(canonical_limbs::<Fr, 4>(high), Some([0, 1u64 << 63, 0, 0]));
        assert_eq!(canonical_limbs::<Fr, 2>(-Fr::from_u64(1)), None);
    }
}
