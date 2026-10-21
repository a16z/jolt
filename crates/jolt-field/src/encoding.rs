//! spongefish [`Encoding`] for the field elements: their canonical
//! little-endian bytes, as [`CanonicalBytes::to_bytes_le`] writes them.
//!
//! Each encoding is fixed-width and canonical, so it is injective and a run
//! of elements of one field is self-delimiting.

use spongefish::Encoding;

use crate::CanonicalBytes;

/// The `N` canonical bytes of `value`; `N` is its `NUM_BYTES`.
fn fixed<const N: usize, T: CanonicalBytes>(value: &T) -> [u8; N] {
    let mut out = [0u8; N];
    value.to_bytes_le(&mut out);
    out
}

macro_rules! fixed_width {
    ($([$($generics:tt)*] $field:ty => $bytes:expr),+ $(,)?) => {$(
        impl<$($generics)*> Encoding for $field {
            fn encode(&self) -> impl AsRef<[u8]> {
                fixed::<{ $bytes }, Self>(self)
            }
        }
    )+};
}

#[cfg(feature = "bn254")]
fixed_width!([] crate::Fr => 32, [] crate::Fq => 32);

#[cfg(feature = "solinas")]
fixed_width!(
    [const P: u32] crate::Fp32<P> => 4,
    [const P: u64] crate::Fp64<P> => 8,
    [const P: u128] crate::Fp128<P> => 16,
);

#[cfg(feature = "binary")]
fixed_width!(
    [] crate::F8 => 1,
    [] crate::F64 => 8,
    [] crate::F128 => 16,
    [] crate::F192 => 24,
);

#[cfg(feature = "solinas")]
impl<F: crate::Field + CanonicalBytes, C: crate::Ext2Config<F>> Encoding for crate::FpExt2<F, C> {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.to_bytes_le_vec()
    }
}

#[cfg(feature = "solinas")]
impl<F: crate::PseudoMersenne> Encoding for crate::FpExt4<F> {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.to_bytes_le_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::Encoding;
    use crate::CanonicalBytes;

    fn matches_canonical_bytes<T: Encoding + CanonicalBytes>(value: &T) {
        assert_eq!(value.encode().as_ref(), value.to_bytes_le_vec());
    }

    #[test]
    fn encodings_are_the_canonical_bytes() {
        #[cfg(feature = "bn254")]
        {
            use crate::Ring;
            matches_canonical_bytes(&crate::Fr::from_u64(0x0102_0304));
            matches_canonical_bytes(&crate::Fq::from_u64(7));
        }
        #[cfg(feature = "solinas")]
        {
            use crate::Ring;
            matches_canonical_bytes(&crate::Prime31Offset19::from_u64(5));
            matches_canonical_bytes(&crate::Prime64Offset59::from_u64(5));
            matches_canonical_bytes(&crate::Prime128OffsetA7F7::from_u64(5));
        }
        #[cfg(feature = "binary")]
        {
            use crate::Ring;
            matches_canonical_bytes(&crate::F8::from_u64(3));
            matches_canonical_bytes(&crate::F192::from_u64(3));
        }
    }
}
