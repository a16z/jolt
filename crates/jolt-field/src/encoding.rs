//! spongefish [`Encoding`] for the field elements: their canonical
//! little-endian bytes, as [`CanonicalBytes::to_bytes_le`] writes them.
//!
//! Each encoding is fixed-width and canonical, so it is injective and a run
//! of elements of one field is self-delimiting.

use spongefish::Encoding;

use crate::CanonicalBytes;
#[cfg(feature = "solinas")]
use crate::{Ext2Config, Field, Fp128, Fp32, Fp64, FpExt2, FpExt4, PseudoMersenne};
#[cfg(feature = "bn254")]
use crate::{Fq, Fr};
#[cfg(feature = "binary")]
use crate::{F128, F192, F64, F8};

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
fixed_width!([] Fr => 32, [] Fq => 32);

#[cfg(feature = "solinas")]
fixed_width!(
    [const P: u32] Fp32<P> => 4,
    [const P: u64] Fp64<P> => 8,
    [const P: u128] Fp128<P> => 16,
);

#[cfg(feature = "binary")]
fixed_width!(
    [] F8 => 1,
    [] F64 => 8,
    [] F128 => 16,
    [] F192 => 24,
);

#[cfg(feature = "solinas")]
impl<F: Field + CanonicalBytes, C: Ext2Config<F>> Encoding for FpExt2<F, C> {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.to_bytes_le_vec()
    }
}

#[cfg(feature = "solinas")]
impl<F: PseudoMersenne> Encoding for FpExt4<F> {
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
            use crate::{Fq, Fr, Ring};
            matches_canonical_bytes(&Fr::from_u64(0x0102_0304));
            matches_canonical_bytes(&Fq::from_u64(7));
        }
        #[cfg(feature = "solinas")]
        {
            use crate::{Prime128OffsetA7F7, Prime31Offset19, Prime64Offset59, Ring};
            matches_canonical_bytes(&Prime31Offset19::from_u64(5));
            matches_canonical_bytes(&Prime64Offset59::from_u64(5));
            matches_canonical_bytes(&Prime128OffsetA7F7::from_u64(5));
        }
        #[cfg(feature = "binary")]
        {
            use crate::{Ring, F192, F8};
            matches_canonical_bytes(&F8::from_u64(3));
            matches_canonical_bytes(&F192::from_u64(3));
        }
    }
}
