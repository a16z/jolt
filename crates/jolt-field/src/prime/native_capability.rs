//! Native capability-trait impls for the prime fields: the canonical
//! byte/transcript surface and the `WithAccumulator` association (native
//! `NaiveAccumulator`).
//!
//! `FromPrimitiveInt`/`FieldCore` carry per-type logic and stay in the prime
//! modules; this module owns the shared derived-capability implementations used
//! directly by both Jolt and Akita.

use std::mem::size_of;

use super::{Fp128, Fp32, Fp64};
use crate::{
    CanonicalField, CanonicalRepr, Field, FieldCore, FromPrimitiveInt, NaiveAccumulator,
    WithAccumulator,
};

macro_rules! impl_prime_native_capability {
    ($ty:ident<$p:ident: $p_ty:ty>, $bytes:expr) => {
        impl<const $p: $p_ty> CanonicalRepr for $ty<$p> {
            const NUM_BYTES: usize = $bytes;

            #[inline(always)]
            fn to_bytes_le(&self, out: &mut [u8]) {
                assert_eq!(out.len(), <Self as CanonicalRepr>::NUM_BYTES);
                out.copy_from_slice(
                    &self.to_canonical_u128().to_le_bytes()[..<Self as CanonicalRepr>::NUM_BYTES],
                );
            }

            #[inline(always)]
            fn from_le_bytes_mod_order(bytes: &[u8]) -> Self {
                if bytes.len() <= size_of::<u128>() {
                    let mut padded = [0u8; size_of::<u128>()];
                    padded[..bytes.len()].copy_from_slice(bytes);
                    return <Self as FromPrimitiveInt>::from_u128(u128::from_le_bytes(padded));
                }

                reduce_le_bytes_mod_order(bytes)
            }

            #[inline]
            fn to_canonical_u64_checked(&self) -> Option<u64> {
                self.to_canonical_u128().try_into().ok()
            }

            #[inline]
            fn num_bits(&self) -> u32 {
                let value = self.to_canonical_u128();
                u128::BITS - value.leading_zeros()
            }
        }

        impl<const $p: $p_ty> WithAccumulator for $ty<$p> {
            type Accumulator = NaiveAccumulator<Self>;
        }

        impl<const $p: $p_ty> Field for $ty<$p> {}
    };
}

/// Horner reduction of arbitrary-length little-endian bytes modulo the field
/// order (the >16-byte path of `CanonicalRepr::from_le_bytes_mod_order`).
#[inline(always)]
fn reduce_le_bytes_mod_order<F: FieldCore + FromPrimitiveInt>(bytes: &[u8]) -> F {
    let base = F::from_u64(256);
    bytes.iter().rev().fold(F::zero(), |acc, &byte| {
        acc * base + F::from_u64(byte as u64)
    })
}

impl_prime_native_capability!(Fp32<P: u32>, 4);
impl_prime_native_capability!(Fp64<P: u64>, 8);
impl_prime_native_capability!(Fp128<P: u128>, 16);

#[cfg(test)]
mod tests {
    //! Native byte / transcript / accumulator capability tests.
    //!
    //! These exercise the Solinas backend directly, so they run under
    //! `--no-default-features --features solinas` as well as combined builds.
    use super::*;
    use crate::Accumulator;
    use crate::Prime128Offset275;

    /// Asserts the full canonical byte round-trip on the native traits.
    fn assert_native_byte_roundtrip<F, const N: usize>(value: F, expected: [u8; N])
    where
        F: CanonicalField + CanonicalRepr + std::fmt::Debug + Eq,
    {
        assert_eq!(<F as CanonicalRepr>::NUM_BYTES, N);

        // to_bytes_le (the audited method) into a correctly sized buffer, plus
        // the vec convenience wrapper — both must agree.
        let mut buf = [0u8; N];
        value.to_bytes_le(&mut buf);
        assert_eq!(buf, expected);
        assert_eq!(value.to_bytes_le_vec(), expected.to_vec());

        // Reducing / challenge constructors all invert the encoding.
        assert_eq!(F::from_le_bytes_mod_order(&buf), value);
        assert_eq!(F::from_challenge_bytes(&buf), value);

        assert_eq!(
            value.num_bits(),
            u128::BITS - value.to_canonical_u128().leading_zeros()
        );
    }

    #[test]
    fn prime_fields_native_byte_capabilities() {
        type F32 = Fp32<251>;
        type F64 = Fp64<4294967197>;
        type F128 = Prime128Offset275;

        assert_native_byte_roundtrip::<F32, 4>(F32::from_u64(42), 42u32.to_le_bytes());
        assert_native_byte_roundtrip::<F64, 8>(F64::from_u64(42), 42u64.to_le_bytes());
        assert_native_byte_roundtrip::<F128, 16>(
            F128::from_canonical_u128(0x0102_0304_0506_0708),
            0x0102_0304_0506_0708u128.to_le_bytes(),
        );

        // Reducing constructor on a short slice: 255 mod 251 == 4.
        assert_eq!(F32::from_le_bytes_mod_order(&[255, 0]), F32::from_u64(4));
        assert_eq!(F32::from_challenge_bytes(&[255, 0]), F32::from_u64(4));

        // Over-long slice (> 16 bytes) takes the Horner-reduction path; trailing
        // zero limbs must not change the value (and must not panic).
        let value = F128::from_canonical_u128(0x00DE_AD00_BEEF);
        let mut over_long = value.to_bytes_le_vec();
        over_long.extend_from_slice(&[0u8; 8]);
        assert!(over_long.len() > 16);
        assert_eq!(F128::from_le_bytes_mod_order(&over_long), value);

        // Bit-length + checked-u64 extraction.
        assert_eq!(F32::zero().num_bits(), 0);
        assert_eq!(F64::from_u64(7).to_canonical_u64_checked(), Some(7));
        assert_eq!(
            F128::from_canonical_u128(1u128 << 65).to_canonical_u64_checked(),
            None
        );
    }

    #[test]
    fn prime_fields_native_mul_capabilities() {
        type F32 = Fp32<251>;
        type F64 = Fp64<4294967197>;

        // mul_pow_2: 3 * 2^4 == 48.
        assert_eq!(F32::from_u64(3).mul_pow_2(4), F32::from_u64(48));
        assert_eq!(F64::from_u64(5).mul_pow_2(0), F64::from_u64(5));
        // Large shift still agrees with repeated doubling.
        let doubled = (0..40).fold(F64::from_u64(1), |acc, _| acc + acc);
        assert_eq!(F64::from_u64(1).mul_pow_2(40), doubled);

        // mul_u64 / mul_i64.
        assert_eq!(F64::from_u64(9).mul_u64(7), F64::from_u64(63));
        assert_eq!(F64::from_u64(9).mul_i64(-1), -F64::from_u64(9));
    }

    #[test]
    fn prime_fields_native_accumulator() {
        type F64 = Fp64<4294967197>;

        let mut acc = <F64 as WithAccumulator>::Accumulator::default();
        acc.fmadd(F64::from_u64(9), F64::from_u64(7));
        acc.add(F64::from_u64(2));
        assert_eq!(acc.reduce(), F64::from_u64(65));
    }
}
