//! BN254 backend: `#[repr(transparent)]` newtypes over arkworks decoupling
//! the public API from the arkworks types.
//!
//! Serde and transcript encodings are the 32-byte little-endian canonical
//! form (identical to jolt-field). Challenges are full-width: the default
//! wide reduction of [`CanonicalEncoding::from_challenge_bytes`].

mod mont;

pub use mont::{FrSignedProductAccumulator, FrSmallScalarAccumulator, WideAccumulator};

use crate::{CanonicalBytes, CanonicalEncoding, Field, NaiveAccumulator, Ring, WithAccumulator};
use ark_ff::{BigInt, BigInteger, PrimeField, UniformRand};
use rand_core::RngCore;

/// Bytes per chunk of [`reduce_le_bytes`]: 248 bits, below both 254-bit moduli.
const CHUNK_BYTES: usize = 31;

/// Reduces little-endian bytes modulo the order of a 254-bit BN254 field.
///
/// Horner's rule over 31-byte chunks, each below the modulus and so converted
/// directly. It returns what arkworks' `from_le_bytes_mod_order` returns, but
/// that folds in one byte at a time past the first 31 bytes, at two Montgomery
/// multiplications per byte; a 52-byte challenge costs about 42 of them there
/// and 3 here.
fn reduce_le_bytes<F: PrimeField<BigInt = BigInt<4>>>(bytes: &[u8]) -> F {
    // 2^248, one chunk.
    let shift = F::from(BigInt([0, 0, 0, 1 << 56]));
    let chunk_to_field = |chunk: &[u8]| {
        let mut limbs = [0u64; 4];
        for (i, &byte) in chunk.iter().enumerate() {
            limbs[i / 8] |= u64::from(byte) << (8 * (i % 8));
        }
        F::from(BigInt(limbs))
    };
    let mut chunks = bytes.chunks(CHUNK_BYTES).rev();
    let top = chunks.next().map_or_else(F::zero, chunk_to_field);
    chunks.fold(top, |acc, chunk| acc * shift + chunk_to_field(chunk))
}

macro_rules! from_primitives {
    ($ty:ident: $via:ident[$($prim:ty),*]) => {
        $(impl From<$prim> for $ty {
            #[inline(always)]
            fn from(v: $prim) -> Self {
                <$ty as Ring>::$via(v as _)
            }
        })*
    };
}

macro_rules! wrap_bn254 {
    ($(#[$doc:meta])* $ty:ident, $inner:ty, accumulators($accum:ty, $small_accum:ty, $signed_accum:ty)) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $ty(pub(crate) $inner);

        impl $ty {
            /// Access the internal Montgomery-form limbs.
            #[inline(always)]
            pub fn inner_limbs(self) -> [u64; 4] {
                (self.0).0 .0
            }
        }

        impl From<$inner> for $ty {
            #[inline(always)]
            fn from(inner: $inner) -> Self {
                $ty(inner)
            }
        }

        impl From<$ty> for $inner {
            #[inline(always)]
            fn from(wrapper: $ty) -> Self {
                wrapper.0
            }
        }

        from_primitives!($ty: from_u128[bool, u8, u16, u32, u64, u128]);
        from_primitives!($ty: from_i128[i8, i16, i32, i64, i128]);

        impl std::fmt::Debug for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(&self.0, f)
            }
        }

        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.0, f)
            }
        }

        $crate::impl_ring_ops!(impl[] $ty {
            add(a, b): $ty(a.0 + b.0),
            sub(a, b): $ty(a.0 - b.0),
            mul(a, b): $ty(a.0 * b.0),
            neg(a): $ty(-a.0),
            zero: $ty(<$inner as ::num_traits::Zero>::zero()),
            one: $ty(<$inner as ::num_traits::One>::one()),
        });

        impl std::ops::Div for $ty {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                $ty(self.0 / rhs.0)
            }
        }

        impl<'a> std::ops::Div<&'a $ty> for $ty {
            type Output = Self;
            #[inline]
            fn div(self, rhs: &'a $ty) -> Self {
                $ty(self.0 / rhs.0)
            }
        }

        impl Field for $ty {
            #[inline]
            fn inverse(&self) -> Option<Self> {
                <$inner as ark_ff::Field>::inverse(&self.0).map($ty)
            }

            #[inline]
            fn random<R: RngCore>(rng: &mut R) -> Self {
                $ty(<$inner as UniformRand>::rand(rng))
            }
        }

        impl CanonicalBytes for $ty {
            const NUM_BYTES: usize = 32;

            #[inline]
            fn to_bytes_le(&self, out: &mut [u8]) {
                assert_eq!(out.len(), <Self as CanonicalBytes>::NUM_BYTES);
                use ark_serialize::CanonicalSerialize;
                self.0
                    .serialize_compressed(out)
                    .expect("BN254 element serializes to 32 bytes");
            }
        }

        impl CanonicalEncoding for $ty {
            const MODULUS_BITS: u32 = 254;
            const CHALLENGE_SET_LIMBS: [u64; 4] = <$inner as PrimeField>::MODULUS.0;

            #[inline]
            fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
                $ty(reduce_le_bytes(bytes))
            }

            #[inline]
            fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
                use ark_serialize::CanonicalDeserialize;
                if bytes.len() != <Self as CanonicalBytes>::NUM_BYTES {
                    return None;
                }
                <$inner>::deserialize_compressed(bytes).ok().map($ty)
            }

            #[inline]
            fn to_u128_checked(&self) -> Option<u128> {
                let bigint = self.0.into_bigint();
                let limbs: &[u64] = bigint.as_ref();
                (limbs[2] == 0 && limbs[3] == 0)
                    .then(|| ((limbs[1] as u128) << 64) | limbs[0] as u128)
            }

            #[inline]
            fn from_u128_checked(v: u128) -> Option<Self> {
                Some(<$ty as Ring>::from_u128(v))
            }

            #[inline]
            fn from_u128_reduced(v: u128) -> Self {
                <$ty as Ring>::from_u128(v)
            }

            #[inline]
            fn num_bits(&self) -> u32 {
                self.0.into_bigint().num_bits()
            }
        }

        $crate::impl_serde_bytes!(impl[] $ty, 32);

        impl ark_serialize::CanonicalSerialize for $ty {
            fn serialize_with_mode<W: ark_serialize::Write>(
                &self,
                writer: W,
                compress: ark_serialize::Compress,
            ) -> Result<(), ark_serialize::SerializationError> {
                self.0.serialize_with_mode(writer, compress)
            }

            fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
                self.0.serialized_size(compress)
            }
        }

        impl ark_serialize::Valid for $ty {
            fn check(&self) -> Result<(), ark_serialize::SerializationError> {
                self.0.check()
            }
        }

        impl ark_serialize::CanonicalDeserialize for $ty {
            fn deserialize_with_mode<R: ark_serialize::Read>(
                reader: R,
                compress: ark_serialize::Compress,
                validate: ark_serialize::Validate,
            ) -> Result<Self, ark_serialize::SerializationError> {
                <$inner>::deserialize_with_mode(reader, compress, validate).map($ty)
            }
        }

        impl UniformRand for $ty {
            fn rand<R: RngCore + ?Sized>(rng: &mut R) -> Self {
                $ty(<$inner as UniformRand>::rand(rng))
            }
        }

        #[cfg(feature = "allocative")]
        impl allocative::Allocative for $ty {
            fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
                visitor.visit_simple_sized::<Self>();
            }
        }

        impl WithAccumulator for $ty {
            type Accumulator = $accum;
            type SmallScalarAccumulator = $small_accum;
            type SignedProductAccumulator = $signed_accum;
        }
    };
}

wrap_bn254!(
    /// BN254 scalar field element (`#[repr(transparent)]` over `ark_bn254::Fr`).
    Fr,
    ark_bn254::Fr,
    accumulators(WideAccumulator, FrSmallScalarAccumulator, FrSignedProductAccumulator)
);

wrap_bn254!(
    /// BN254 base field element (`#[repr(transparent)]` over `ark_bn254::Fq`).
    Fq,
    ark_bn254::Fq,
    accumulators(NaiveAccumulator<Fq>, NaiveAccumulator<Fq>, NaiveAccumulator<Fq>)
);

impl Fr {
    /// The modulus as canonical little-endian limbs.
    pub const MODULUS_LIMBS: [u64; 4] = <ark_bn254::Fr as PrimeField>::MODULUS.0;
}

impl Ring for Fr {
    #[inline]
    fn from_u64(v: u64) -> Self {
        Fr(mont::from_u64(v))
    }

    #[inline]
    fn from_i64(v: i64) -> Self {
        if v < 0 {
            -Fr(mont::from_u64(v.unsigned_abs()))
        } else {
            Fr(mont::from_u64(v as u64))
        }
    }

    #[inline]
    fn from_u128(v: u128) -> Self {
        Fr(mont::from_u128(v))
    }

    #[inline]
    fn from_i128(v: i128) -> Self {
        if v < 0 {
            -Fr(mont::from_u128(v.unsigned_abs()))
        } else {
            Fr(mont::from_u128(v as u128))
        }
    }

    #[inline]
    fn square(&self) -> Self {
        Fr(ark_ff::Field::square(&self.0))
    }

    #[inline]
    fn mul_u64(&self, n: u64) -> Self {
        Fr(mont::mul_u64(self.0, n))
    }

    #[inline(always)]
    fn mul_i64(&self, n: i64) -> Self {
        let res = self.mul_u64(n.unsigned_abs());
        if n < 0 {
            -res
        } else {
            res
        }
    }

    #[inline(always)]
    fn mul_u128(&self, n: u128) -> Self {
        Fr(mont::mul_u128(self.0, n))
    }

    #[inline]
    fn mul_i128(&self, n: i128) -> Self {
        let res = self.mul_u128(n.unsigned_abs());
        if n < 0 {
            -res
        } else {
            res
        }
    }
}

impl Ring for Fq {
    #[inline]
    fn from_u64(v: u64) -> Self {
        Fq(ark_bn254::Fq::from(v))
    }

    #[inline]
    fn from_i64(v: i64) -> Self {
        if v < 0 {
            -Self::from_u64(v.unsigned_abs())
        } else {
            Self::from_u64(v as u64)
        }
    }

    #[inline]
    fn from_u128(v: u128) -> Self {
        Fq(ark_bn254::Fq::from(v))
    }

    #[inline]
    fn from_i128(v: i128) -> Self {
        if v < 0 {
            -Self::from_u128(v.unsigned_abs())
        } else {
            Self::from_u128(v as u128)
        }
    }

    #[inline]
    fn square(&self) -> Self {
        Fq(ark_ff::Field::square(&self.0))
    }
}
