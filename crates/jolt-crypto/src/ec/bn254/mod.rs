//! Concrete BN254 curve implementation.
//!
//! This module wraps the arkworks `ark-bn254` crate behind the generic
//! `JoltGroup` and `PairingGroup` traits. Arkworks types never appear in
//! the public API — all conversions happen internally.

/// Generates a `#[repr(transparent)]` wrapper over an arkworks projective curve type,
/// with all operator impls, serde, `AppendToTranscript`, `JoltGroup`, compile-time
/// size assertions, and a safe `into_inner` accessor.
///
/// Paths are fully qualified so the macro does not inject `use` items into the caller's
/// module scope — callers can expand the macro multiple times in the same module or
/// alongside unrelated imports without conflicts.
macro_rules! impl_jolt_group_wrapper {
    ($wrapper:ident, $projective:ty, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Default, Eq, PartialEq)]
        #[repr(transparent)]
        pub struct $wrapper(pub(crate) $projective);

        // SAFETY: $wrapper is #[repr(transparent)] over $projective.
        // Unsafe pointer casts in batch_addition and glv rely on this.
        const _: () =
            assert!(::std::mem::size_of::<$wrapper>() == ::std::mem::size_of::<$projective>());

        impl $wrapper {
            /// Unwraps into the inner arkworks projective type.
            #[inline(always)]
            pub fn into_inner(self) -> $projective {
                self.0
            }
        }

        impl ::std::fmt::Debug for $wrapper {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                let affine = <$projective as ::ark_ec::CurveGroup>::into_affine(self.0);
                f.debug_tuple(stringify!($wrapper)).field(&affine).finish()
            }
        }

        impl From<$projective> for $wrapper {
            #[inline(always)]
            fn from(inner: $projective) -> Self {
                Self(inner)
            }
        }

        impl From<$wrapper> for $projective {
            #[inline(always)]
            fn from(w: $wrapper) -> Self {
                w.0
            }
        }

        impl ::std::ops::Add for $wrapper {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl<'a> ::std::ops::Add<&'a $wrapper> for $wrapper {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: &'a $wrapper) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl ::std::ops::Sub for $wrapper {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl<'a> ::std::ops::Sub<&'a $wrapper> for $wrapper {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: &'a $wrapper) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl ::std::ops::Neg for $wrapper {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }

        impl ::std::ops::AddAssign for $wrapper {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }

        impl ::std::ops::SubAssign for $wrapper {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0;
            }
        }

        impl ::serde::Serialize for $wrapper {
            fn serialize<S: ::serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                use ::ark_serialize::CanonicalSerialize;
                let mut buf = Vec::with_capacity(self.0.compressed_size());
                self.0
                    .serialize_compressed(&mut buf)
                    .map_err(::serde::ser::Error::custom)?;
                serializer.serialize_bytes(&buf)
            }
        }

        impl<'de> ::serde::Deserialize<'de> for $wrapper {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> Result<Self, D::Error> {
                use ::ark_serialize::CanonicalDeserialize;
                let buf = <Vec<u8>>::deserialize(deserializer)?;
                let inner = <$projective>::deserialize_compressed(&buf[..])
                    .map_err(::serde::de::Error::custom)?;
                Ok(Self(inner))
            }
        }

        impl ::jolt_transcript::AppendToTranscript for $wrapper {
            fn append_to_transcript<T: ::jolt_transcript::Transcript>(&self, transcript: &mut T) {
                use ::ark_serialize::CanonicalSerialize;
                let mut buf = Vec::with_capacity(self.0.uncompressed_size());
                self.0
                    .serialize_uncompressed(&mut buf)
                    .expect(concat!(stringify!($wrapper), " serialization cannot fail"));
                buf.reverse();
                transcript.append_bytes(&buf);
            }
        }

        impl $crate::JoltGroup for $wrapper {
            #[inline(always)]
            fn identity() -> Self {
                Self(<$projective as ::ark_ff::Zero>::zero())
            }

            #[inline(always)]
            fn is_identity(&self) -> bool {
                <$projective as ::ark_ff::Zero>::is_zero(&self.0)
            }

            #[inline(always)]
            fn double(&self) -> Self {
                Self(<$projective as ::ark_ec::AdditiveGroup>::double(&self.0))
            }

            #[inline]
            fn scalar_mul<F: ::jolt_field::Field>(&self, scalar: &F) -> Self {
                Self(self.0 * super::field_to_fr(scalar))
            }

            #[inline]
            fn msm<F: ::jolt_field::Field>(bases: &[Self], scalars: &[F]) -> Self {
                use ::ark_ec::{CurveGroup, VariableBaseMSM};
                use ::ark_ff::PrimeField;
                debug_assert_eq!(bases.len(), scalars.len());
                // SAFETY: $wrapper is #[repr(transparent)] over $projective — the
                // `impl_jolt_group_wrapper!` macro asserts this invariant at compile
                // time. `normalize_batch` amortizes a single field inversion across
                // all points, replacing per-point `into_affine` (which inverts z per
                // point) with one batch inversion — 10–100× cheaper at MSM sizes
                // used in Dory tier-1 commitment.
                let projective: &[$projective] = unsafe {
                    ::std::slice::from_raw_parts(bases.as_ptr().cast::<$projective>(), bases.len())
                };
                let affines = <$projective as CurveGroup>::normalize_batch(projective);
                let bigints: Vec<_> = scalars
                    .iter()
                    .map(|s| super::field_to_fr(s).into_bigint())
                    .collect();
                Self(<$projective as VariableBaseMSM>::msm_bigint(
                    &affines, &bigints,
                ))
            }
        }
    };
}

pub(crate) use impl_jolt_group_wrapper;

mod g1;
mod g2;
mod gt;

#[doc(hidden)]
pub mod batch_addition;
#[doc(hidden)]
pub mod glv;

pub use g1::Bn254G1;
pub use g2::Bn254G2;
pub use gt::Bn254GT;

use ark_bn254::Bn254 as ArkBn254;
use ark_ec::pairing::Pairing;
use ark_ec::CurveGroup;
use ark_ff::PrimeField as _;
use jolt_field::Field;

use crate::PairingGroup;

/// BN254 pairing-friendly curve.
#[derive(Clone, Debug, Default)]
pub struct Bn254;

impl Bn254 {
    /// Standard G1 generator. Useful for tests and PCS setup code.
    pub fn g1_generator() -> Bn254G1 {
        use ark_ec::AffineRepr;
        Bn254G1(ark_bn254::G1Affine::generator().into())
    }

    /// Standard G2 generator. Useful for tests and PCS setup code.
    pub fn g2_generator() -> Bn254G2 {
        use ark_ec::AffineRepr;
        Bn254G2(ark_bn254::G2Affine::generator().into())
    }

    /// Samples a uniformly random G1 element.
    pub fn random_g1<R: rand_core::RngCore>(rng: &mut R) -> Bn254G1 {
        use ark_std::UniformRand;
        Bn254G1(ark_bn254::G1Projective::rand(rng))
    }
}

impl PairingGroup for Bn254 {
    type ScalarField = jolt_field::Fr;
    type G1 = Bn254G1;
    type G2 = Bn254G2;
    type GT = Bn254GT;

    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::GT {
        Bn254GT(ArkBn254::pairing(g1.0, g2.0).0)
    }

    fn multi_pairing(g1s: &[Self::G1], g2s: &[Self::G2]) -> Self::GT {
        debug_assert_eq!(g1s.len(), g2s.len());
        // Batched projective → affine normalization (one inversion for all points)
        // is 10-100× faster than per-point `into_affine` for typical Dory/KZG verifier
        // sizes.
        let g1_projs: Vec<ark_bn254::G1Projective> = g1s.iter().map(|g| g.0).collect();
        let g2_projs: Vec<ark_bn254::G2Projective> = g2s.iter().map(|g| g.0).collect();
        let g1_affines = ark_bn254::G1Projective::normalize_batch(&g1_projs);
        let g2_affines = ark_bn254::G2Projective::normalize_batch(&g2_projs);
        Bn254GT(ArkBn254::multi_pairing(&g1_affines, &g2_affines).0)
    }
}

/// Converts a generic `Field` element to an arkworks `Fr` via serialization.
///
/// This is the bridge between jolt-field's backend-agnostic `Field` trait and
/// arkworks' concrete scalar type. The conversion goes through little-endian
/// byte serialization.
///
/// When the concrete `F` is `jolt_field::Fr` (itself a `#[repr(transparent)]`
/// newtype over `ark_bn254::Fr`), the `TypeId` fast path skips the byte
/// serialization roundtrip entirely — the monomorphized branch is optimized
/// away by the compiler. See spec: jolt-crypto-perf-optimizations (#1368
/// follow-up) for details.
///
/// In debug builds (generic path only), asserts that the source value fits in
/// the BN254 Fr modulus — catches silent modular reduction when `F` has a
/// larger modulus than BN254 Fr.
#[inline]
pub(crate) fn field_to_fr<F: Field>(f: &F) -> ark_bn254::Fr {
    use std::any::TypeId;
    if TypeId::of::<F>() == TypeId::of::<jolt_field::Fr>() {
        let ptr = std::ptr::from_ref::<F>(f).cast::<ark_bn254::Fr>();
        // SAFETY: TypeId equality implies F and jolt_field::Fr are the same
        // type; jolt_field::Fr is `#[repr(transparent)]` over ark_bn254::Fr,
        // so the pointer cast preserves layout bit-for-bit.
        return unsafe { *ptr };
    }
    let bytes = f.to_bytes();
    #[cfg(debug_assertions)]
    {
        use ark_ff::{BigInteger, PrimeField as _};
        let value = num_bigint::BigUint::from_bytes_le(&bytes);
        let modulus = num_bigint::BigUint::from_bytes_le(&ark_bn254::Fr::MODULUS.to_bytes_le());
        debug_assert!(
            value < modulus,
            "field_to_fr: source value >= BN254 Fr modulus (silent reduction)",
        );
    }
    ark_bn254::Fr::from_le_bytes_mod_order(&bytes)
}
