//! Concrete BN254 curve implementation.
//!
//! This module wraps the arkworks `ark-bn254` crate behind the generic
//! `JoltGroup` and `PairingGroup` traits. Arkworks types never appear in
//! the public API — all conversions happen internally.

/// Generates a `#[repr(transparent)]` wrapper over an arkworks projective curve type,
/// with all operator impls, serde, `AppendToTranscript`, `JoltGroup`, compile-time
/// size assertions, and a safe `into_inner` accessor.
macro_rules! impl_jolt_group_wrapper {
    ($wrapper:ident, $projective:ty, $affine:ty, $doc:literal) => {
        use std::fmt::Debug;
        use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

        use ark_ec::{AdditiveGroup, CurveGroup, VariableBaseMSM};
        use ark_ff::{PrimeField, Zero};
        use jolt_field::Field;

        use jolt_transcript::{AppendToTranscript, Transcript};

        use crate::JoltGroup;

        use super::field_to_fr;

        #[doc = $doc]
        #[derive(Clone, Copy, Default, Eq, PartialEq)]
        #[repr(transparent)]
        pub struct $wrapper(pub(crate) $projective);

        // SAFETY: $wrapper is #[repr(transparent)] over $projective.
        // Unsafe pointer casts in batch_addition and glv rely on this.
        const _: () =
            assert!(std::mem::size_of::<$wrapper>() == std::mem::size_of::<$projective>());

        impl $wrapper {
            /// Unwraps into the inner arkworks projective type.
            #[inline(always)]
            pub fn into_inner(self) -> $projective {
                self.0
            }
        }

        impl Debug for $wrapper {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let affine = self.0.into_affine();
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

        impl Add for $wrapper {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl<'a> Add<&'a $wrapper> for $wrapper {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: &'a $wrapper) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl Sub for $wrapper {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl<'a> Sub<&'a $wrapper> for $wrapper {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: &'a $wrapper) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl Neg for $wrapper {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }

        impl AddAssign for $wrapper {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }

        impl SubAssign for $wrapper {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0;
            }
        }

        impl serde::Serialize for $wrapper {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                use ark_serialize::CanonicalSerialize;
                let mut buf = Vec::new();
                self.0
                    .serialize_compressed(&mut buf)
                    .map_err(serde::ser::Error::custom)?;
                serializer.serialize_bytes(&buf)
            }
        }

        impl<'de> serde::Deserialize<'de> for $wrapper {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                use ark_serialize::CanonicalDeserialize;
                let buf = <Vec<u8>>::deserialize(deserializer)?;
                let inner = <$projective>::deserialize_compressed(&buf[..])
                    .map_err(serde::de::Error::custom)?;
                Ok(Self(inner))
            }
        }

        impl AppendToTranscript for $wrapper {
            fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
                use ark_serialize::CanonicalSerialize;
                let mut buf = Vec::new();
                self.0
                    .serialize_compressed(&mut buf)
                    .expect(concat!(stringify!($wrapper), " serialization cannot fail"));
                transcript.append_bytes(&buf);
            }
        }

        impl JoltGroup for $wrapper {
            #[inline(always)]
            fn identity() -> Self {
                Self(<$projective>::zero())
            }

            #[inline(always)]
            fn is_identity(&self) -> bool {
                self.0.is_zero()
            }

            #[inline(always)]
            fn double(&self) -> Self {
                Self(AdditiveGroup::double(&self.0))
            }

            #[inline]
            fn scalar_mul<F: Field>(&self, scalar: &F) -> Self {
                Self(self.0 * field_to_fr(scalar))
            }

            #[inline]
            fn msm<F: Field>(bases: &[Self], scalars: &[F]) -> Self {
                debug_assert_eq!(bases.len(), scalars.len());
                let affines: Vec<$affine> = bases.iter().map(|b| b.0.into_affine()).collect();
                let fr_scalars: Vec<ark_bn254::Fr> = scalars.iter().map(field_to_fr).collect();
                let bigints: Vec<_> = fr_scalars.iter().map(|s| s.into_bigint()).collect();
                Self(<$projective>::msm_bigint(&affines, &bigints))
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
        let g1_affines: Vec<ark_bn254::G1Affine> = g1s.iter().map(|g| g.0.into_affine()).collect();
        let g2_affines: Vec<ark_bn254::G2Affine> = g2s.iter().map(|g| g.0.into_affine()).collect();
        Bn254GT(ArkBn254::multi_pairing(&g1_affines, &g2_affines).0)
    }
}

/// Converts a generic `Field` element to an arkworks `Fr` via serialization.
///
/// This is the bridge between jolt-field's backend-agnostic `Field` trait and
/// arkworks' concrete scalar type. The conversion goes through little-endian
/// byte serialization.
#[inline]
pub(crate) fn field_to_fr<F: Field>(f: &F) -> ark_bn254::Fr {
    let bytes = f.to_bytes();
    ark_bn254::Fr::from_le_bytes_mod_order(&bytes)
}
