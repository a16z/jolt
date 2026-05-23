//! Concrete Grumpkin curve implementation.
//!
//! This module wraps `ark-grumpkin` behind `JoltGroup`. Arkworks types stay
//! internal to this backend module.

use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, UniformRand, Zero};
use jolt_field::{CanonicalBytes, FixedByteSize, Fq};

/// Grumpkin prime-order group element.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct GrumpkinPoint(pub(crate) ark_grumpkin::Projective);

const _: () = assert!(
    std::mem::size_of::<GrumpkinPoint>() == std::mem::size_of::<ark_grumpkin::Projective>()
);

impl std::fmt::Debug for GrumpkinPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let affine = self.0.into_affine();
        f.debug_tuple("GrumpkinPoint").field(&affine).finish()
    }
}

impl std::ops::Add for GrumpkinPoint {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<'a> std::ops::Add<&'a GrumpkinPoint> for GrumpkinPoint {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: &'a GrumpkinPoint) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Sub for GrumpkinPoint {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl<'a> std::ops::Sub<&'a GrumpkinPoint> for GrumpkinPoint {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: &'a GrumpkinPoint) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl std::ops::Neg for GrumpkinPoint {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl std::ops::AddAssign for GrumpkinPoint {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl std::ops::SubAssign for GrumpkinPoint {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl serde::Serialize for GrumpkinPoint {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use ark_serialize::CanonicalSerialize;
        let mut buf = Vec::with_capacity(self.0.compressed_size());
        self.0
            .serialize_compressed(&mut buf)
            .map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&buf)
    }
}

impl<'de> serde::Deserialize<'de> for GrumpkinPoint {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use ark_serialize::CanonicalDeserialize;
        let buf = <Vec<u8>>::deserialize(deserializer)?;
        let inner = ark_grumpkin::Projective::deserialize_compressed(&buf[..])
            .map_err(serde::de::Error::custom)?;
        Ok(Self(inner))
    }
}

impl jolt_transcript::AppendToTranscript for GrumpkinPoint {
    #[expect(clippy::expect_used, reason = "serialization into Vec cannot fail")]
    fn append_to_transcript<T: jolt_transcript::Transcript>(&self, transcript: &mut T) {
        use ark_serialize::CanonicalSerialize;
        let mut buf = Vec::with_capacity(self.0.compressed_size());
        self.0
            .serialize_compressed(&mut buf)
            .expect("GrumpkinPoint serialization cannot fail");
        transcript.append_bytes(&buf);
    }
}

impl crate::JoltGroup for GrumpkinPoint {
    type ScalarField = Fq;

    #[inline(always)]
    fn identity() -> Self {
        Self(<ark_grumpkin::Projective as Zero>::zero())
    }

    #[inline(always)]
    fn is_identity(&self) -> bool {
        <ark_grumpkin::Projective as Zero>::is_zero(&self.0)
    }

    #[inline(always)]
    fn double(&self) -> Self {
        Self(<ark_grumpkin::Projective as ark_ec::AdditiveGroup>::double(
            &self.0,
        ))
    }

    #[inline]
    fn scalar_mul(&self, scalar: &Self::ScalarField) -> Self {
        Self(self.0 * fq_to_grumpkin_fr(scalar))
    }

    #[inline]
    fn msm(bases: &[Self], scalars: &[Self::ScalarField]) -> Self {
        debug_assert_eq!(bases.len(), scalars.len());
        let affines: Vec<ark_grumpkin::Affine> = bases.iter().map(|b| b.0.into_affine()).collect();
        let ark_scalars: Vec<ark_grumpkin::Fr> = scalars.iter().map(fq_to_grumpkin_fr).collect();
        let bigints: Vec<_> = ark_scalars.iter().map(|s| s.into_bigint()).collect();
        Self(<ark_grumpkin::Projective as VariableBaseMSM>::msm_bigint(
            &affines, &bigints,
        ))
    }
}

/// Grumpkin curve marker and constructors.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Grumpkin;

impl Grumpkin {
    /// Standard Grumpkin generator.
    pub fn generator() -> GrumpkinPoint {
        GrumpkinPoint(ark_grumpkin::Affine::generator().into())
    }

    /// Samples a uniformly random Grumpkin group element.
    pub fn random<R: rand_core::RngCore>(rng: &mut R) -> GrumpkinPoint {
        GrumpkinPoint(ark_grumpkin::Projective::rand(rng))
    }
}

#[inline]
pub(crate) fn fq_to_grumpkin_fr(f: &Fq) -> ark_grumpkin::Fr {
    let mut bytes = vec![0u8; Fq::NUM_BYTES];
    f.to_bytes_le(&mut bytes);
    #[cfg(debug_assertions)]
    {
        use ark_ff::{BigInteger, PrimeField as _};
        let value = num_bigint::BigUint::from_bytes_le(&bytes);
        let modulus = num_bigint::BigUint::from_bytes_le(&ark_grumpkin::Fr::MODULUS.to_bytes_le());
        debug_assert!(
            value < modulus,
            "fq_to_grumpkin_fr: source value >= Grumpkin scalar modulus",
        );
    }
    ark_grumpkin::Fr::from_le_bytes_mod_order(&bytes)
}

#[cfg(test)]
mod tests {
    use jolt_field::{Fq, FromPrimitiveInt};

    use super::*;
    use crate::{JoltGroup, Pedersen, PedersenSetup, VectorCommitment};

    #[test]
    fn scalar_mul_and_msm_match() {
        let generator = Grumpkin::generator();
        let a = Fq::from_u64(11);
        let b = Fq::from_u64(19);
        let p = generator.scalar_mul(&a);
        let q = generator.scalar_mul(&b);

        assert_eq!(GrumpkinPoint::msm(&[generator, generator], &[a, b]), p + q);
    }

    #[test]
    fn pedersen_over_grumpkin_uses_fq_scalars() {
        type VC = Pedersen<GrumpkinPoint>;

        let generator = Grumpkin::generator();
        let setup = PedersenSetup::new(
            vec![
                generator,
                generator.scalar_mul(&Fq::from_u64(2)),
                generator.scalar_mul(&Fq::from_u64(3)),
            ],
            generator.scalar_mul(&Fq::from_u64(99)),
        );
        let values = [Fq::from_u64(4), Fq::from_u64(5), Fq::from_u64(6)];
        let opening_scalar = Fq::from_u64(7);
        let commitment = VC::commit(&setup, &values, &opening_scalar);

        assert!(VC::verify(&setup, &commitment, &values, &opening_scalar));
        assert!(!VC::verify(
            &setup,
            &commitment,
            &values,
            &(opening_scalar + Fq::from_u64(1)),
        ));
    }
}
