use std::fmt::Debug;
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use ark_bn254::{G2Affine, G2Projective};
use ark_ec::{AdditiveGroup, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, Zero};
use jolt_field::Field;

use jolt_transcript::{AppendToTranscript, Transcript};

use crate::JoltGroup;

use super::field_to_fr;

/// BN254 G2 group element (projective coordinates).
#[derive(Clone, Copy, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct Bn254G2(pub(crate) G2Projective);

impl Debug for Bn254G2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let affine = self.0.into_affine();
        f.debug_tuple("Bn254G2").field(&affine).finish()
    }
}

impl From<G2Projective> for Bn254G2 {
    #[inline(always)]
    fn from(inner: G2Projective) -> Self {
        Self(inner)
    }
}

impl Add for Bn254G2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<'a> Add<&'a Bn254G2> for Bn254G2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: &'a Bn254G2) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Bn254G2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl<'a> Sub<&'a Bn254G2> for Bn254G2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: &'a Bn254G2) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Neg for Bn254G2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl AddAssign for Bn254G2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Bn254G2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl serde::Serialize for Bn254G2 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use ark_serialize::CanonicalSerialize;
        let mut buf = Vec::new();
        self.0
            .serialize_compressed(&mut buf)
            .map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&buf)
    }
}

impl<'de> serde::Deserialize<'de> for Bn254G2 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use ark_serialize::CanonicalDeserialize;
        let buf = <Vec<u8>>::deserialize(deserializer)?;
        let inner =
            G2Projective::deserialize_compressed(&buf[..]).map_err(serde::de::Error::custom)?;
        Ok(Self(inner))
    }
}

impl AppendToTranscript for Bn254G2 {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        use ark_serialize::CanonicalSerialize;
        let mut buf = Vec::new();
        self.0
            .serialize_compressed(&mut buf)
            .expect("G2 serialization cannot fail");
        transcript.append_bytes(&buf);
    }
}

impl JoltGroup for Bn254G2 {
    #[inline(always)]
    fn identity() -> Self {
        Self(G2Projective::zero())
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
        let affines: Vec<G2Affine> = bases.iter().map(|b| b.0.into_affine()).collect();
        let fr_scalars: Vec<ark_bn254::Fr> = scalars.iter().map(field_to_fr).collect();
        let bigints: Vec<_> = fr_scalars.iter().map(|s| s.into_bigint()).collect();
        Self(G2Projective::msm_bigint(&affines, &bigints))
    }
}
