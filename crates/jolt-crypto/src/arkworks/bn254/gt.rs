use std::fmt::Debug;
use std::ops::{Mul, MulAssign};

use ark_bn254::Fq12;
use ark_ff::Field as _;

/// BN254 target group element (pairing output).
///
/// GT uses **multiplicative** notation: the group operation is Fq12
/// multiplication, and the identity is `Fq12::one()`.
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(transparent)]
pub struct Bn254GT(pub(crate) Fq12);

impl Debug for Bn254GT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Bn254GT").field(&self.0).finish()
    }
}

impl Bn254GT {
    /// The multiplicative identity `1` in GT.
    pub fn one() -> Self {
        Self(Fq12::ONE)
    }
}

impl Mul for Bn254GT {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl MulAssign for Bn254GT {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl serde::Serialize for Bn254GT {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use ark_serialize::CanonicalSerialize;
        let mut buf = Vec::new();
        self.0
            .serialize_compressed(&mut buf)
            .map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&buf)
    }
}

impl<'de> serde::Deserialize<'de> for Bn254GT {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use ark_serialize::CanonicalDeserialize;
        let buf = <Vec<u8>>::deserialize(deserializer)?;
        let inner = Fq12::deserialize_compressed(&buf[..]).map_err(serde::de::Error::custom)?;
        Ok(Self(inner))
    }
}
