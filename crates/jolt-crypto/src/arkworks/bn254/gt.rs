use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_bn254::Fq12;
use ark_ff::{Field as ArkField, PrimeField};
use jolt_field::Field;

use jolt_transcript::{AppendToTranscript, Transcript};

use crate::JoltGroup;

use super::field_to_fr;

/// BN254 target group element (pairing output).
///
/// GT is mathematically multiplicative (Fq12 multiplication), but we expose it
/// with **additive notation** via `JoltGroup` for uniformity with G1/G2:
///
/// | JoltGroup operation | GT semantics        |
/// |---------------------|---------------------|
/// | `Add` (`+`)         | Fq12 multiplication |
/// | `Neg` (`-x`)        | Fq12 inverse        |
/// | `Sub` (`-`)         | Fq12 mul-by-inverse |
/// | `identity()`        | `Fq12::ONE`         |
/// | `double()`          | Fq12 squaring       |
///
/// `Mul`/`MulAssign` are also provided as convenience aliases that map directly
/// to the same Fq12 multiplication, for callers who prefer multiplicative
/// notation in pairing contexts.
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(transparent)]
pub struct Bn254GT(pub(crate) Fq12);

impl Debug for Bn254GT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Bn254GT").field(&self.0).finish()
    }
}

impl Default for Bn254GT {
    #[inline(always)]
    fn default() -> Self {
        Self(Fq12::ONE)
    }
}

// GT's additive notation maps to Fq12 multiplication by design.
#[allow(clippy::suspicious_arithmetic_impl, clippy::suspicious_op_assign_impl)]
const _: () = {
    impl Add for Bn254GT {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            Self(self.0 * rhs.0)
        }
    }

    impl<'a> Add<&'a Bn254GT> for Bn254GT {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: &'a Bn254GT) -> Self {
            Self(self.0 * rhs.0)
        }
    }

    impl Sub for Bn254GT {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            Self(self.0 * rhs.0.inverse().expect("GT element has no inverse"))
        }
    }

    impl<'a> Sub<&'a Bn254GT> for Bn254GT {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: &'a Bn254GT) -> Self {
            Self(self.0 * rhs.0.inverse().expect("GT element has no inverse"))
        }
    }

    impl Neg for Bn254GT {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            Self(self.0.inverse().expect("GT element has no inverse"))
        }
    }

    impl AddAssign for Bn254GT {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            self.0 *= rhs.0;
        }
    }

    impl SubAssign for Bn254GT {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            self.0 *= rhs.0.inverse().expect("GT element has no inverse");
        }
    }
}; // end #[allow(clippy::suspicious_*)]

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

impl AppendToTranscript for Bn254GT {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        use ark_serialize::CanonicalSerialize;
        let mut buf = Vec::new();
        self.0
            .serialize_compressed(&mut buf)
            .expect("GT serialization cannot fail");
        transcript.append_bytes(&buf);
    }
}

impl JoltGroup for Bn254GT {
    #[inline(always)]
    fn identity() -> Self {
        Self(Fq12::ONE)
    }

    #[inline(always)]
    fn is_identity(&self) -> bool {
        self.0 == Fq12::ONE
    }

    #[inline(always)]
    fn double(&self) -> Self {
        Self(self.0.square())
    }

    #[inline]
    fn scalar_mul<F: Field>(&self, scalar: &F) -> Self {
        // GT exponentiation: self^scalar (written additively as scalar * self).
        let fr = field_to_fr(scalar);
        Self(self.0.pow(fr.into_bigint()))
    }

    #[inline]
    fn msm<F: Field>(bases: &[Self], scalars: &[F]) -> Self {
        debug_assert_eq!(bases.len(), scalars.len());
        // GT "MSM" is Π bases[i]^scalars[i] (written additively as Σ scalars[i] * bases[i]).
        let mut acc = Fq12::ONE;
        for (base, scalar) in bases.iter().zip(scalars.iter()) {
            let fr = field_to_fr(scalar);
            acc *= base.0.pow(fr.into_bigint());
        }
        Self(acc)
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
