use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_bn254::{Fq12, Fr};
use ark_ff::{AdditiveGroup, Field as ArkField, PrimeField};
use ark_serialize::{CanonicalSerialize, Compress, SerializationError, Write};
use jolt_field::JoltField;

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

impl From<Bn254GT> for Fq12 {
    #[inline(always)]
    fn from(w: Bn254GT) -> Self {
        w.0
    }
}

impl From<Fq12> for Bn254GT {
    #[inline(always)]
    fn from(value: Fq12) -> Self {
        Self(value)
    }
}

impl Default for Bn254GT {
    #[inline(always)]
    fn default() -> Self {
        Self(Fq12::ONE)
    }
}

impl CanonicalSerialize for Bn254GT {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

// GT's additive notation maps to Fq12 multiplication by design.
#[expect(
    clippy::suspicious_arithmetic_impl,
    clippy::suspicious_op_assign_impl,
    clippy::expect_used
)]
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

#[expect(clippy::expect_used)]
impl AppendToTranscript for Bn254GT {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        use ark_serialize::CanonicalSerialize;
        let mut buf = Vec::with_capacity(self.0.uncompressed_size());
        self.0
            .serialize_uncompressed(&mut buf)
            .expect("GT serialization cannot fail");
        buf.reverse();
        transcript.append_bytes(&buf);
    }

    fn transcript_payload_len(&self) -> Option<u64> {
        use ark_serialize::CanonicalSerialize;
        Some(self.0.uncompressed_size() as u64)
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
    fn scalar_mul<F: JoltField>(&self, scalar: &F) -> Self {
        // GT exponentiation: self^scalar (written additively as scalar * self).
        let fr = field_to_fr(scalar);
        Self(self.0.pow(fr.into_bigint()))
    }

    #[inline]
    fn msm<F: JoltField>(bases: &[Self], scalars: &[F]) -> Self {
        // zip would silently truncate to the shorter slice.
        assert_eq!(
            bases.len(),
            scalars.len(),
            "msm: bases/scalars length mismatch"
        );
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
        let mut buf = Vec::with_capacity(self.0.compressed_size());
        self.0
            .serialize_compressed(&mut buf)
            .map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&buf)
    }
}

impl<'de> serde::Deserialize<'de> for Bn254GT {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
        let buf = <Vec<u8>>::deserialize(deserializer)?;
        // Exact-size gate: rejects oversized/truncated payloads before any
        // field parsing and makes the encoding canonical (no trailing bytes).
        let expected_len = Fq12::ONE.compressed_size();
        if buf.len() != expected_len {
            return Err(serde::de::Error::custom(format!(
                "GT element encoding must be exactly {expected_len} bytes, got {}",
                buf.len()
            )));
        }
        let inner = Fq12::deserialize_compressed(&buf[..]).map_err(serde::de::Error::custom)?;
        // Reject Fq12::ZERO: not in any multiplicative subgroup, and later
        // Neg/Sub/SubAssign would call .inverse().expect(...) and panic.
        if inner == Fq12::ZERO {
            return Err(serde::de::Error::custom(
                "GT element is zero (not in r-torsion subgroup)",
            ));
        }
        // Unitarity pre-filter: GT ⊂ the norm-1 (unitary) subgroup of Fq12
        // over Fq6, i.e. x^(q^6+1) = conj(x)·x = 1. A non-GT Fq12 element is
        // unitary with probability ~q^-6, so one conjugation + multiplication
        // rejects virtually all malformed input before the 254-bit
        // exponentiation below (which would otherwise be attacker-triggerable
        // per element).
        let mut conj = inner;
        let _ = conj.conjugate_in_place();
        if conj * inner != Fq12::ONE {
            return Err(serde::de::Error::custom(
                "GT element is not unitary (not in the r-torsion subgroup)",
            ));
        }
        // Subgroup membership: GT is the r-torsion subgroup, so x^r == 1.
        // Unitarity is necessary but not sufficient; this check is exact.
        if inner.pow(Fr::MODULUS) != Fq12::ONE {
            return Err(serde::de::Error::custom(
                "GT element is not in the r-torsion subgroup",
            ));
        }
        Ok(Self(inner))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests may fail loudly")]
mod tests {
    use super::*;
    use crate::PairingGroup;
    use ark_serialize::CanonicalSerialize;
    use ark_std::UniformRand;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn encode_as_json(fq12: &Fq12) -> String {
        let mut buf = Vec::new();
        fq12.serialize_compressed(&mut buf).unwrap();
        serde_json::to_string(&buf).unwrap()
    }

    #[test]
    fn deserialize_accepts_pairing_output() {
        let gt =
            crate::Bn254::pairing(&crate::Bn254::g1_generator(), &crate::Bn254::g2_generator());
        let recovered: Bn254GT = serde_json::from_str(&encode_as_json(&gt.0)).unwrap();
        assert_eq!(recovered, gt);
    }

    #[test]
    fn deserialize_rejects_zero() {
        let err = serde_json::from_str::<Bn254GT>(&encode_as_json(&Fq12::ZERO)).unwrap_err();
        assert!(err.to_string().contains("zero"), "{err}");
    }

    #[test]
    fn deserialize_rejects_non_unitary_element() {
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        // A random Fq12 element is unitary with probability ~q^-6.
        let z = Fq12::rand(&mut rng);
        let err = serde_json::from_str::<Bn254GT>(&encode_as_json(&z)).unwrap_err();
        assert!(err.to_string().contains("not unitary"), "{err}");
    }

    #[test]
    fn deserialize_rejects_unitary_non_r_torsion_element() {
        let mut rng = ChaCha20Rng::seed_from_u64(8);
        // u = z^(q^6-1) = conj(z)/z is unitary by construction but lies in the
        // full norm-1 subgroup (order q^6+1), outside GT w.o.p. This must pass
        // the unitarity pre-filter and be caught by the exact x^r check.
        let z = Fq12::rand(&mut rng);
        let mut conj = z;
        let _ = conj.conjugate_in_place();
        let u = conj * z.inverse().unwrap();
        assert_ne!(u.pow(Fr::MODULUS), Fq12::ONE, "unlucky sample landed in GT");
        let err = serde_json::from_str::<Bn254GT>(&encode_as_json(&u)).unwrap_err();
        assert!(err.to_string().contains("r-torsion"), "{err}");
    }

    #[test]
    fn deserialize_rejects_wrong_length_encoding() {
        let gt =
            crate::Bn254::pairing(&crate::Bn254::g1_generator(), &crate::Bn254::g2_generator());
        let mut buf = Vec::new();
        gt.0.serialize_compressed(&mut buf).unwrap();
        buf.push(0);
        let json = serde_json::to_string(&buf).unwrap();
        let err = serde_json::from_str::<Bn254GT>(&json).unwrap_err();
        assert!(err.to_string().contains("exactly"), "{err}");
    }
}
