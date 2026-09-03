use std::error::Error as StdError;
use std::fmt::{Display, Formatter, Result as FmtResult};

use ark_bn254::{
    compressible_fq12_to_fq12, fq12_to_compressible_fq12, CompressedFq12, CompressibleFq12, Fq12,
    Fq2, Fq6, Fq6Config, Fr,
};
use ark_ff::{Field, Fp6Config, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::de::{Error as _, SeqAccess, Visitor};
use serde::ser::SerializeTuple;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::Bn254GT;

pub const COMPRESSED_GT_SIZE: usize = 128;
const IDENTITY_ENCODING: [u8; COMPRESSED_GT_SIZE] = [0; COMPRESSED_GT_SIZE];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GtCompressionError {
    /// One or more base-field limbs are not canonically encoded.
    InvalidFieldEncoding,
    /// The torus denominator is zero.
    InvalidTorusEncoding,
    /// The decoded element is outside the prime-order target group.
    NotInSubgroup,
    /// Re-encoding the decoded element produces different bytes.
    NonCanonicalEncoding,
}

impl Display for GtCompressionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::InvalidFieldEncoding => f.write_str("invalid compressed GT field encoding"),
            Self::InvalidTorusEncoding => f.write_str("invalid compressed GT torus encoding"),
            Self::NotInSubgroup => {
                f.write_str("compressed GT element is not in the r-torsion subgroup")
            }
            Self::NonCanonicalEncoding => f.write_str("non-canonical compressed GT encoding"),
        }
    }
}

impl StdError for GtCompressionError {}

/// Canonical BN254 target-group encoding.
///
/// Stores two `Fq2` coordinates of the torus `Fq6`; the cyclotomic relation
/// recovers the omitted third coordinate without a square root or sign bit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompressedBn254GT([u8; COMPRESSED_GT_SIZE]);

impl CompressedBn254GT {
    /// Encodes a target-group element.
    #[must_use]
    pub fn from_gt(gt: &Bn254GT) -> Self {
        Self(compress_gt(gt))
    }

    /// Validates and wraps fixed-width wire bytes.
    pub fn from_bytes(bytes: [u8; COMPRESSED_GT_SIZE]) -> Result<Self, GtCompressionError> {
        let _gt = decompress_gt(&bytes)?;
        Ok(Self(bytes))
    }

    /// Returns the fixed-width wire bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; COMPRESSED_GT_SIZE] {
        &self.0
    }

    /// Decodes and validates the target-group element.
    pub fn decompress(&self) -> Result<Bn254GT, GtCompressionError> {
        decompress_gt(&self.0)
    }
}

impl From<&Bn254GT> for CompressedBn254GT {
    fn from(gt: &Bn254GT) -> Self {
        Self::from_gt(gt)
    }
}

impl TryFrom<[u8; COMPRESSED_GT_SIZE]> for CompressedBn254GT {
    type Error = GtCompressionError;

    fn try_from(bytes: [u8; COMPRESSED_GT_SIZE]) -> Result<Self, Self::Error> {
        Self::from_bytes(bytes)
    }
}

impl Serialize for CompressedBn254GT {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(COMPRESSED_GT_SIZE)?;
        for byte in self.0 {
            tuple.serialize_element(&byte)?;
        }
        tuple.end()
    }
}

struct CompressedGtVisitor;

impl<'de> Visitor<'de> for CompressedGtVisitor {
    type Value = CompressedBn254GT;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        write!(
            formatter,
            "a canonical {COMPRESSED_GT_SIZE}-byte GT encoding"
        )
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut bytes = [0u8; COMPRESSED_GT_SIZE];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = seq
                .next_element()?
                .ok_or_else(|| A::Error::invalid_length(index, &self))?;
        }
        Self::Value::from_bytes(bytes).map_err(A::Error::custom)
    }
}

impl<'de> Deserialize<'de> for CompressedBn254GT {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_tuple(COMPRESSED_GT_SIZE, CompressedGtVisitor)
    }
}

#[must_use]
#[expect(
    clippy::expect_used,
    reason = "serialization into an exact-size in-memory buffer cannot fail"
)]
/// Encodes a target-group element in 128 bytes, with all-zero reserved for identity.
pub fn compress_gt(gt: &Bn254GT) -> [u8; COMPRESSED_GT_SIZE] {
    if gt.0 == Fq12::ONE {
        return IDENTITY_ENCODING;
    }

    let element = fq12_to_compressible_fq12(gt.0);
    let Some(c1_inv) = element.c1.inverse() else {
        return IDENTITY_ENCODING;
    };
    let torus = (-Fq6::ONE - element.c0) * c1_inv;
    let compressed = CompressedFq12((torus.c0, torus.c1));
    let mut bytes = [0u8; COMPRESSED_GT_SIZE];
    compressed
        .serialize_compressed(&mut bytes[..])
        .expect("fixed compressed GT buffer has the canonical serialized size");
    bytes
}

/// Decodes canonical wire bytes and checks prime-order subgroup membership.
pub fn decompress_gt(bytes: &[u8; COMPRESSED_GT_SIZE]) -> Result<Bn254GT, GtCompressionError> {
    if bytes == &IDENTITY_ENCODING {
        return Ok(Bn254GT(Fq12::ONE));
    }

    let compressed = CompressedFq12::deserialize_compressed(&bytes[..])
        .map_err(|_| GtCompressionError::InvalidFieldEncoding)?;
    let (c0, c1) = compressed.0;
    let denominator = Fq2::from(3u64) * c1 * Fq6Config::NONRESIDUE;
    let denominator_inv = denominator
        .inverse()
        .ok_or(GtCompressionError::InvalidTorusEncoding)?;
    let c2 = (Fq2::from(3u64) * c0.square() + Fq6Config::NONRESIDUE) * denominator_inv;
    let torus = Fq6 { c0, c1, c2 };
    let numerator = CompressibleFq12::new(torus, -Fq6::ONE);
    let denominator = CompressibleFq12::new(torus, Fq6::ONE);
    let denominator_inv = denominator
        .inverse()
        .ok_or(GtCompressionError::InvalidTorusEncoding)?;
    let inner = compressible_fq12_to_fq12(numerator * denominator_inv);

    if inner.pow(Fr::MODULUS) != Fq12::ONE {
        return Err(GtCompressionError::NotInSubgroup);
    }

    let gt = Bn254GT(inner);
    if compress_gt(&gt) != *bytes {
        return Err(GtCompressionError::NonCanonicalEncoding);
    }
    Ok(gt)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests may fail loudly")]
mod tests {
    use ark_bn254::Config;
    use ark_ec::bn::raise_to_psi_six_pow;
    use ark_ff::{AdditiveGroup, UniformRand};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use crate::{Bn254, JoltGroup, PairingGroup};

    use super::*;

    #[test]
    fn pairing_outputs_round_trip() {
        let mut rng = ChaCha20Rng::seed_from_u64(70);
        let g2 = Bn254::g2_generator();
        for _ in 0..16 {
            let gt = Bn254::pairing(&Bn254::random_g1(&mut rng), &g2);
            let bytes = compress_gt(&gt);
            assert_eq!(decompress_gt(&bytes).unwrap(), gt);
        }

        let identity = Bn254GT::identity();
        assert_eq!(compress_gt(&identity), IDENTITY_ENCODING);
        assert_eq!(decompress_gt(&IDENTITY_ENCODING).unwrap(), identity);
    }

    #[test]
    fn serde_is_fixed_width() {
        let gt = Bn254::pairing(&Bn254::g1_generator(), &Bn254::g2_generator());
        let compressed = CompressedBn254GT::from_gt(&gt);
        let encoded =
            bincode::serde::encode_to_vec(compressed, bincode::config::standard()).unwrap();
        assert_eq!(encoded.len(), COMPRESSED_GT_SIZE);

        let (decoded, consumed): (CompressedBn254GT, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        assert_eq!(consumed, COMPRESSED_GT_SIZE);
        assert_eq!(decoded.decompress().unwrap(), gt);
    }

    #[test]
    fn rejects_invalid_encodings() {
        let noncanonical_field = [u8::MAX; COMPRESSED_GT_SIZE];
        assert_eq!(
            decompress_gt(&noncanonical_field),
            Err(GtCompressionError::InvalidFieldEncoding)
        );

        let malformed = CompressedFq12((Fq2::ONE, Fq2::ZERO));
        let mut malformed_bytes = [0u8; COMPRESSED_GT_SIZE];
        malformed
            .serialize_compressed(&mut malformed_bytes[..])
            .unwrap();
        assert_eq!(
            decompress_gt(&malformed_bytes),
            Err(GtCompressionError::InvalidTorusEncoding)
        );

        let mut rng = ChaCha20Rng::seed_from_u64(71);
        let non_gt = loop {
            let candidate = raise_to_psi_six_pow::<Config>(Fq12::rand(&mut rng)).unwrap();
            if candidate != Fq12::ONE && candidate.pow(Fr::MODULUS) != Fq12::ONE {
                break candidate;
            }
        };
        let non_subgroup_bytes = compress_gt(&Bn254GT::from(non_gt));
        assert_eq!(
            decompress_gt(&non_subgroup_bytes),
            Err(GtCompressionError::NotInSubgroup)
        );
    }
}
