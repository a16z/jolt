use std::fmt::Debug;

use ark_bn254::{Bn254 as ArkBn254, Fq12};
use ark_ec::{pairing::MillerLoopOutput, pairing::Pairing};
use ark_ff::{Field as ArkField, Zero};
use jolt_field::{FixedByteSize, Fq};
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{Deserialize, Serialize};

use super::gt::Bn254GT;

/// BN254 Fq12 element used for raw Miller-loop outputs before final exponentiation.
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(transparent)]
pub struct Bn254Fq12(pub(crate) Fq12);

impl Debug for Bn254Fq12 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Bn254Fq12").field(&self.0).finish()
    }
}

impl Default for Bn254Fq12 {
    #[inline(always)]
    fn default() -> Self {
        Self(Fq12::ONE)
    }
}

impl From<Bn254GT> for Bn254Fq12 {
    #[inline(always)]
    fn from(value: Bn254GT) -> Self {
        Self(value.0)
    }
}

impl Bn254Fq12 {
    pub const COEFFICIENTS: usize = 12;

    /// Returns the BN254 Fq12 tower coefficients in canonical basis order.
    ///
    /// The order is `(c0.c0, c0.c1, c0.c2, c1.c0, c1.c1, c1.c2)`, with each
    /// Fq2 coefficient emitted as `(c0, c1)`.
    pub fn coefficients(&self) -> [Fq; Self::COEFFICIENTS] {
        [
            field_to_fq(&self.0.c0.c0.c0),
            field_to_fq(&self.0.c0.c0.c1),
            field_to_fq(&self.0.c0.c1.c0),
            field_to_fq(&self.0.c0.c1.c1),
            field_to_fq(&self.0.c0.c2.c0),
            field_to_fq(&self.0.c0.c2.c1),
            field_to_fq(&self.0.c1.c0.c0),
            field_to_fq(&self.0.c1.c0.c1),
            field_to_fq(&self.0.c1.c1.c0),
            field_to_fq(&self.0.c1.c1.c1),
            field_to_fq(&self.0.c1.c2.c0),
            field_to_fq(&self.0.c1.c2.c1),
        ]
    }

    /// Applies BN254 final exponentiation to this raw Miller-loop output.
    pub fn final_exponentiation(&self) -> Option<Bn254GT> {
        ArkBn254::final_exponentiation(MillerLoopOutput(self.0)).map(|value| Bn254GT(value.0))
    }
}

#[expect(
    clippy::expect_used,
    reason = "canonical BN254 Fq serialization into a fixed 32-byte buffer cannot fail"
)]
fn field_to_fq(value: &ark_bn254::Fq) -> Fq {
    use ark_serialize::CanonicalSerialize;

    let mut bytes = [0_u8; Fq::NUM_BYTES];
    value
        .serialize_compressed(&mut bytes[..])
        .expect("BN254 Fq serialization cannot fail");
    Fq::from_le_bytes_mod_order(&bytes)
}

#[expect(clippy::expect_used)]
impl AppendToTranscript for Bn254Fq12 {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        use ark_serialize::CanonicalSerialize;
        let mut buf = Vec::with_capacity(self.0.uncompressed_size());
        self.0
            .serialize_uncompressed(&mut buf)
            .expect("Fq12 serialization cannot fail");
        buf.reverse();
        transcript.append_bytes(&buf);
    }

    fn transcript_payload_len(&self) -> Option<u64> {
        use ark_serialize::CanonicalSerialize;
        Some(self.0.uncompressed_size() as u64)
    }
}

impl Serialize for Bn254Fq12 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use ark_serialize::CanonicalSerialize;
        let mut buf = Vec::with_capacity(self.0.compressed_size());
        self.0
            .serialize_compressed(&mut buf)
            .map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&buf)
    }
}

impl<'de> Deserialize<'de> for Bn254Fq12 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use ark_serialize::CanonicalDeserialize;
        let buf = <Vec<u8>>::deserialize(deserializer)?;
        let inner = Fq12::deserialize_compressed(&buf[..]).map_err(serde::de::Error::custom)?;
        if inner.is_zero() {
            return Err(serde::de::Error::custom("Fq12 Miller-loop output is zero"));
        }
        Ok(Self(inner))
    }
}
