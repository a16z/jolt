use std::fmt::{Debug, Formatter, Result as FmtResult};

use ark_bn254::G1Affine;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::de::Error as DeserializeError;
use serde::ser::Error as SerializeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// BN254 G1 point prepared for multi-scalar multiplication.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct Bn254G1Affine(pub(crate) G1Affine);

impl Bn254G1Affine {
    #[inline(always)]
    pub(crate) fn as_inner_slice(slice: &[Self]) -> &[G1Affine] {
        // SAFETY: Bn254G1Affine is repr(transparent) over G1Affine.
        unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<G1Affine>(), slice.len()) }
    }
}

impl Debug for Bn254G1Affine {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_tuple("Bn254G1Affine").field(&self.0).finish()
    }
}

impl Serialize for Bn254G1Affine {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut buf = Vec::with_capacity(self.0.compressed_size());
        self.0
            .serialize_compressed(&mut buf)
            .map_err(SerializeError::custom)?;
        serializer.serialize_bytes(&buf)
    }
}

impl<'de> Deserialize<'de> for Bn254G1Affine {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let buf = <Vec<u8>>::deserialize(deserializer)?;
        G1Affine::deserialize_compressed(&buf[..])
            .map(Self)
            .map_err(DeserializeError::custom)
    }
}
