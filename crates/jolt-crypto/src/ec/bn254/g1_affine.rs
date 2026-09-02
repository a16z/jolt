use ark_bn254::G1Affine;

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

impl std::fmt::Debug for Bn254G1Affine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Bn254G1Affine").field(&self.0).finish()
    }
}

impl serde::Serialize for Bn254G1Affine {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use ark_serialize::CanonicalSerialize;

        let mut buf = Vec::with_capacity(self.0.compressed_size());
        self.0
            .serialize_compressed(&mut buf)
            .map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&buf)
    }
}

impl<'de> serde::Deserialize<'de> for Bn254G1Affine {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use ark_serialize::CanonicalDeserialize;

        let buf = <Vec<u8>>::deserialize(deserializer)?;
        G1Affine::deserialize_compressed(&buf[..])
            .map(Self)
            .map_err(serde::de::Error::custom)
    }
}
