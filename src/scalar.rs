use ark_bls12_381::Fr;

/// Scalar
pub type Scalar = Fr;

// #[derive(Debug, Clone, Copy)]
// pub struct ScalarBytes([u8; 32]);

// impl CanonicalSerialize for ScalarBytes {
//   fn serialize<W: Write>(&self, writer: W) -> Result<(), SerializationError> {
//     Ok(writer.write_all(self.0.as_ref())?)
//   }

//   fn serialized_size(&self) -> usize {
//     32
//   }
// }

// impl CanonicalDeserialize for ScalarBytes {
//   /// Reads `Self` from `reader`.
//   fn deserialize<R: Read>(reader: R) -> Result<Self, SerializationError> {
//     let mut data = [0u8; 32];
//     reader.read_exact(&mut data)?;
//     Ok(Self(data))
//   }
// }

// impl From<&ScalarBytes> for Scalar {
//   fn from(input: &ScalarBytes) -> Self {
//     Scalar::deserialize(input.0.as_ref()).unwrap()
//   }
// }

// impl From<&Scalar> for ScalarBytes {
//   fn from(input: &Scalar) -> Self {
//     let mut res = [0u8; 32];
//     let mut buf = vec![];
//     input.serialize(&mut buf).unwrap();
//     res.copy_from_slice(buf.as_slice());
//     Self(res)
//   }
// }
