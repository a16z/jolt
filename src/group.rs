use ark_bls12_381::G1Projective;
use ark_ec::ProjectiveCurve;
use ark_serialize::*;

pub type GroupElement = G1Projective;

#[derive(Debug, Clone, Copy)]
pub struct CompressedGroup([u8; 48]);

// GROUP_BASEPOINT_COMPRESSED
pub(crate) fn group_basepoint_compressed() -> Vec<u8> {
  let mut buf = vec![];
  GroupElement::prime_subgroup_generator()
    .serialize(&mut buf)
    .unwrap();
  buf
}

impl CanonicalSerialize for CompressedGroup {
  fn serialize<W: Write>(&self, mut writer: W) -> Result<(), SerializationError> {
    Ok(writer.write_all(self.0.as_ref())?)
  }

  fn serialized_size(&self) -> usize {
    48
  }
}

impl CanonicalDeserialize for CompressedGroup {
  /// Reads `Self` from `reader`.
  fn deserialize<R: Read>(mut reader: R) -> Result<Self, SerializationError> {
    let mut data = [0u8; 48];
    reader.read_exact(&mut data)?;
    Ok(Self(data))
  }
}

impl From<&CompressedGroup> for GroupElement {
  fn from(input: &CompressedGroup) -> Self {
    GroupElement::deserialize(input.0.as_ref()).unwrap()
  }
}

impl From<&GroupElement> for CompressedGroup {
  fn from(input: &GroupElement) -> Self {
    let mut res = [0u8; 48];
    let mut buf = vec![];
    input.serialize(&mut buf).unwrap();
    res.copy_from_slice(buf.as_slice());
    Self(res)
  }
}
