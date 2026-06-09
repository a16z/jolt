use ark_bn254::{G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::Zero;
use jolt_field::{FixedByteSize, Fq, FromPrimitiveInt};

super::impl_jolt_group_wrapper!(
    Bn254G1,
    G1Projective,
    G1Affine,
    "BN254 G1 group element (projective coordinates)."
);

impl Bn254G1 {
    /// Returns affine `(x, y, infinity)` coordinates over BN254 Fq.
    pub fn affine_coordinates_with_infinity(&self) -> [Fq; 3] {
        let affine = self.0.into_affine();
        if affine.infinity {
            return [Fq::default(), Fq::default(), Fq::from_u64(1)];
        }

        [
            field_to_fq(&affine.x),
            field_to_fq(&affine.y),
            Fq::default(),
        ]
    }

    pub fn from_affine_coordinates_with_infinity(coordinates: [Fq; 3]) -> Option<Self> {
        let [x, y, infinity] = coordinates;
        if infinity == Fq::from_u64(1) {
            if x != Fq::default() || y != Fq::default() {
                return None;
            }
            return Some(Self(G1Projective::zero()));
        }
        if infinity != Fq::default() {
            return None;
        }

        let affine = G1Affine::new_unchecked(super::fq_to_field(&x), super::fq_to_field(&y));
        if !affine.is_on_curve() || !affine.is_in_correct_subgroup_assuming_on_curve() {
            return None;
        }

        Some(Self(affine.into_group()))
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
