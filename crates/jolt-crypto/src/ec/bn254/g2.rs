use ark_bn254::{G2Affine, G2Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::Zero;
use jolt_field::{FixedByteSize, Fq, FromPrimitiveInt};

super::impl_jolt_group_wrapper!(
    Bn254G2,
    G2Projective,
    G2Affine,
    "BN254 G2 group element (projective coordinates)."
);

impl Bn254G2 {
    /// Returns affine `(x.c0, x.c1, y.c0, y.c1, infinity)` coordinates over BN254 Fq.
    pub fn affine_coordinates_with_infinity(&self) -> [Fq; 5] {
        let affine = self.0.into_affine();
        if affine.infinity {
            return [
                Fq::default(),
                Fq::default(),
                Fq::default(),
                Fq::default(),
                Fq::from_u64(1),
            ];
        }

        [
            field_to_fq(&affine.x.c0),
            field_to_fq(&affine.x.c1),
            field_to_fq(&affine.y.c0),
            field_to_fq(&affine.y.c1),
            Fq::default(),
        ]
    }

    pub fn from_affine_coordinates_with_infinity(coordinates: [Fq; 5]) -> Option<Self> {
        let [x0, x1, y0, y1, infinity] = coordinates;
        if infinity == Fq::from_u64(1) {
            if x0 != Fq::default()
                || x1 != Fq::default()
                || y0 != Fq::default()
                || y1 != Fq::default()
            {
                return None;
            }
            return Some(Self(G2Projective::zero()));
        }
        if infinity != Fq::default() {
            return None;
        }

        let affine = G2Affine::new_unchecked(
            ark_bn254::Fq2::new(super::fq_to_field(&x0), super::fq_to_field(&x1)),
            ark_bn254::Fq2::new(super::fq_to_field(&y0), super::fq_to_field(&y1)),
        );
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
