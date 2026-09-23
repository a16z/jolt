use ark_bn254::{G2Affine, G2Projective};

super::impl_jolt_group_wrapper!(
    Bn254G2,
    G2Projective,
    G2Affine,
    "BN254 G2 group element (projective coordinates)."
);

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests may fail loudly")]
mod tests {
    use super::*;
    use ark_serialize::CanonicalSerialize;
    use ark_std::UniformRand;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// Regression guard for the untrusted-bytes boundary: BN254 G2 has a
    /// non-trivial cofactor, so an on-curve point outside the order-r subgroup
    /// exists and must be rejected by serde deserialization (arkworks
    /// `Validate::Yes`). Protects against a future arkworks bump or a switch
    /// to `deserialize_*_unchecked` silently dropping the subgroup check.
    #[test]
    fn deserialize_rejects_on_curve_non_subgroup_point() {
        let mut rng = ChaCha20Rng::seed_from_u64(9);
        let point = loop {
            let x = ark_bn254::Fq2::rand(&mut rng);
            if let Some(p) = G2Affine::get_point_from_x_unchecked(x, true) {
                // Cofactor ~2^254: a random curve point is in the r-subgroup
                // only with negligible probability.
                if !p.is_in_correct_subgroup_assuming_on_curve() {
                    break p;
                }
            }
        };
        let mut buf = Vec::new();
        point.serialize_compressed(&mut buf).unwrap();
        let json = serde_json::to_string(&buf).unwrap();
        assert!(serde_json::from_str::<Bn254G2>(&json).is_err());
    }

    #[test]
    fn deserialize_accepts_subgroup_point() {
        let g = crate::Bn254::g2_generator();
        let json = serde_json::to_string(&g).unwrap();
        let recovered: Bn254G2 = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered, g);
    }
}
