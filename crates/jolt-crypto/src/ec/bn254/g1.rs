use ark_bn254::{G1Affine, G1Projective};

super::impl_jolt_group_wrapper!(
    Bn254G1,
    G1Projective,
    G1Affine,
    "BN254 G1 group element (projective coordinates)."
);
