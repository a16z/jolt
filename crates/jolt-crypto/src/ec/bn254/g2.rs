use ark_bn254::G2Projective;

super::impl_jolt_group_wrapper!(
    Bn254G2,
    G2Projective,
    "BN254 G2 group element (projective coordinates)."
);
