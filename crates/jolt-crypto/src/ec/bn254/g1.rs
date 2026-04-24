use ark_bn254::G1Projective;

super::impl_jolt_group_wrapper!(
    Bn254G1,
    G1Projective,
    "BN254 G1 group element (projective coordinates)."
);
