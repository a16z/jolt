use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[derive(Clone, Copy, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct KZGVerifierKey<P: Pairing> {
    pub g1: P::G1Affine,
    pub g2: P::G2Affine,
    pub beta_g2: P::G2Affine,
}
