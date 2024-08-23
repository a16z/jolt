use crate::poly::commitment::kzg::KZGVerifierKey;
use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[derive(Copy, Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct HyperKZGVerifierKey<P: Pairing> {
    pub kzg_vk: KZGVerifierKey<P>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct HyperKZGProof<P: Pairing> {
    pub com: Vec<P::G1Affine>,
    pub w: Vec<P::G1Affine>,
    pub v: Vec<Vec<P::ScalarField>>,
}

#[derive(Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyperKZGCommitment<P: Pairing>(pub P::G1Affine);
