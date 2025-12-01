//! Pedersen commitment scheme implementation

use ark_ec::{pairing::Pairing, CurveGroup};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{vec::Vec, UniformRand};
use rand::SeedableRng;

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct PedersenCommitment<G: CurveGroup> {
    pub commitment: G,
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct PedersenGenerators<G: CurveGroup> {
    pub generators: Vec<G::Affine>,
}

impl<G: CurveGroup> PedersenGenerators<G> {
    pub fn new(size: usize, label: &[u8]) -> Self {
        // Deterministic generation using label as seed
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(
            label.iter().fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64))
        );

        Self {
            generators: (0..size).map(|_| G::rand(&mut rng).into_affine()).collect(),
        }
    }
}

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct PedersenGeneratorsPairing<P: Pairing> {
    pub generators_g1: Vec<P::G1Affine>,
    pub generators_g2: Vec<P::G2Affine>,
    pub h_g1: P::G1Affine,
    pub h_g2: P::G2Affine,
}

impl<P: Pairing> PedersenGeneratorsPairing<P> {
    pub fn new(size: usize, label: &[u8]) -> Self {
        // Deterministic generation using label as seed
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(
            label.iter().fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64))
        );

        Self {
            generators_g1: (0..size).map(|_| P::G1::rand(&mut rng).into_affine()).collect(),
            generators_g2: (0..size).map(|_| P::G2::rand(&mut rng).into_affine()).collect(),
            h_g1: P::G1::rand(&mut rng).into_affine(),
            h_g2: P::G2::rand(&mut rng).into_affine(),
        }
    }
}

impl<G: CurveGroup> PedersenCommitment<G> {
    pub fn commit_vector(values: &[G::ScalarField], gens: &[G::Affine]) -> G {
        // Multi-scalar multiplication
        G::msm_unchecked(gens, values)
    }
}