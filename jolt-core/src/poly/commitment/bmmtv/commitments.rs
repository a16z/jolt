//! This crate defines what is a Double Homomorphic Commitment and
//! exports some implementations of it like Afgho, Identity and Pedersen

use ark_ec::CurveGroup;
use ark_std::rand::Rng;

pub mod afgho16;
pub mod identity;

pub type Error = anyhow::Error;

/// Helpers for generator commitment keys used by Pedersen and AFGHO16
pub fn random_generators<R: Rng, G: CurveGroup>(rng: &mut R, num: usize) -> Vec<G> {
    (0..num).map(|_| G::rand(rng)).collect()
}
