//! Concrete BN254 curve implementation.
//!
//! This module wraps the arkworks `ark-bn254` crate behind the generic
//! `JoltGroup` and `PairingGroup` traits. Arkworks types never appear in
//! the public API — all conversions happen internally.

mod g1;
mod g2;
mod gt;

pub use g1::Bn254G1;
pub use g2::Bn254G2;
pub use gt::Bn254GT;

use ark_bn254::{Bn254 as ArkBn254, G1Affine, G1Projective, G2Affine};
use ark_ec::pairing::Pairing;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::PrimeField as _;
use ark_std::UniformRand;
use jolt_field::Field;
use rand_core::RngCore;

use crate::PairingGroup;

/// BN254 pairing-friendly curve.
#[derive(Clone, Debug, Default)]
pub struct Bn254;

impl PairingGroup for Bn254 {
    type G1 = Bn254G1;
    type G2 = Bn254G2;
    type GT = Bn254GT;

    fn gt_one() -> Self::GT {
        Bn254GT::one()
    }

    #[inline]
    fn g1_generator() -> Self::G1 {
        Bn254G1(G1Affine::generator().into())
    }

    #[inline]
    fn g2_generator() -> Self::G2 {
        Bn254G2(G2Affine::generator().into())
    }

    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::GT {
        Bn254GT(ArkBn254::pairing(g1.0, g2.0).0)
    }

    fn multi_pairing(g1s: &[Self::G1], g2s: &[Self::G2]) -> Self::GT {
        debug_assert_eq!(g1s.len(), g2s.len());
        let g1_affines: Vec<G1Affine> = g1s.iter().map(|g| g.0.into_affine()).collect();
        let g2_affines: Vec<ark_bn254::G2Affine> = g2s.iter().map(|g| g.0.into_affine()).collect();
        Bn254GT(ArkBn254::multi_pairing(&g1_affines, &g2_affines).0)
    }

    fn random_g1<R: RngCore>(rng: &mut R) -> Self::G1 {
        Bn254G1(G1Projective::rand(rng))
    }
}

/// Converts a generic `Field` element to an arkworks `Fr` via serialization.
///
/// This is the bridge between jolt-field's backend-agnostic `Field` trait and
/// arkworks' concrete scalar type. The conversion goes through little-endian
/// byte serialization.
#[inline]
pub(crate) fn field_to_fr<F: Field>(f: &F) -> ark_bn254::Fr {
    let bytes = f.to_bytes();
    ark_bn254::Fr::from_le_bytes_mod_order(&bytes)
}
