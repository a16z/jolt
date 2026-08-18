//! BN254-specific GLV optimizations for elliptic curve scalar multiplication.
//!
//! Provides GLV scalar multiplication (2D for G1, 4D for G2 via Frobenius endomorphism)
//! and vector-scalar operations used in Dory's inner-product argument rounds.
//!
//! References:
//! - GLV: Gallant, Lambert, Vanstone. [Faster Point Multiplication on Elliptic Curves](https://link.springer.com/chapter/10.1007/3-540-44647-8_11) (CRYPTO 2001)
//! - Dory: Lee. [Dory: Efficient, Transparent arguments for Generalised Inner Products](https://eprint.iacr.org/2020/1274)

mod constants;
mod decomp_2d;
mod decomp_4d;
mod frobenius;
pub mod glv_four;
pub mod glv_two;

pub mod dory_g1;
pub mod dory_g2;

use ark_bn254::G1Projective;
use jolt_field::Fr;

use super::{field_to_fr, Bn254G1, Bn254G2};

/// `v[i] += scalar * generators[i]`, using 2D GLV decomposition + rayon.
#[inline]
pub fn vector_add_scalar_mul_g1(v: &mut [Bn254G1], generators: &[Bn254G1], scalar: Fr) {
    let v_inner = Bn254G1::as_inner_slice_mut(v);
    let gens_inner = Bn254G1::as_inner_slice(generators);
    dory_g1::vector_add_scalar_mul_g1_online(v_inner, gens_inner, field_to_fr(&scalar));
}

/// `v[i] = scalar * v[i] + gamma[i]`, using 2D GLV decomposition + rayon.
#[inline]
pub fn vector_scalar_mul_add_gamma_g1(v: &mut [Bn254G1], scalar: Fr, gamma: &[Bn254G1]) {
    let v_inner = Bn254G1::as_inner_slice_mut(v);
    let gamma_inner = Bn254G1::as_inner_slice(gamma);
    dory_g1::vector_scalar_mul_add_gamma_g1_online(v_inner, field_to_fr(&scalar), gamma_inner);
}

/// Fixed-base MSM: `[base * scalars[0], ..., base * scalars[n-1]]`, using precomputed Shamir table.
#[inline]
pub fn fixed_base_vector_msm_g1(base: &Bn254G1, scalars: &[Fr]) -> Vec<Bn254G1> {
    let ark_scalars: Vec<ark_bn254::Fr> = scalars.iter().map(field_to_fr).collect();
    let inner: G1Projective = (*base).into();
    let results = glv_two::fixed_base_vector_msm_g1(&inner, &ark_scalars);
    results.into_iter().map(Bn254G1::from).collect()
}

/// `v[i] += scalar * generators[i]`, using 4D GLV with Frobenius endomorphism + rayon.
#[inline]
pub fn vector_add_scalar_mul_g2(v: &mut [Bn254G2], generators: &[Bn254G2], scalar: Fr) {
    let v_inner = Bn254G2::as_inner_slice_mut(v);
    let gens_inner = Bn254G2::as_inner_slice(generators);
    dory_g2::vector_add_scalar_mul_g2_online(v_inner, gens_inner, field_to_fr(&scalar));
}

/// `v[i] = scalar * v[i] + gamma[i]`, using 4D GLV with Frobenius + rayon.
#[inline]
pub fn vector_scalar_mul_add_gamma_g2(v: &mut [Bn254G2], scalar: Fr, gamma: &[Bn254G2]) {
    let v_inner = Bn254G2::as_inner_slice_mut(v);
    let gamma_inner = Bn254G2::as_inner_slice(gamma);
    dory_g2::vector_scalar_mul_add_gamma_g2_online(v_inner, field_to_fr(&scalar), gamma_inner);
}

/// Scalar multiplication of multiple G2 points by a single scalar, using 4D GLV.
#[inline]
pub fn glv_four_scalar_mul(scalar: Fr, points: &[Bn254G2]) -> Vec<Bn254G2> {
    let points_inner = Bn254G2::as_inner_slice(points);
    let results = glv_four::glv_four_scalar_mul_online(field_to_fr(&scalar), points_inner);
    results.into_iter().map(Bn254G2::from).collect()
}
