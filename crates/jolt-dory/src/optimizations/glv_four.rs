//! 4D GLV scalar multiplication for BN254 G2.
//!
//! Uses Strauss-Shamir interleaving with four ~64-bit scalars from the 4D decomposition.
//! The four bases are `P, psi(P), psi^2(P), psi^3(P)` where `psi` is the Frobenius endomorphism.

use ark_bn254::{Fr, G2Projective};
use ark_ec::AdditiveGroup;
use ark_ff::{BigInteger, PrimeField};
use ark_std::Zero;
use rayon::prelude::*;

use super::decomp_4d::decompose_scalar_4d;
use super::frobenius::frobenius_psi_power_projective;

/// Online 4D GLV scalar multiplication for BN254 G2.
/// Decomposes scalar into 4 ~66-bit components using Frobenius endomorphism,
/// then uses Strauss-Shamir interleaving.
pub fn glv_four_scalar_mul_online(scalar: Fr, points: &[G2Projective]) -> Vec<G2Projective> {
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    points
        .par_iter()
        .map(|point| {
            let bases = [
                *point,
                frobenius_psi_power_projective(point, 1),
                frobenius_psi_power_projective(point, 2),
                frobenius_psi_power_projective(point, 3),
            ];
            shamir_glv_mul_4d(&bases, &coeffs, &signs)
        })
        .collect()
}

/// Shamir's trick for 4-point scalar multiplication with sign handling.
pub(crate) fn shamir_glv_mul_4d(
    bases: &[G2Projective; 4],
    coeffs: &[<Fr as PrimeField>::BigInt; 4],
    signs: &[bool; 4],
) -> G2Projective {
    let mut result = G2Projective::zero();
    let max_bits = coeffs
        .iter()
        .map(|c| c.num_bits() as usize)
        .max()
        .unwrap_or(0);

    for bit_idx in (0..max_bits).rev() {
        result = result.double();

        for (i, (coeff, &base)) in coeffs.iter().zip(bases.iter()).enumerate() {
            if coeff.get_bit(bit_idx) {
                if signs[i] {
                    // signs[i] = true means negative in 4D decomposition
                    result -= base;
                } else {
                    result += base;
                }
            }
        }
    }

    result
}
