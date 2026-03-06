//! Vector-scalar operations on G2 using 4D GLV with Frobenius, for Dory inner-product argument rounds.

use ark_bn254::{Fr, G2Projective};
use rayon::prelude::*;

use super::decomp_4d::decompose_scalar_4d;
use super::frobenius::frobenius_psi_power_projective;
use super::glv_four::shamir_glv_mul_4d;

/// `v[i] += scalar * generators[i]`, using 4D GLV decomposition with Frobenius.
pub fn vector_add_scalar_mul_g2_online(
    v: &mut [G2Projective],
    generators: &[G2Projective],
    scalar: Fr,
) {
    assert_eq!(v.len(), generators.len());
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    v.par_iter_mut()
        .zip(generators.par_iter())
        .for_each(|(vi, gen)| {
            let bases = [
                *gen,
                frobenius_psi_power_projective(gen, 1),
                frobenius_psi_power_projective(gen, 2),
                frobenius_psi_power_projective(gen, 3),
            ];
            *vi += shamir_glv_mul_4d(&bases, &coeffs, &signs);
        });
}

/// `v[i] = scalar * v[i] + gamma[i]`, using 4D GLV decomposition with Frobenius.
pub fn vector_scalar_mul_add_gamma_g2_online(
    v: &mut [G2Projective],
    scalar: Fr,
    gamma: &[G2Projective],
) {
    assert_eq!(v.len(), gamma.len());
    let (coeffs, signs) = decompose_scalar_4d(scalar);

    v.par_iter_mut()
        .zip(gamma.par_iter())
        .for_each(|(vi, &gamma_i)| {
            let bases = [
                *vi,
                frobenius_psi_power_projective(vi, 1),
                frobenius_psi_power_projective(vi, 2),
                frobenius_psi_power_projective(vi, 3),
            ];
            *vi = shamir_glv_mul_4d(&bases, &coeffs, &signs) + gamma_i;
        });
}
