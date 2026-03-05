use ark_bn254::{Fr, G1Projective};
use rayon::prelude::*;

use super::decomp_2d::{decompose_scalar_2d, glv_endomorphism};
use super::glv_two::shamir_glv_mul_2d;

/// `v[i] += scalar * generators[i]`, using 2D GLV decomposition.
pub fn vector_add_scalar_mul_g1_online(
    v: &mut [G1Projective],
    generators: &[G1Projective],
    scalar: Fr,
) {
    assert_eq!(v.len(), generators.len());
    let (coeffs, signs) = decompose_scalar_2d(scalar);

    v.par_iter_mut()
        .zip(generators.par_iter())
        .for_each(|(vi, gen)| {
            let bases = [*gen, glv_endomorphism(gen)];
            *vi += shamir_glv_mul_2d(&bases, &coeffs, &signs);
        });
}

/// `v[i] = scalar * v[i] + gamma[i]`, using 2D GLV decomposition.
pub fn vector_scalar_mul_add_gamma_g1_online(
    v: &mut [G1Projective],
    scalar: Fr,
    gamma: &[G1Projective],
) {
    assert_eq!(v.len(), gamma.len());
    let (coeffs, signs) = decompose_scalar_2d(scalar);

    v.par_iter_mut()
        .zip(gamma.par_iter())
        .for_each(|(vi, &gamma_i)| {
            let bases = [*vi, glv_endomorphism(vi)];
            *vi = shamir_glv_mul_2d(&bases, &coeffs, &signs) + gamma_i;
        });
}
