use ark_bn254::{Fr, G1Projective};
use ark_ec::AdditiveGroup;
use ark_ff::{BigInteger, PrimeField};
use ark_std::Zero;
use rayon::prelude::*;

use super::decomp_2d::{decompose_scalar_2d, glv_endomorphism};

/// Shamir's trick for 2-point scalar multiplication with sign handling.
pub(crate) fn shamir_glv_mul_2d(
    bases: &[G1Projective; 2],
    coeffs: &[<Fr as PrimeField>::BigInt; 2],
    signs: &[bool; 2],
) -> G1Projective {
    let mut result = G1Projective::zero();
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
                    result += base;
                } else {
                    result -= base;
                }
            }
        }
    }

    result
}

/// Precomputed Shamir lookup table for 2D GLV: all 16 combinations
/// of [P, λ(P)] with sign bits.
struct PrecomputedShamir2Table {
    table: [G1Projective; 16],
}

impl PrecomputedShamir2Table {
    fn new(bases: &[G1Projective; 2]) -> Self {
        let mut table = [G1Projective::zero(); 16];

        table.par_iter_mut().enumerate().for_each(|(idx, point)| {
            let point_mask = idx & 0x3;
            let sign_mask = idx >> 2;

            *point = G1Projective::zero();
            for (i, &base) in bases.iter().enumerate() {
                if (point_mask >> i) & 1 == 1 {
                    if (sign_mask >> i) & 1 == 1 {
                        *point -= base;
                    } else {
                        *point += base;
                    }
                }
            }
        });

        Self { table }
    }

    #[inline]
    fn get(&self, point_mask: usize, sign_mask: usize) -> G1Projective {
        self.table[point_mask | (sign_mask << 2)]
    }
}

/// Shamir's trick using precomputed table.
fn shamir_glv_mul_2d_precomputed(
    table: &PrecomputedShamir2Table,
    coeffs: &[<Fr as PrimeField>::BigInt; 2],
    signs: &[bool; 2],
) -> G1Projective {
    let mut result = G1Projective::zero();
    let max_bits = coeffs
        .iter()
        .map(|c| c.num_bits() as usize)
        .max()
        .unwrap_or(0);

    for bit_idx in (0..max_bits).rev() {
        result = result.double();

        let mut point_mask = 0;
        let mut sign_mask = 0;

        for (i, coeff) in coeffs.iter().enumerate() {
            if coeff.get_bit(bit_idx) {
                point_mask |= 1 << i;
                if !signs[i] {
                    sign_mask |= 1 << i;
                }
            }
        }

        if point_mask != 0 {
            result += table.get(point_mask, sign_mask);
        }
    }

    result
}

struct DecomposedScalar2D {
    coeffs: [<Fr as PrimeField>::BigInt; 2],
    signs: [bool; 2],
}

impl DecomposedScalar2D {
    fn from_scalar(scalar: Fr) -> Self {
        let (coeffs, signs) = decompose_scalar_2d(scalar);
        Self { coeffs, signs }
    }
}

struct FixedBasePrecomputedG1 {
    shamir_table: PrecomputedShamir2Table,
}

impl FixedBasePrecomputedG1 {
    fn new(base: &G1Projective) -> Self {
        let glv_bases = [*base, glv_endomorphism(base)];
        let shamir_table = PrecomputedShamir2Table::new(&glv_bases);
        Self { shamir_table }
    }

    fn mul_scalar(&self, scalar: Fr) -> G1Projective {
        let decomposed = DecomposedScalar2D::from_scalar(scalar);
        shamir_glv_mul_2d_precomputed(&self.shamir_table, &decomposed.coeffs, &decomposed.signs)
    }

    fn mul_scalars(&self, scalars: &[Fr]) -> Vec<G1Projective> {
        scalars
            .par_iter()
            .map(|&scalar| self.mul_scalar(scalar))
            .collect()
    }
}

/// Fixed-base vector MSM: compute base * scalars[i] for all i.
/// Used in Dory for g2_scaling by g_fin in eval_vmv_re.
pub fn fixed_base_vector_msm_g1(base: &G1Projective, scalars: &[Fr]) -> Vec<G1Projective> {
    let precomputed = FixedBasePrecomputedG1::new(base);
    precomputed.mul_scalars(scalars)
}
