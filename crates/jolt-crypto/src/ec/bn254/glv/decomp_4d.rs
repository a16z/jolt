//! 4D GLV scalar decomposition for BN254 G2.
//!
//! Uses a precomputed lookup table of power-of-2 decompositions to split a scalar
//! into four ~64-bit components for the Frobenius-based 4D GLV multiplication.

use super::constants::POWER_OF_2_DECOMPOSITIONS;
use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use num_bigint::{BigInt, Sign};

fn fr_to_bigint(fr: Fr) -> BigInt {
    let bytes = fr.into_bigint().to_bytes_be();
    BigInt::from_bytes_be(Sign::Plus, &bytes)
}

/// Table-based 4-dimensional scalar decomposition.
/// Decomposes k into (k0, k1, k2, k3) such that
/// k ≡ k0 + k1·λ + k2·λ² + k3·λ³ (mod r).
/// Each coefficient is at most ~66 bits.
/// Returns `(|kᵢ|, signsᵢ)` with `signsᵢ = true` meaning positive — matching
/// the 2D convention in [`super::decomp_2d::decompose_scalar_2d`].
fn decompose_scalar_table_based(scalar: &BigInt) -> ([u128; 4], [bool; 4]) {
    // Sign detection via `(k as i128) < 0` is sound only when |k| < 2^127.
    // For BN254, each table row contributes at most ~66 bits and the loop
    // runs at most `TABLE_LEN` times, bounding the accumulator to ~2^74.
    // A debug_assert at the end catches any future violation (e.g., a
    // corrupted table or a port to a curve with a larger endomorphism).
    const SIGN_SAFE_MAGNITUDE_BITS: u32 = 100;

    let mut k0 = 0u128;
    let mut k1 = 0u128;
    let mut k2 = 0u128;
    let mut k3 = 0u128;

    let mut temp_scalar = scalar.clone();
    let mut bit_position = 0;

    while temp_scalar > BigInt::from(0) && bit_position < POWER_OF_2_DECOMPOSITIONS.len() {
        if &temp_scalar & BigInt::from(1) == BigInt::from(1) {
            let (decomp_k0, decomp_k1, decomp_k2, decomp_k3, neg0, neg1, neg2, neg3) =
                POWER_OF_2_DECOMPOSITIONS[bit_position];

            if neg0 {
                k0 = k0.wrapping_sub(decomp_k0);
            } else {
                k0 = k0.wrapping_add(decomp_k0);
            }
            if neg1 {
                k1 = k1.wrapping_sub(decomp_k1);
            } else {
                k1 = k1.wrapping_add(decomp_k1);
            }
            if neg2 {
                k2 = k2.wrapping_sub(decomp_k2);
            } else {
                k2 = k2.wrapping_add(decomp_k2);
            }
            if neg3 {
                k3 = k3.wrapping_sub(decomp_k3);
            } else {
                k3 = k3.wrapping_add(decomp_k3);
            }
        }

        temp_scalar >>= 1;
        bit_position += 1;
    }

    let (final_k0, sign0) = if (k0 as i128) < 0 {
        (k0.wrapping_neg(), false)
    } else {
        (k0, true)
    };

    let (final_k1, sign1) = if (k1 as i128) < 0 {
        (k1.wrapping_neg(), false)
    } else {
        (k1, true)
    };

    let (final_k2, sign2) = if (k2 as i128) < 0 {
        (k2.wrapping_neg(), false)
    } else {
        (k2, true)
    };

    let (final_k3, sign3) = if (k3 as i128) < 0 {
        (k3.wrapping_neg(), false)
    } else {
        (k3, true)
    };

    let sign_safe_max: u128 = 1u128 << SIGN_SAFE_MAGNITUDE_BITS;
    debug_assert!(
        final_k0 < sign_safe_max
            && final_k1 < sign_safe_max
            && final_k2 < sign_safe_max
            && final_k3 < sign_safe_max,
        "4D GLV decomposition exceeded safe magnitude; sign detection may be wrong",
    );

    (
        [final_k0, final_k1, final_k2, final_k3],
        [sign0, sign1, sign2, sign3],
    )
}

/// Decompose a BN254 scalar for 4D GLV multiplication (G2).
/// Returns `(coefficients, signs)` where `signs[i] = true` means positive,
/// mirroring the convention used by [`super::decomp_2d::decompose_scalar_2d`].
pub fn decompose_scalar_4d(scalar: Fr) -> ([<Fr as PrimeField>::BigInt; 4], [bool; 4]) {
    let scalar_bigint = fr_to_bigint(scalar);
    let (coeffs_u128, signs) = decompose_scalar_table_based(&scalar_bigint);

    let coeffs = [
        Fr::from(coeffs_u128[0]).into_bigint(),
        Fr::from(coeffs_u128[1]).into_bigint(),
        Fr::from(coeffs_u128[2]).into_bigint(),
        Fr::from(coeffs_u128[3]).into_bigint(),
    ];

    (coeffs, signs)
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests convert decomposition magnitudes that are below the modulus by construction"
)]
mod tests {
    use super::super::frobenius::frobenius_psi_power_projective;
    use super::*;
    use ark_bn254::{Fq, G2Projective};
    use ark_ec::AffineRepr;
    use ark_ff::PrimeField;
    use ark_std::{UniformRand, Zero};
    use num_bigint::BigUint;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// The 4D GLV eigenvalue: on G2 the twisted Frobenius ψ acts as
    /// multiplication by q mod r (the base-field characteristic reduced into
    /// the scalar field). Anchored against the group in
    /// [`lambda_powers_match_frobenius_powers_on_g2`].
    fn lambda() -> Fr {
        Fr::from(BigUint::from_bytes_be(&Fq::MODULUS.to_bytes_be()))
    }

    // Ties the scalar-field lambda used by the reconstruction check to the
    // actual psi powers the table test verifies the table against.
    #[test]
    fn lambda_powers_match_frobenius_powers_on_g2() {
        let lambda = lambda();
        let g = ark_bn254::G2Affine::generator().into_group();
        let mut power = Fr::from(1u64);
        for exponent in 1usize..4 {
            power *= lambda;
            assert_eq!(
                frobenius_psi_power_projective(&g, exponent),
                g * power,
                "psi^{exponent} must act as multiplication by lambda^{exponent}"
            );
        }
    }

    #[test]
    fn decomposition_reconstructs_scalar_within_documented_bounds() {
        let lambda = lambda();

        // Worst-case per-component magnitude: each scalar bit adds at most one
        // table row, so |k_i| is bounded by the column-wise sum of row
        // magnitudes. The sum must stay far below 2^100, the documented
        // sign-detection safety margin.
        let mut bounds = [0u128; 4];
        for &(k0, k1, k2, k3, ..) in &POWER_OF_2_DECOMPOSITIONS {
            bounds[0] += k0;
            bounds[1] += k1;
            bounds[2] += k2;
            bounds[3] += k3;
        }
        for (component, bound) in bounds.iter().enumerate() {
            assert!(
                *bound < 1u128 << 100,
                "component {component} worst case {bound} breaches the sign-safety margin"
            );
        }

        let mut rng = ChaCha20Rng::seed_from_u64(0x4d61);
        let mut scalars: Vec<Fr> = (0..64).map(|_| Fr::rand(&mut rng)).collect();
        scalars.extend([
            Fr::from(0u64),
            Fr::from(1u64),
            -Fr::from(1u64), // r - 1
            lambda,
            lambda * lambda,
            lambda * lambda * lambda,
            lambda - Fr::from(1u64),
            // magnitudes at the ~64-66 bit component boundary
            Fr::from(u64::MAX),
            Fr::from((1u128 << 66) - 1),
            Fr::from(1u128 << 66),
            Fr::from(u128::MAX),
        ]);

        for scalar in scalars {
            let (coeffs, signs) = decompose_scalar_4d(scalar);
            let mut reconstructed = Fr::from(0u64);
            let mut power = Fr::from(1u64);
            for (coeff, sign) in coeffs.iter().zip(signs) {
                let value = Fr::from_bigint(*coeff).unwrap();
                reconstructed += power * if sign { value } else { -value };
                power *= lambda;
            }
            assert_eq!(
                reconstructed, scalar,
                "sum k_i*lambda^i must reconstruct {scalar}"
            );
            for (component, coeff) in coeffs.iter().enumerate() {
                let mag = BigUint::from_bytes_be(&coeff.to_bytes_be());
                assert!(
                    mag <= BigUint::from(bounds[component]),
                    "|k{component}| = {mag} exceeds the table-derived bound for {scalar}"
                );
            }
        }
    }

    // Independent per-row verification of the power-of-2 table. Catches table
    // corruption that end-to-end GLV tests can miss when wrong rows cancel out
    // for random scalars.
    #[test]
    fn power_of_2_decomposition_table_matches_reference() {
        let g = ark_bn254::G2Affine::generator().into_group();
        let psi = [
            g,
            frobenius_psi_power_projective(&g, 1),
            frobenius_psi_power_projective(&g, 2),
            frobenius_psi_power_projective(&g, 3),
        ];

        let mut power_of_2 = Fr::from(1u64);
        for (i, &(k0, k1, k2, k3, neg0, neg1, neg2, neg3)) in
            POWER_OF_2_DECOMPOSITIONS.iter().enumerate()
        {
            let coeffs = [k0, k1, k2, k3];
            let negs = [neg0, neg1, neg2, neg3];

            let mut lhs = G2Projective::zero();
            for ((coeff, neg), base) in coeffs.iter().zip(negs.iter()).zip(psi.iter()) {
                let term = *base * Fr::from(*coeff);
                lhs += if *neg { -term } else { term };
            }

            let rhs = g * power_of_2;
            assert_eq!(
                lhs, rhs,
                "power-of-2 decomposition row {i} does not reproduce 2^{i}·G",
            );

            power_of_2 += power_of_2;
        }
    }
}
