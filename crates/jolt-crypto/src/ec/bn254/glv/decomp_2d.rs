//! 2D GLV scalar decomposition for BN254 G1.
//!
//! Decomposes a scalar `k` into `k = k0 + k1 * lambda (mod n)` where `lambda`
//! is the GLV endomorphism eigenvalue, halving the bit-length of each component.

use ark_bn254::{Fq, Fr, G1Projective};
use ark_ff::{BigInteger, MontFp, PrimeField};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, Signed};
use std::ops::AddAssign;

/// GLV endomorphism coefficient for BN254 G1: `β` such that `[λ]P = (β·x, y)`
const ENDO_COEFF: Fq =
    MontFp!("21888242871839275220042445260109153167277707414472061641714758635765020556616");

/// Lattice coefficients for the BN254 GLV decomposition
const SCALAR_DECOMP_COEFFS: [(bool, <Fr as PrimeField>::BigInt); 4] = [
    (
        false,
        ark_ff::BigInt!("147946756881789319000765030803803410728"),
    ),
    (true, ark_ff::BigInt!("9931322734385697763")),
    (false, ark_ff::BigInt!("9931322734385697763")),
    (
        false,
        ark_ff::BigInt!("147946756881789319010696353538189108491"),
    ),
];

/// Decompose a BN254 scalar into two ~128-bit components via GLV lattice reduction.
/// Returns (coefficients, signs) where `signs[i]` = true means positive.
#[expect(clippy::unwrap_used)]
pub fn decompose_scalar_2d(scalar: Fr) -> ([<Fr as PrimeField>::BigInt; 2], [bool; 2]) {
    let scalar_bytes = scalar.into_bigint().to_bytes_be();
    let scalar_bigint = BigInt::from_bytes_be(Sign::Plus, &scalar_bytes);

    let coeff_bigints: [BigInt; 4] = SCALAR_DECOMP_COEFFS.map(|x| {
        let bytes = x.1.to_bytes_be();
        let sign = if x.0 { Sign::Plus } else { Sign::Minus };
        BigInt::from_bytes_be(sign, &bytes)
    });

    let [n11, n12, n21, n22] = coeff_bigints;

    let r_bytes = Fr::MODULUS.to_bytes_be();
    let r = BigInt::from_bytes_be(Sign::Plus, &r_bytes);

    // β = (k·n22, -k·n12) / r
    let beta_1 = {
        let (mut div, rem) = (&scalar_bigint * &n22).div_rem(&r);
        if (&rem + &rem) > r {
            div.add_assign(BigInt::one());
        }
        div
    };
    let beta_2 = {
        let (mut div, rem) = (&scalar_bigint * &(-&n12)).div_rem(&r);
        if (&rem + &rem) > r {
            div.add_assign(BigInt::one());
        }
        div
    };

    // b = β · N
    let b1 = &beta_1 * &n11 + &beta_2 * &n21;
    let b2 = &beta_1 * &n12 + &beta_2 * &n22;

    let k1 = &scalar_bigint - b1;
    let k1_abs = BigUint::try_from(k1.abs()).unwrap();

    let k2 = -b2;
    let k2_abs = BigUint::try_from(k2.abs()).unwrap();

    let k1_fr = Fr::from(k1_abs);
    let k2_fr = Fr::from(k2_abs);

    let k_bigint = [k1_fr.into_bigint(), k2_fr.into_bigint()];
    let signs = [k1.sign() == Sign::Plus, k2.sign() == Sign::Plus];

    (k_bigint, signs)
}

/// Apply the GLV endomorphism to a G1 point: (x, y) → (β·x, y)
pub fn glv_endomorphism(point: &G1Projective) -> G1Projective {
    let mut res = *point;
    res.x *= ENDO_COEFF;
    res
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests invert fixed curve constants that are nonzero by construction"
)]
mod tests {
    use super::*;
    use ark_bn254::G1Affine;
    use ark_ec::AffineRepr;
    use ark_ff::Field;
    use ark_std::UniformRand;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// Signed lattice basis entries in Fr, matching `decompose_scalar_2d`'s
    /// sign convention (`true` = positive).
    fn basis_fr() -> [Fr; 4] {
        SCALAR_DECOMP_COEFFS.map(|(positive, magnitude)| {
            let value = Fr::from_bigint(magnitude).unwrap();
            if positive {
                value
            } else {
                -value
            }
        })
    }

    /// The GLV eigenvalue λ derived from the lattice basis itself: every
    /// basis row (a, b) lies in {(a, b) : a + b·λ ≡ 0 (mod r)}, so
    /// λ = -n11/n12. Anchored by the second basis row and the cube-root
    /// identity before use.
    fn lambda() -> Fr {
        let [n11, n12, n21, n22] = basis_fr();
        let lambda = -n11 * n12.inverse().unwrap();
        assert_eq!(
            n21 + n22 * lambda,
            Fr::from(0u64),
            "second basis row must vanish"
        );
        assert_eq!(
            lambda * lambda * lambda,
            Fr::from(1u64),
            "lambda must be a cube root of unity"
        );
        assert_ne!(lambda, Fr::from(1u64), "lambda must be nontrivial");
        lambda
    }

    fn signed_fr(magnitude: <Fr as PrimeField>::BigInt, positive: bool) -> Fr {
        let value = Fr::from_bigint(magnitude).unwrap();
        if positive {
            value
        } else {
            -value
        }
    }

    fn magnitude(value: <Fr as PrimeField>::BigInt) -> BigUint {
        BigUint::from_bytes_be(&value.to_bytes_be())
    }

    // Ties the scalar-field lambda used by the reconstruction check to the
    // actual curve endomorphism the multiplication routines apply.
    #[test]
    fn lattice_lambda_is_the_endomorphism_eigenvalue_on_g1() {
        let g = G1Affine::generator().into_group();
        assert_eq!(glv_endomorphism(&g), g * lambda());
    }

    #[test]
    fn decomposition_reconstructs_scalar_within_half_bitlength_bounds() {
        let lambda = lambda();
        let [n11_abs, n12_abs, n21_abs, n22_abs] =
            SCALAR_DECOMP_COEFFS.map(|(_, value)| magnitude(value));
        // The Babai coefficients round to nearest for positive products but
        // truncate toward zero for negative ones (the `2*rem > r` round-up
        // never fires on a negative remainder), so each coefficient error is
        // below 1 and |k0| <= |n11| + |n21|, |k1| <= |n12| + |n22|, with one
        // unit of slack. Both sums are ~2^128 — the documented halving of the
        // 254-bit scalar.
        let bound_0: BigUint = n11_abs.clone() + n21_abs + BigUint::one();
        let bound_1: BigUint = n12_abs + n22_abs.clone() + BigUint::one();
        assert!(
            bound_0.bits() <= 128 && bound_1.bits() <= 128,
            "lattice bounds must confirm the documented ~128-bit components"
        );

        let mut rng = ChaCha20Rng::seed_from_u64(0x2d61);
        let mut scalars: Vec<Fr> = (0..64).map(|_| Fr::rand(&mut rng)).collect();
        scalars.extend([
            Fr::from(0u64),
            Fr::from(1u64),
            -Fr::from(1u64), // r - 1
            lambda,
            lambda - Fr::from(1u64),
            lambda + Fr::from(1u64),
            -lambda,
            Fr::from(BigUint::one() << 127),
            Fr::from(BigUint::one() << 128),
            // scalars sitting on lattice basis magnitudes
            Fr::from(n11_abs),
            Fr::from(n22_abs.clone()),
            Fr::from(n22_abs) + Fr::from(1u64),
            Fr::from(bound_0.clone()),
            Fr::from(bound_1.clone()),
        ]);

        for scalar in scalars {
            let (coeffs, signs) = decompose_scalar_2d(scalar);
            let k0 = signed_fr(coeffs[0], signs[0]);
            let k1 = signed_fr(coeffs[1], signs[1]);
            assert_eq!(
                k0 + k1 * lambda,
                scalar,
                "k0 + k1*lambda must reconstruct {scalar}"
            );
            assert!(
                magnitude(coeffs[0]) <= bound_0,
                "|k0| exceeds the lattice bound for {scalar}"
            );
            assert!(
                magnitude(coeffs[1]) <= bound_1,
                "|k1| exceeds the lattice bound for {scalar}"
            );
        }
    }
}
