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
