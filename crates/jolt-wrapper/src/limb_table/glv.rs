//! GLV decompositions and the signed-digit Straus recoding that form the
//! public digit schedule: the 4-dimensional Frobenius decomposition for GT
//! and G2 (eigenvalue `q mod r = 6u²`), the 2-dimensional one for G1, and
//! centered radix-`2^WINDOW` digits of fixed length.

use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{Signed, ToPrimitive, Zero};

pub const WINDOW: usize = 5;
/// Digits per mini-scalar: 4D components are below `2^64`, G1's 2D components
/// below `2^128`; `WINDOW · windows` bits cover the centered range.
pub const GT_WINDOWS: usize = 13;
pub const G2_WINDOWS: usize = 13;
pub const G1_WINDOWS: usize = 26;
/// Stored table magnitudes `1..=2^(WINDOW-1)`.
pub const TABLE_ENTRIES: usize = 1 << (WINDOW - 1);

const BN_U: u64 = 4_965_661_367_192_848_881;

fn modulus() -> BigInt {
    BigInt::from_biguint(
        Sign::Plus,
        BigUint::from_bytes_be(&Fr::MODULUS.to_bytes_be()),
    )
}

fn to_bigint(scalar: Fr) -> BigInt {
    BigInt::from_biguint(
        Sign::Plus,
        BigUint::from_bytes_be(&scalar.into_bigint().to_bytes_be()),
    )
}

fn round_div(numerator: BigInt, denominator: &BigInt) -> BigInt {
    let (mut quotient, remainder) = numerator.div_rem(denominator);
    if (&remainder + &remainder).abs() >= *denominator {
        quotient += if remainder.sign() == Sign::Minus {
            -1
        } else {
            1
        };
    }
    quotient
}

/// `s = Σ_p k_p · λ^p (mod r)` with `λ = q mod r`, the reduced kernel lattice
/// of lane M1 (`|k_p| < 2^64`).
pub fn decompose_4d(scalar: Fr) -> [BigInt; 4] {
    const BASIS: [[i128; 4]; 4] = [
        [
            9_931_322_734_385_697_762,
            4_965_661_367_192_848_882,
            -4_965_661_367_192_848_881,
            4_965_661_367_192_848_881,
        ],
        [
            -4_965_661_367_192_848_881,
            4_965_661_367_192_848_881,
            -4_965_661_367_192_848_881,
            -9_931_322_734_385_697_763,
        ],
        [
            4_965_661_367_192_848_882,
            4_965_661_367_192_848_881,
            4_965_661_367_192_848_881,
            -9_931_322_734_385_697_762,
        ],
        [
            9_931_322_734_385_697_763,
            -4_965_661_367_192_848_881,
            -4_965_661_367_192_848_882,
            -4_965_661_367_192_848_881,
        ],
    ];
    const INVERSE_ROW_NUMERATORS: [&str; 4] = [
        "734653495049373973658254490726798021314063399421879442165",
        "-734653495049373973806201247608587340319794091592875701774",
        "734653495049373973806201247608587340329725414327261399537",
        "734653495049373973806201247608587340314828430225682852893",
    ];
    let modulus = modulus();
    let scalar = to_bigint(scalar);
    let coefficients: Vec<BigInt> = INVERSE_ROW_NUMERATORS
        .iter()
        .map(|numerator| {
            let numerator: BigInt = numerator
                .parse()
                .unwrap_or_else(|_| unreachable!("literal lattice constant"));
            round_div(&scalar * numerator, &modulus)
        })
        .collect();
    std::array::from_fn(|coordinate| {
        let lattice: BigInt = coefficients
            .iter()
            .zip(BASIS)
            .map(|(coefficient, row)| coefficient * BigInt::from(row[coordinate]))
            .sum();
        let target = if coordinate == 0 {
            scalar.clone()
        } else {
            BigInt::zero()
        };
        target - lattice
    })
}

/// `s = k_0 + k_1 · λ_1 (mod r)` for G1's cube-root-of-unity endomorphism
/// (arkworks' `SCALAR_DECOMP_COEFFS`), `|k_i| < 2^128`.
pub fn decompose_2d_g1(scalar: Fr) -> [BigInt; 2] {
    let [n11, n12, n21, n22] = [
        BigInt::from(-147_946_756_881_789_319_000_765_030_803_803_410_728i128),
        BigInt::from(9_931_322_734_385_697_763i128),
        BigInt::from(-9_931_322_734_385_697_763i128),
        BigInt::from(-147_946_756_881_789_319_010_696_353_538_189_108_491i128),
    ];
    let modulus = modulus();
    let scalar = to_bigint(scalar);
    let half_round = |value: BigInt| {
        let (mut quotient, remainder) = value.div_rem(&modulus);
        if &remainder + &remainder > modulus {
            quotient += 1;
        }
        quotient
    };
    let beta1 = half_round(&scalar * &n22);
    let beta2 = half_round(&scalar * -&n12);
    let b1 = &beta1 * n11 + &beta2 * n21;
    let b2 = beta1 * n12 + beta2 * n22;
    [scalar - b1, -b2]
}

/// `λ = q mod r = 6u²`, the eigenvalue of Frobenius on GT and of `ψ` on G2.
pub fn frobenius_eigenvalue() -> Fr {
    Fr::from(6u64) * Fr::from(BN_U) * Fr::from(BN_U)
}

pub fn fr_from_bigint(value: &BigInt) -> Fr {
    let magnitude = Fr::from_be_bytes_mod_order(&value.magnitude().to_bytes_be());
    if value.sign() == Sign::Minus {
        -magnitude
    } else {
        magnitude
    }
}

/// Centered radix-`2^WINDOW` digits (`[-16, 16)`), least significant first,
/// exactly `windows` of them.
///
/// # Panics
/// If the value does not fit `WINDOW · windows` centered bits; the GLV bounds
/// make that impossible for honest inputs.
pub fn centered_digits(value: &BigInt, windows: usize) -> Vec<i8> {
    let radix = BigInt::from(1u32 << WINDOW);
    let half = BigInt::from(1u32 << (WINDOW - 1));
    let mut value = value.clone();
    let mut digits = Vec::with_capacity(windows);
    for _ in 0..windows {
        let mut residue = &value % &radix;
        if residue.sign() == Sign::Minus {
            residue += &radix;
        }
        let digit = if residue >= half {
            residue - &radix
        } else {
            residue
        };
        let digit = digit
            .to_i8()
            .unwrap_or_else(|| unreachable!("centered digit fits i8"));
        digits.push(digit);
        value = (value - BigInt::from(digit)) / &radix;
    }
    assert!(
        value.is_zero(),
        "mini-scalar exceeds the fixed window count"
    );
    digits
}

/// Public digit schedule of one multi-exponentiation: `digits[base][dim][window]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Digits {
    pub digits: Vec<Vec<Vec<i8>>>,
    pub windows: usize,
}

impl Digits {
    pub fn four_dimensional(scalars: &[Fr], windows: usize) -> Self {
        Self {
            digits: scalars
                .iter()
                .map(|s| {
                    decompose_4d(*s)
                        .iter()
                        .map(|k| centered_digits(k, windows))
                        .collect()
                })
                .collect(),
            windows,
        }
    }

    pub fn two_dimensional_g1(scalars: &[Fr], windows: usize) -> Self {
        Self {
            digits: scalars
                .iter()
                .map(|s| {
                    decompose_2d_g1(*s)
                        .iter()
                        .map(|k| centered_digits(k, windows))
                        .collect()
                })
                .collect(),
            windows,
        }
    }

    pub fn dims(&self) -> usize {
        self.digits.first().map_or(0, Vec::len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ec::scalar_mul::glv::GLVConfig;
    use ark_ff::UniformRand;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn recompose(components: &[BigInt], lambda: Fr) -> Fr {
        let mut power = Fr::from(1u64);
        let mut sum = Fr::from(0u64);
        for component in components {
            sum += fr_from_bigint(component) * power;
            power *= lambda;
        }
        sum
    }

    fn digits_value(digits: &[i8]) -> BigInt {
        digits.iter().rev().fold(BigInt::zero(), |acc, &d| {
            acc * BigInt::from(1u32 << WINDOW) + BigInt::from(d)
        })
    }

    #[test]
    fn decompositions_recompose_and_fit_the_windows() {
        let mut rng = ChaCha20Rng::seed_from_u64(9);
        for _ in 0..50 {
            let s = Fr::rand(&mut rng);
            let four = decompose_4d(s);
            assert_eq!(recompose(&four, frobenius_eigenvalue()), s);
            for k in &four {
                assert!(k.bits() <= 64);
                assert_eq!(digits_value(&centered_digits(k, GT_WINDOWS)), *k);
            }
            let two = decompose_2d_g1(s);
            assert_eq!(recompose(&two, ark_bn254::g1::Config::LAMBDA), s);
            for k in &two {
                assert!(k.bits() <= 128);
                assert_eq!(digits_value(&centered_digits(k, G1_WINDOWS)), *k);
            }
        }
        assert_eq!(centered_digits(&BigInt::from(-16), 2), vec![-16, 0]);
        assert_eq!(centered_digits(&BigInt::from(16), 2), vec![-16, 1]);
    }
}
