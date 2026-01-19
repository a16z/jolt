//! GT exponentiation witness generation for packed base-4 recursion.

use ark_bn254::{Fq, Fq12, Fr};
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_optimizations::{fq12_to_multilinear_evals, get_g_mle};

/// Base-4 exponentiation witness generation.
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct Base4ExponentiationSteps {
    pub base: Fq12,
    pub exponent: Fr,
    pub result: Fq12,
    pub rho_mles: Vec<Vec<Fq>>,
    pub quotient_mles: Vec<Vec<Fq>>,
    pub bits: Vec<bool>, // MSB-first bits (no leading zeros)
}

impl Base4ExponentiationSteps {
    /// Generate MLE witness for base^exponent using MSB-first base-4 digits.
    pub fn new(base: Fq12, exponent: Fr) -> Self {
        let digits = digits_from_exponent_msb(exponent);
        let bits_msb = bits_from_base4_digits_msb(&digits);

        let base2 = base * base;
        let base3 = base2 * base;
        let base_mle = fq12_to_multilinear_evals(&base);
        let base2_mle = fq12_to_multilinear_evals(&base2);
        let base3_mle = fq12_to_multilinear_evals(&base3);

        let digit_bits: Vec<(bool, bool)> = digits
            .iter()
            .map(|digit| ((digit & 2) != 0, (digit & 1) != 0))
            .collect();

        let mut rho = Fq12::one();
        let mut rho_mles = vec![fq12_to_multilinear_evals(&rho)];
        let mut quotient_mles = Vec::with_capacity(digit_bits.len());

        for &(digit_hi, digit_lo) in &digit_bits {
            let rho_prev = rho;
            let rho4 = rho_prev.square().square();

            let mul = match (digit_hi, digit_lo) {
                (false, false) => Fq12::one(),
                (false, true) => base,
                (true, false) => base2,
                (true, true) => base3,
            };

            let rho_next = rho4 * mul;
            let q_i = compute_step_quotient_base4(
                rho_prev,
                rho_next,
                &base_mle,
                &base2_mle,
                &base3_mle,
                digit_lo,
                digit_hi,
            );

            quotient_mles.push(q_i);
            rho = rho_next;
            rho_mles.push(fq12_to_multilinear_evals(&rho));
        }

        Self {
            base,
            exponent,
            result: rho,
            rho_mles,
            quotient_mles,
            bits: bits_msb,
        }
    }

    /// Verify that the final result matches base^exponent.
    pub fn verify_result(&self) -> bool {
        self.result == self.base.pow(self.exponent.into_bigint())
    }

    pub fn num_steps(&self) -> usize {
        self.quotient_mles.len()
    }
}

fn digits_from_exponent_msb(exponent: Fr) -> Vec<u8> {
    let mut n = exponent.into_bigint();
    if n.is_zero() {
        return vec![];
    }

    let mut digits_lsb = Vec::new();
    while !n.is_zero() {
        let limb0 = n.as_ref()[0];
        digits_lsb.push((limb0 & 3) as u8);
        n.divn(2);
    }

    digits_lsb.reverse();
    digits_lsb
}

fn bits_from_base4_digits_msb(digits: &[u8]) -> Vec<bool> {
    let mut bits = Vec::with_capacity(digits.len() * 2);
    let mut started = false;

    for &digit in digits {
        let hi = (digit & 2) != 0;
        let lo = (digit & 1) != 0;

        if !started {
            if hi {
                bits.push(true);
                bits.push(lo);
                started = true;
            } else if lo {
                bits.push(true);
                started = true;
            }
        } else {
            bits.push(hi);
            bits.push(lo);
        }
    }

    bits
}

fn compute_step_quotient_base4(
    rho_prev: Fq12,
    rho_next: Fq12,
    base_mle: &[Fq],
    base2_mle: &[Fq],
    base3_mle: &[Fq],
    digit_lo: bool,
    digit_hi: bool,
) -> Vec<Fq> {
    let rho_prev_mle = fq12_to_multilinear_evals(&rho_prev);
    let rho_next_mle = fq12_to_multilinear_evals(&rho_next);
    let g_mle = get_g_mle();

    let u = if digit_lo { Fq::one() } else { Fq::zero() };
    let v = if digit_hi { Fq::one() } else { Fq::zero() };
    let w0 = (Fq::one() - u) * (Fq::one() - v);
    let w1 = u * (Fq::one() - v);
    let w2 = (Fq::one() - u) * v;
    let w3 = u * v;

    let mut quotient_mle = vec![Fq::zero(); 16];
    for j in 0..16 {
        let base_pow = w0 + w1 * base_mle[j] + w2 * base2_mle[j] + w3 * base3_mle[j];
        let rho4 = rho_prev_mle[j].square().square();
        let expected = rho4 * base_pow;

        if !g_mle[j].is_zero() {
            quotient_mle[j] = (rho_next_mle[j] - expected) / g_mle[j];
        }
    }

    quotient_mle
}
