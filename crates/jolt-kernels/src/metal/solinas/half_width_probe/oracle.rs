//! Independent limb oracle for the Akita-specialized half-width primitive.

use super::super::Fp128;
use super::HALF_WIDTH_AKITA_OFFSET;
#[cfg(any(test, feature = "test-utils"))]
use super::{HalfWidthDomain, HalfWidthOperand};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HalfWidthWide192 {
    limbs: [u32; 6],
}

impl HalfWidthWide192 {
    const fn from_limbs(limbs: [u32; 6]) -> Self {
        Self { limbs }
    }

    pub const fn limbs(self) -> [u32; 6] {
        self.limbs
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HalfWidthReductionTrace {
    pub first_fold_carry: u32,
    pub carry_fold_carry: u32,
    pub canonical_subtracted: bool,
}

/// Computes the unreduced 192-bit product using eight 32-bit coefficient
/// products. This deliberately does not use the field implementation.
pub fn product_u64_limbs(coefficient: Fp128, scalar: u64) -> HalfWidthWide192 {
    let coefficient = coefficient.limbs();
    let scalar = [scalar as u32, (scalar >> 32) as u32];
    let mut product = [0u32; 6];

    for (i, coefficient_limb) in coefficient.into_iter().enumerate() {
        let mut carry = 0u64;
        for (j, scalar_limb) in scalar.into_iter().enumerate() {
            let k = i + j;
            let word = u64::from(coefficient_limb) * u64::from(scalar_limb)
                + u64::from(product[k])
                + carry;
            product[k] = word as u32;
            carry = word >> 32;
        }
        product[i + 2] = carry as u32;
    }

    HalfWidthWide192::from_limbs(product)
}

/// Reduces `L + H * 2^128` for Akita's `p = 2^128 - 0xffff_a7f7`.
///
/// For this 128-by-64 product, `H < 2^64`, so `H * offset < 2^96`.
/// The first fold can carry only one bit past 128. Once that bit is folded,
/// the value is below 2^96 plus `offset`, hence the carry fold cannot overflow.
pub fn reduce_u192_akita(product: HalfWidthWide192) -> (Fp128, HalfWidthReductionTrace) {
    let product = product.limbs();
    let mut folded = [0u32; 4];
    let mut carry = 0u64;

    for i in 0..2 {
        let word = u64::from(product[i + 4]) * u64::from(HALF_WIDTH_AKITA_OFFSET)
            + u64::from(product[i])
            + carry;
        folded[i] = word as u32;
        carry = word >> 32;
    }
    for i in 2..4 {
        let word = u64::from(product[i]) + carry;
        folded[i] = word as u32;
        carry = word >> 32;
    }
    let first_fold_carry = carry as u32;

    let word =
        u64::from(folded[0]) + u64::from(first_fold_carry) * u64::from(HALF_WIDTH_AKITA_OFFSET);
    folded[0] = word as u32;
    carry = word >> 32;
    for limb in folded.iter_mut().skip(1) {
        let word = u64::from(*limb) + carry;
        *limb = word as u32;
        carry = word >> 32;
    }
    let carry_fold_carry = carry as u32;
    debug_assert_eq!(carry_fold_carry, 0);

    let mut corrected = folded;
    let word = u64::from(corrected[0]) + u64::from(HALF_WIDTH_AKITA_OFFSET);
    corrected[0] = word as u32;
    carry = word >> 32;
    for limb in corrected.iter_mut().skip(1) {
        let word = u64::from(*limb) + carry;
        *limb = word as u32;
        carry = word >> 32;
    }
    let canonical_subtracted = carry != 0;
    let output = if canonical_subtracted {
        corrected
    } else {
        folded
    };

    (
        Fp128::from_limbs(output),
        HalfWidthReductionTrace {
            first_fold_carry,
            carry_fold_carry,
            canonical_subtracted,
        },
    )
}

pub fn mul_u64_oracle(coefficient: Fp128, scalar: u64) -> Fp128 {
    reduce_u192_akita(product_u64_limbs(coefficient, scalar)).0
}

pub fn mul_signed_u64_oracle(coefficient: Fp128, magnitude: u64, negative: bool) -> Fp128 {
    let positive = mul_u64_oracle(coefficient, magnitude);
    if !negative || positive == Fp128::ZERO {
        positive
    } else {
        Fp128::from_u128(super::HALF_WIDTH_AKITA_MODULUS - positive.to_u128())
    }
}

pub fn mul_u64_delta_oracle(coefficient: Fp128, minuend: u64, subtrahend: u64) -> Fp128 {
    if minuend >= subtrahend {
        mul_u64_oracle(coefficient, minuend - subtrahend)
    } else {
        mul_signed_u64_oracle(coefficient, subtrahend - minuend, true)
    }
}

#[cfg(any(test, feature = "test-utils"))]
pub fn reference_outputs(
    probe: super::HalfWidthProbe,
    coefficients: &[Fp128],
    operands: &[HalfWidthOperand],
    iterations: u32,
) -> Result<Vec<Fp128>, super::HalfWidthProbeError> {
    use jolt_field::AkitaField;

    let _ = super::checked_probe_shape(
        probe,
        coefficients,
        operands,
        iterations,
        HALF_WIDTH_AKITA_OFFSET,
        u64::MAX,
    )?;

    let rounds = if probe.is_chain() { iterations } else { 1 };
    Ok(coefficients
        .iter()
        .copied()
        .zip(operands.iter().copied())
        .map(|(coefficient, operand)| {
            let factor = match probe.domain() {
                HalfWidthDomain::Unsigned => AkitaField::from_u64(operand.primary),
                HalfWidthDomain::SignedMagnitude => {
                    let magnitude = AkitaField::from_u64(operand.primary);
                    if operand.secondary == 0 {
                        magnitude
                    } else {
                        -magnitude
                    }
                }
                HalfWidthDomain::UnsignedDelta => {
                    AkitaField::from_u64(operand.primary) - AkitaField::from_u64(operand.secondary)
                }
            };
            let mut accumulator = coefficient.into_jolt_field::<AkitaField>();
            for _ in 0..rounds {
                accumulator *= factor;
            }
            Fp128::from_jolt_field(&accumulator)
        })
        .collect())
}
