//! Non-native field representations for verifier circuits.
//!
//! This module currently targets the wrapper path that constrains BN254 `Fq`
//! values inside an R1CS over BN254 `Fr`. Values are represented as canonical
//! little-endian integer limbs, not as Montgomery-form field internals.

use jolt_field::{CanonicalBytes, FixedByteSize, Fq, Fr, FromPrimitiveInt, Invertible};
use num_bigint::BigUint;
use num_traits::Zero;

use crate::{AssignedScalar, LinearCombination, R1csBuilder};

const LIMB_BITS: usize = 64;
const NUM_LIMBS: usize = 4;
const MUL_LIMBS: usize = 2 * NUM_LIMBS;
const RADIX: u128 = 1u128 << LIMB_BITS;
const CARRY_BITS: usize = 68;

const FQ_MODULUS_LIMBS: [u64; NUM_LIMBS] = [
    4_332_616_871_279_656_263,
    10_917_124_144_477_883_021,
    13_281_191_951_274_694_749,
    3_486_998_266_802_970_665,
];

const FQ_MODULUS_MINUS_ONE_LIMBS: [u64; NUM_LIMBS] = [
    4_332_616_871_279_656_262,
    10_917_124_144_477_883_021,
    13_281_191_951_274_694_749,
    3_486_998_266_802_970_665,
];

const FR_MODULUS_MINUS_ONE_LIMBS: [u64; NUM_LIMBS] = [
    4_891_460_686_036_598_784,
    2_896_914_383_306_846_353,
    13_281_191_951_274_694_749,
    3_486_998_266_802_970_665,
];

/// A canonical BN254 `Fq` integer represented by four 64-bit limbs in an
/// `Fr`-native R1CS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FqVar {
    limbs: [AssignedScalar<Fr>; NUM_LIMBS],
}

impl FqVar {
    pub const LIMB_BITS: usize = LIMB_BITS;
    pub const NUM_LIMBS: usize = NUM_LIMBS;

    pub fn constant(value: Fq) -> Self {
        let limbs = fq_to_u64_limbs(value).map(|limb| AssignedScalar::constant(fr(limb)));
        Self { limbs }
    }

    pub fn alloc(builder: &mut R1csBuilder<Fr>, value: Fq) -> Self {
        let limbs = fq_to_u64_limbs(value).map(|limb| AssignedScalar::alloc(builder, fr(limb)));
        Self::from_checked_limbs(builder, limbs)
    }

    pub fn from_checked_limbs(
        builder: &mut R1csBuilder<Fr>,
        limbs: [AssignedScalar<Fr>; NUM_LIMBS],
    ) -> Self {
        for limb in &limbs {
            assert_u64(builder, limb);
        }
        assert_limbs_less_or_equal(builder, &limbs, FQ_MODULUS_MINUS_ONE_LIMBS);

        Self { limbs }
    }

    /// Injects a native `Fr` scalar into `Fq` as an integer.
    ///
    /// This is the canonical integer injection `Fr -> Fq`: constrain the
    /// `Fr` value's little-endian integer limbs to be canonical modulo `q_Fr`,
    /// then reuse those same limbs as an `Fq` value. This is valid because
    /// BN254 has `q_Fr < q_Fq`; it is not a residue reinterpretation.
    pub fn inject_fr(builder: &mut R1csBuilder<Fr>, value: &AssignedScalar<Fr>) -> Self {
        let limbs =
            fr_to_u64_limbs(value.value).map(|limb| AssignedScalar::alloc(builder, fr(limb)));
        for limb in &limbs {
            assert_u64(builder, limb);
        }
        assert_limbs_less_or_equal(builder, &limbs, FR_MODULUS_MINUS_ONE_LIMBS);
        builder.assert_equal(value.lc.clone(), compose_limbs(&limbs));

        Self { limbs }
    }

    /// Converts a Poseidon-over-`Fr` challenge into an `Fq` challenge by the
    /// canonical integer injection `Fr -> Fq`.
    ///
    /// The transcript remains an `Fr` transcript. Callers should domain-separate
    /// in the transcript before squeezing `challenge`, then call this at the
    /// protocol boundary where `Fq` arithmetic begins.
    pub fn inject_fr_challenge(
        builder: &mut R1csBuilder<Fr>,
        challenge: &AssignedScalar<Fr>,
    ) -> Self {
        Self::inject_fr(builder, challenge)
    }

    pub fn limbs(&self) -> &[AssignedScalar<Fr>; NUM_LIMBS] {
        &self.limbs
    }

    pub fn witness_value(&self) -> Fq {
        self.value()
    }

    /// Returns a constrained little-endian bit decomposition of this `Fq`
    /// variable's canonical integer representation.
    pub fn bits_le(&self, builder: &mut R1csBuilder<Fr>) -> Vec<AssignedScalar<Fr>> {
        self.limbs
            .iter()
            .flat_map(|limb| assert_unsigned_bits(builder, limb, LIMB_BITS))
            .collect()
    }

    pub fn assert_equal(&self, builder: &mut R1csBuilder<Fr>, rhs: &Self) {
        for (lhs_limb, rhs_limb) in self.limbs.iter().zip(&rhs.limbs) {
            builder.assert_equal(lhs_limb.lc.clone(), rhs_limb.lc.clone());
        }
    }

    pub fn add(&self, builder: &mut R1csBuilder<Fr>, rhs: &Self) -> Self {
        let output = Self::alloc(builder, self.value() + rhs.value());
        assert_add_relation(builder, self, rhs, &output);
        output
    }

    pub fn sub(&self, builder: &mut R1csBuilder<Fr>, rhs: &Self) -> Self {
        let output = Self::alloc(builder, self.value() - rhs.value());
        assert_sub_relation(builder, self, rhs, &output);
        output
    }

    pub fn mul(&self, builder: &mut R1csBuilder<Fr>, rhs: &Self) -> Self {
        let output = Self::alloc(builder, self.value() * rhs.value());
        assert_mul_relation(builder, self, rhs, &output);
        output
    }

    pub fn inverse(&self, builder: &mut R1csBuilder<Fr>) -> Option<Self> {
        let output = Self::alloc(builder, self.value().inverse()?);
        let one = Self::constant(Fq::from_u64(1));
        assert_mul_relation(builder, self, &output, &one);
        Some(output)
    }

    pub fn select(
        builder: &mut R1csBuilder<Fr>,
        selector: &AssignedScalar<Fr>,
        when_true: &Self,
        when_false: &Self,
    ) -> Self {
        assert_boolean(builder, selector);

        let output = if selector.value == fr(1) {
            Self::alloc(builder, when_true.value())
        } else {
            Self::alloc(builder, when_false.value())
        };

        for ((output_limb, true_limb), false_limb) in output
            .limbs
            .iter()
            .zip(&when_true.limbs)
            .zip(&when_false.limbs)
        {
            let selected_delta = builder.multiply(
                selector.lc.clone(),
                true_limb.lc.clone() - false_limb.lc.clone(),
            );
            builder.assert_equal(
                output_limb.lc.clone(),
                false_limb.lc.clone() + selected_delta,
            );
        }

        output
    }

    fn value(&self) -> Fq {
        fq_from_limbs(self.value_limbs())
    }

    fn value_limbs(&self) -> [u64; NUM_LIMBS] {
        std::array::from_fn(|index| scalar_low_u64(self.limbs[index].value))
    }
}

fn assert_u64(builder: &mut R1csBuilder<Fr>, value: &AssignedScalar<Fr>) {
    let _ = assert_unsigned_bits(builder, value, LIMB_BITS);
}

fn assert_unsigned_bits(
    builder: &mut R1csBuilder<Fr>,
    value: &AssignedScalar<Fr>,
    bit_len: usize,
) -> Vec<AssignedScalar<Fr>> {
    let bits = scalar_low_u128(value.value);
    let bit_vars = (0..bit_len)
        .map(|index| {
            let bit = Fr::from_bool(((bits >> index) & 1) == 1);
            let assigned = AssignedScalar::alloc(builder, bit);
            assert_boolean(builder, &assigned);
            assigned
        })
        .collect::<Vec<_>>();

    builder.assert_equal(value.lc.clone(), compose_bits(&bit_vars));
    bit_vars
}

fn assert_add_relation(builder: &mut R1csBuilder<Fr>, lhs: &FqVar, rhs: &FqVar, output: &FqVar) {
    let lhs_limbs = lhs.value_limbs();
    let rhs_limbs = rhs.value_limbs();
    let output_limbs = output.value_limbs();
    let sum = limbs_to_biguint(lhs_limbs) + limbs_to_biguint(rhs_limbs);
    let wraps_modulus = sum >= fq_modulus();
    let quotient = AssignedScalar::alloc(builder, Fr::from_bool(wraps_modulus));
    assert_boolean(builder, &quotient);
    let normalized = alloc_u64_limbs(builder, biguint_to_u64_limbs::<NUM_LIMBS>(&sum));

    let lhs_terms =
        std::array::from_fn(|index| lhs.limbs[index].lc.clone() + rhs.limbs[index].lc.clone());
    let lhs_raw_terms = std::array::from_fn(|index| {
        BigUint::from(lhs_limbs[index]) + BigUint::from(rhs_limbs[index])
    });
    assert_terms_normalize_to(builder, lhs_terms, lhs_raw_terms, &normalized);

    let rhs_terms = std::array::from_fn(|index| {
        output.limbs[index].lc.clone() + quotient.lc.clone().scale(fr(FQ_MODULUS_LIMBS[index]))
    });
    let quotient_value = u64::from(wraps_modulus);
    let rhs_raw_terms = std::array::from_fn(|index| {
        BigUint::from(output_limbs[index]) + BigUint::from(FQ_MODULUS_LIMBS[index]) * quotient_value
    });
    assert_terms_normalize_to(builder, rhs_terms, rhs_raw_terms, &normalized);
}

fn assert_sub_relation(builder: &mut R1csBuilder<Fr>, lhs: &FqVar, rhs: &FqVar, output: &FqVar) {
    let lhs_limbs = lhs.value_limbs();
    let rhs_limbs = rhs.value_limbs();
    let output_limbs = output.value_limbs();
    let lhs_value = limbs_to_biguint(lhs_limbs);
    let rhs_value = limbs_to_biguint(rhs_limbs);
    let borrow = lhs_value < rhs_value;
    let adjusted_lhs = if borrow {
        lhs_value + fq_modulus()
    } else {
        lhs_value
    };
    let quotient = AssignedScalar::alloc(builder, Fr::from_bool(borrow));
    assert_boolean(builder, &quotient);
    let normalized = alloc_u64_limbs(builder, biguint_to_u64_limbs::<NUM_LIMBS>(&adjusted_lhs));

    let lhs_terms = std::array::from_fn(|index| {
        lhs.limbs[index].lc.clone() + quotient.lc.clone().scale(fr(FQ_MODULUS_LIMBS[index]))
    });
    let quotient_value = u64::from(borrow);
    let lhs_raw_terms = std::array::from_fn(|index| {
        BigUint::from(lhs_limbs[index]) + BigUint::from(FQ_MODULUS_LIMBS[index]) * quotient_value
    });
    assert_terms_normalize_to(builder, lhs_terms, lhs_raw_terms, &normalized);

    let rhs_terms =
        std::array::from_fn(|index| rhs.limbs[index].lc.clone() + output.limbs[index].lc.clone());
    let rhs_raw_terms = std::array::from_fn(|index| {
        BigUint::from(rhs_limbs[index]) + BigUint::from(output_limbs[index])
    });
    assert_terms_normalize_to(builder, rhs_terms, rhs_raw_terms, &normalized);
}

fn assert_mul_relation(builder: &mut R1csBuilder<Fr>, lhs: &FqVar, rhs: &FqVar, output: &FqVar) {
    let lhs_limbs = lhs.value_limbs();
    let rhs_limbs = rhs.value_limbs();
    let output_limbs = output.value_limbs();
    let product = limbs_to_biguint(lhs_limbs) * limbs_to_biguint(rhs_limbs);
    let output_value = limbs_to_biguint(output_limbs);
    let quotient = if product >= output_value {
        (product.clone() - output_value) / fq_modulus()
    } else {
        BigUint::zero()
    };
    let quotient_limbs = alloc_limbs(builder, biguint_to_u64_limbs::<NUM_LIMBS>(&quotient));
    let quotient = FqVar::from_checked_limbs(builder, quotient_limbs);
    let normalized = alloc_u64_limbs(builder, biguint_to_u64_limbs::<MUL_LIMBS>(&product));

    let mut product_terms = std::array::from_fn(|_| LinearCombination::zero());
    for (lhs_index, lhs_limb) in lhs.limbs.iter().enumerate() {
        for (rhs_index, rhs_limb) in rhs.limbs.iter().enumerate() {
            let product_limb = builder.multiply(lhs_limb.lc.clone(), rhs_limb.lc.clone());
            product_terms[lhs_index + rhs_index] =
                product_terms[lhs_index + rhs_index].clone() + product_limb;
        }
    }
    let product_raw_terms = convolution_terms(lhs_limbs, rhs_limbs);
    assert_terms_normalize_to(builder, product_terms, product_raw_terms, &normalized);

    let mut reduction_terms = std::array::from_fn(|_| LinearCombination::zero());
    for (modulus_index, modulus_limb) in FQ_MODULUS_LIMBS.into_iter().enumerate() {
        for (quotient_index, quotient_limb) in quotient.limbs.iter().enumerate() {
            reduction_terms[modulus_index + quotient_index] =
                reduction_terms[modulus_index + quotient_index].clone()
                    + quotient_limb.lc.clone().scale(fr(modulus_limb));
        }
    }
    for (index, output_limb) in output.limbs.iter().enumerate() {
        reduction_terms[index] = reduction_terms[index].clone() + output_limb.lc.clone();
    }
    let mut reduction_raw_terms = constant_mul_terms(FQ_MODULUS_LIMBS, quotient.value_limbs());
    for (index, output_limb) in output_limbs.into_iter().enumerate() {
        reduction_raw_terms[index] += BigUint::from(output_limb);
    }
    assert_terms_normalize_to(builder, reduction_terms, reduction_raw_terms, &normalized);
}

fn alloc_u64_limbs<const N: usize>(
    builder: &mut R1csBuilder<Fr>,
    limbs: [u64; N],
) -> [AssignedScalar<Fr>; N] {
    let assigned = alloc_limbs(builder, limbs);
    for limb in &assigned {
        assert_u64(builder, limb);
    }
    assigned
}

fn alloc_limbs<const N: usize>(
    builder: &mut R1csBuilder<Fr>,
    limbs: [u64; N],
) -> [AssignedScalar<Fr>; N] {
    limbs.map(|limb| AssignedScalar::alloc(builder, fr(limb)))
}

fn assert_terms_normalize_to<const N: usize>(
    builder: &mut R1csBuilder<Fr>,
    terms: [LinearCombination<Fr>; N],
    raw_terms: [BigUint; N],
    normalized: &[AssignedScalar<Fr>; N],
) {
    let mut carry_value = BigUint::zero();
    let mut carry = AssignedScalar::constant(fr(0));

    for ((term, raw_term), normalized_limb) in terms.into_iter().zip(raw_terms).zip(normalized) {
        let total = raw_term + &carry_value;
        let limb = BigUint::from(scalar_low_u64(normalized_limb.value));
        let next_carry_value = if total >= limb {
            (total - limb) / BigUint::from(RADIX)
        } else {
            BigUint::zero()
        };
        let next_carry = AssignedScalar::alloc(builder, fr_from_biguint(&next_carry_value));
        let _ = assert_unsigned_bits(builder, &next_carry, CARRY_BITS);

        let lhs = term + carry.lc;
        let rhs = normalized_limb.lc.clone() + next_carry.lc.clone().scale(radix_fr());
        builder.assert_equal(lhs, rhs);

        carry_value = next_carry_value;
        carry = next_carry;
    }

    builder.assert_equal(carry.lc, LinearCombination::zero());
}

fn assert_boolean(builder: &mut R1csBuilder<Fr>, value: &AssignedScalar<Fr>) {
    builder.assert_product(
        value.lc.clone(),
        value.lc.clone() - LinearCombination::one(),
        LinearCombination::zero(),
    );
}

fn assert_limbs_less_or_equal(
    builder: &mut R1csBuilder<Fr>,
    limbs: &[AssignedScalar<Fr>; NUM_LIMBS],
    bound: [u64; NUM_LIMBS],
) {
    let difference = wrapping_difference_limbs(bound, limbs);
    let mut borrow = AssignedScalar::constant(fr(0));

    for ((limb, bound_limb), difference_limb) in limbs.iter().zip(bound).zip(difference) {
        let next_borrow = AssignedScalar::alloc(builder, Fr::from_bool(difference_limb.borrow));
        assert_boolean(builder, &next_borrow);

        let difference = AssignedScalar::alloc(builder, fr(difference_limb.value));
        assert_u64(builder, &difference);

        let lhs =
            LinearCombination::constant(fr(bound_limb)) + next_borrow.lc.clone().scale(radix_fr());
        let rhs = limb.lc.clone() + borrow.lc + difference.lc;
        builder.assert_equal(lhs, rhs);

        borrow = next_borrow;
    }

    builder.assert_equal(borrow.lc, LinearCombination::zero());
}

fn compose_bits(bits: &[AssignedScalar<Fr>]) -> LinearCombination<Fr> {
    bits.iter()
        .enumerate()
        .fold(LinearCombination::zero(), |acc, (index, bit)| {
            acc + bit.lc.clone().scale(Fr::from_u128(1u128 << index))
        })
}

fn compose_limbs(limbs: &[AssignedScalar<Fr>; NUM_LIMBS]) -> LinearCombination<Fr> {
    let mut coefficient = fr(1);
    let radix = radix_fr();
    limbs.iter().fold(LinearCombination::zero(), |acc, limb| {
        let term = limb.lc.clone().scale(coefficient);
        coefficient *= radix;
        acc + term
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DifferenceLimb {
    value: u64,
    borrow: bool,
}

fn wrapping_difference_limbs(
    bound: [u64; NUM_LIMBS],
    limbs: &[AssignedScalar<Fr>; NUM_LIMBS],
) -> [DifferenceLimb; NUM_LIMBS] {
    let mut borrow = 0u128;
    std::array::from_fn(|index| {
        let bound = u128::from(bound[index]);
        let limb = u128::from(scalar_low_u64(limbs[index].value));
        let subtrahend = limb + borrow;
        let (value, next_borrow) = if bound >= subtrahend {
            (bound - subtrahend, 0u128)
        } else {
            (bound + RADIX - subtrahend, 1u128)
        };
        borrow = next_borrow;

        DifferenceLimb {
            value: value as u64,
            borrow: next_borrow == 1,
        }
    })
}

fn fq_to_u64_limbs(value: Fq) -> [u64; NUM_LIMBS] {
    let mut bytes = [0u8; Fq::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    bytes_to_u64_limbs(bytes)
}

fn fq_from_limbs(limbs: [u64; NUM_LIMBS]) -> Fq {
    Fq::from_le_bytes_mod_order(&limbs_to_bytes(limbs))
}

fn fr_to_u64_limbs(value: Fr) -> [u64; NUM_LIMBS] {
    let mut bytes = [0u8; Fr::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    bytes_to_u64_limbs(bytes)
}

fn bytes_to_u64_limbs(bytes: [u8; 32]) -> [u64; NUM_LIMBS] {
    std::array::from_fn(|index| {
        let offset = index * 8;
        let mut limb = [0u8; 8];
        limb.copy_from_slice(&bytes[offset..offset + 8]);
        u64::from_le_bytes(limb)
    })
}

fn scalar_low_u64(value: Fr) -> u64 {
    let mut bytes = [0u8; Fr::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    let mut limb = [0u8; 8];
    limb.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(limb)
}

fn scalar_low_u128(value: Fr) -> u128 {
    let mut bytes = [0u8; Fr::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    let mut limbs = [0u8; 16];
    limbs.copy_from_slice(&bytes[..16]);
    u128::from_le_bytes(limbs)
}

fn limbs_to_biguint<const N: usize>(limbs: [u64; N]) -> BigUint {
    BigUint::from_bytes_le(&limbs_to_bytes(limbs))
}

fn biguint_to_u64_limbs<const N: usize>(value: &BigUint) -> [u64; N] {
    let bytes = value.to_bytes_le();
    std::array::from_fn(|index| {
        let offset = index * 8;
        let mut limb = [0u8; 8];
        if offset < bytes.len() {
            let available = (bytes.len() - offset).min(8);
            limb[..available].copy_from_slice(&bytes[offset..offset + available]);
        }
        u64::from_le_bytes(limb)
    })
}

fn limbs_to_bytes<const N: usize>(limbs: [u64; N]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(N * 8);
    for limb in limbs {
        bytes.extend_from_slice(&limb.to_le_bytes());
    }
    bytes
}

fn fq_modulus() -> BigUint {
    limbs_to_biguint(FQ_MODULUS_LIMBS)
}

fn convolution_terms(lhs: [u64; NUM_LIMBS], rhs: [u64; NUM_LIMBS]) -> [BigUint; MUL_LIMBS] {
    let mut terms = std::array::from_fn(|_| BigUint::zero());
    for (lhs_index, lhs_limb) in lhs.into_iter().enumerate() {
        for (rhs_index, rhs_limb) in rhs.into_iter().enumerate() {
            terms[lhs_index + rhs_index] += BigUint::from(lhs_limb) * BigUint::from(rhs_limb);
        }
    }
    terms
}

fn constant_mul_terms(lhs: [u64; NUM_LIMBS], rhs: [u64; NUM_LIMBS]) -> [BigUint; MUL_LIMBS] {
    let mut terms = std::array::from_fn(|_| BigUint::zero());
    for (lhs_index, lhs_limb) in lhs.into_iter().enumerate() {
        for (rhs_index, rhs_limb) in rhs.into_iter().enumerate() {
            terms[lhs_index + rhs_index] += BigUint::from(lhs_limb) * BigUint::from(rhs_limb);
        }
    }
    terms
}

fn fr_from_biguint(value: &BigUint) -> Fr {
    Fr::from_le_bytes_mod_order(&value.to_bytes_le())
}

fn radix_fr() -> Fr {
    Fr::from_u128(RADIX)
}

fn fr(value: u64) -> Fr {
    Fr::from_u64(value)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;
    use crate::Variable;

    #[test]
    fn accepts_fq_values_at_canonical_edges() {
        assert_fq_value_satisfies(Fq::from_u64(0));
        assert_fq_value_satisfies(fq_from_limbs(FQ_MODULUS_MINUS_ONE_LIMBS));
    }

    #[test]
    fn rejects_limb_out_of_u64_range() {
        let mut builder = R1csBuilder::new();
        let limbs = [
            AssignedScalar::alloc(&mut builder, Fr::from_u128(RADIX)),
            AssignedScalar::alloc(&mut builder, fr(0)),
            AssignedScalar::alloc(&mut builder, fr(0)),
            AssignedScalar::alloc(&mut builder, fr(0)),
        ];

        let _ = FqVar::from_checked_limbs(&mut builder, limbs);

        assert!(builder_rejects(builder));
    }

    #[test]
    fn rejects_fq_modulus_as_noncanonical() {
        let mut builder = R1csBuilder::new();
        let limbs = [
            AssignedScalar::alloc(&mut builder, fr(4_332_616_871_279_656_263)),
            AssignedScalar::alloc(&mut builder, fr(10_917_124_144_477_883_021)),
            AssignedScalar::alloc(&mut builder, fr(13_281_191_951_274_694_749)),
            AssignedScalar::alloc(&mut builder, fr(3_486_998_266_802_970_665)),
        ];

        let _ = FqVar::from_checked_limbs(&mut builder, limbs);

        assert!(builder_rejects(builder));
    }

    #[test]
    fn injects_native_fr_as_canonical_fq_limbs() {
        assert_fr_injection_satisfies(fr(0));
        assert_fr_injection_satisfies(Fr::from_u64(17));
        assert_fr_injection_satisfies(fr_from_limbs(FR_MODULUS_MINUS_ONE_LIMBS));
    }

    #[test]
    fn fq_bits_le_match_canonical_integer_bits() {
        let mut builder = R1csBuilder::new();
        let value = Fq::from_u64(0xdead_beef);
        let fq = FqVar::alloc(&mut builder, value);

        let bits = fq.bits_le(&mut builder);

        assert_eq!(bits.len(), FqVar::NUM_LIMBS * FqVar::LIMB_BITS);
        for (index, bit) in bits.iter().enumerate().take(32) {
            let expected = Fr::from_bool(((0xdead_beefu64 >> index) & 1) == 1);
            assert_eq!(bit.value, expected);
        }
        assert!(bits[32..].iter().all(|bit| bit.value.is_zero()));
        assert!(builder_accepts(builder));
    }

    #[test]
    fn fq_bits_le_rejects_tampered_bit() {
        let mut builder = R1csBuilder::new();
        let fq = FqVar::alloc(&mut builder, Fq::from_u64(0xdead_beef));
        let bits = fq.bits_le(&mut builder);
        let bit = variable(&bits[0]);
        let mut witness = builder.witness().expect("witness is assigned");
        witness[bit.index()] += fr(1);

        assert!(builder.into_matrices().check_witness(&witness).is_err());
    }

    #[test]
    fn rejects_tampered_fr_injection_limb() {
        let mut builder = R1csBuilder::new();
        let value = AssignedScalar::alloc(&mut builder, Fr::from_u64(17));
        let fq = FqVar::inject_fr(&mut builder, &value);
        let first_limb = variable(&fq.limbs()[0]);
        let mut witness = builder.witness().expect("witness is assigned");
        witness[first_limb.index()] += fr(1);

        assert!(builder.into_matrices().check_witness(&witness).is_err());
    }

    #[test]
    fn fr_challenge_injection_uses_canonical_integer_embedding() {
        assert_fr_challenge_injection_satisfies(fr(0));
        assert_fr_challenge_injection_satisfies(Fr::from_u64(99));
        assert_fr_challenge_injection_satisfies(fr_from_limbs(FR_MODULUS_MINUS_ONE_LIMBS));
    }

    #[test]
    fn fr_challenge_injection_rejects_tampered_source_challenge() {
        let mut builder = R1csBuilder::new();
        let challenge = AssignedScalar::alloc(&mut builder, Fr::from_u64(99));
        let _ = FqVar::inject_fr_challenge(&mut builder, &challenge);
        let challenge_variable = variable(&challenge);
        let mut witness = builder.witness().expect("witness is assigned");
        witness[challenge_variable.index()] += fr(1);

        assert!(builder.into_matrices().check_witness(&witness).is_err());
    }

    #[test]
    fn fr_challenge_injection_rejects_tampered_fq_limb() {
        let mut builder = R1csBuilder::new();
        let challenge = AssignedScalar::alloc(&mut builder, Fr::from_u64(99));
        let fq = FqVar::inject_fr_challenge(&mut builder, &challenge);
        let first_limb = variable(&fq.limbs()[0]);
        let mut witness = builder.witness().expect("witness is assigned");
        witness[first_limb.index()] += fr(1);

        assert!(builder.into_matrices().check_witness(&witness).is_err());
    }

    #[test]
    fn fq_arithmetic_completeness_matches_native_field() {
        let values = interesting_fq_values();
        for &lhs in &values {
            for &rhs in &values {
                let mut builder = R1csBuilder::new();
                let lhs_var = FqVar::alloc(&mut builder, lhs);
                let rhs_var = FqVar::alloc(&mut builder, rhs);

                let sum = lhs_var.add(&mut builder, &rhs_var);
                let difference = lhs_var.sub(&mut builder, &rhs_var);
                let product = lhs_var.mul(&mut builder, &rhs_var);

                assert_eq!(sum.value(), lhs + rhs);
                assert_eq!(difference.value(), lhs - rhs);
                assert_eq!(product.value(), lhs * rhs);
                assert!(builder_accepts(builder));
            }
        }
    }

    #[test]
    fn fq_inverse_completeness_matches_native_field() {
        for value in interesting_fq_values()
            .into_iter()
            .filter(|value| *value != Fq::from_u64(0))
        {
            let mut builder = R1csBuilder::new();
            let value_var = FqVar::alloc(&mut builder, value);
            let inverse = value_var
                .inverse(&mut builder)
                .expect("nonzero field element has an inverse");

            assert_eq!(value * inverse.value(), Fq::from_u64(1));
            assert!(builder_accepts(builder));
        }
    }

    #[test]
    fn fq_inverse_returns_none_for_zero() {
        let mut builder = R1csBuilder::new();
        let zero = FqVar::alloc(&mut builder, Fq::from_u64(0));

        assert!(zero.inverse(&mut builder).is_none());
    }

    #[test]
    fn fq_select_completeness_matches_selector() {
        let mut builder = R1csBuilder::new();
        let when_true = FqVar::alloc(&mut builder, Fq::from_u64(11));
        let when_false = FqVar::alloc(&mut builder, fq_from_limbs(FQ_MODULUS_MINUS_ONE_LIMBS));
        let true_selector = AssignedScalar::alloc(&mut builder, fr(1));
        let false_selector = AssignedScalar::alloc(&mut builder, fr(0));

        let selected_true = FqVar::select(&mut builder, &true_selector, &when_true, &when_false);
        let selected_false = FqVar::select(&mut builder, &false_selector, &when_true, &when_false);

        assert_eq!(selected_true.value(), when_true.value());
        assert_eq!(selected_false.value(), when_false.value());
        assert!(builder_accepts(builder));
    }

    #[test]
    fn fq_select_rejects_non_boolean_selector() {
        let mut builder = R1csBuilder::new();
        let when_true = FqVar::alloc(&mut builder, Fq::from_u64(11));
        let when_false = FqVar::alloc(&mut builder, Fq::from_u64(7));
        let selector = AssignedScalar::alloc(&mut builder, fr(2));

        let _ = FqVar::select(&mut builder, &selector, &when_true, &when_false);

        assert!(builder_rejects(builder));
    }

    #[test]
    fn fq_add_soundness_rejects_single_variable_tampering() {
        let mut builder = R1csBuilder::new();
        let lhs = FqVar::alloc(&mut builder, fq_from_limbs(FQ_MODULUS_MINUS_ONE_LIMBS));
        let rhs = FqVar::alloc(&mut builder, Fq::from_u64(9));

        let _ = lhs.add(&mut builder, &rhs);

        assert_single_variable_tampering_rejected(builder);
    }

    #[test]
    fn fq_sub_soundness_rejects_single_variable_tampering() {
        let mut builder = R1csBuilder::new();
        let lhs = FqVar::alloc(&mut builder, Fq::from_u64(3));
        let rhs = FqVar::alloc(&mut builder, Fq::from_u64(9));

        let _ = lhs.sub(&mut builder, &rhs);

        assert_single_variable_tampering_rejected(builder);
    }

    #[test]
    fn fq_mul_soundness_rejects_single_variable_tampering() {
        let mut builder = R1csBuilder::new();
        let lhs = FqVar::alloc(&mut builder, fq_from_limbs(FQ_MODULUS_MINUS_ONE_LIMBS));
        let rhs = FqVar::alloc(&mut builder, fq_from_limbs([17, 9_876_543_210, 11, 3]));

        let _ = lhs.mul(&mut builder, &rhs);

        assert_single_variable_tampering_rejected(builder);
    }

    #[test]
    fn fq_inverse_soundness_rejects_single_variable_tampering() {
        let mut builder = R1csBuilder::new();
        let value = FqVar::alloc(&mut builder, fq_from_limbs([17, 9_876_543_210, 11, 3]));

        let _ = value
            .inverse(&mut builder)
            .expect("nonzero field element has an inverse");

        assert_single_variable_tampering_rejected(builder);
    }

    #[test]
    fn fq_select_soundness_rejects_single_variable_tampering() {
        let mut builder = R1csBuilder::new();
        let when_true = FqVar::alloc(&mut builder, fq_from_limbs([17, 9_876_543_210, 11, 3]));
        let when_false = FqVar::alloc(&mut builder, Fq::from_u64(7));
        let selector = AssignedScalar::alloc(&mut builder, fr(1));

        let _ = FqVar::select(&mut builder, &selector, &when_true, &when_false);

        assert_single_variable_tampering_rejected(builder);
    }

    fn assert_fq_value_satisfies(value: Fq) {
        let mut builder = R1csBuilder::new();
        let fq = FqVar::alloc(&mut builder, value);

        assert_eq!(
            fq.limbs()
                .iter()
                .map(|limb| scalar_low_u64(limb.value))
                .collect::<Vec<_>>(),
            fq_to_u64_limbs(value)
        );
        assert!(builder_accepts(builder));
    }

    fn assert_fr_injection_satisfies(value: Fr) {
        let mut builder = R1csBuilder::new();
        let scalar = AssignedScalar::alloc(&mut builder, value);
        let fq = FqVar::inject_fr(&mut builder, &scalar);

        assert_eq!(
            fq.limbs()
                .iter()
                .map(|limb| scalar_low_u64(limb.value))
                .collect::<Vec<_>>(),
            fr_to_u64_limbs(value)
        );
        assert!(builder_accepts(builder));
    }

    fn assert_fr_challenge_injection_satisfies(value: Fr) {
        let mut builder = R1csBuilder::new();
        let challenge = AssignedScalar::alloc(&mut builder, value);
        let fq = FqVar::inject_fr_challenge(&mut builder, &challenge);

        assert_eq!(
            fq.limbs()
                .iter()
                .map(|limb| scalar_low_u64(limb.value))
                .collect::<Vec<_>>(),
            fr_to_u64_limbs(value)
        );
        assert!(builder_accepts(builder));
    }

    fn builder_accepts(builder: R1csBuilder<Fr>) -> bool {
        let witness = builder.witness().expect("witness is assigned");
        builder.into_matrices().check_witness(&witness).is_ok()
    }

    fn builder_rejects(builder: R1csBuilder<Fr>) -> bool {
        let witness = builder.witness().expect("witness is assigned");
        builder.into_matrices().check_witness(&witness).is_err()
    }

    fn assert_single_variable_tampering_rejected(builder: R1csBuilder<Fr>) {
        let witness = builder.witness().expect("witness is assigned");
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());

        for index in 1..witness.len() {
            let mut tampered = witness.clone();
            tampered[index] += fr(1);
            assert!(
                matrices.check_witness(&tampered).is_err(),
                "variable {index} accepted after single-variable tampering"
            );
        }
    }

    fn interesting_fq_values() -> Vec<Fq> {
        vec![
            Fq::from_u64(0),
            Fq::from_u64(1),
            Fq::from_u64(2),
            Fq::from_u64(17),
            fq_from_limbs([123, 456, 789, 101_112]),
            fq_from_limbs([
                FQ_MODULUS_MINUS_ONE_LIMBS[0] - 1,
                FQ_MODULUS_MINUS_ONE_LIMBS[1],
                FQ_MODULUS_MINUS_ONE_LIMBS[2],
                FQ_MODULUS_MINUS_ONE_LIMBS[3],
            ]),
            fq_from_limbs(FQ_MODULUS_MINUS_ONE_LIMBS),
        ]
    }

    fn fq_from_limbs(limbs: [u64; NUM_LIMBS]) -> Fq {
        Fq::from_le_bytes_mod_order(&limbs_to_bytes(limbs))
    }

    fn fr_from_limbs(limbs: [u64; NUM_LIMBS]) -> Fr {
        Fr::from_le_bytes_mod_order(&limbs_to_bytes(limbs))
    }

    fn limbs_to_bytes(limbs: [u64; NUM_LIMBS]) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (index, limb) in limbs.into_iter().enumerate() {
            bytes[index * 8..index * 8 + 8].copy_from_slice(&limb.to_le_bytes());
        }
        bytes
    }

    fn variable(value: &AssignedScalar<Fr>) -> Variable {
        assert_eq!(value.lc.terms.len(), 1);
        let (variable, coefficient) = value
            .lc
            .terms
            .first()
            .copied()
            .expect("linear combination has one term");
        assert_eq!(coefficient, fr(1));
        variable
    }
}
