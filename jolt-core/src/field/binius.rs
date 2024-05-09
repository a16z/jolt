use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use binius_field::{BinaryField, BinaryField128b, BinaryField128bPolyval};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use super::JoltField;

use once_cell::sync::Lazy;

static PRECOMPUTED_LOW_128B: Lazy<[BinaryField128b; TABLE_SIZE]> = Lazy::new(|| {
    compute_powers::<BinaryField128b, TABLE_SIZE>(BinaryField128b::MULTIPLICATIVE_GENERATOR)
});
static PRECOMPUTED_HIGH_128B: Lazy<[BinaryField128b; TABLE_SIZE]> = Lazy::new(|| {
    compute_powers_starting_from::<BinaryField128b, TABLE_SIZE>(
        BinaryField128b::MULTIPLICATIVE_GENERATOR,
        16,
    )
});
impl BiniusConstructable for BinaryField128b {
    fn new(n: u64) -> Self {
        Self::new(n as u128)
    }

    fn precomputed_generator_multiples(
    ) -> (&'static [Self; TABLE_SIZE], &'static [Self; TABLE_SIZE]) {
        (&PRECOMPUTED_LOW_128B, &PRECOMPUTED_HIGH_128B)
    }
}

static PRECOMPUTED_LOW_128B_POLYVAL: Lazy<[BinaryField128bPolyval; TABLE_SIZE]> = Lazy::new(|| {
    compute_powers::<BinaryField128bPolyval, TABLE_SIZE>(
        BinaryField128bPolyval::MULTIPLICATIVE_GENERATOR,
    )
});
static PRECOMPUTED_HIGH_128B_POLYVAL: Lazy<[BinaryField128bPolyval; TABLE_SIZE]> =
    Lazy::new(|| {
        compute_powers_starting_from::<BinaryField128bPolyval, TABLE_SIZE>(
            BinaryField128bPolyval::MULTIPLICATIVE_GENERATOR,
            16,
        )
    });
impl BiniusConstructable for BinaryField128bPolyval {
    fn new(n: u64) -> Self {
        Self::new(n as u128)
    }

    fn precomputed_generator_multiples(
    ) -> (&'static [Self; TABLE_SIZE], &'static [Self; TABLE_SIZE]) {
        (
            &PRECOMPUTED_LOW_128B_POLYVAL,
            &PRECOMPUTED_HIGH_128B_POLYVAL,
        )
    }
}

impl BiniusSpecific for BinaryField128b {}
impl BiniusSpecific for BinaryField128bPolyval {}

/// Trait for BiniusField functionality specific to each impl.
pub trait BiniusSpecific: binius_field::TowerField + BiniusConstructable + bytemuck::Pod {}

const LOG_TABLE_SIZE: usize = 16;
const TABLE_SIZE: usize = 1 << LOG_TABLE_SIZE;

pub trait BiniusConstructable: BinaryField {
    fn new(n: u64) -> Self;

    /// Binius counts are constructed from multiplicities of a Binary Field multiplicative generator.
    /// Precomputing all required counts [0, 2^32] is prohibitively expensive and using iterative multiplication
    /// or square and multiply is still excessively costly.
    /// Utilizes a two-table lookup method to handle counts up to `2^32` efficiently:
    /// - Decompose count `x` as `x = a * 2^16 + b` where `a` and `b` are within the range `[0, 2^16]`.
    /// - Two Precomptued Lookup Tables:
    ///   - One table stores `2^16` powers of `g^{2^16}`.
    ///   - Another table stores `2^16` powers of `g`.
    /// - Computes any count up to `2^32` using: `g^x = (g^{2^16})^a * g^b`
    /// This is achieved with two lookups (one from each table) and a single multiplication.
    ///
    /// `precomputed_generator_multiples() -> (&low_table, &high_table)`
    fn precomputed_generator_multiples(
    ) -> (&'static [Self; TABLE_SIZE], &'static [Self; TABLE_SIZE]);

    fn from_count_index(n: u64) -> Self {
        const MAX_COUNTS: usize = 1 << (2 * LOG_TABLE_SIZE);
        let n = (n) as usize;
        assert!(n <= MAX_COUNTS);
        assert!(n != 0);

        let high_index = (n >> LOG_TABLE_SIZE) as usize;
        let low_index = (n & ((1 << LOG_TABLE_SIZE) - 1)) as usize;
        let (precomputed_low, precomputed_high) = Self::precomputed_generator_multiples();
        // TODO(sragss): mul_0_optimized
        if n < TABLE_SIZE {
            precomputed_low[n]
        } else {
            println!("low[{low_index}] * high[{high_index}]");
            precomputed_low[low_index] * precomputed_high[high_index]
        }
    }
}

#[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
pub struct BiniusField<F: BiniusSpecific>(F);

/// Wrapper for all generic BiniusField functionality.
impl<F: BiniusSpecific> JoltField for BiniusField<F> {
    const NUM_BYTES: usize = 16;

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        Self(F::random(rng))
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero().into()
    }

    fn is_one(&self) -> bool {
        self.0 == Self::one().0
    }

    fn zero() -> Self {
        Self(F::ZERO)
    }

    fn one() -> Self {
        Self(F::ONE)
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self(F::new(n)))
    }

    fn square(&self) -> Self {
        Self(self.0.square())
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::NUM_BYTES);

        let field_element = bytemuck::try_from_bytes::<F>(bytes).unwrap();
        Self(field_element.to_owned())
    }

    fn from_count_index(index: u64) -> Self {
        Self(<F as BiniusConstructable>::from_count_index(index))
    }
}

impl<F: BiniusSpecific> Neg for BiniusField<F> {
    type Output = Self;

    fn neg(self) -> Self {
        BiniusField(self.0.neg())
    }
}

impl<F: BiniusSpecific> Add for BiniusField<F> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        BiniusField(self.0.add(other.0))
    }
}

impl<F: BiniusSpecific> Sub for BiniusField<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        BiniusField(self.0.sub(other.0))
    }
}

impl<F: BiniusSpecific> Mul for BiniusField<F> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        BiniusField(self.0.mul(other.0))
    }
}

impl<F: BiniusSpecific> Div for BiniusField<F> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self(self.0.mul(other.0.invert().unwrap()))
    }
}

impl<F: BiniusSpecific> AddAssign for BiniusField<F> {
    fn add_assign(&mut self, other: Self) {
        self.0.add_assign(other.0);
    }
}

impl<F: BiniusSpecific> SubAssign for BiniusField<F> {
    fn sub_assign(&mut self, other: Self) {
        self.0.sub_assign(other.0);
    }
}

impl<F: BiniusSpecific> MulAssign for BiniusField<F> {
    fn mul_assign(&mut self, other: Self) {
        self.0.mul_assign(other.0);
    }
}

impl<F: BiniusSpecific> core::iter::Sum for BiniusField<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| BiniusField(acc.0.add(x.0)))
    }
}

impl<'a, F: BiniusSpecific> core::iter::Sum<&'a Self> for BiniusField<F> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| BiniusField(acc.0.add(x.0)))
    }
}

impl<F: BiniusSpecific> core::iter::Product for BiniusField<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| BiniusField(acc.0.mul(x.0)))
    }
}

impl<'a, F: BiniusSpecific> core::iter::Product<&'a Self> for BiniusField<F> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| BiniusField(acc.0.mul(x.0)))
    }
}

impl<F: BiniusSpecific> CanonicalSerialize for BiniusField<F> {
    fn serialize_with_mode<W: ark_std::io::Write>(
        &self,
        mut writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        let bytes = bytemuck::bytes_of(&self.0);
        writer.write_all(bytes)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        Self::NUM_BYTES
    }
}

impl<F: BiniusSpecific> CanonicalDeserialize for BiniusField<F> {
    // Required method
    fn deserialize_with_mode<R: std::io::prelude::Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        todo!()
    }
}

impl<F: BiniusSpecific> ark_serialize::Valid for BiniusField<F> {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        todo!()
    }
}

/// Computes `N` sequential powers of base^{[0, ... N]}
const fn compute_powers<F: binius_field::BinaryField, const N: usize>(base: F) -> [F; N] {
    let mut powers = [F::ZERO; N];
    powers[0] = F::ONE;
    powers[1] = base;
    let mut i = 2;
    while i < N {
        powers[i] = powers[i - 1] * base;
        i += 1;
    }
    powers
}

/// Computes `N` sequential powers of base^{2^starting_power}: {base^{2^starting_power}}^{[1, ... N]}
const fn compute_powers_starting_from<F: binius_field::BinaryField, const N: usize>(
    base: F,
    starting_power: usize,
) -> [F; N] {
    let mut starting = base;
    let mut count = 0;
    // // Repeated squaring
    while count < starting_power {
        starting = starting.mul(starting);
        count += 1;
    }
    compute_powers::<F, N>(starting)
}

#[cfg(test)]
mod tests {
    use super::*;
    use binius_field::Field;

    #[test]
    fn test_compute_powers() {
        let base = BinaryField128b::MULTIPLICATIVE_GENERATOR;
        let powers = compute_powers::<BinaryField128b, 5>(base);
        let expected_powers = [
            BinaryField128b::ONE,
            base,
            base * base,
            base * base * base,
            base * base * base * base,
        ];
        assert_eq!(powers, expected_powers);
    }

    #[test]
    fn test_compute_powers_starting_from() {
        let base = BinaryField128b::MULTIPLICATIVE_GENERATOR;
        let powers = compute_powers_starting_from::<BinaryField128b, 5>(base, 2);
        let expected_starting_base = base * base * base * base;
        let expected_powers = [
            BinaryField128b::ONE,
            expected_starting_base,
            expected_starting_base * expected_starting_base,
            expected_starting_base * expected_starting_base * expected_starting_base,
            expected_starting_base
                * expected_starting_base
                * expected_starting_base
                * expected_starting_base,
        ];
        assert_eq!(powers, expected_powers);
    }

    #[test]
    fn test_from_count_index() {
        let actual = BiniusField::<BinaryField128b>::from_count_index(1);
        let expected = BinaryField128b::MULTIPLICATIVE_GENERATOR;
        assert_eq!(actual.0, expected);

        let actual = BiniusField::<BinaryField128b>::from_count_index(2);
        let expected =
            BinaryField128b::MULTIPLICATIVE_GENERATOR * BinaryField128b::MULTIPLICATIVE_GENERATOR;
        assert_eq!(actual.0, expected);

        let actual = BiniusField::<BinaryField128b>::from_count_index(3);
        let expected = BinaryField128b::MULTIPLICATIVE_GENERATOR
            * BinaryField128b::MULTIPLICATIVE_GENERATOR
            * BinaryField128b::MULTIPLICATIVE_GENERATOR;
        assert_eq!(actual.0, expected);

        let actual = BiniusField::<BinaryField128b>::from_count_index(1 << 17);
        let mut expected = BinaryField128b::MULTIPLICATIVE_GENERATOR;
        for _ in 0..17 {
            expected = expected.square();
        }
        assert_eq!(actual.0, expected);

        let actual = BiniusField::<BinaryField128b>::from_count_index((1 << 17) + 1);
        let mut expected = BinaryField128b::MULTIPLICATIVE_GENERATOR;
        for _ in 0..17 {
            expected = expected.square();
        }
        expected *= BinaryField128b::MULTIPLICATIVE_GENERATOR;
        assert_eq!(actual.0, expected);

        let actual = BiniusField::<BinaryField128b>::from_count_index((1 << 18) + 2);
        let mut expected = BinaryField128b::MULTIPLICATIVE_GENERATOR;
        for _ in 0..18 {
            expected = expected.square();
        }
        expected *= BinaryField128b::MULTIPLICATIVE_GENERATOR;
        expected *= BinaryField128b::MULTIPLICATIVE_GENERATOR;
        assert_eq!(actual.0, expected);
    }
}
