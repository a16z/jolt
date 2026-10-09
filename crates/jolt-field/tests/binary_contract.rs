#![cfg(feature = "binary")]
#![expect(clippy::unwrap_used, reason = "contract assertions in test code")]

use jolt_field::signed::S256;
use jolt_field::{
    Accumulator, CanonicalEncoding, ExtField, Field, JoltField, NaiveAccumulator, One, Ring,
    WithAccumulator, Zero, F128, F192, F64,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_core::{Error, RngCore};
use std::panic::AssertUnwindSafe;

fn assert_field_bounds<F>()
where
    F: JoltField
        + WithAccumulator<
            Accumulator = NaiveAccumulator<F>,
            SmallScalarAccumulator = NaiveAccumulator<F>,
            SignedProductAccumulator = NaiveAccumulator<F>,
        >,
{
}

fn check_metadata<F: JoltField>(top: F, width: usize, bits: u32) {
    assert_eq!(F::NUM_BYTES, width);
    assert_eq!(F::MODULUS_BITS, bits + 1);
    assert_eq!(F::zero().num_bits(), 0);
    assert_eq!(F::one().num_bits(), 1);
    assert_eq!(top.num_bits(), bits);
}

#[test]
fn trait_bounds_and_metadata() {
    assert_field_bounds::<F64>();
    assert_field_bounds::<F128>();
    assert_field_bounds::<F192>();
    check_metadata(F64::from_raw(1 << 63), 8, 64);
    check_metadata(F128::from_raw(1 << 127), 16, 128);
    check_metadata(
        F192::from_base_slice(&[F64::zero(), F64::zero(), F64::from_raw(1 << 63)]),
        24,
        192,
    );
}

fn check_algebra<F: JoltField>(bits: usize) {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6269_6e61_7279_0001);
    assert_eq!(F::zero().inverse(), None);
    assert_eq!(F::zero().inv_or_zero(), F::zero());
    for _ in 0..64 {
        let a = F::random(&mut rng);
        let b = F::random(&mut rng);
        let c = F::random(&mut rng);
        assert_eq!((a + b) + c, a + (b + c));
        assert_eq!(a + b, b + a);
        assert_eq!((a * b) * c, a * (b * c));
        assert_eq!(a * b, b * a);
        assert_eq!(a * (b + c), a * b + a * c);
        assert_eq!(a + a, F::zero());
        assert_eq!(-a, a);
        assert_eq!(a - b, a + b);
        assert_eq!(a * F::one(), a);
        assert_eq!(a * F::zero(), F::zero());
        assert_eq!(a.square(), a * a);
        assert_eq!(a.mul_add(b, c), a * b + c);
        if !a.is_zero() {
            assert_eq!(a * a.inverse().unwrap(), F::one());
        }
        let mut frobenius = a;
        for _ in 0..bits {
            frobenius = frobenius.square();
        }
        assert_eq!(frobenius, a);
    }
}

#[test]
fn seeded_field_axioms() {
    check_algebra::<F64>(64);
    check_algebra::<F128>(128);
    check_algebra::<F192>(192);
}

fn check_integer_maps<F: JoltField>() {
    assert_eq!(F::from_u64(2), F::zero());
    assert_eq!(F::from_u64(3), F::one());
    assert_eq!(F::from_u64(u64::MAX), F::one());
    assert_eq!(F::from_i64(-1), F::one());
    assert_eq!(F::from_i64(i64::MIN), F::zero());
    assert_eq!(F::from_i64(i64::MAX), F::one());
    assert_eq!(F::from_u128(1 << 64), F::zero());
    assert_eq!(F::from_u128(u128::MAX), F::one());
    assert_eq!(F::from_i128(i128::MIN), F::zero());
    assert_eq!(F::from_i128(-3), F::one());
    assert_eq!(F::from_i128(-2), F::zero());
    assert_eq!(F::pow2(0), F::one());
    assert_eq!(F::one().mul_pow_2(0), F::one());
    for exponent in 1..=255 {
        assert_eq!(F::pow2(exponent), F::zero());
        assert_eq!(F::one().mul_pow_2(exponent), F::zero());
    }
    assert!(std::panic::catch_unwind(|| F::one().mul_pow_2(256)).is_err());
    assert!(std::panic::catch_unwind(F::two_inv).is_err());
    assert!(std::panic::catch_unwind(|| F::one().half()).is_err());
}

#[test]
fn integer_maps_and_characteristic_two_defaults() {
    check_integer_maps::<F64>();
    check_integer_maps::<F128>();
    check_integer_maps::<F192>();
}

fn check_signed_accumulator<A: Accumulator>(value: A::Element) {
    for (scalar, odd) in [
        (1, true),
        (2, false),
        (-1, true),
        (-2, false),
        (1 << 64, false),
        ((1 << 64) + 1, true),
        (i128::MIN, false),
    ] {
        let mut acc = A::default();
        acc.fmadd_i128(value, scalar);
        assert_eq!(acc.reduce(), if odd { value } else { A::Element::zero() });
    }
    for is_positive in [true, false] {
        for (magnitude, odd) in [(1, true), (2, false), (u64::MAX, true)] {
            let mut acc = A::default();
            acc.fmadd_signed_u64(value, magnitude, is_positive);
            assert_eq!(acc.reduce(), if odd { value } else { A::Element::zero() });
        }
        for (limbs, odd) in [
            ([1, 0, 0, 0], true),
            ([2, 0, 0, 0], false),
            ([0, 1, 0, 0], false),
            ([0, 0, 0, 1], false),
            ([1, u64::MAX, u64::MAX, u64::MAX], true),
        ] {
            let mut acc = A::default();
            acc.fmadd_s256(value, &S256::new(limbs, is_positive));
            assert_eq!(acc.reduce(), if odd { value } else { A::Element::zero() });
        }
    }
}

fn check_accumulators<F: JoltField>(value: F) {
    check_signed_accumulator::<F::Accumulator>(value);
    check_signed_accumulator::<F::SmallScalarAccumulator>(value);
    check_signed_accumulator::<F::SignedProductAccumulator>(value);
    let mut acc = F::Accumulator::default();
    acc.add(value);
    let mut partial = F::Accumulator::default();
    partial.fmadd(value, F::one());
    acc.merge(partial);
    assert_eq!(acc.reduce(), F::zero());
}

#[test]
fn signed_accumulators_reduce_scalar_parity() {
    check_accumulators(F64::from_raw(0xfedc_ba98_7654_3210));
    check_accumulators(F128::from_raw(0xfedc_ba98_7654_3210_0123_4567_89ab_cdef));
    check_accumulators(F192::from_base_slice(&[
        F64::from_raw(0xfedc_ba98_7654_3210),
        F64::from_raw(0x0123_4567_89ab_cdef),
        F64::from_raw(0x1020_3040_5060_7080),
    ]));
}

fn check_encoding<F: JoltField>(value: F, frozen: &[u8], short: F, display: &str) {
    assert_eq!(value.to_string(), display);
    assert_eq!(value.to_bytes_le_vec(), frozen);
    let config = bincode::config::standard();
    assert_eq!(
        bincode::serde::encode_to_vec(value, config).unwrap(),
        frozen
    );
    let (decoded, read): (F, usize) = bincode::serde::decode_from_slice(frozen, config).unwrap();
    assert_eq!(decoded, value);
    assert_eq!(read, F::NUM_BYTES);
    assert_eq!(F::from_bytes_le_checked(frozen), Some(value));
    assert_eq!(
        F::from_bytes_le_checked(&vec![0xff; F::NUM_BYTES]),
        Some(F::from_bytes_le_reduced(&vec![0xff; F::NUM_BYTES]))
    );
    for length in (0..F::NUM_BYTES).chain([F::NUM_BYTES + 1, F::NUM_BYTES * 2 + 7]) {
        assert_eq!(F::from_bytes_le_checked(&vec![0xff; length]), None);
        let mut out = vec![0u8; length];
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| value.to_bytes_le(&mut out))).is_err()
        );
    }
    let mut long = frozen.to_vec();
    long.extend_from_slice(&[0xaa; 31]);
    for (bytes, expected) in [
        (&[][..], F::zero()),
        (&[0x01, 0x23, 0x45][..], short),
        (frozen, value),
        (long.as_slice(), value),
    ] {
        assert_eq!(F::from_bytes_le_reduced(bytes), expected);
        assert_eq!(F::from_challenge_bytes(bytes), expected);
        assert_eq!(F::from_scalar_challenge_bytes(bytes), expected);
    }
}

#[test]
fn frozen_encoding_and_decode_contracts() {
    check_encoding(
        F64::from_raw(0x0123_4567_89ab_cdef),
        &[0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01],
        F64::from_raw(0x0045_2301),
        "123456789abcdef",
    );
    check_encoding(
        F128::from_raw(0xfedc_ba98_7654_3210_0123_4567_89ab_cdef),
        &[
            0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01, 0x10, 0x32, 0x54, 0x76, 0x98, 0xba,
            0xdc, 0xfe,
        ],
        F128::from_raw(0x0045_2301),
        "fedcba98765432100123456789abcdef",
    );
    check_encoding(
        F192::from_base_slice(&[
            F64::from_raw(0x0123_4567_89ab_cdef),
            F64::from_raw(0xfedc_ba98_7654_3210),
            F64::from_raw(0x1020_3040_5060_7080),
        ]),
        &[
            0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01, 0x10, 0x32, 0x54, 0x76, 0x98, 0xba,
            0xdc, 0xfe, 0x80, 0x70, 0x60, 0x50, 0x40, 0x30, 0x20, 0x10,
        ],
        F192::lift_base(F64::from_raw(0x0045_2301)),
        "1020304050607080fedcba98765432100123456789abcdef",
    );
}

#[test]
fn raw_coordinates_and_u128_boundaries() {
    assert_eq!(F64::from_raw(u64::MAX).to_raw(), u64::MAX);
    assert_eq!(F128::from_raw(u128::MAX).to_raw(), u128::MAX);
    for raw in [0, 1, u128::from(u64::MAX)] {
        let base = F64::from_raw(raw as u64);
        assert_eq!(F64::from_u128_checked(raw), Some(base));
        assert_eq!(base.to_u128_checked(), Some(raw));
        assert_eq!(F64::from_u128_reduced(raw), base);
        let extension = F192::lift_base(base);
        assert_eq!(F192::from_u128_checked(raw), Some(extension));
        assert_eq!(extension.to_u128_checked(), Some(raw));
        assert_eq!(F192::from_u128_reduced(raw), extension);
    }
    for raw in [1 << 64, (1 << 64) + 1, u128::MAX] {
        assert_eq!(F64::from_u128_checked(raw), None);
        assert_eq!(F192::from_u128_checked(raw), None);
        assert_eq!(F64::from_u128_reduced(raw), F64::from_raw(raw as u64));
        assert_eq!(
            F192::from_u128_reduced(raw),
            F192::lift_base(F64::from_raw(raw as u64))
        );
    }
    for raw in [0, 1, 1 << 64, 1 << 127, u128::MAX] {
        let element = F128::from_raw(raw);
        assert_eq!(F128::from_u128_checked(raw), Some(element));
        assert_eq!(element.to_u128_checked(), Some(raw));
        assert_eq!(F128::from_u128_reduced(raw), element);
    }
    for coefficients in [[0, 1, 0], [0, 0, 1]] {
        assert_eq!(
            F192::from_base_fn(|i| F64::from_raw(coefficients[i])).to_u128_checked(),
            None
        );
    }
    assert_eq!((F64::from_raw(0x55) + F64::from_raw(0xaa)).to_raw(), 0xff);
    assert_eq!((F128::from_raw(0x55) + F128::from_raw(0xaa)).to_raw(), 0xff);
}

#[derive(Default)]
struct CountingRng {
    consumed: usize,
}

impl RngCore for CountingRng {
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0; 4];
        self.fill_bytes(&mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0; 8];
        self.fill_bytes(&mut bytes);
        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        dest.fill(0xa5);
        self.consumed += dest.len();
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

fn check_sampling<F: JoltField>() {
    let mut rng = CountingRng::default();
    let expected = F::from_bytes_le_checked(&vec![0xa5; F::NUM_BYTES]).unwrap();
    for sample in 1..=3 {
        assert_eq!(F::random(&mut rng), expected);
        assert_eq!(rng.consumed, sample * F::NUM_BYTES);
    }
}

#[test]
fn sampling_consumes_exact_width() {
    check_sampling::<F64>();
    check_sampling::<F128>();
    check_sampling::<F192>();
}

#[test]
fn extension_basis_contract() {
    assert_eq!(F192::DEGREE, 3);
    let coefficients = [
        F64::from_raw(0x1234),
        F64::from_raw(0x5678),
        F64::from_raw(0x9abc),
    ];
    let mut calls = Vec::new();
    let extension = F192::from_base_fn(|index| {
        calls.push(index);
        coefficients[index]
    });
    assert_eq!(calls, [0, 1, 2]);
    assert_eq!(extension.to_base_vec(), coefficients);
    let addend = F192::from_base_slice(&[F64::from_raw(0xffff); 3]);
    assert_eq!(
        (extension + addend).to_base_vec(),
        [
            F64::from_raw(0xedcb),
            F64::from_raw(0xa987),
            F64::from_raw(0x6543)
        ],
    );
    for index in [3, 4, usize::MAX] {
        assert!(std::panic::catch_unwind(|| extension.base_coefficient(index)).is_err());
    }
    for length in [0, 1, 2, 4] {
        assert!(
            std::panic::catch_unwind(|| F192::from_base_slice(&vec![F64::one(); length])).is_err()
        );
    }
    let mut rng = ChaCha20Rng::seed_from_u64(0x6261_7365_0000_0001);
    for _ in 0..64 {
        let base = F64::random(&mut rng);
        assert_eq!(
            F192::lift_base(base).to_base_vec(),
            [base, F64::zero(), F64::zero()]
        );
        let element = F192::random(&mut rng);
        let scaled = element.mul_base(base);
        for index in 0..F192::DEGREE {
            assert_eq!(
                scaled.base_coefficient(index),
                element.base_coefficient(index) * base
            );
        }
        assert_eq!(scaled, element * F192::lift_base(base));
    }
}

#[test]
fn relative_frobenius_contract() {
    let y = F192::from_base_slice(&[F64::zero(), F64::one(), F64::zero()]);
    let y_squared = F192::from_base_slice(&[F64::zero(), F64::zero(), F64::one()]);
    assert_eq!(y.frobenius_pow(1), y_squared);
    let mut rng = ChaCha20Rng::seed_from_u64(0x6672_6f62_0000_0001);
    for _ in 0..16 {
        let base = F64::random(&mut rng);
        assert_eq!(
            F192::lift_base(base).frobenius_pow(1),
            F192::lift_base(base)
        );
        let element = F192::random(&mut rng);
        assert_eq!(element.frobenius_pow(0), element);
        assert_eq!(element.frobenius_pow(3), element);
        assert_eq!(element.frobenius_pow(usize::MAX), element);
        let mut exponentiated = element;
        for _ in 0..64 {
            exponentiated = exponentiated.square();
        }
        assert_eq!(element.frobenius_pow(1), exponentiated);
    }
}
