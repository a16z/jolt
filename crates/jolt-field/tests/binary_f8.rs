#![cfg(feature = "binary")]

use jolt_field::signed::S256;
use jolt_field::{
    Accumulator, CanonicalBytes, CanonicalEncoding, ExtField, Field, JoltField, NaiveAccumulator,
    One, Ring, WithAccumulator, Zero, F128, F192, F64, F8,
};
use rand_core::{Error, RngCore};
use std::collections::HashSet;
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

fn assert_embedding_bound<E: From<F8>>() {}

#[test]
fn trait_bounds() {
    assert_field_bounds::<F8>();
    assert_embedding_bound::<F64>();
    assert_embedding_bound::<F128>();
    assert_embedding_bound::<F192>();
}

fn power<F: Ring>(value: F, exponent: usize) -> F {
    (0..exponent).fold(F::one(), |product, _| product * value)
}

#[test]
fn aes_vectors() {
    assert_eq!(F8::from_raw(0x57) * F8::from_raw(0x83), F8::from_raw(0xc1));
    assert_eq!(F8::from_raw(0x57) * F8::from_raw(0x13), F8::from_raw(0xfe));
    assert_eq!(F8::from_raw(0x53).inverse(), Some(F8::from_raw(0xca)));
    assert_eq!(F8::zero().inverse(), None);
}

#[test]
fn exhaustive_field_axioms() {
    let c = F8::from_raw(0xa7);
    for raw_a in 0..=u8::MAX {
        let a = F8::from_raw(raw_a);
        if !a.is_zero() {
            assert_eq!(a.inverse().map(|inverse| a * inverse), Some(F8::one()));
        }
        assert_eq!(a.square(), a * a);
        assert_eq!(power(a, 256), a);
        for raw_b in 0..=u8::MAX {
            let b = F8::from_raw(raw_b);
            assert_eq!(a + b, b + a);
            assert_eq!(a * b, b * a);
            assert_eq!(a * (b + c), a * b + a * c);
        }
    }
}

#[test]
fn multiplicative_orders() {
    let x = F8::from_raw(2);
    assert_eq!(power(x, 51), F8::one());
    for exponent in [17, 3] {
        assert_ne!(power(x, exponent), F8::one());
    }
    let primitive = F8::from_raw(3);
    assert_eq!(power(primitive, 255), F8::one());
    for exponent in [85, 51, 15] {
        assert_ne!(power(primitive, exponent), F8::one());
    }
}

#[test]
fn spine_contract() {
    assert_eq!(F8::NUM_BYTES, 1);
    assert_eq!(F8::MODULUS_BITS, 9);
    assert_eq!(F8::from_u64(2), F8::zero());
    assert_eq!(F8::pow2(1), F8::zero());
    assert_eq!(F8::one().mul_pow_2(1), F8::zero());
    assert_eq!(F8::from_u64(3), F8::one());
    assert_eq!(F8::from_i64(-1), F8::one());
    assert!(std::panic::catch_unwind(F8::two_inv).is_err());
    assert!(std::panic::catch_unwind(|| F8::one().half()).is_err());

    let value = F8::from_raw(0xab);
    let mut bytes = [0];
    value.to_bytes_le(&mut bytes);
    assert_eq!(bytes, [0xab]);
    assert_eq!(
        bincode::serde::encode_to_vec(value, bincode::config::standard()).ok(),
        Some(vec![0xab])
    );
    for length in [0, 2] {
        let mut out = vec![0; length];
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| value.to_bytes_le(&mut out))).is_err()
        );
    }
    for (input, expected) in [(&[][..], F8::zero()), (&[0xab, 0xcd][..], value)] {
        assert_eq!(F8::from_bytes_le_reduced(input), expected);
        assert_eq!(F8::from_challenge_bytes(input), expected);
        assert_eq!(F8::from_scalar_challenge_bytes(input), expected);
    }
    assert_eq!(F8::from_bytes_le_checked(&[]), None);
    assert_eq!(F8::from_bytes_le_checked(&[0xab]), Some(value));
    assert_eq!(F8::from_bytes_le_checked(&[0xab, 0xcd]), None);
    assert_eq!(F8::from_u128_checked(0x100), None);
    assert_eq!(F8::from_u128_reduced(0x1ab), value);
    assert_eq!(value.to_u128_checked(), Some(0xab));
    assert_eq!(F8::zero().num_bits(), 0);
    assert_eq!(F8::one().num_bits(), 1);
    assert_eq!(F8::from_raw(0x80).num_bits(), 8);
    assert_eq!(F8::zero().to_string(), "00");
    assert_eq!(F8::one().to_string(), "01");

    let mut rng = CountingRng::default();
    assert_eq!(F8::random(&mut rng), F8::from_raw(0xa5));
    assert_eq!(rng.consumed, 1);
}

#[test]
fn signed_accumulator_parity() {
    let value = F8::from_raw(0xab);
    for (scalar, odd) in [
        (1, true),
        (2, false),
        (-1, true),
        (-2, false),
        (1 << 64, false),
        ((1 << 64) + 1, true),
        (i128::MIN, false),
    ] {
        let mut acc = NaiveAccumulator::<F8>::default();
        acc.fmadd_i128(value, scalar);
        assert_eq!(acc.reduce(), if odd { value } else { F8::zero() });
    }
    for is_positive in [true, false] {
        for (magnitude, odd) in [(1, true), (2, false), (u64::MAX, true)] {
            let mut acc = NaiveAccumulator::<F8>::default();
            acc.fmadd_signed_u64(value, magnitude, is_positive);
            assert_eq!(acc.reduce(), if odd { value } else { F8::zero() });
        }
        for (limbs, odd) in [
            ([1, 0, 0, 0], true),
            ([2, 0, 0, 0], false),
            ([0, 1, 0, 0], false),
            ([0, 0, 0, 1], false),
            ([1, u64::MAX, u64::MAX, u64::MAX], true),
        ] {
            let mut acc = NaiveAccumulator::<F8>::default();
            acc.fmadd_s256(value, &S256::new(limbs, is_positive));
            assert_eq!(acc.reduce(), if odd { value } else { F8::zero() });
        }
    }
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

fn check_embedding<E: JoltField + From<F8>>(beta: E) {
    assert_eq!(E::from(F8::from_raw(2)), beta);
    assert_eq!(
        power(beta, 8) + power(beta, 4) + power(beta, 3) + beta + E::one(),
        E::zero()
    );
    let mut conjugates = HashSet::new();
    let mut conjugate = beta;
    for _ in 0..8 {
        assert!(conjugates.insert(conjugate.to_u128_checked()));
        assert!(conjugate.to_u128_checked() >= beta.to_u128_checked());
        conjugate = conjugate.square();
    }
    assert_eq!(conjugate, beta);
    assert_eq!(E::from(F8::one()), E::one());
    let mut images = HashSet::new();
    for raw_a in 0..=u8::MAX {
        let a = F8::from_raw(raw_a);
        let image_a = E::from(a);
        assert!(images.insert(image_a.to_u128_checked()));
        for raw_b in 0..=u8::MAX {
            let b = F8::from_raw(raw_b);
            let image_b = E::from(b);
            assert_eq!(E::from(a + b), image_a + image_b);
            assert_eq!(E::from(a * b), image_a * image_b);
        }
    }
}

#[test]
fn f64_embedding() {
    check_embedding(F64::from_raw(0x033c_e8be_ddc8_a656));
    assert_eq!(
        F64::from(F8::from_raw(0x53)).to_raw(),
        0xff05_4c3f_7cef_0cca
    );
    assert_eq!(
        F64::from(F8::from_raw(0xff)).to_raw(),
        0x5c83_4678_7364_a654
    );
}

#[test]
fn f128_embedding() {
    check_embedding(F128::from_raw(0x053d_8555_a997_9a1c_a13f_e8ac_5560_ce0d));
    assert_eq!(
        F128::from(F8::from_raw(0x53)).to_raw(),
        0xde77_167b_8539_a797_0d97_2a0b_4c6f_a967
    );
    assert_eq!(
        F128::from(F8::from_raw(0xff)).to_raw(),
        0xfae6_e08c_31e8_9f90_017e_b6dc_d4f3_3a26
    );
}

#[test]
fn f192_embedding_factors_through_f64() {
    for raw in 0..=u8::MAX {
        let a = F8::from_raw(raw);
        assert_eq!(F192::from(a), F192::lift_base(F64::from(a)));
    }
}
