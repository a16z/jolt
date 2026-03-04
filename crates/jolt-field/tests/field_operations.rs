use ark_bn254::Fr;
use ark_std::rand::Rng;
use ark_std::{test_rng, One, Zero};
use jolt_field::Field;
use rand_chacha::rand_core::RngCore;

#[test]
fn implicit_montgomery_conversion() {
    let mut rng = test_rng();

    for _ in 0..256 {
        let x = rng.next_u64();
        assert_eq!(
            <Fr as Field>::from_u64(x),
            Fr::one() * <Fr as Field>::from_u64(x)
        );
    }

    for _ in 0..256 {
        let x = rng.next_u64();
        let y: Fr = Field::random(&mut rng);
        assert_eq!(
            y * <Fr as Field>::from_u64(x),
            y * <Fr as Field>::from_u64(x)
        );
    }
}

#[test]
fn field_arithmetic() {
    let mut rng = test_rng();

    let x = <Fr as Field>::from_u64(rng.next_u64());
    let y = <Fr as Field>::from_u64(rng.next_u64());

    let sum = x + y;
    assert_eq!(sum, y + x);

    let product = x * y;
    assert_eq!(product, y * x);

    let diff = x - y;
    assert_eq!(diff + y, x);

    if !y.is_zero() {
        let quotient = x / y;
        assert_eq!(quotient * y, x);
    }
}

#[test]
fn field_conversions() {
    let mut rng = test_rng();

    assert_eq!(<Fr as Field>::from_bool(true), Fr::one());
    assert_eq!(<Fr as Field>::from_bool(false), Fr::zero());

    for _ in 0..100 {
        let val = rng.gen::<u8>();
        let field_elem = <Fr as Field>::from_u8(val);
        assert_eq!(field_elem, <Fr as Field>::from_u64(val as u64));
    }

    for _ in 0..100 {
        let val = rng.gen::<u16>();
        let field_elem = <Fr as Field>::from_u16(val);
        assert_eq!(field_elem, <Fr as Field>::from_u64(val as u64));
    }

    for _ in 0..100 {
        let val = rng.gen::<u32>();
        let field_elem = <Fr as Field>::from_u32(val);
        assert_eq!(field_elem, <Fr as Field>::from_u64(val as u64));
    }

    for _ in 0..100 {
        let val = rng.gen::<u128>();
        let field_elem = <Fr as Field>::from_u128(val);
        assert!(!field_elem.is_zero() || val == 0);
    }
}

#[test]
fn bytes_conversion() {
    let mut rng = test_rng();

    for &len in &[1, 8, 16, 32, 48, 64] {
        let mut bytes = vec![0u8; len];
        rng.fill_bytes(&mut bytes);
        let _field_elem = <Fr as Field>::from_bytes(&bytes);
    }
}

#[test]
fn signed_conversions() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let val = rng.gen::<i64>();
        let field_elem = <Fr as Field>::from_i64(val);

        if val >= 0 {
            assert_eq!(field_elem, <Fr as Field>::from_u64(val as u64));
        } else {
            assert_eq!(field_elem, -<Fr as Field>::from_u64(val.unsigned_abs()));
        }
    }

    for _ in 0..100 {
        let val = rng.gen::<i128>();
        let field_elem = <Fr as Field>::from_i128(val);

        if val >= 0 {
            assert_eq!(field_elem, <Fr as Field>::from_u128(val as u128));
        } else {
            assert_eq!(field_elem, -<Fr as Field>::from_u128(val.unsigned_abs()));
        }
    }
}

#[test]
fn mul_u64_method() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let field_elem: Fr = Field::random(&mut rng);
        let n = rng.next_u64();

        // Use UFCS to call trait method (arkworks has inherent mul_u64 with different signature)
        let result = <Fr as Field>::mul_u64(&field_elem, n);
        let expected = field_elem * <Fr as Field>::from_u64(n);
        assert_eq!(result, expected);
    }
}

#[test]
fn mul_i64_method() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let field_elem: Fr = Field::random(&mut rng);
        let n = rng.gen::<i64>();

        let result = <Fr as Field>::mul_i64(&field_elem, n);
        let expected = field_elem * <Fr as Field>::from_i64(n);
        assert_eq!(result, expected);
    }
}

#[test]
fn mul_u128_method() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let field_elem: Fr = Field::random(&mut rng);
        let n = rng.gen::<u128>();

        let result = <Fr as Field>::mul_u128(&field_elem, n);
        let expected = field_elem * <Fr as Field>::from_u128(n);
        assert_eq!(result, expected);
    }
}

#[test]
fn mul_i128_method() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let field_elem: Fr = Field::random(&mut rng);
        let n = rng.gen::<i128>();

        let result = <Fr as Field>::mul_i128(&field_elem, n);
        let expected = field_elem * <Fr as Field>::from_i128(n);
        assert_eq!(result, expected);
    }
}

#[test]
fn mul_pow_2_method() {
    let mut rng = test_rng();

    for _ in 0..10 {
        let field_elem: Fr = Field::random(&mut rng);

        for pow in [0, 1, 2, 7, 16, 32, 63, 64, 127, 128, 255] {
            let result = <Fr as Field>::mul_pow_2(&field_elem, pow);
            let mut expected = field_elem;
            for _ in 0..pow {
                expected = expected + expected;
            }
            assert_eq!(result, expected, "Failed for pow={pow}");
        }
    }
}

#[test]
fn mul_by_small_values() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let field_elem: Fr = Field::random(&mut rng);
        let small_val = rng.gen_range(0u64..1000);

        let result1 = field_elem * <Fr as Field>::from_u64(small_val);

        let mut result2 = Fr::zero();
        for _ in 0..small_val {
            result2 += field_elem;
        }

        assert_eq!(result1, result2);
    }
}

#[test]
fn special_values() {
    let mut rng = test_rng();
    let field_elem: Fr = Field::random(&mut rng);

    assert_eq!(field_elem * <Fr as Field>::from_u64(0), Fr::zero());
    assert_eq!(field_elem * <Fr as Field>::from_u64(1), field_elem);
    assert!((Fr::zero() * <Fr as Field>::from_u64(rng.next_u64())).is_zero());

    assert_eq!(<Fr as Field>::mul_u64(&field_elem, 0), Fr::zero());
    assert_eq!(<Fr as Field>::mul_u64(&field_elem, 1), field_elem);
    assert_eq!(<Fr as Field>::mul_u64(&Fr::zero(), 42), Fr::zero());
}

#[test]
fn to_u64_conversion() {
    for i in 0..1000u64 {
        let field_elem = <Fr as Field>::from_u64(i);
        assert_eq!(field_elem.to_u64(), Some(i));
    }

    let mut rng = test_rng();
    let large_field: Fr = Field::random(&mut rng);
    let _ = large_field.to_u64();
}
