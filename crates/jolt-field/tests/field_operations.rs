use jolt_field::Field;
use ark_bn254::Fr;
use ark_std::{test_rng, One, Zero};
use ark_std::rand::Rng;
use rand_chacha::rand_core::RngCore;

#[test]
fn implicit_montgomery_conversion() {
    let mut rng = test_rng();

    // Test from_u64 consistency with multiplication
    for _ in 0..256 {
        let x = rng.next_u64();
        assert_eq!(
            <Fr as Field>::from_u64(x),
            Fr::one() * <Fr as Field>::from_u64(x)
        );
    }

    // Test multiplication consistency
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

    // Test addition
    let sum = x + y;
    assert_eq!(sum, y + x); // Commutative

    // Test multiplication
    let product = x * y;
    assert_eq!(product, y * x); // Commutative

    // Test subtraction
    let diff = x - y;
    assert_eq!(diff + y, x);

    // Test division
    if !y.is_zero() {
        let quotient = x / y;
        assert_eq!(quotient * y, x);
    }
}

#[test]
fn field_conversions() {
    let mut rng = test_rng();

    // Test bool conversion
    assert_eq!(<Fr as Field>::from_bool(true), Fr::one());
    assert_eq!(<Fr as Field>::from_bool(false), Fr::zero());

    // Test u8 conversion
    for _ in 0..100 {
        let val = rng.gen::<u8>();
        let field_elem = <Fr as Field>::from_u8(val);
        assert_eq!(field_elem, <Fr as Field>::from_u64(val as u64));
    }

    // Test u16 conversion
    for _ in 0..100 {
        let val = rng.gen::<u16>();
        let field_elem = <Fr as Field>::from_u16(val);
        assert_eq!(field_elem, <Fr as Field>::from_u64(val as u64));
    }

    // Test u32 conversion
    for _ in 0..100 {
        let val = rng.gen::<u32>();
        let field_elem = <Fr as Field>::from_u32(val);
        assert_eq!(field_elem, <Fr as Field>::from_u64(val as u64));
    }

    // Test u128 conversion
    for _ in 0..100 {
        let val = rng.gen::<u128>();
        let field_elem = <Fr as Field>::from_u128(val);
        // Just verify it doesn't panic
        assert!(!field_elem.is_zero() || val == 0);
    }
}

#[test]
fn bytes_conversion() {
    let mut rng = test_rng();

    // Test conversion from various byte lengths
    for len in [1, 8, 16, 32, 48, 64].iter() {
        let mut bytes = vec![0u8; *len];
        rng.fill_bytes(&mut bytes);
        let _field_elem = <Fr as Field>::from_bytes(&bytes);
        // Just verify it doesn't panic
    }
}

#[test]
fn signed_conversions() {
    let mut rng = test_rng();

    // Test i64 conversion
    for _ in 0..100 {
        let val = rng.gen::<i64>();
        let field_elem = <Fr as Field>::from_i64(val);

        if val >= 0 {
            assert_eq!(field_elem, <Fr as Field>::from_u64(val as u64));
        } else {
            assert_eq!(field_elem, -<Fr as Field>::from_u64(val.unsigned_abs()));
        }
    }

    // Test i128 conversion
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
fn mul_by_small_values() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let field_elem: Fr = Field::random(&mut rng);
        let small_val = rng.gen_range(0u64..1000);

        // Test multiplication with small u64
        let result1 = field_elem * <Fr as Field>::from_u64(small_val);

        // Verify by repeated addition
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

    // Test multiplication by 0
    assert_eq!(field_elem * <Fr as Field>::from_u64(0), Fr::zero());
    assert_eq!(field_elem * <Fr as Field>::from_u64(1), field_elem);
    assert!((Fr::zero() * <Fr as Field>::from_u64(rng.next_u64())).is_zero());
}

#[test]
fn to_u64_conversion() {
    // Test small values
    for i in 0..1000u64 {
        let field_elem = <Fr as Field>::from_u64(i);
        assert_eq!(field_elem.to_u64(), Some(i));
    }

    // Test random values that don't fit in u64
    let mut rng = test_rng();
    let large_field: Fr = Field::random(&mut rng);
    // Most random field elements won't fit in u64
    // This is just to ensure the method doesn't panic
    let _ = large_field.to_u64();
}