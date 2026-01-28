use jolt_field::challenge::MontU128Challenge;
#[cfg(feature = "challenge-254-bit")]
use jolt_field::challenge::Mont254BitChallenge;
use jolt_field::{Field, WithChallenge, OptimizedMul};
use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::test_rng;
use ark_std::rand::Rng;
use num_traits::{Zero, One};

#[test]
fn mont_u128_challenge_basic() {
    let mut rng = test_rng();

    // Test from u128
    let val = rng.gen::<u128>();
    let challenge = MontU128Challenge::<Fr>::from(val);

    // Test that top 3 bits are always zero
    assert!(challenge.high < (1u64 << 61));

    // Test conversion to field element
    let field_elem: Fr = challenge.into();
    assert!(!field_elem.is_zero() || val == 0);
}

#[test]
fn mont_u128_challenge_arithmetic() {
    let mut rng = test_rng();

    // Test challenge + challenge operations
    for _ in 0..100 {
        let a = MontU128Challenge::<Fr>::rand(&mut rng);
        let b = MontU128Challenge::<Fr>::rand(&mut rng);
        let c: Fr = Field::random(&mut rng);

        // Challenge + Challenge -> Field
        let sum: Fr = a + b;
        assert_eq!(sum, Into::<Fr>::into(a) + Into::<Fr>::into(b));

        // Challenge - Challenge -> Field
        let diff: Fr = a - b;
        assert_eq!(diff, Into::<Fr>::into(a) - Into::<Fr>::into(b));

        // Challenge * Challenge -> Field
        let prod: Fr = a * b;
        assert_eq!(prod, Into::<Fr>::into(a) * Into::<Fr>::into(b));

        // Challenge + Field -> Field
        let sum2: Fr = a + c;
        assert_eq!(sum2, Into::<Fr>::into(a) + c);

        // Field + Challenge -> Field
        let sum3: Fr = c + a;
        assert_eq!(sum3, c + Into::<Fr>::into(a));

        // Challenge * Field -> Field
        let prod2: Fr = a * c;
        assert_eq!(prod2, Into::<Fr>::into(a) * c);

        // Field * Challenge -> Field
        let prod3: Fr = c * a;
        assert_eq!(prod3, c * Into::<Fr>::into(a));
    }
}

#[test]
fn mont_u128_challenge_serialization() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let challenge = MontU128Challenge::<Fr>::rand(&mut rng);

        // Serialize
        let mut bytes = Vec::new();
        challenge.serialize_compressed(&mut bytes).unwrap();

        // Deserialize
        let recovered = MontU128Challenge::<Fr>::deserialize_compressed(&bytes[..]).unwrap();

        assert_eq!(challenge, recovered);
    }
}

#[test]
fn mont_u128_challenge_special_values() {
    // Test zero
    let zero_challenge = MontU128Challenge::<Fr>::from(0u128);
    assert_eq!(zero_challenge.low, 0);
    assert_eq!(zero_challenge.high, 0);
    assert!(Into::<Fr>::into(zero_challenge).is_zero());

    // Test one
    let one_challenge = MontU128Challenge::<Fr>::from(1u128);
    assert_eq!(one_challenge.low, 1);
    assert_eq!(one_challenge.high, 0);
    assert_eq!(Into::<Fr>::into(one_challenge), Fr::one());

    // Test max 125-bit value
    let max_125_bit = (1u128 << 125) - 1;
    let max_challenge = MontU128Challenge::<Fr>::from(max_125_bit);
    assert!(max_challenge.high < (1u64 << 61));
}

#[test]
fn challenge_type_consistency() {
    // Verify that the challenge type is correctly associated
    type Challenge = <Fr as WithChallenge>::Challenge;

    #[cfg(not(feature = "challenge-254-bit"))]
    {
        // Default should be MontU128Challenge
        let _challenge: Challenge = MontU128Challenge::<Fr>::from(42u128);
    }

    #[cfg(feature = "challenge-254-bit")]
    {
        // With feature flag should be Mont254BitChallenge
        let _challenge: Challenge = Mont254BitChallenge::<Fr>::from(42u128);
    }
}

#[test]
fn optimized_mul_trait() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let challenge = MontU128Challenge::<Fr>::rand(&mut rng);
        let field: Fr = Field::random(&mut rng);

        // Test mul_0_optimized
        {
            let zero_challenge = MontU128Challenge::<Fr>::from(0u128);
            let result = challenge.mul_0_optimized(Fr::zero());
            assert!(result.is_zero());

            let result2 = zero_challenge.mul_0_optimized(field);
            assert!(result2.is_zero());
        }

        // Test mul_1_optimized
        {
            let one = Fr::one();
            let result = challenge.mul_1_optimized(one);
            assert_eq!(result, challenge.into());

            let one_challenge = MontU128Challenge::<Fr>::from(1u128);
            let result2 = one_challenge.mul_1_optimized(field);
            assert_eq!(result2, field);
        }

        // Test mul_01_optimized
        {
            let result = challenge.mul_01_optimized(Fr::zero());
            assert!(result.is_zero());

            let result2 = challenge.mul_01_optimized(Fr::one());
            assert_eq!(result2, challenge.into());
        }
    }
}

#[cfg(feature = "challenge-254-bit")]
#[test]
fn mont_254bit_challenge_basic() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let val = rng.gen::<u128>();
        let challenge = Mont254BitChallenge::<Fr>::from(val);

        // Test conversion to field
        let field_elem: Fr = challenge.into();
        assert!(!field_elem.is_zero() || val == 0);

        // Test arithmetic
        let a = Mont254BitChallenge::<Fr>::rand(&mut rng);
        let b = Mont254BitChallenge::<Fr>::rand(&mut rng);
        let c: Fr = Field::random(&mut rng);

        // Challenge arithmetic
        let sum: Fr = a + b;
        assert_eq!(sum, Into::<Fr>::into(a) + Into::<Fr>::into(b));

        let prod: Fr = a * b;
        assert_eq!(prod, Into::<Fr>::into(a) * Into::<Fr>::into(b));

        // Mixed arithmetic
        let mixed_sum: Fr = a + c;
        assert_eq!(mixed_sum, Into::<Fr>::into(a) + c);

        let mixed_prod: Fr = a * c;
        assert_eq!(mixed_prod, Into::<Fr>::into(a) * c);
    }
}