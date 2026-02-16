use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;
use ark_std::test_rng;
#[cfg(feature = "challenge-254-bit")]
use jolt_field::challenge::Mont254BitChallenge;
use jolt_field::challenge::MontU128Challenge;
use jolt_field::{Field, OptimizedMul, WithChallenge};
use num_traits::{One, Zero};

#[test]
fn mont_u128_challenge_basic() {
    let mut rng = test_rng();

    let val = rng.gen::<u128>();
    let challenge = MontU128Challenge::<Fr>::from(val);

    // Top 3 bits are always zero
    assert!(challenge.high < (1u64 << 61));

    let field_elem: Fr = challenge.into();
    assert!(!field_elem.is_zero() || val == 0);
}

#[test]
fn mont_u128_challenge_bigint_layout() {
    // Verify the [0, 0, low, high] layout
    let challenge = MontU128Challenge::<Fr>::new(0x1234_5678_9ABC_DEF0_FEDC_BA98_7654_3210u128);
    let arr = challenge.to_bigint_array();
    assert_eq!(arr[0], 0);
    assert_eq!(arr[1], 0);
    assert_eq!(arr[2], challenge.low);
    assert_eq!(arr[3], challenge.high);
}

#[test]
fn mont_u128_challenge_arithmetic() {
    let mut rng = test_rng();

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

        // Challenge * Field -> Field (uses optimized mul_by_hi_2limbs)
        let prod2: Fr = a * c;
        assert_eq!(prod2, Into::<Fr>::into(a) * c);

        // Field * Challenge -> Field (uses optimized mul_by_hi_2limbs)
        let prod3: Fr = c * a;
        assert_eq!(prod3, c * Into::<Fr>::into(a));
    }
}

#[test]
fn mont_u128_challenge_serialization() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let challenge = MontU128Challenge::<Fr>::rand(&mut rng);

        let mut bytes = Vec::new();
        challenge.serialize_compressed(&mut bytes).unwrap();

        let recovered = MontU128Challenge::<Fr>::deserialize_compressed(&bytes[..]).unwrap();

        assert_eq!(challenge, recovered);
    }
}

#[test]
fn mont_u128_challenge_deserialization_masks_high_bits() {
    // Create a challenge with max 61-bit high limb
    let val = ((u64::MAX >> 3) as u128) << 64 | u64::MAX as u128;
    let challenge = MontU128Challenge::<Fr>::from(val);
    assert_eq!(challenge.high, u64::MAX >> 3);

    let mut bytes = Vec::new();
    challenge.serialize_compressed(&mut bytes).unwrap();

    let recovered = MontU128Challenge::<Fr>::deserialize_compressed(&bytes[..]).unwrap();
    assert_eq!(challenge.low, recovered.low);
    assert_eq!(challenge.high, recovered.high);
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

    // Test max 125-bit value
    let max_125_bit = (1u128 << 125) - 1;
    let max_challenge = MontU128Challenge::<Fr>::from(max_125_bit);
    assert!(max_challenge.high < (1u64 << 61));
}

#[test]
fn challenge_type_consistency() {
    type Challenge = <Fr as WithChallenge>::Challenge;

    #[cfg(not(feature = "challenge-254-bit"))]
    {
        let _challenge: Challenge = MontU128Challenge::<Fr>::from(42u128);
    }

    #[cfg(feature = "challenge-254-bit")]
    {
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
            let result = challenge.mul_0_optimized(Fr::zero());
            assert!(result.is_zero());
        }

        // Test mul_1_optimized
        {
            let one = Fr::one();
            let result = challenge.mul_1_optimized(one);
            assert_eq!(result, challenge.into());
        }

        // Test mul_01_optimized
        {
            let result = challenge.mul_01_optimized(Fr::zero());
            assert!(result.is_zero());

            let result2 = challenge.mul_01_optimized(Fr::one());
            assert_eq!(result2, challenge.into());
        }

        // Verify optimized mul gives same result as standard
        let optimized_result: Fr = challenge * field;
        let standard_result: Fr = Into::<Fr>::into(challenge) * field;
        assert_eq!(optimized_result, standard_result);
    }
}

#[cfg(feature = "challenge-254-bit")]
#[test]
fn mont_254bit_challenge_basic() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let challenge = Mont254BitChallenge::<Fr>::random(&mut rng);

        let field_elem: Fr = challenge.into();
        assert!(!field_elem.is_zero()); // full-range random should almost never be zero

        let a = Mont254BitChallenge::<Fr>::rand(&mut rng);
        let b = Mont254BitChallenge::<Fr>::rand(&mut rng);
        let c: Fr = Field::random(&mut rng);

        let sum: Fr = a + b;
        assert_eq!(sum, Into::<Fr>::into(a) + Into::<Fr>::into(b));

        let prod: Fr = a * b;
        assert_eq!(prod, Into::<Fr>::into(a) * Into::<Fr>::into(b));

        let mixed_sum: Fr = a + c;
        assert_eq!(mixed_sum, Into::<Fr>::into(a) + c);

        let mixed_prod: Fr = a * c;
        assert_eq!(mixed_prod, Into::<Fr>::into(a) * c);
    }
}
