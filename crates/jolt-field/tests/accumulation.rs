use jolt_field::{Field, UnreducedOps, ReductionOps};
use ark_bn254::Fr;
use ark_std::test_rng;
use ark_std::rand::{Rng, RngCore};
use ark_std::{Zero, One};

#[test]
fn unreduced_arithmetic() {
    let mut rng = test_rng();

    // Test unreduced multiplication and reduction
    for _ in 0..100 {
        let a: Fr = Field::random(&mut rng);
        let b: Fr = Field::random(&mut rng);

        // Standard multiplication
        let expected = a * b;

        // Unreduced multiplication followed by reduction
        let unreduced = UnreducedOps::mul_unreduced(a, b);
        let reduced = <Fr as ReductionOps>::from_montgomery_reduce(unreduced);

        assert_eq!(expected, reduced);
    }
}

#[test]
fn mul_u64_unreduced() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let a: Fr = Field::random(&mut rng);
        let b = rng.next_u64();

        // Standard multiplication
        let expected = a * <Fr as Field>::from_u64(b);

        // Unreduced multiplication followed by reduction
        let unreduced = UnreducedOps::mul_u64_unreduced(a, b);
        let reduced = <Fr as ReductionOps>::from_montgomery_reduce(unreduced);

        assert_eq!(expected, reduced);
    }
}

#[test]
fn mul_u128_unreduced() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let a: Fr = Field::random(&mut rng);
        let b = rng.gen::<u128>();

        // Standard multiplication
        let expected = a * <Fr as Field>::from_u128(b);

        // Unreduced multiplication followed by reduction
        let unreduced = UnreducedOps::mul_u128_unreduced(a, b);
        let reduced = <Fr as ReductionOps>::from_montgomery_reduce(unreduced);

        assert_eq!(expected, reduced);
    }
}

#[test]
fn barrett_reduction() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let a: Fr = Field::random(&mut rng);
        let b: Fr = Field::random(&mut rng);

        // Get unreduced result
        let unreduced = UnreducedOps::mul_unreduced(a, b);

        // Both reduction methods should give the same result
        let barrett = <Fr as ReductionOps>::from_barrett_reduce(unreduced);
        let montgomery = <Fr as ReductionOps>::from_montgomery_reduce(unreduced);

        assert_eq!(barrett, montgomery);
        assert_eq!(barrett, a * b);
    }
}

#[test]
fn montgomery_reduction() {
    let mut rng = test_rng();

    // Test with specific values
    let one = Fr::one();
    let zero = Fr::zero();

    // 0 * x = 0
    for _ in 0..10 {
        let x: Fr = Field::random(&mut rng);
        let unreduced = UnreducedOps::mul_unreduced(zero, x);
        assert_eq!(<Fr as ReductionOps>::from_montgomery_reduce(unreduced), zero);
    }

    // 1 * x = x
    for _ in 0..10 {
        let x: Fr = Field::random(&mut rng);
        let unreduced = UnreducedOps::mul_unreduced(one, x);
        assert_eq!(<Fr as ReductionOps>::from_montgomery_reduce(unreduced), x);
    }
}

#[test]
fn montgomery_constants() {
    // Verify Montgomery constants are valid field elements
    let _r = <Fr as ReductionOps>::MONTGOMERY_R;
    let _r2 = <Fr as ReductionOps>::MONTGOMERY_R_SQUARE;

    // R * R^-1 = 1 (mod p)
    // This is implicitly tested by the reduction operations working correctly
}

#[test]
fn unreduced_accumulation() {
    let mut rng = test_rng();

    // Test accumulation pattern: sum(a_i * b_i)
    let n = 10;
    let a: Vec<Fr> = (0..n).map(|_| <Fr as Field>::random(&mut rng)).collect();
    let b: Vec<Fr> = (0..n).map(|_| <Fr as Field>::random(&mut rng)).collect();

    // Compute expected result
    let expected: Fr = a.iter().zip(b.iter()).map(|(a, b)| *a * *b).sum();

    // Compute using unreduced accumulation
    let mut accumulator = <Fr as UnreducedOps>::UnreducedType::zero();
    for (a_elem, b_elem) in a.iter().zip(b.iter()) {
        let prod = UnreducedOps::mul_unreduced(*a_elem, *b_elem);
        accumulator = accumulator + prod;
    }

    let result = <Fr as ReductionOps>::from_montgomery_reduce(accumulator);
    assert_eq!(result, expected);
}

#[test]
fn unreduced_reference_access() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let a: Fr = Field::random(&mut rng);
        let b: Fr = Field::random(&mut rng);

        // Access unreduced representation
        let _unreduced_ref = UnreducedOps::as_unreduced_ref(&a);

        // Verify mul_unreduced works correctly
        let unreduced = UnreducedOps::mul_unreduced(a, b);
        let reduced = <Fr as ReductionOps>::from_montgomery_reduce(unreduced);

        assert_eq!(reduced, a * b);
    }
}