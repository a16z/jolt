use ark_bn254::Fr;
use ark_ff::BigInt;
use ark_std::rand::Rng;
use ark_std::test_rng;
use ark_std::{One, Zero};
use jolt_field::{Field, ReductionOps, UnreducedOps};
use rand_chacha::rand_core::RngCore;

#[test]
fn unreduced_mul_montgomery_reduce() {
    let mut rng = test_rng();

    // mul_unreduced(a, b) produces aR * bR in the raw bigint.
    // Montgomery reduction: aR * bR / R = abR = Montgomery form of a*b.
    for _ in 0..100 {
        let a: Fr = Field::random(&mut rng);
        let b: Fr = Field::random(&mut rng);

        let expected = a * b;

        let unreduced: BigInt<8> = UnreducedOps::mul_unreduced(a, b);
        let reduced = <Fr as ReductionOps>::from_montgomery_reduce(unreduced);

        assert_eq!(expected, reduced);
    }
}

#[test]
fn mul_u64_unreduced_barrett_reduce() {
    let mut rng = test_rng();

    // mul_u64_unreduced(a, n) produces aR * n in the raw bigint.
    // Barrett reduction: aR * n mod p = the Montgomery form of a*n.
    for _ in 0..100 {
        let a: Fr = Field::random(&mut rng);
        let b = rng.next_u64();

        let expected = a * <Fr as Field>::from_u64(b);

        let unreduced = UnreducedOps::mul_u64_unreduced(a, b);
        let reduced = <Fr as ReductionOps>::from_barrett_reduce(unreduced);

        assert_eq!(expected, reduced);
    }
}

#[test]
fn mul_u128_unreduced_barrett_reduce() {
    let mut rng = test_rng();

    // mul_u128_unreduced(a, n) produces aR * n in the raw bigint.
    // Barrett reduction: aR * n mod p = the Montgomery form of a*n.
    for _ in 0..100 {
        let a: Fr = Field::random(&mut rng);
        let b = rng.gen::<u128>();

        let expected = a * <Fr as Field>::from_u128(b);

        let unreduced = UnreducedOps::mul_u128_unreduced(a, b);
        let reduced = <Fr as ReductionOps>::from_barrett_reduce(unreduced);

        assert_eq!(expected, reduced);
    }
}

#[test]
fn montgomery_reduction_identity() {
    let mut rng = test_rng();

    let one = Fr::one();
    let zero = Fr::zero();

    // 0 * x = 0
    for _ in 0..10 {
        let x: Fr = Field::random(&mut rng);
        let unreduced: BigInt<8> = UnreducedOps::mul_unreduced(zero, x);
        assert_eq!(
            <Fr as ReductionOps>::from_montgomery_reduce(unreduced),
            zero
        );
    }

    // 1 * x = x
    for _ in 0..10 {
        let x: Fr = Field::random(&mut rng);
        let unreduced: BigInt<8> = UnreducedOps::mul_unreduced(one, x);
        assert_eq!(<Fr as ReductionOps>::from_montgomery_reduce(unreduced), x);
    }
}

#[test]
fn montgomery_constants() {
    let _r = <Fr as ReductionOps>::MONTGOMERY_R;
    let _r2 = <Fr as ReductionOps>::MONTGOMERY_R_SQUARE;
}

#[test]
fn unreduced_accumulation() {
    let mut rng = test_rng();

    let n = 10;
    let a: Vec<Fr> = (0..n).map(|_| <Fr as Field>::random(&mut rng)).collect();
    let b: Vec<Fr> = (0..n).map(|_| <Fr as Field>::random(&mut rng)).collect();

    let expected: Fr = a.iter().zip(b.iter()).map(|(a, b)| *a * *b).sum();

    // Accumulate unreduced products and Montgomery reduce at the end
    let mut accumulator = BigInt::<8>::zero();
    for (a_elem, b_elem) in a.iter().zip(b.iter()) {
        let prod: BigInt<8> = UnreducedOps::mul_unreduced(*a_elem, *b_elem);
        accumulator += prod;
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

        // Verify as_unreduced_ref returns valid BigInt<4>
        let unreduced_ref = UnreducedOps::as_unreduced_ref(&a);
        let _ = unreduced_ref.0;

        let unreduced: BigInt<8> = UnreducedOps::mul_unreduced(a, b);
        let reduced = <Fr as ReductionOps>::from_montgomery_reduce(unreduced);

        assert_eq!(reduced, a * b);
    }
}

#[test]
fn reduction_with_bigint_9() {
    let mut rng = test_rng();

    for _ in 0..100 {
        let a: Fr = Field::random(&mut rng);
        let b: Fr = Field::random(&mut rng);

        let expected = a * b;

        // BigInt<9> has extra headroom for accumulated products
        let unreduced_9: BigInt<9> = UnreducedOps::mul_unreduced(a, b);
        let reduced_9 = <Fr as ReductionOps>::from_montgomery_reduce(unreduced_9);
        assert_eq!(reduced_9, expected);
    }
}
