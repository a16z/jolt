use super::super::butterfly::{
    forward_ntt as scalar_forward_ntt, forward_ntt_cyclic as scalar_forward_ntt_cyclic,
    inverse_ntt as scalar_inverse_ntt, inverse_ntt_cyclic as scalar_inverse_ntt_cyclic,
    NttTwiddles,
};
use super::super::prime::{MontCoeff, NttPrime, I32_LAZY_DOT_BATCH};
use super::super::tables::Q128_RAW_PRIMES;
use super::super::NttKernelPlan;
use super::*;

fn random_mont_array_i32<const D: usize>(prime: NttPrime<i32>, seed: u64) -> [MontCoeff<i32>; D] {
    let mut state = seed;
    std::array::from_fn(|_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((state >> 33) as i64 % prime.p as i64) as i32;
        prime.from_canonical(val)
    })
}

fn random_raw_mont_array_i32<const D: usize>(
    prime: NttPrime<i32>,
    seed: u64,
) -> [MontCoeff<i32>; D] {
    let mut state = seed;
    let width = i64::from(prime.p) * 2 - 1;
    std::array::from_fn(|_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let raw = (state as i64).rem_euclid(width) - (i64::from(prime.p) - 1);
        MontCoeff::from_raw(raw as i32)
    })
}

fn random_mont_array_i16<const D: usize>(prime: NttPrime<i16>, seed: u64) -> [MontCoeff<i16>; D] {
    let mut state = seed;
    std::array::from_fn(|_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((state >> 33) as i64 % prime.p as i64) as i16;
        prime.from_canonical(val)
    })
}

const TEST_PRIME_I32: i32 = 1073692673;
const TEST_PRIME_I16: i16 = crate::ntt::tables::I16_TAIL_PRIME.p;

fn assert_neon_ntt_i32_matches_scalar<const D: usize>() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    let tw = NttTwiddles::<i32, D>::compute(prime);
    let input = random_mont_array_i32::<D>(prime, 0xCAFE);

    let mut neon_result = input;
    unsafe { forward_ntt_i32(&mut neon_result, prime, &tw) };

    let mut scalar_result = input;
    scalar_forward_ntt(&mut scalar_result, prime, &tw, NttKernelPlan::SCALAR);

    for i in 0..D {
        let n = prime.to_canonical(neon_result[i]);
        let s = prime.to_canonical(scalar_result[i]);
        assert_eq!(n, s, "D={D} forward mismatch at {i}: neon={n}, scalar={s}");
    }

    let mut neon_result = input;
    unsafe { inverse_ntt_i32(&mut neon_result, prime, &tw) };

    let mut scalar_result = input;
    scalar_inverse_ntt(&mut scalar_result, prime, &tw, NttKernelPlan::SCALAR);

    for i in 0..D {
        let n = prime.to_canonical(neon_result[i]);
        let s = prime.to_canonical(scalar_result[i]);
        assert_eq!(n, s, "D={D} inverse mismatch at {i}: neon={n}, scalar={s}");
    }
}

#[test]
fn neon_ntt_i32_matches_scalar_at_target_dimensions() {
    assert_neon_ntt_i32_matches_scalar::<64>();
    assert_neon_ntt_i32_matches_scalar::<128>();
    assert_neon_ntt_i32_matches_scalar::<256>();
    assert_neon_ntt_i32_matches_scalar::<512>();
    assert_neon_ntt_i32_matches_scalar::<1024>();
    assert_neon_ntt_i32_matches_scalar::<2048>();
}

#[test]
fn neon_forward_inverse_roundtrip_i32() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    let tw = NttTwiddles::<i32, 512>::compute(prime);
    let input = random_mont_array_i32::<512>(prime, 0xDEAD);
    let canonical_input: Vec<i32> = input.iter().map(|c| prime.to_canonical(*c)).collect();

    let mut a = input;
    unsafe {
        forward_ntt_i32(&mut a, prime, &tw);
        inverse_ntt_i32(&mut a, prime, &tw);
    }

    for i in 0..512 {
        let result = prime.to_canonical(a[i]);
        assert_eq!(
            result, canonical_input[i],
            "roundtrip mismatch at index {i}"
        );
    }
}

fn assert_neon_i8_forward_matches_scalar<const D: usize>() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    let tw = NttTwiddles::<i32, D>::compute(prime);
    let digits = std::array::from_fn(|i| (i as i8).wrapping_mul(37).wrapping_add(11));

    let mut neon = [MontCoeff::from_raw(0); D];
    unsafe { forward_ntt_i8_i32(&mut neon, &digits, prime, &tw) };

    let mut scalar = std::array::from_fn(|i| prime.from_canonical(i32::from(digits[i])));
    scalar_forward_ntt(&mut scalar, prime, &tw, NttKernelPlan::SCALAR);

    for i in 0..D {
        assert_eq!(
            prime.to_canonical(neon[i]),
            prime.to_canonical(scalar[i]),
            "D={D} fused i8 forward mismatch at {i}"
        );
    }
}

#[test]
fn neon_i8_forward_ntt_matches_scalar() {
    assert_neon_i8_forward_matches_scalar::<8>();
    assert_neon_i8_forward_matches_scalar::<64>();
    assert_neon_i8_forward_matches_scalar::<512>();
    assert_neon_i8_forward_matches_scalar::<1024>();
}

#[test]
fn neon_cyclic_ntt_i32_matches_scalar() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    let tw = NttTwiddles::<i32, 512>::compute(prime);
    let input = random_mont_array_i32::<512>(prime, 0xFACE);

    let mut neon_fwd = input;
    unsafe { forward_ntt_cyclic_i32(&mut neon_fwd, prime, &tw) };

    let mut scalar_fwd = input;
    scalar_forward_ntt_cyclic(&mut scalar_fwd, prime, &tw, NttKernelPlan::SCALAR);

    for i in 0..512 {
        let n = prime.to_canonical(neon_fwd[i]);
        let s = prime.to_canonical(scalar_fwd[i]);
        assert_eq!(n, s, "forward cyclic mismatch at {i}: neon={n}, scalar={s}");
    }

    let mut neon_inv = neon_fwd;
    unsafe { inverse_ntt_cyclic_i32(&mut neon_inv, prime, &tw) };

    let mut scalar_inv = scalar_fwd;
    scalar_inverse_ntt_cyclic(&mut scalar_inv, prime, &tw, NttKernelPlan::SCALAR);

    for i in 0..512 {
        let n = prime.to_canonical(neon_inv[i]);
        let s = prime.to_canonical(scalar_inv[i]);
        assert_eq!(n, s, "inverse cyclic mismatch at {i}: neon={n}, scalar={s}");
    }
}

#[test]
fn neon_pointwise_mul_acc_i32_matches_scalar() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    const D: usize = 512;
    let acc_init = random_mont_array_i32::<D>(prime, 0x1111);
    let lhs = random_mont_array_i32::<D>(prime, 0x2222);
    let rhs = random_mont_array_i32::<D>(prime, 0x3333);

    let mut neon_acc = acc_init;
    unsafe {
        pointwise_mul_acc_i32(
            neon_acc.as_mut_ptr() as *mut i32,
            lhs.as_ptr() as *const i32,
            rhs.as_ptr() as *const i32,
            D,
            prime.p,
            prime.pinv,
        );
    }

    let mut scalar_acc = acc_init;
    for i in 0..D {
        let prod = prime.mul(lhs[i], rhs[i]);
        let sum = MontCoeff::from_raw(scalar_acc[i].raw().wrapping_add(prod.raw()));
        scalar_acc[i] = prime.reduce_range(sum);
    }

    for i in 0..D {
        let n = prime.to_canonical(neon_acc[i]);
        let s = prime.to_canonical(scalar_acc[i]);
        assert_eq!(n, s, "pointwise mul acc mismatch at {i}");
    }
}

#[test]
fn neon_pointwise_mul_acc_i32_handles_scalar_tail() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    const D: usize = 6;
    let acc_init = random_mont_array_i32::<D>(prime, 0x4444);
    let lhs = random_mont_array_i32::<D>(prime, 0x5555);
    let rhs = random_mont_array_i32::<D>(prime, 0x6666);

    let mut neon_acc = acc_init;
    unsafe {
        pointwise_mul_acc_i32(
            neon_acc.as_mut_ptr() as *mut i32,
            lhs.as_ptr() as *const i32,
            rhs.as_ptr() as *const i32,
            D,
            prime.p,
            prime.pinv,
        );
    }

    let mut scalar_acc = acc_init;
    for i in 0..D {
        let prod = prime.mul(lhs[i], rhs[i]);
        let sum = MontCoeff::from_raw(scalar_acc[i].raw().wrapping_add(prod.raw()));
        scalar_acc[i] = prime.reduce_range(sum);
    }

    assert_eq!(neon_acc, scalar_acc);
}

#[test]
fn neon_pointwise_dot_acc_i32_matches_scalar() {
    const D: usize = 19;
    for raw_prime in Q128_RAW_PRIMES {
        let prime = NttPrime::compute(raw_prime);
        let edge_values = [
            0,
            1,
            -1,
            prime.p - 1,
            1 - prime.p,
            prime.p / 2,
            -(prime.p / 2),
            0x4000_1234_i32,
            -0x3fff_4321_i32,
        ];
        let lhs: [[MontCoeff<i32>; D]; I32_LAZY_DOT_BATCH] = std::array::from_fn(|index| {
            if index == 0 {
                std::array::from_fn(|i| MontCoeff::from_raw(edge_values[i % edge_values.len()]))
            } else {
                random_raw_mont_array_i32(prime, 0x1000 + index as u64)
            }
        });
        let rhs: [[MontCoeff<i32>; D]; I32_LAZY_DOT_BATCH] = std::array::from_fn(|index| {
            if index + 1 == I32_LAZY_DOT_BATCH {
                std::array::from_fn(|i| MontCoeff::from_raw(edge_values[i % edge_values.len()]))
            } else {
                random_raw_mont_array_i32(prime, 0x2000 + index as u64)
            }
        });
        let lhs_pointers: [*const i32; I32_LAZY_DOT_BATCH] =
            std::array::from_fn(|index| lhs[index].as_ptr().cast::<i32>());
        let rhs_pointers: [*const i32; I32_LAZY_DOT_BATCH] =
            std::array::from_fn(|index| rhs[index].as_ptr().cast::<i32>());

        for count in 1..=I32_LAZY_DOT_BATCH {
            let initial = random_mont_array_i32::<D>(prime, 0x3000 + count as u64);
            let mut actual = initial;
            unsafe {
                pointwise_dot_acc_i32(
                    actual.as_mut_ptr().cast::<i32>(),
                    lhs_pointers.as_ptr(),
                    rhs_pointers.as_ptr(),
                    count,
                    D,
                    prime.p,
                    prime.pinv,
                );
            }
            let mut expected = initial;
            for product in 0..count {
                for i in 0..D {
                    let value = prime.mul(lhs[product][i], rhs[product][i]);
                    let sum = MontCoeff::from_raw(expected[i].raw().wrapping_add(value.raw()));
                    expected[i] = prime.reduce_range(sum);
                }
            }
            assert_eq!(actual, expected, "prime={raw_prime}, count={count}");
        }
    }
}

#[test]
fn neon_centered_i8_to_mont_i32_matches_scalar() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    let mut coefficients: Vec<i8> = (i8::MIN..=i8::MAX).collect();
    coefficients.extend_from_slice(&[i8::MIN, 0, i8::MAX]);
    let mut actual = vec![0i32; coefficients.len()];
    // SAFETY: both buffers contain 259 non-overlapping elements.
    unsafe {
        centered_i8_to_mont_i32(
            actual.as_mut_ptr(),
            coefficients.as_ptr(),
            coefficients.len(),
            prime.p,
            prime.pinv,
            prime.montsq,
        );
    }
    let expected: Vec<i32> = coefficients
        .iter()
        .map(|&coefficient| prime.from_canonical(i32::from(coefficient)).raw())
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn neon_centered_i16_to_mont_i16_matches_scalar() {
    let prime = NttPrime::compute(TEST_PRIME_I16);
    let coefficients = [
        i16::MIN,
        -30_000,
        -13_698,
        -1,
        0,
        1,
        13_696,
        30_000,
        i16::MAX,
        -17,
        42,
        -2048,
        4096,
    ];
    let mut actual = [0i16; 13];
    unsafe {
        centered_i16_to_mont_i16(
            actual.as_mut_ptr(),
            coefficients.as_ptr(),
            coefficients.len(),
            prime.p,
            prime.pinv,
            prime.montsq,
        );
    }

    for (index, (&coefficient, &converted)) in coefficients.iter().zip(&actual).enumerate() {
        let expected = prime.from_canonical(coefficient);
        assert_eq!(
            prime.to_canonical(MontCoeff::from_raw(converted)),
            prime.to_canonical(expected),
            "Montgomery conversion mismatch at {index}",
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn neon_add_reduce_i32_handles_scalar_tail() {
    let prime = NttPrime::compute(TEST_PRIME_I32);
    const D: usize = 6;
    let acc_init = random_mont_array_i32::<D>(prime, 0x7777);
    let other = random_mont_array_i32::<D>(prime, 0x8888);

    let mut neon_acc = acc_init;
    unsafe {
        add_reduce_i32(
            neon_acc.as_mut_ptr() as *mut i32,
            other.as_ptr() as *const i32,
            D,
            prime.p,
        );
    }

    let mut scalar_acc = acc_init;
    for i in 0..D {
        let sum = MontCoeff::from_raw(scalar_acc[i].raw().wrapping_add(other[i].raw()));
        scalar_acc[i] = prime.reduce_range(sum);
    }

    assert_eq!(neon_acc, scalar_acc);
}

fn assert_neon_ntt_i16_matches_scalar<const D: usize>() {
    let prime = NttPrime::compute(TEST_PRIME_I16);
    let tw = NttTwiddles::<i16, D>::compute(prime);
    let input = random_mont_array_i16::<D>(prime, 0xABCD);

    let mut neon_result = input;
    unsafe { forward_ntt_i16(&mut neon_result, prime, &tw) };

    let mut scalar_result = input;
    scalar_forward_ntt(&mut scalar_result, prime, &tw, NttKernelPlan::SCALAR);

    for i in 0..D {
        let n = prime.to_canonical(neon_result[i]);
        let s = prime.to_canonical(scalar_result[i]);
        assert_eq!(
            n, s,
            "D={D} i16 forward mismatch at {i}: neon={n}, scalar={s}"
        );
    }

    let mut neon_result = input;
    unsafe { inverse_ntt_i16(&mut neon_result, prime, &tw) };

    let mut scalar_result = input;
    scalar_inverse_ntt(&mut scalar_result, prime, &tw, NttKernelPlan::SCALAR);

    for i in 0..D {
        let n = prime.to_canonical(neon_result[i]);
        let s = prime.to_canonical(scalar_result[i]);
        assert_eq!(
            n, s,
            "D={D} i16 inverse mismatch at {i}: neon={n}, scalar={s}"
        );
    }
}

#[test]
fn neon_ntt_i16_matches_scalar_at_fallback_and_target_dimensions() {
    assert_neon_ntt_i16_matches_scalar::<16>();
    assert_neon_ntt_i16_matches_scalar::<64>();
    assert_neon_ntt_i16_matches_scalar::<128>();
    assert_neon_ntt_i16_matches_scalar::<256>();
    assert_neon_ntt_i16_matches_scalar::<512>();
    assert_neon_ntt_i16_matches_scalar::<1024>();
    assert_neon_ntt_i16_matches_scalar::<2048>();
}

#[test]
fn neon_forward_inverse_roundtrip_i16() {
    let prime = NttPrime::compute(TEST_PRIME_I16);
    let tw = NttTwiddles::<i16, 64>::compute(prime);
    let input = random_mont_array_i16::<64>(prime, 0x7777);
    let canonical_input: Vec<i16> = input.iter().map(|c| prime.to_canonical(*c)).collect();

    let mut a = input;
    unsafe {
        forward_ntt_i16(&mut a, prime, &tw);
        inverse_ntt_i16(&mut a, prime, &tw);
    }

    for i in 0..64 {
        let result = prime.to_canonical(a[i]);
        assert_eq!(result, canonical_input[i], "i16 roundtrip mismatch at {i}");
    }
}

#[test]
fn neon_cyclic_i16_matches_scalar() {
    let prime = NttPrime::compute(TEST_PRIME_I16);
    let tw = NttTwiddles::<i16, 64>::compute(prime);
    let input = random_mont_array_i16::<64>(prime, 0x9999);

    let mut neon_fwd = input;
    unsafe { forward_ntt_cyclic_i16(&mut neon_fwd, prime, &tw) };

    let mut scalar_fwd = input;
    scalar_forward_ntt_cyclic(&mut scalar_fwd, prime, &tw, NttKernelPlan::SCALAR);

    for i in 0..64 {
        let n = prime.to_canonical(neon_fwd[i]);
        let s = prime.to_canonical(scalar_fwd[i]);
        assert_eq!(n, s, "i16 fwd cyclic mismatch at {i}");
    }

    let mut neon_inv = neon_fwd;
    unsafe { inverse_ntt_cyclic_i16(&mut neon_inv, prime, &tw) };

    let mut scalar_inv = scalar_fwd;
    scalar_inverse_ntt_cyclic(&mut scalar_inv, prime, &tw, NttKernelPlan::SCALAR);

    for i in 0..64 {
        let n = prime.to_canonical(neon_inv[i]);
        let s = prime.to_canonical(scalar_inv[i]);
        assert_eq!(n, s, "i16 inv cyclic mismatch at {i}");
    }
}

#[test]
fn neon_pointwise_mul_acc_i16_matches_scalar() {
    let prime = NttPrime::compute(TEST_PRIME_I16);
    const D: usize = 64;
    let acc_init = random_mont_array_i16::<D>(prime, 0xAAAA);
    let lhs = random_mont_array_i16::<D>(prime, 0xBBBB);
    let rhs = random_mont_array_i16::<D>(prime, 0xCCCC);

    let mut neon_acc = acc_init;
    unsafe {
        pointwise_mul_acc_i16(
            neon_acc.as_mut_ptr() as *mut i16,
            lhs.as_ptr() as *const i16,
            rhs.as_ptr() as *const i16,
            D,
            prime.p,
            prime.pinv,
        );
    }

    let mut scalar_acc = acc_init;
    for i in 0..D {
        let prod = prime.mul(lhs[i], rhs[i]);
        let sum = MontCoeff::from_raw(scalar_acc[i].raw().wrapping_add(prod.raw()));
        scalar_acc[i] = prime.reduce_range(sum);
    }

    for i in 0..D {
        let n = prime.to_canonical(neon_acc[i]);
        let s = prime.to_canonical(scalar_acc[i]);
        assert_eq!(n, s, "i16 pointwise mul acc mismatch at {i}");
    }
}
