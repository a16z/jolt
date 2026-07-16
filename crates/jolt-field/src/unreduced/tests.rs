#![expect(
    clippy::unreadable_literal,
    reason = "regression tests retain copied modulus constants"
)]

use super::*;
use crate::RandomSampling;
use crate::{Prime128Offset275, Prime24Offset3, Prime40Offset195};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_core::RngCore;

type F128 = Prime128Offset275;
type F32 = Prime24Offset3;
type F64 = Prime40Offset195;

const P128: u128 = 0xfffffffffffffffffffffffffffffeed;
const P32: u32 = (1 << 24) - 3;
const P64: u64 = (1 << 40) - 195;

#[test]
fn fp128_roundtrip() {
    let mut rng = StdRng::seed_from_u64(0xdead_1234);
    for _ in 0..1000 {
        let a: F128 = RandomSampling::random(&mut rng);
        let wide = Fp128x8i32::from(a);
        let back = wide.reduce::<P128>();
        assert_eq!(a, back, "roundtrip failed for {a:?}");
    }
}

#[test]
fn fp128_accumulate_matches_scalar() {
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe_4321);
    let n = 1000;
    let vals: Vec<F128> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

    let scalar_sum = vals.iter().fold(F128::zero(), |acc, &x| acc + x);

    let wide_sum = vals
        .iter()
        .fold(Fp128x8i32::zero(), |acc, &x| acc + Fp128x8i32::from(x));
    let reduced = wide_sum.reduce::<P128>();

    assert_eq!(scalar_sum, reduced);
}

#[test]
fn fp128_add_sub_neg_match_scalar() {
    let mut rng = StdRng::seed_from_u64(0x1122_3344_5566);
    for _ in 0..500 {
        let a: F128 = RandomSampling::random(&mut rng);
        let b: F128 = RandomSampling::random(&mut rng);

        let wa = Fp128x8i32::from(a);
        let wb = Fp128x8i32::from(b);

        assert_eq!((wa + wb).reduce::<P128>(), a + b);
        assert_eq!((wa - wb).reduce::<P128>(), a - b);
        assert_eq!((-wa).reduce::<P128>(), -a);
    }
}

#[test]
fn fp128_mixed_add_sub_stress() {
    let mut rng = StdRng::seed_from_u64(0xaaaa_bbbb_cccc);
    let n = 500;
    let vals: Vec<F128> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

    let mut scalar = F128::zero();
    let mut wide = Fp128x8i32::zero();
    for (i, &v) in vals.iter().enumerate() {
        let wv = Fp128x8i32::from(v);
        if i % 3 == 0 {
            scalar -= v;
            wide -= wv;
        } else {
            scalar += v;
            wide += wv;
        }
    }
    assert_eq!(wide.reduce::<P128>(), scalar);
}

#[test]
fn fp32_roundtrip() {
    let mut rng = StdRng::seed_from_u64(0x3232_3232);
    for _ in 0..1000 {
        let a: F32 = RandomSampling::random(&mut rng);
        let wide = Fp32x2i32::from(a);
        let back = wide.reduce::<P32>();
        assert_eq!(a, back);
    }
}

#[test]
fn fp32_accumulate_matches_scalar() {
    let mut rng = StdRng::seed_from_u64(0x3232_abcd);
    let n = 1000;
    let vals: Vec<F32> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

    let scalar_sum = vals.iter().fold(F32::zero(), |acc, &x| acc + x);
    let wide_sum = vals
        .iter()
        .fold(Fp32x2i32::zero(), |acc, &x| acc + Fp32x2i32::from(x));
    assert_eq!(wide_sum.reduce::<P32>(), scalar_sum);
}

#[test]
fn fp64_roundtrip() {
    let mut rng = StdRng::seed_from_u64(0x6464_6464);
    for _ in 0..1000 {
        let a: F64 = RandomSampling::random(&mut rng);
        let wide = Fp64x4i32::from(a);
        let back = wide.reduce::<P64>();
        assert_eq!(a, back);
    }
}

#[test]
fn fp64_accumulate_matches_scalar() {
    let mut rng = StdRng::seed_from_u64(0x6464_beef);
    let n = 1000;
    let vals: Vec<F64> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

    let scalar_sum = vals.iter().fold(F64::zero(), |acc, &x| acc + x);
    let wide_sum = vals
        .iter()
        .fold(Fp64x4i32::zero(), |acc, &x| acc + Fp64x4i32::from(x));
    assert_eq!(wide_sum.reduce::<P64>(), scalar_sum);
}

#[test]
fn fp64_product_accum_matches_scalar() {
    let mut rng = StdRng::seed_from_u64(0x6464_4444);
    let n = 500;
    let a_vals: Vec<F64> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
    let b_vals: Vec<F64> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

    let scalar_sum: F64 = a_vals
        .iter()
        .zip(b_vals.iter())
        .fold(F64::zero(), |acc, (&a, &b)| acc + a * b);

    let accum_sum = a_vals
        .iter()
        .zip(b_vals.iter())
        .fold(Fp64ProductAccum::ZERO, |acc, (&a, &b)| {
            acc + a.mul_to_product_accum(b)
        });
    assert_eq!(F64::reduce_product_accum(accum_sum), scalar_sum);
}

#[test]
fn fp64_ext2_product_accum_matches_scalar() {
    use crate::Ext2;

    type E = Ext2<F64>;

    let mut rng = StdRng::seed_from_u64(0x6464_4445);
    let n = 500;
    let a_vals: Vec<E> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
    let b_vals: Vec<E> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

    let scalar_sum: E = a_vals
        .iter()
        .zip(b_vals.iter())
        .fold(E::zero(), |acc, (&a, &b)| acc + a * b);

    let accum_sum = a_vals.iter().zip(b_vals.iter()).fold(
        <<E as HasUnreducedOps>::ProductAccum as num_traits::Zero>::zero(),
        |acc, (&a, &b)| acc + a.mul_to_product_accum(b),
    );
    assert_eq!(E::reduce_product_accum(accum_sum), scalar_sum);
}

#[test]
fn fp64_mul_u64_accum_matches_scalar() {
    let mut rng = StdRng::seed_from_u64(0x6464_5555);
    let n = 500;
    let a_vals: Vec<F64> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
    let b_vals: Vec<u64> = (0..n).map(|_| rng.next_u64() >> 32).collect();

    let scalar_sum: F64 = a_vals
        .iter()
        .zip(b_vals.iter())
        .fold(F64::zero(), |acc, (&a, &b)| acc + a * F64::from_u64(b));

    let accum_sum = a_vals
        .iter()
        .zip(b_vals.iter())
        .fold(Fp64ProductAccum::ZERO, |acc, (&a, &b)| {
            acc + a.mul_u64_unreduced(b)
        });
    assert_eq!(F64::reduce_mul_u64_accum(accum_sum), scalar_sum);
}

#[test]
fn fp128_product_accum_matches_scalar() {
    let mut rng = StdRng::seed_from_u64(0x0128_6666);
    let n = 500;
    let a_vals: Vec<F128> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
    let b_vals: Vec<F128> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

    let scalar_sum: F128 = a_vals
        .iter()
        .zip(b_vals.iter())
        .fold(F128::zero(), |acc, (&a, &b)| acc + a * b);

    let accum_sum = a_vals
        .iter()
        .zip(b_vals.iter())
        .fold(Fp128ProductAccum::ZERO, |acc, (&a, &b)| {
            acc + a.mul_to_product_accum(b)
        });
    assert_eq!(F128::reduce_product_accum(accum_sum), scalar_sum);
}

#[test]
fn fp128_mul_u64_accum_matches_scalar() {
    let mut rng = StdRng::seed_from_u64(0x0128_7777);
    let n = 500;
    let a_vals: Vec<F128> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
    let b_vals: Vec<u64> = (0..n).map(|_| rng.next_u64()).collect();

    let scalar_sum: F128 = a_vals
        .iter()
        .zip(b_vals.iter())
        .fold(F128::zero(), |acc, (&a, &b)| acc + a * F128::from_u64(b));

    let accum_sum = a_vals
        .iter()
        .zip(b_vals.iter())
        .fold(Fp128MulU64Accum::ZERO, |acc, (&a, &b)| {
            acc + a.mul_u64_unreduced(b)
        });
    assert_eq!(F128::reduce_mul_u64_accum(accum_sum), scalar_sum);
}

#[test]
fn fp128_product_accum_sub_neg() {
    let mut rng = StdRng::seed_from_u64(0x0128_8888);
    let n = 500;
    let a_vals: Vec<F128> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();
    let b_vals: Vec<F128> = (0..n).map(|_| RandomSampling::random(&mut rng)).collect();

    let mut scalar_sum = F128::zero();
    let mut accum_pos = Fp128ProductAccum::ZERO;
    let mut accum_neg = Fp128ProductAccum::ZERO;
    for (i, (&a, &b)) in a_vals.iter().zip(b_vals.iter()).enumerate() {
        let prod = a.mul_to_product_accum(b);
        if i % 2 == 0 {
            scalar_sum += a * b;
            accum_pos += prod;
        } else {
            scalar_sum -= a * b;
            accum_neg += prod;
        }
    }
    let result = F128::reduce_product_accum(accum_pos) - F128::reduce_product_accum(accum_neg);
    assert_eq!(result, scalar_sum);
}
