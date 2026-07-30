//! Differential tests for the deferred-reduction machinery (`Unreduced`,
//! `Fold`, `MulBaseUnreduced`) against jolt-field, with an independent
//! schoolbook oracle (256-bit limb multiply + binary long division — no
//! Solinas folding, no shared code).
//!
//! Coverage per accumulator type: exactness of delayed sums vs direct
//! reduced multiplication over random batches AND adversarial batches
//! (all-max operands, wrap-through add/sub sequences), plus strict parity
//! with the baseline's `HasUnreducedOps`/`HasWide`/`ReduceTo`/
//! `HasOptimizedFold`/`MulBaseUnreduced` machinery under identical inputs
//! and challenge constants.
//!
//! Headroom boundaries: the `i32`-lane bound (32768 max-lane accumulations)
//! is tested exactly, with the one-past case asserted to panic in debug
//! builds. The `u128`-slot headrooms (≥ 2^61 terms) are analytically
//! derived in `solinas/unreduced.rs` and computationally untestable; the
//! adversarial all-max batches here exercise the worst per-term slot
//! contributions those derivations bound.

#![cfg(feature = "solinas")]
// NB: no `expect(clippy::unwrap_used)` — every unwrap here sits inside a
// local `macro_rules!` expansion, where the lint does not fire.

use jolt_field as base;
use jolt_field_two as two;

use base::unreduced::{HasOptimizedFold, HasUnreducedOps, HasWide, ReduceTo};
use base::{CanonicalField, MulBaseUnreduced as BaseMulBaseUnreduced};
use num_traits::Zero;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use two::{CanonicalEncoding, ExtField, Fold, MulBaseUnreduced, Ring, Unreduced};

const M61: u64 = (1 << 61) - 1;

/// 128×128 → 256-bit schoolbook multiply over 64-bit halves (independent of
/// both crates' `mul_wide`).
fn oracle_mul_256(a: u128, b: u128) -> [u64; 4] {
    let (a0, a1) = (a as u64 as u128, a >> 64);
    let (b0, b1) = (b as u64 as u128, b >> 64);
    let (p00, p01, p10, p11) = (a0 * b0, a0 * b1, a1 * b0, a1 * b1);
    const LO: u128 = u64::MAX as u128;
    let mid = (p00 >> 64) + (p01 & LO) + (p10 & LO);
    let hi = (p01 >> 64) + (p10 >> 64) + (p11 & LO) + (mid >> 64);
    let top = (p11 >> 64) + (hi >> 64);
    [p00 as u64, mid as u64, hi as u64, top as u64]
}

/// Little-endian limbs mod `p` by binary long division — no Solinas folding.
fn oracle_mod(limbs: &[u64], p: u128) -> u128 {
    let mut r: u128 = 0;
    for &limb in limbs.iter().rev() {
        for i in (0..64).rev() {
            let top = r >> 127;
            let mut v = (r << 1) | ((limb >> i) & 1) as u128;
            if top == 1 {
                v = v.wrapping_add(0u128.wrapping_sub(p));
            } else if v >= p {
                v -= p;
            }
            r = v;
        }
    }
    r
}

fn mulmod(a: u128, b: u128, p: u128) -> u128 {
    oracle_mod(&oracle_mul_256(a, b), p)
}

fn addmod(a: u128, b: u128, p: u128) -> u128 {
    let (s, overflow) = a.overflowing_add(b);
    oracle_mod(&[s as u64, (s >> 64) as u64, overflow as u64], p)
}

fn submod(a: u128, b: u128, p: u128) -> u128 {
    addmod(a, p - b, p)
}

/// Full `Unreduced` + `HasWide` sweep for one paired base-field
/// instantiation: product/small-product batch exactness (random,
/// all-max, wrap-through), wide-lane roundtrip/group-ops/scaling — all
/// against the field ops, the schoolbook oracle, and the baseline.
macro_rules! base_field_suite {
    ($name:ident, $F2:ty, $FB:ty, $p:expr, $seed:expr) => {
        #[test]
        fn $name() {
            let p: u128 = $p;
            let mut rng = ChaCha20Rng::seed_from_u64($seed);
            let f2 = |v: u128| <$F2 as CanonicalEncoding>::from_u128_checked(v).unwrap();
            let fb = |v: u128| <$FB as CanonicalField>::from_canonical_u128_checked(v).unwrap();
            let val2 = |x: &$F2| x.to_u128_checked().unwrap();
            let valb = |x: &$FB| x.to_canonical_u128();

            assert_eq!(
                <$F2 as Unreduced>::SUM_IS_EXACT,
                <$FB as HasUnreducedOps>::DELAYED_PRODUCT_SUM_IS_EXACT,
                "SUM_IS_EXACT parity"
            );

            // Σ aᵢ·bᵢ: delayed vs per-term vs oracle vs baseline.
            let check_products = |pairs: &[(u128, u128)]| {
                let expect = pairs
                    .iter()
                    .fold(0u128, |acc, &(a, b)| addmod(acc, mulmod(a, b, p), p));
                let acc2 = pairs.iter().fold(
                    <<$F2 as Unreduced>::Product as Zero>::zero(),
                    |acc, &(a, b)| acc + f2(a).mul_unreduced(f2(b)),
                );
                assert_eq!(val2(&<$F2 as Unreduced>::reduce_product(acc2)), expect);
                let per_term = pairs
                    .iter()
                    .fold(<$F2 as Zero>::zero(), |acc, &(a, b)| acc + f2(a) * f2(b));
                assert_eq!(val2(&per_term), expect, "per-term vs oracle");
                let accb = pairs.iter().fold(
                    <<$FB as HasUnreducedOps>::ProductAccum as Zero>::zero(),
                    |acc, &(a, b)| acc + fb(a).mul_to_product_accum(fb(b)),
                );
                assert_eq!(
                    valb(&<$FB as HasUnreducedOps>::reduce_product_accum(accb)),
                    expect,
                    "baseline parity"
                );
            };
            for &n in &[1usize, 2, 7, 501] {
                let pairs: Vec<(u128, u128)> = (0..n)
                    .map(|_| (rng.gen::<u128>() % p, rng.gen::<u128>() % p))
                    .collect();
                check_products(&pairs);
            }
            check_products(&vec![(p - 1, p - 1); 512]);

            // Σ aᵢ·bᵢ with raw u64 scalars (including u64::MAX and 0).
            let check_small = |pairs: &[(u128, u64)]| {
                let expect = pairs.iter().fold(0u128, |acc, &(a, b)| {
                    addmod(acc, mulmod(a, b as u128, p), p)
                });
                let acc2 = pairs.iter().fold(
                    <<$F2 as Unreduced>::SmallProduct as Zero>::zero(),
                    |acc, &(a, b)| acc + f2(a).mul_u64_unreduced(b),
                );
                assert_eq!(
                    val2(&<$F2 as Unreduced>::reduce_small_product(acc2)),
                    expect
                );
                let accb = pairs.iter().fold(
                    <<$FB as HasUnreducedOps>::MulU64Accum as Zero>::zero(),
                    |acc, &(a, b)| acc + fb(a).mul_u64_unreduced(b),
                );
                assert_eq!(
                    valb(&<$FB as HasUnreducedOps>::reduce_mul_u64_accum(accb)),
                    expect,
                    "baseline small-product parity"
                );
            };
            for &n in &[1usize, 3, 400] {
                let pairs: Vec<(u128, u64)> = (0..n)
                    .map(|_| (rng.gen::<u128>() % p, rng.gen::<u64>()))
                    .collect();
                check_small(&pairs);
            }
            check_small(&[(p - 1, u64::MAX), (p - 1, 0), (0, u64::MAX)]);
            check_small(&vec![(p - 1, u64::MAX); 400]);

            // Wrap-through subtraction: t1 − t2 + t2 = t1 must be exact even
            // though the intermediate slots dip below zero (wrapping group).
            let (a1, b1) = (rng.gen::<u128>() % p, rng.gen::<u128>() % p);
            let t1 = f2(a1).mul_unreduced(f2(b1));
            let t2 = f2(p - 1).mul_unreduced(f2(p - 1));
            assert_eq!(
                val2(&<$F2 as Unreduced>::reduce_product(t1 - t2 + t2)),
                mulmod(a1, b1, p),
                "wrap-through sub/add"
            );
            assert_eq!(
                val2(
                    &(<$F2 as Unreduced>::reduce_product(t1)
                        - <$F2 as Unreduced>::reduce_product(t2))
                ),
                submod(mulmod(a1, b1, p), mulmod(p - 1, p - 1, p), p),
                "separate pos/neg accumulators"
            );

            // Wide lanes: roundtrip, group ops, scaling; vs baseline.
            let mut vals: Vec<u128> = vec![0, 1, 2, p / 2, p - 2, p - 1];
            vals.extend((0..200).map(|_| rng.gen::<u128>() % p));
            for &x in &vals {
                let w2 = <$F2 as Unreduced>::Wide::from(f2(x));
                assert_eq!(val2(&<$F2 as Unreduced>::reduce_wide(w2)), x, "roundtrip");
                let wb = <$FB as HasWide>::Wide::from(fb(x));
                assert_eq!(valb(&ReduceTo::<$FB>::reduce(wb)), x, "baseline roundtrip");

                let y = rng.gen::<u128>() % p;
                let wy = <$F2 as Unreduced>::Wide::from(f2(y));
                assert_eq!(
                    val2(&<$F2 as Unreduced>::reduce_wide(w2 + wy)),
                    addmod(x, y, p)
                );
                assert_eq!(
                    val2(&<$F2 as Unreduced>::reduce_wide(w2 - wy)),
                    submod(x, y, p)
                );
                assert_eq!(val2(&<$F2 as Unreduced>::reduce_wide(-w2)), submod(0, x, p));

                for s in [-32768i32, -12345, -1, 0, 1, 2, 12345, 32768] {
                    let got = <$F2 as Unreduced>::reduce_wide(f2(x).scale_wide(s));
                    assert_eq!(got, f2(x) * <$F2 as Ring>::from_i64(s as i64), "scale_wide");
                    let gotb = ReduceTo::<$FB>::reduce(fb(x).mul_small_to_wide(s));
                    assert_eq!(val2(&got), valb(&gotb), "baseline scale parity");
                }
            }

            // Mixed-sign wide accumulation (magnitudes stay within lane
            // headroom: ≤ 300 canonical terms).
            let mut w2 = <<$F2 as Unreduced>::Wide as Zero>::zero();
            let mut expect = 0u128;
            for (i, &x) in vals.iter().enumerate() {
                if i % 3 == 0 {
                    w2 -= <$F2 as Unreduced>::Wide::from(f2(x));
                    expect = submod(expect, x, p);
                } else {
                    w2 += <$F2 as Unreduced>::Wide::from(f2(x));
                    expect = addmod(expect, x, p);
                }
            }
            assert_eq!(
                val2(&<$F2 as Unreduced>::reduce_wide(w2)),
                expect,
                "mixed signs"
            );

            // Degree-1 MulBaseUnreduced blanket: default body is the plain
            // unreduced product.
            let (x, s) = (rng.gen::<u128>() % p, rng.gen::<u128>() % p);
            assert_eq!(
                val2(&<$F2 as Unreduced>::reduce_product(
                    f2(x).mul_base_unreduced(f2(s))
                )),
                mulmod(x, s, p),
                "degree-1 mul_base_unreduced"
            );
        }
    };
}

base_field_suite!(
    fp32_prime24_unreduced,
    two::Prime24Offset3,
    base::Prime24Offset3,
    (1 << 24) - 3,
    0x0724_0001
);
base_field_suite!(
    fp32_prime32_unreduced,
    two::Prime32Offset99,
    base::Prime32Offset99,
    (1 << 32) - 99,
    0x0732_0002
);
base_field_suite!(
    fp64_prime40_unreduced,
    two::Prime40Offset195,
    base::Prime40Offset195,
    (1 << 40) - 195,
    0x0740_0003
);
base_field_suite!(
    fp64_prime64_unreduced,
    two::Prime64Offset59,
    base::Prime64Offset59,
    u64::MAX as u128 - 58,
    0x0764_0004
);
base_field_suite!(
    fp128_prime275_unreduced,
    two::Prime128Offset275,
    base::Prime128Offset275,
    u128::MAX - 274,
    0x0728_0005
);
base_field_suite!(
    fp128_prime_a7f7_unreduced,
    two::Prime128OffsetA7F7,
    base::Prime128OffsetA7F7,
    u128::MAX - 0xFFFF_A7F6,
    0x0728_0006
);

/// The `i32`-lane headroom boundary: 32768 all-max-lane accumulations are
/// exact (32768 · 0xFFFF = 2147450880 ≤ i32::MAX).
#[test]
fn wide_lane_headroom_boundary_is_exact() {
    type F = two::Prime128Offset275;
    let unit = two::Fp128x8i32([0xFFFF; 8]); // value = 2^128 − 1
    let mut acc = two::Fp128x8i32([0; 8]);
    for _ in 0..32768 {
        acc += unit;
    }
    let expect = <F as Ring>::from_u128(u128::MAX) * <F as Ring>::from_u64(32768);
    assert_eq!(<F as Unreduced>::reduce_wide(acc), expect);
}

/// One past the lane headroom overflows an `i32` lane; the non-wrapping
/// lane ops turn that into a debug-build panic (the only runtime
/// enforcement the contract has).
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "attempt to add with overflow")]
fn wide_lane_one_past_headroom_panics_in_debug() {
    let unit = two::Fp128x8i32([0xFFFF; 8]);
    let mut acc = two::Fp128x8i32([0; 8]);
    for _ in 0..32769 {
        acc += unit;
    }
    let _ = std::hint::black_box(acc);
}

/// `FpExt4<Fp32>` fused accumulator: delayed batch sums vs per-term ring
/// multiplication and the baseline, plus the coordinate-scaling
/// `MulBaseUnreduced` override.
macro_rules! ext4_fp32_suite {
    ($name:ident, $F2:ty, $FB:ty, $p:expr, $seed:expr) => {
        #[test]
        fn $name() {
            type E2 = two::FpExt4<$F2>;
            type EB = base::FpExt4<$FB>;
            let p: u128 = $p;
            let mut rng = ChaCha20Rng::seed_from_u64($seed);
            let f2 = |v: u128| <$F2 as CanonicalEncoding>::from_u128_checked(v).unwrap();
            let fb = |v: u128| <$FB as CanonicalField>::from_canonical_u128_checked(v).unwrap();
            let mk2 = |v: [u128; 4]| E2::new(v.map(f2));
            let mkb = |v: [u128; 4]| EB::new(v.map(fb));
            let vec2 = |e: &E2| e.coeffs.map(|c| c.to_u128_checked().unwrap());
            let vecb = |e: &EB| e.coeffs.map(|c| c.to_canonical_u128());
            let sample = |rng: &mut ChaCha20Rng| -> [u128; 4] {
                std::array::from_fn(|_| rng.gen::<u128>() % p)
            };

            assert!(<E2 as Unreduced>::SUM_IS_EXACT);
            assert!(<EB as HasUnreducedOps>::DELAYED_PRODUCT_SUM_IS_EXACT);

            let check = |pairs: &[([u128; 4], [u128; 4])]| {
                let acc2 = pairs.iter().fold(
                    <<E2 as Unreduced>::Product as Zero>::zero(),
                    |acc, &(a, b)| acc + mk2(a).mul_unreduced(mk2(b)),
                );
                let per_term = pairs
                    .iter()
                    .fold(<E2 as Zero>::zero(), |acc, &(a, b)| acc + mk2(a) * mk2(b));
                assert_eq!(
                    <E2 as Unreduced>::reduce_product(acc2),
                    per_term,
                    "delayed sum vs per-term"
                );
                let accb = pairs.iter().fold(
                    <<EB as HasUnreducedOps>::ProductAccum as Zero>::zero(),
                    |acc, &(a, b)| acc + mkb(a).mul_to_product_accum(mkb(b)),
                );
                assert_eq!(
                    vec2(&per_term),
                    vecb(&<EB as HasUnreducedOps>::reduce_product_accum(accb)),
                    "baseline parity"
                );
            };
            for &n in &[1usize, 2, 33, 512] {
                let pairs: Vec<_> = (0..n)
                    .map(|_| (sample(&mut rng), sample(&mut rng)))
                    .collect();
                check(&pairs);
            }
            // Adversarial: all coefficients at p − 1 maximizes every fused
            // column sum (the 7·P² per-term worst case).
            check(&vec![([p - 1; 4], [p - 1; 4]); 512]);

            // Wrap-through subtraction on the fused accumulator.
            let (a, b) = (sample(&mut rng), sample(&mut rng));
            let t1 = mk2(a).mul_unreduced(mk2(b));
            let t2 = mk2([p - 1; 4]).mul_unreduced(mk2([p - 1; 4]));
            assert_eq!(
                <E2 as Unreduced>::reduce_product(t1 - t2 + t2),
                mk2(a) * mk2(b),
                "wrap-through sub/add"
            );

            // MulBaseUnreduced override: vs mul_base, vs the default
            // lift-then-mul body, vs the baseline; batched.
            let mut acc2 = <<E2 as Unreduced>::Product as Zero>::zero();
            let mut accb = <<EB as HasUnreducedOps>::ProductAccum as Zero>::zero();
            let mut per_term = <E2 as Zero>::zero();
            for _ in 0..300 {
                let (xv, sv) = (sample(&mut rng), rng.gen::<u128>() % p);
                let (x2, xb) = (mk2(xv), mkb(xv));
                let over = x2.mul_base_unreduced(f2(sv));
                assert_eq!(
                    <E2 as Unreduced>::reduce_product(over),
                    x2.mul_base(f2(sv)),
                    "override vs mul_base"
                );
                assert_eq!(
                    <E2 as Unreduced>::reduce_product(over),
                    <E2 as Unreduced>::reduce_product(
                        x2.mul_unreduced(<E2 as ExtField<$F2>>::lift_base(f2(sv)))
                    ),
                    "override vs default lift-then-mul"
                );
                acc2 += over;
                accb += xb.mul_base_to_product_accum(fb(sv));
                per_term += x2.mul_base(f2(sv));
            }
            assert_eq!(<E2 as Unreduced>::reduce_product(acc2), per_term);
            assert_eq!(
                vec2(&per_term),
                vecb(&<EB as HasUnreducedOps>::reduce_product_accum(accb)),
                "baseline mul-base batch parity"
            );
        }
    };
}

ext4_fp32_suite!(
    ext4_fp32_prime24_accum,
    two::Prime24Offset3,
    base::Prime24Offset3,
    (1 << 24) - 3,
    0x0E44_0001
);
ext4_fp32_suite!(
    ext4_fp32_prime32_accum,
    two::Prime32Offset99,
    base::Prime32Offset99,
    (1 << 32) - 99,
    0x0E44_0002
);

/// `FpExt2<Fp64>` carry-tracked accumulator: batch exactness vs per-term
/// multiplication and the baseline (both non-residue configs), plus the
/// `AccumPair` small-product path.
macro_rules! ext2_fp64_suite {
    ($name:ident, $F2:ty, $FB:ty, $C2:ty, $CB:ty, $p:expr, $seed:expr) => {
        #[test]
        fn $name() {
            type E2 = two::FpExt2<$F2, $C2>;
            type EB = base::FpExt2<$FB, $CB>;
            let p: u128 = $p;
            let mut rng = ChaCha20Rng::seed_from_u64($seed);
            let f2 = |v: u128| <$F2 as CanonicalEncoding>::from_u128_checked(v).unwrap();
            let fb = |v: u128| <$FB as CanonicalField>::from_canonical_u128_checked(v).unwrap();
            let mk2 = |v: [u128; 2]| E2::new(f2(v[0]), f2(v[1]));
            let mkb = |v: [u128; 2]| EB::new(fb(v[0]), fb(v[1]));
            let vec2 = |e: &E2| e.coeffs.map(|c| c.to_u128_checked().unwrap());
            let vecb = |e: &EB| e.coeffs.map(|c| c.to_canonical_u128());
            let sample = |rng: &mut ChaCha20Rng| -> [u128; 2] {
                std::array::from_fn(|_| rng.gen::<u128>() % p)
            };

            assert!(<E2 as Unreduced>::SUM_IS_EXACT);
            assert!(<EB as HasUnreducedOps>::DELAYED_PRODUCT_SUM_IS_EXACT);

            let check = |pairs: &[([u128; 2], [u128; 2])]| {
                let acc2 = pairs.iter().fold(
                    <<E2 as Unreduced>::Product as Zero>::zero(),
                    |acc, &(a, b)| acc + mk2(a).mul_unreduced(mk2(b)),
                );
                let per_term = pairs
                    .iter()
                    .fold(<E2 as Zero>::zero(), |acc, &(a, b)| acc + mk2(a) * mk2(b));
                assert_eq!(
                    <E2 as Unreduced>::reduce_product(acc2),
                    per_term,
                    "delayed sum vs per-term"
                );
                let accb = pairs.iter().fold(
                    <<EB as HasUnreducedOps>::ProductAccum as Zero>::zero(),
                    |acc, &(a, b)| acc + mkb(a).mul_to_product_accum(mkb(b)),
                );
                assert_eq!(
                    vec2(&per_term),
                    vecb(&<EB as HasUnreducedOps>::reduce_product_accum(accb)),
                    "baseline parity"
                );
            };
            for &n in &[1usize, 2, 33, 512] {
                let pairs: Vec<_> = (0..n)
                    .map(|_| (sample(&mut rng), sample(&mut rng)))
                    .collect();
                check(&pairs);
            }
            // All-max coefficients maximize p00/p11 and force the c0 carry
            // paths (P² bias wrap for NR = −1, double-add carries for
            // NR = 2) on every term.
            check(&vec![([p - 1; 2], [p - 1; 2]); 512]);
            // Single products at the corners of the carry analysis.
            for corner in [[0u128, p - 1], [p - 1, 0], [1, p - 1], [p - 1, 1]] {
                check(&[(corner, [p - 1; 2])]);
            }

            // Wrap-through subtraction.
            let (a, b) = (sample(&mut rng), sample(&mut rng));
            let t1 = mk2(a).mul_unreduced(mk2(b));
            let t2 = mk2([p - 1; 2]).mul_unreduced(mk2([p - 1; 2]));
            assert_eq!(
                <E2 as Unreduced>::reduce_product(t1 - t2 + t2),
                mk2(a) * mk2(b),
                "wrap-through sub/add"
            );

            // Small products through the AccumPair path.
            let pairs: Vec<([u128; 2], u64)> = (0..300)
                .map(|_| (sample(&mut rng), rng.gen::<u64>()))
                .collect();
            let acc2 = pairs.iter().fold(
                <<E2 as Unreduced>::SmallProduct as Zero>::zero(),
                |acc, &(x, s)| acc + mk2(x).mul_u64_unreduced(s),
            );
            let per_term = pairs.iter().fold(<E2 as Zero>::zero(), |acc, &(x, s)| {
                acc + mk2(x) * <E2 as Ring>::from_u64(s)
            });
            assert_eq!(
                <E2 as Unreduced>::reduce_small_product(acc2),
                per_term,
                "small-product delayed sum"
            );
            let accb = pairs.iter().fold(
                <<EB as HasUnreducedOps>::MulU64Accum as Zero>::zero(),
                |acc, &(x, s)| acc + mkb(x).mul_u64_unreduced(s),
            );
            assert_eq!(
                vec2(&per_term),
                vecb(&<EB as HasUnreducedOps>::reduce_mul_u64_accum(accb)),
                "baseline small-product parity"
            );

            // MulBaseUnreduced default body routes through the fused accum.
            let (xv, sv) = (sample(&mut rng), rng.gen::<u128>() % p);
            assert_eq!(
                <E2 as Unreduced>::reduce_product(mk2(xv).mul_base_unreduced(f2(sv))),
                mk2(xv).mul_base(f2(sv)),
                "mul_base_unreduced"
            );
        }
    };
}

ext2_fp64_suite!(
    ext2_fp64_prime40_two_nr_accum,
    two::Prime40Offset195,
    base::Prime40Offset195,
    two::TwoNr,
    base::TwoNr,
    (1 << 40) - 195,
    0x0E22_0001
);
ext2_fp64_suite!(
    ext2_fp64_prime64_two_nr_accum,
    two::Prime64Offset59,
    base::Prime64Offset59,
    two::TwoNr,
    base::TwoNr,
    u64::MAX as u128 - 58,
    0x0E22_0002
);
// Mersenne-61 is the one convenient u64 prime with p ≡ 3 (mod 4), where
// NegOneNr is a genuine non-residue — exercises the P²-bias carry branch.
ext2_fp64_suite!(
    ext2_fp64_m61_neg_one_nr_accum,
    two::Fp64<M61>,
    base::Fp64<M61>,
    two::NegOneNr,
    base::NegOneNr,
    M61 as u128,
    0x0E22_0003
);

/// Fold parity for one paired instantiation: `fold_one(precompute(r), e, o)`
/// must equal the field identity `e + r·(o − e)` AND the baseline's
/// optimized fold under the same challenge constants.
macro_rules! fold_parity {
    ($name:ident, $E2:ty, $EB:ty, $F2:ty, $FB:ty, $d:expr, $p:expr, $seed:expr) => {
        #[test]
        fn $name() {
            let p: u128 = $p;
            let d: usize = $d;
            let mut rng = ChaCha20Rng::seed_from_u64($seed);
            let f2 = |v: u128| <$F2 as CanonicalEncoding>::from_u128_checked(v).unwrap();
            let fb = |v: u128| <$FB as CanonicalField>::from_canonical_u128_checked(v).unwrap();
            let mk2 = |vals: &[u128]| {
                <$E2 as ExtField<$F2>>::from_base_slice(
                    &vals.iter().map(|&v| f2(v)).collect::<Vec<_>>(),
                )
            };
            let mkb = |vals: &[u128]| {
                <$EB as base::ExtField<$FB>>::from_base_slice(
                    &vals.iter().map(|&v| fb(v)).collect::<Vec<_>>(),
                )
            };
            let vec2 = |e: &$E2| {
                <$E2 as ExtField<$F2>>::to_base_vec(e)
                    .iter()
                    .map(|c| c.to_u128_checked().unwrap())
                    .collect::<Vec<u128>>()
            };
            let vecb = |e: &$EB| {
                <$EB as base::ExtField<$FB>>::to_base_vec(e)
                    .iter()
                    .map(|c| c.to_canonical_u128())
                    .collect::<Vec<u128>>()
            };
            let sample = |rng: &mut ChaCha20Rng| -> Vec<u128> {
                (0..d).map(|_| rng.gen::<u128>() % p).collect()
            };

            for _ in 0..8 {
                let rv = sample(&mut rng);
                let (r2, rb) = (mk2(&rv), mkb(&rv));
                let ctx2 = <$E2 as Fold>::precompute(r2);
                let ctxb = <$EB as HasOptimizedFold>::precompute_fold(rb);

                let mut cases: Vec<(Vec<u128>, Vec<u128>)> = vec![
                    (vec![0; d], vec![p - 1; d]),
                    (vec![p - 1; d], vec![0; d]),
                    (vec![p - 1; d], vec![p - 1; d]),
                    (vec![0; d], vec![0; d]),
                ];
                cases.extend((0..24).map(|_| (sample(&mut rng), sample(&mut rng))));
                for (ev, ov) in cases {
                    let (e2, o2) = (mk2(&ev), mk2(&ov));
                    let got = <$E2 as Fold>::fold_one(&ctx2, e2, o2);
                    assert_eq!(got, e2 + r2 * (o2 - e2), "fold vs field identity");
                    let gotb = <$EB as HasOptimizedFold>::fold_one(&ctxb, mkb(&ev), mkb(&ov));
                    assert_eq!(vec2(&got), vecb(&gotb), "fold vs baseline");
                }
            }
        }
    };
}

fold_parity!(
    fold_fp32_prime24,
    two::Prime24Offset3,
    base::Prime24Offset3,
    two::Prime24Offset3,
    base::Prime24Offset3,
    1,
    (1 << 24) - 3,
    0x0F01
);
fold_parity!(
    fold_fp32_prime32,
    two::Prime32Offset99,
    base::Prime32Offset99,
    two::Prime32Offset99,
    base::Prime32Offset99,
    1,
    (1 << 32) - 99,
    0x0F02
);
fold_parity!(
    fold_fp64_prime40,
    two::Prime40Offset195,
    base::Prime40Offset195,
    two::Prime40Offset195,
    base::Prime40Offset195,
    1,
    (1 << 40) - 195,
    0x0F03
);
fold_parity!(
    fold_fp64_prime64,
    two::Prime64Offset59,
    base::Prime64Offset59,
    two::Prime64Offset59,
    base::Prime64Offset59,
    1,
    u64::MAX as u128 - 58,
    0x0F04
);
fold_parity!(
    fold_fp128_prime275,
    two::Prime128Offset275,
    base::Prime128Offset275,
    two::Prime128Offset275,
    base::Prime128Offset275,
    1,
    u128::MAX - 274,
    0x0F05
);
fold_parity!(
    fold_fp128_prime_a7f7,
    two::Prime128OffsetA7F7,
    base::Prime128OffsetA7F7,
    two::Prime128OffsetA7F7,
    base::Prime128OffsetA7F7,
    1,
    u128::MAX - 0xFFFF_A7F6,
    0x0F06
);
// FpExt2<Fp64> fold matrices (the specialized EOR fold), all three configs.
fold_parity!(
    fold_ext2_fp64_prime40,
    two::Ext2<two::Prime40Offset195>,
    base::Ext2<base::Prime40Offset195>,
    two::Prime40Offset195,
    base::Prime40Offset195,
    2,
    (1 << 40) - 195,
    0x0F07
);
fold_parity!(
    fold_ext2_fp64_prime64,
    two::Ext2<two::Prime64Offset59>,
    base::Ext2<base::Prime64Offset59>,
    two::Prime64Offset59,
    base::Prime64Offset59,
    2,
    u64::MAX as u128 - 58,
    0x0F08
);
fold_parity!(
    fold_ext2_fp64_m61_neg_one,
    two::FpExt2<two::Fp64<M61>, two::NegOneNr>,
    base::FpExt2<base::Fp64<M61>, base::NegOneNr>,
    two::Fp64<M61>,
    base::Fp64<M61>,
    2,
    M61 as u128,
    0x0F09
);
// FpExt2 default folds over the other bases.
fold_parity!(
    fold_ext2_fp32_prime32,
    two::Ext2<two::Prime32Offset99>,
    base::Ext2<base::Prime32Offset99>,
    two::Prime32Offset99,
    base::Prime32Offset99,
    2,
    (1 << 32) - 99,
    0x0F0A
);
fold_parity!(
    fold_ext2_fp128_prime275,
    two::Ext2<two::Prime128Offset275>,
    base::Ext2<base::Prime128Offset275>,
    two::Prime128Offset275,
    base::Prime128Offset275,
    2,
    u128::MAX - 274,
    0x0F0B
);
// FpExt4<Fp32> fold matrix, both reduction paths (P < 2^31 and P ≥ 2^31).
fold_parity!(
    fold_ext4_fp32_prime24,
    two::FpExt4<two::Prime24Offset3>,
    base::FpExt4<base::Prime24Offset3>,
    two::Prime24Offset3,
    base::Prime24Offset3,
    4,
    (1 << 24) - 3,
    0x0F0C
);
fold_parity!(
    fold_ext4_fp32_prime32,
    two::FpExt4<two::Prime32Offset99>,
    base::FpExt4<base::Prime32Offset99>,
    two::Prime32Offset99,
    base::Prime32Offset99,
    4,
    (1 << 32) - 99,
    0x0F0D
);
// FpExt4 default folds over the other bases.
fold_parity!(
    fold_ext4_fp64_prime40,
    two::FpExt4<two::Prime40Offset195>,
    base::FpExt4<base::Prime40Offset195>,
    two::Prime40Offset195,
    base::Prime40Offset195,
    4,
    (1 << 40) - 195,
    0x0F0E
);
fold_parity!(
    fold_ext4_fp128_prime275,
    two::FpExt4<two::Prime128Offset275>,
    base::FpExt4<base::Prime128Offset275>,
    two::Prime128Offset275,
    base::Prime128Offset275,
    4,
    u128::MAX - 274,
    0x0F0F
);
// FpExt8 default folds across all three widths.
fold_parity!(
    fold_ext8_fp32_prime24,
    two::FpExt8<two::Prime24Offset3>,
    base::FpExt8<base::Prime24Offset3>,
    two::Prime24Offset3,
    base::Prime24Offset3,
    8,
    (1 << 24) - 3,
    0x0F10
);
fold_parity!(
    fold_ext8_fp64_prime40,
    two::FpExt8<two::Prime40Offset195>,
    base::FpExt8<base::Prime40Offset195>,
    two::Prime40Offset195,
    base::Prime40Offset195,
    8,
    (1 << 40) - 195,
    0x0F11
);
fold_parity!(
    fold_ext8_fp128_prime275,
    two::FpExt8<two::Prime128Offset275>,
    base::FpExt8<base::Prime128Offset275>,
    two::Prime128Offset275,
    base::Prime128Offset275,
    8,
    u128::MAX - 274,
    0x0F12
);

/// Identity-shape extensions: the `MulBaseUnreduced` default body must
/// match `mul_base`, and the identity `Unreduced` ops must match plain
/// ring arithmetic.
macro_rules! identity_unreduced_suite {
    ($name:ident, $E2:ty, $F2:ty, $d:expr, $p:expr, $seed:expr) => {
        #[test]
        fn $name() {
            let p: u128 = $p;
            let d: usize = $d;
            let mut rng = ChaCha20Rng::seed_from_u64($seed);
            let f2 = |v: u128| <$F2 as CanonicalEncoding>::from_u128_checked(v).unwrap();
            let mk2 = |vals: &[u128]| {
                <$E2 as ExtField<$F2>>::from_base_slice(
                    &vals.iter().map(|&v| f2(v)).collect::<Vec<_>>(),
                )
            };
            assert!(!<$E2 as Unreduced>::SUM_IS_EXACT);
            for _ in 0..32 {
                let xv: Vec<u128> = (0..d).map(|_| rng.gen::<u128>() % p).collect();
                let yv: Vec<u128> = (0..d).map(|_| rng.gen::<u128>() % p).collect();
                let (x, y) = (mk2(&xv), mk2(&yv));
                let (s64, s32, sf) = (rng.gen::<u64>(), rng.gen::<i32>(), rng.gen::<u128>() % p);
                assert_eq!(
                    <$E2 as Unreduced>::reduce_product(x.mul_unreduced(y)),
                    x * y
                );
                assert_eq!(
                    <$E2 as Unreduced>::reduce_small_product(x.mul_u64_unreduced(s64)),
                    x * <$E2 as Ring>::from_u64(s64)
                );
                assert_eq!(
                    <$E2 as Unreduced>::reduce_wide(x.scale_wide(s32)),
                    x * <$E2 as Ring>::from_i64(s32 as i64)
                );
                assert_eq!(
                    <$E2 as Unreduced>::reduce_product(x.mul_base_unreduced(f2(sf))),
                    x.mul_base(f2(sf))
                );
            }
        }
    };
}

identity_unreduced_suite!(
    identity_ext2_fp32,
    two::Ext2<two::Prime32Offset99>,
    two::Prime32Offset99,
    2,
    (1 << 32) - 99,
    0x1D01
);
identity_unreduced_suite!(
    identity_ext2_fp128,
    two::Ext2<two::Prime128Offset275>,
    two::Prime128Offset275,
    2,
    u128::MAX - 274,
    0x1D02
);
identity_unreduced_suite!(
    identity_ext4_fp64,
    two::FpExt4<two::Prime40Offset195>,
    two::Prime40Offset195,
    4,
    (1 << 40) - 195,
    0x1D03
);
identity_unreduced_suite!(
    identity_ext4_fp128,
    two::FpExt4<two::Prime128OffsetA7F7>,
    two::Prime128OffsetA7F7,
    4,
    u128::MAX - 0xFFFF_A7F6,
    0x1D04
);
identity_unreduced_suite!(
    identity_ext8_fp32,
    two::FpExt8<two::Prime24Offset3>,
    two::Prime24Offset3,
    8,
    (1 << 24) - 3,
    0x1D05
);
identity_unreduced_suite!(
    identity_ext8_fp64,
    two::FpExt8<two::Prime64Offset59>,
    two::Prime64Offset59,
    8,
    u64::MAX as u128 - 58,
    0x1D06
);
identity_unreduced_suite!(
    identity_ext8_fp128,
    two::FpExt8<two::Prime128Offset275>,
    two::Prime128Offset275,
    8,
    u128::MAX - 274,
    0x1D07
);
