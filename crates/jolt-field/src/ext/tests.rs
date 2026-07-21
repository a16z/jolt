#![expect(
    clippy::expect_used,
    clippy::unreadable_literal,
    clippy::unwrap_used,
    reason = "tests assert field identities and retain copied field constants"
)]

use super::*;
use crate::ext::lift::{
    canonical_frobenius_thetas, solve_frobenius_moore, validate_canonical_frobenius_thetas,
    ExtField, FrobeniusExtField,
};
use crate::Fp64;
use crate::{FromPrimitiveInt, Invertible};
use rand::rngs::StdRng;
use rand::SeedableRng;

type F = Fp64<4294967197>;
type E2 = Ext2<F>;
type E4 = FpExt4<F>;
type R4 = FpExt4<F>;
type R8 = FpExt8<F>;

#[test]
fn fp_ext2_add_sub_identity() {
    let a = E2::new(F::from_u64(3), F::from_u64(5));
    let b = E2::new(F::from_u64(7), F::from_u64(11));
    let c = a + b;
    assert_eq!(c - b, a);
    assert_eq!(c - a, b);
}

#[test]
fn fp_ext2_mul_one() {
    let a = E2::new(F::from_u64(42), F::from_u64(13));
    assert_eq!(a * E2::one(), a);
    assert_eq!(E2::one() * a, a);
}

#[test]
fn fp_ext2_mul_commutativity() {
    let mut rng = StdRng::seed_from_u64(1234);
    let a = E2::random(&mut rng);
    let b = E2::random(&mut rng);
    assert_eq!(a * b, b * a);
}

#[test]
fn fp_ext2_karatsuba_matches_schoolbook() {
    let mut rng = StdRng::seed_from_u64(5678);
    for _ in 0..100 {
        let a = E2::random(&mut rng);
        let b = E2::random(&mut rng);
        let nr = <TwoNr as FpExt2Config<F>>::non_residue();
        let expected = E2::new(
            (a.coeffs[0] * b.coeffs[0]) + (nr * (a.coeffs[1] * b.coeffs[1])),
            (a.coeffs[0] * b.coeffs[1]) + (a.coeffs[1] * b.coeffs[0]),
        );
        assert_eq!(a * b, expected);
    }
}

#[test]
fn fp_ext2_square_matches_mul() {
    let mut rng = StdRng::seed_from_u64(9012);
    for _ in 0..100 {
        let a = E2::random(&mut rng);
        assert_eq!(a.square(), a * a, "square mismatch for {a:?}");
    }
}

#[test]
fn fp_ext2_inv() {
    let mut rng = StdRng::seed_from_u64(3456);
    for _ in 0..50 {
        let a = E2::random(&mut rng);
        if !a.is_zero() {
            let inv = a.inverse().unwrap();
            assert_eq!(a * inv, E2::one());
        }
    }
}

#[test]
fn fp_ext4_mul_commutativity() {
    let mut rng = StdRng::seed_from_u64(7890);
    let a = E4::random(&mut rng);
    let b = E4::random(&mut rng);
    assert_eq!(a * b, b * a);
}

#[test]
fn fp_ext4_square_matches_mul() {
    let mut rng = StdRng::seed_from_u64(1111);
    for _ in 0..50 {
        let a = E4::random(&mut rng);
        assert_eq!(a.square(), a * a);
    }
}

#[test]
fn fp_ext4_inv() {
    let mut rng = StdRng::seed_from_u64(2222);
    for _ in 0..50 {
        let a = E4::random(&mut rng);
        if !a.is_zero() {
            let inv = a.inverse().unwrap();
            assert_eq!(a * inv, E4::one());
        }
    }
}

#[test]
fn fp_ext4_multiplication_table() {
    let two = F::from_u64(2);
    let e1 = R4::new([F::zero(), F::one(), F::zero(), F::zero()]);
    let e2 = R4::new([F::zero(), F::zero(), F::one(), F::zero()]);
    let e3 = R4::new([F::zero(), F::zero(), F::zero(), F::one()]);
    let two_const = R4::new([two, F::zero(), F::zero(), F::zero()]);

    assert_eq!(e1 * e1, two_const + e2);
    assert_eq!(e1 * e2, e1 + e3);
    assert_eq!(e1 * e3, e2);
    assert_eq!(e2 * e2, two_const);
    assert_eq!(e2 * e3, e1 - e3);
    assert_eq!(e3 * e3, two_const - e2);
}

#[test]
fn fp_ext8_multiplication_table_spot_checks() {
    let two = F::from_u64(2);
    let e = |idx: usize| {
        R8::new(std::array::from_fn(|i| {
            if i == idx {
                F::one()
            } else {
                F::zero()
            }
        }))
    };
    let two_const = R8::new([
        two,
        F::zero(),
        F::zero(),
        F::zero(),
        F::zero(),
        F::zero(),
        F::zero(),
        F::zero(),
    ]);

    assert_eq!(e(1) * e(1), two_const + e(2));
    assert_eq!(e(2) * e(2), two_const + e(4));
    assert_eq!(e(4) * e(4), two_const);
    assert_eq!(e(7) * e(7), two_const - e(2));
    assert_eq!(e(5) * e(7), e(2) - e(4));
}

#[test]
fn fp_ext8_square_matches_mul() {
    let mut rng = StdRng::seed_from_u64(7777);
    for _ in 0..50 {
        let a = R8::random(&mut rng);
        assert_eq!(a.square(), a * a);
    }
}

#[test]
fn fp_ext8_inv() {
    let mut rng = StdRng::seed_from_u64(8888);
    for _ in 0..50 {
        let a = R8::random(&mut rng);
        if !a.is_zero() {
            let inv = a.inverse().unwrap();
            assert_eq!(a * inv, R8::one());
        }
    }
}

#[test]
fn frobenius_fp_ext2_is_conjugation() {
    let x = E2::new(F::from_u64(13), F::from_u64(21));
    assert_eq!(<E2 as FrobeniusExtField<F>>::frobenius_pow(x, 0), x);
    assert_eq!(
        <E2 as FrobeniusExtField<F>>::frobenius_pow(x, 1),
        x.conjugate()
    );
    assert_eq!(<E2 as FrobeniusExtField<F>>::frobenius_pow(x, 2), x);
    assert_eq!(
        <E2 as FrobeniusExtField<F>>::frobenius_inv_pow(x, 1),
        x.conjugate()
    );
}

#[test]
fn canonical_moore_thetas_solve_fp_ext2() {
    validate_canonical_frobenius_thetas::<F, E2>(2).unwrap();
    let thetas = canonical_frobenius_thetas::<F, E2>(2).unwrap();
    let z = [
        E2::new(F::from_u64(3), F::from_u64(5)),
        E2::new(F::from_u64(7), F::from_u64(11)),
    ];
    let r = (0..2)
        .map(|row| {
            thetas
                .iter()
                .zip(z.iter())
                .fold(E2::zero(), |acc, (&theta, &z_h)| {
                    acc + <E2 as FrobeniusExtField<F>>::frobenius_inv_pow(theta, row) * z_h
                })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        solve_frobenius_moore::<F, E2>(&thetas, &r).unwrap(),
        z.to_vec()
    );
}

#[test]
fn canonical_ring_subfield_thetas_are_the_packing_basis() {
    let thetas = canonical_frobenius_thetas::<F, R4>(4).unwrap();
    assert_eq!(
        thetas[0],
        R4::new([F::one(), F::zero(), F::zero(), F::zero()])
    );
    assert_eq!(
        thetas[1],
        R4::new([F::zero(), F::one(), F::zero(), F::zero()])
    );
    assert_eq!(
        thetas[2],
        R4::new([F::zero(), F::zero(), F::one(), F::zero()])
    );
    assert_eq!(
        thetas[3],
        R4::new([F::zero(), F::zero(), F::zero(), F::one()])
    );
    validate_canonical_frobenius_thetas::<F, R4>(4).unwrap();
}

#[test]
fn canonical_fp_ext8_thetas_are_the_packing_basis() {
    let thetas = canonical_frobenius_thetas::<F, R8>(8).unwrap();
    for (idx, theta) in thetas.iter().enumerate().take(8) {
        assert_eq!(
            *theta,
            R8::new(std::array::from_fn(|i| {
                if i == idx {
                    F::one()
                } else {
                    F::zero()
                }
            }))
        );
    }
    validate_canonical_frobenius_thetas::<F, R8>(8).unwrap();
}

#[test]
fn duplicate_moore_theta_rejects() {
    let theta = E2::one();
    let err = solve_frobenius_moore::<F, E2>(&[theta, theta], &[E2::one(), E2::one()])
        .expect_err("duplicate theta should be singular");
    assert!(format!("{err}").contains("singular"));
}

#[test]
fn from_small_int_fp_ext2() {
    let a = E2::from_u64(42);
    assert_eq!(a, E2::new(F::from_u64(42), F::zero()));

    let b = E2::from_i64(-3);
    assert_eq!(b, E2::new(F::from_i64(-3), F::zero()));

    let c = E2::from_u8(7);
    assert_eq!(c, E2::from_u64(7));

    let d = E2::from_u32(100_000);
    assert_eq!(d, E2::from_u64(100_000));
}

#[test]
fn from_small_int_fp_ext4() {
    let a = E4::from_u64(42);
    assert_eq!(
        a,
        E4::new([F::from_u64(42), F::zero(), F::zero(), F::zero(),])
    );

    let b = E4::from_i64(-7);
    assert_eq!(
        b,
        E4::new([F::from_i64(-7), F::zero(), F::zero(), F::zero(),])
    );
}

#[test]
fn ext_field_degree() {
    assert_eq!(<F as ExtField<F>>::EXT_DEGREE, 1);
    assert_eq!(<E2 as ExtField<F>>::EXT_DEGREE, 2);
    assert_eq!(<E4 as ExtField<F>>::EXT_DEGREE, 4);
    assert_eq!(<R4 as ExtField<F>>::EXT_DEGREE, 4);
    assert_eq!(<R8 as ExtField<F>>::EXT_DEGREE, 8);
}

#[test]
fn ext_field_from_base_slice() {
    let c0 = F::from_u64(3);
    let c1 = F::from_u64(5);
    let e2 = E2::from_base_slice(&[c0, c1]);
    assert_eq!(e2, E2::new(c0, c1));

    let c2 = F::from_u64(7);
    let c3 = F::from_u64(11);
    let e4 = E4::from_base_slice(&[c0, c1, c2, c3]);
    assert_eq!(e4, E4::new([c0, c1, c2, c3]));

    let r4 = R4::from_base_slice(&[c0, c1, c2, c3]);
    assert_eq!(r4, R4::new([c0, c1, c2, c3]));

    let c4 = F::from_u64(13);
    let c5 = F::from_u64(17);
    let c6 = F::from_u64(19);
    let c7 = F::from_u64(23);
    let r8 = R8::from_base_slice(&[c0, c1, c2, c3, c4, c5, c6, c7]);
    assert_eq!(r8, R8::new([c0, c1, c2, c3, c4, c5, c6, c7]));
}

#[test]
fn extension_fields_are_array_layouts() {
    assert_eq!(core::mem::size_of::<E2>(), core::mem::size_of::<[F; 2]>());
    assert_eq!(core::mem::align_of::<E2>(), core::mem::align_of::<[F; 2]>());
    assert_eq!(core::mem::size_of::<E4>(), core::mem::size_of::<[F; 4]>());
    assert_eq!(core::mem::align_of::<E4>(), core::mem::align_of::<[F; 4]>());
}

#[test]
fn eq_impl() {
    let a = E2::new(F::from_u64(1), F::from_u64(2));
    let b = E2::new(F::from_u64(1), F::from_u64(2));
    let c = E2::new(F::from_u64(1), F::from_u64(3));
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn fp_ext4_fp32_product_accum_matches_direct_mul() {
    use super::fp_ext4::fp_ext4_mul_to_accum_fp32;
    use crate::unreduced::FpExt4Fp32ProductAccum;
    use crate::Fp32;
    use num_traits::Zero;

    type Fp = Fp32<251>;
    type R4Fp32 = FpExt4<Fp>;

    let mut rng = StdRng::seed_from_u64(0xACC0);
    for _ in 0..200 {
        let a = R4Fp32::random(&mut rng);
        let b = R4Fp32::random(&mut rng);
        let direct = a * b;
        let accum = fp_ext4_mul_to_accum_fp32(a.coeffs, b.coeffs);
        let reduced = R4Fp32::new(accum.reduce::<251>());
        assert_eq!(direct, reduced, "accum mismatch for a={a:?} b={b:?}");
    }

    let zero_accum = FpExt4Fp32ProductAccum::ZERO;
    assert!(zero_accum.is_zero());
    let reduced_zero = R4Fp32::new(zero_accum.reduce::<251>());
    assert_eq!(reduced_zero, R4Fp32::zero());
}

#[test]
fn fp_ext4_fp32_accum_summation() {
    use crate::Fp32;
    use num_traits::Zero;

    type Fp = Fp32<251>;
    type R4Fp32 = FpExt4<Fp>;

    let mut rng = StdRng::seed_from_u64(0xACC1);
    let n = 1024;
    let pairs: Vec<(R4Fp32, R4Fp32)> = (0..n)
        .map(|_| (R4Fp32::random(&mut rng), R4Fp32::random(&mut rng)))
        .collect();

    let direct_sum: R4Fp32 = pairs
        .iter()
        .map(|(a, b)| *a * *b)
        .fold(R4Fp32::zero(), |s, p| s + p);

    let accum_sum = pairs.iter().fold(
        <R4Fp32 as HasUnreducedOps>::ProductAccum::zero(),
        |s, (a, b)| s + a.mul_to_product_accum(*b),
    );
    let reduced = R4Fp32::reduce_product_accum(accum_sum);

    assert_eq!(
        direct_sum, reduced,
        "accumulated sum of {n} products mismatched"
    );
}

#[test]
fn mul_base_to_product_accum_matches_mul_base_sum() {
    use crate::{Fp32, MulBaseUnreduced};
    use num_traits::Zero;

    fn check<Base, Ext>(seed: u64)
    where
        Base: FieldCore + RandomSampling,
        Ext: MulBaseUnreduced<Base> + Zero + RandomSampling,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = 1024;
        let pairs: Vec<(Ext, Base)> = (0..n)
            .map(|_| (Ext::random(&mut rng), Base::random(&mut rng)))
            .collect();

        let direct: Ext = pairs
            .iter()
            .map(|(w, x)| w.mul_base(*x))
            .fold(Ext::zero(), |s, p| s + p);

        let accum = pairs.iter().fold(
            <Ext as HasUnreducedOps>::ProductAccum::zero(),
            |s, (w, x)| s + w.mul_base_to_product_accum(*x),
        );

        assert_eq!(
            direct,
            Ext::reduce_product_accum(accum),
            "delayed base-scaling mismatch over {n} terms"
        );
    }

    // fp_ext4/Fp32 takes the optimal coordinate-scaling override; fp_ext2/Fp64
    // takes the lifted default body. Both defer reduction.
    check::<Fp32<251>, FpExt4<Fp32<251>>>(0xB001);
    check::<F, Ext2<F>>(0xB002);
}

// Regression guard for the `FpExt2<Fp64>` delayed-reduction accumulator. The earlier
// bug dropped the carry into bit 128 because each FpExt2 coefficient (c0 up to ~2^130,
// c1 up to ~2^129) was formed in a single `u128`. It only surfaces with near-`p`
// operands -- products around 2^128 -- which the small-modulus tests never reach,
// so these use the real 2^64-59 prime and cover both FpExt2 configs.
#[test]
fn fp_ext2_fp64_product_accum_matches_direct_mul_large_operands() {
    use crate::Prime64Offset59;

    let mut rng = StdRng::seed_from_u64(0xF64A);
    for _ in 0..256 {
        // TwoNr (IS_NEG_ONE = false): c0 = p00 + 2*p11.
        let a = Ext2::<Prime64Offset59>::random(&mut rng);
        let b = Ext2::<Prime64Offset59>::random(&mut rng);
        assert_eq!(
            a * b,
            Ext2::<Prime64Offset59>::reduce_product_accum(a.mul_to_product_accum(b)),
            "TwoNr accum mismatch a={a:?} b={b:?}"
        );

        // NegOneNr (IS_NEG_ONE = true): c0 = p00 + p^2 - p11.
        let c = FpExt2::<Prime64Offset59, NegOneNr>::random(&mut rng);
        let d = FpExt2::<Prime64Offset59, NegOneNr>::random(&mut rng);
        assert_eq!(
            c * d,
            FpExt2::<Prime64Offset59, NegOneNr>::reduce_product_accum(c.mul_to_product_accum(d)),
            "NegOneNr accum mismatch c={c:?} d={d:?}"
        );
    }
}

#[test]
fn fp_ext2_fp64_accum_summation_large_operands() {
    use crate::Prime64Offset59;
    use num_traits::Zero;

    type E = Ext2<Prime64Offset59>;

    let mut rng = StdRng::seed_from_u64(0xF64C);
    let n = 1024;
    let pairs: Vec<(E, E)> = (0..n)
        .map(|_| (E::random(&mut rng), E::random(&mut rng)))
        .collect();

    let direct_sum: E = pairs
        .iter()
        .map(|(a, b)| *a * *b)
        .fold(E::zero(), |s, p| s + p);

    let accum_sum = pairs
        .iter()
        .fold(<E as HasUnreducedOps>::ProductAccum::zero(), |s, (a, b)| {
            s + a.mul_to_product_accum(*b)
        });

    assert_eq!(
        direct_sum,
        E::reduce_product_accum(accum_sum),
        "fp_ext2<fp64> accumulated sum of {n} products mismatched"
    );
}

// The specialized `FpExt2<Fp64>` EOR fold must be byte-identical to the generic
// `even + r·(odd − even)`. Full-word `Prime64Offset59` exercises the
// carry-folding reduction path (products near 2^128, sum near 2^129);
// sub-word `Prime40Offset195` exercises the no-overflow path. Random operands
// reach carry=1 roughly half the time; the explicit max-coordinate cases pin
// the worst case. Covers both `FpExt2Config`s (TwoNr and NegOneNr).
#[test]
fn fp_ext2_fp64_optimized_fold_matches_generic() {
    use crate::{Prime40Offset195, Prime64Offset59};

    macro_rules! check_fold {
        ($E:ty, $r:expr, $even:expr, $odd:expr) => {{
            let r: $E = $r;
            let even: $E = $even;
            let odd: $E = $odd;
            let generic = even + r * (odd - even);
            let ctx = <$E as HasOptimizedFold>::precompute_fold(r);
            let optimized = <$E as HasOptimizedFold>::fold_one(&ctx, even, odd);
            assert_eq!(
                generic,
                optimized,
                "{} fold mismatch r={r:?} even={even:?} odd={odd:?}",
                stringify!($E)
            );
        }};
    }

    let mut rng = StdRng::seed_from_u64(0xF01D);
    for _ in 0..512 {
        check_fold!(
            Ext2<Prime64Offset59>,
            Ext2::random(&mut rng),
            Ext2::random(&mut rng),
            Ext2::random(&mut rng)
        );
        check_fold!(
            FpExt2<Prime64Offset59, NegOneNr>,
            FpExt2::random(&mut rng),
            FpExt2::random(&mut rng),
            FpExt2::random(&mut rng)
        );
        check_fold!(
            Ext2<Prime40Offset195>,
            Ext2::random(&mut rng),
            Ext2::random(&mut rng),
            Ext2::random(&mut rng)
        );
        check_fold!(
            FpExt2<Prime40Offset195, NegOneNr>,
            FpExt2::random(&mut rng),
            FpExt2::random(&mut rng),
            FpExt2::random(&mut rng)
        );
    }

    // Worst case for the full-word carry fold: all coordinates at p-1, so each
    // base product is ≈ p² ≈ 2^128 and the per-coordinate sum is ≈ 2^129.
    let max64 = Prime64Offset59::zero() - Prime64Offset59::one();
    check_fold!(
        Ext2<Prime64Offset59>,
        Ext2::new(max64, max64),
        Ext2::zero(),
        Ext2::new(max64, max64)
    );
    check_fold!(
        FpExt2<Prime64Offset59, NegOneNr>,
        FpExt2::new(max64, max64),
        FpExt2::zero(),
        FpExt2::new(max64, max64)
    );
}
