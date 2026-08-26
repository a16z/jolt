//! Differential tests for the packed SIMD backends.
//!
//! Packed-vs-scalar equivalence on the native ISA for every width
//! (32/64/128) and every packed extension type, over random inputs and
//! boundary lane patterns (all-max lanes, mixed canonical extremes,
//! single-lane-nonzero); lane-access and slice-helper laws;
//! `WithPacking` associated-type sanity for every field type; `NoPacking`
//! equivalence; and, on aarch64/NEON, the expected lane widths. Scalar
//! arithmetic is verified against independent oracles in the other suites,
//! so packed-vs-scalar equivalence transitively pins the packed kernels.

#![cfg(feature = "solinas")]
#![expect(clippy::unwrap_used, reason = "test code")]

use jolt_field as two;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use two::{
    pseudo_mersenne_modulus, CanonicalEncoding, ExtField, Field, NoPacking, Packed, WithPacking,
};

/// Packed ops must equal per-lane scalar ops (add/sub/mul/square/inverse).
fn check_packed_matches_scalar<PF: Packed>(lhs: &[PF::Scalar], rhs: &[PF::Scalar]) {
    let w = PF::WIDTH;
    assert_eq!(lhs.len() % w, 0);
    assert_eq!(lhs.len(), rhs.len());
    for (la, ra) in lhs.chunks_exact(w).zip(rhs.chunks_exact(w)) {
        let a = PF::from_fn(|i| la[i]);
        let b = PF::from_fn(|i| ra[i]);
        let (sum, diff, prod, sq) = (a + b, a - b, a * b, a.square());
        for i in 0..w {
            assert_eq!(sum.extract(i), la[i] + ra[i], "add lane {i}");
            assert_eq!(diff.extract(i), la[i] - ra[i], "sub lane {i}");
            assert_eq!(prod.extract(i), la[i] * ra[i], "mul lane {i}");
            assert_eq!(sq.extract(i), la[i] * la[i], "square lane {i}");
        }
        let packed_inv = a.inverse();
        let scalar_inv: Option<Vec<_>> = la.iter().map(|x| x.inverse()).collect();
        assert_eq!(packed_inv.is_some(), scalar_inv.is_some(), "inverse parity");
        if let (Some(pi), Some(si)) = (packed_inv, scalar_inv) {
            for (i, s) in si.iter().enumerate() {
                assert_eq!(pi.extract(i), *s, "inverse lane {i}");
            }
        }
    }
}

/// Boundary lane patterns for a prime field with modulus `p`: all-max
/// lanes, mixed canonical extremes, and single-lane-nonzero, crossed.
fn check_boundary_patterns<PF>(p: u128)
where
    PF: Packed,
    PF::Scalar: CanonicalEncoding,
{
    let w = PF::WIDTH;
    let f = |v: u128| PF::Scalar::from_u128_checked(v).unwrap();
    let all_max = vec![f(p - 1); w];
    let mixed: Vec<_> = (0..w).map(|i| f([0, 1, p - 2, p - 1][i % 4])).collect();
    check_packed_matches_scalar::<PF>(&all_max, &all_max);
    check_packed_matches_scalar::<PF>(&mixed, &all_max);
    check_packed_matches_scalar::<PF>(&all_max, &mixed);
    check_packed_matches_scalar::<PF>(&mixed, &mixed);
    for lane in 0..w {
        let single: Vec<_> = (0..w)
            .map(|i| if i == lane { f(p - 1) } else { f(0) })
            .collect();
        check_packed_matches_scalar::<PF>(&single, &all_max);
        check_packed_matches_scalar::<PF>(&single, &single);
    }
}

/// Random packed-vs-scalar equivalence plus boundary patterns.
fn check_prime_field<PF>(p: u128, seed: u64)
where
    PF: Packed,
    PF::Scalar: CanonicalEncoding,
{
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let n = PF::WIDTH * 16;
    let lhs: Vec<PF::Scalar> = (0..n).map(|_| Field::random(&mut rng)).collect();
    let rhs: Vec<PF::Scalar> = (0..n).map(|_| Field::random(&mut rng)).collect();
    check_packed_matches_scalar::<PF>(&lhs, &rhs);
    check_boundary_patterns::<PF>(p);
}

/// Packed Fp64 arithmetic checked directly against integer modular
/// arithmetic, without routing the expectation through the scalar kernel.
fn check_fp64_integer_oracle<const P: u64, PF>(lhs: &[u64], rhs: &[u64])
where
    PF: Packed<Scalar = two::Fp64<P>>,
{
    assert_eq!(lhs.len(), PF::WIDTH);
    assert_eq!(rhs.len(), PF::WIDTH);
    let a = PF::from_fn(|i| two::Fp64::<P>::from_u128_checked(lhs[i] as u128).unwrap());
    let b = PF::from_fn(|i| two::Fp64::<P>::from_u128_checked(rhs[i] as u128).unwrap());
    let (sum, diff, product) = (a + b, a - b, a * b);
    for lane in 0..PF::WIDTH {
        let x = lhs[lane] as u128;
        let y = rhs[lane] as u128;
        let p = P as u128;
        assert_eq!(sum.extract(lane).to_u128_checked(), Some((x + y) % p));
        assert_eq!(diff.extract(lane).to_u128_checked(), Some((x + p - y) % p));
        assert_eq!(product.extract(lane).to_u128_checked(), Some((x * y) % p));
    }
}

fn check_fp64_wide_sub_word<const P: u64>(seed: u64)
where
    two::Fp64<P>: WithPacking,
{
    type F<const Q: u64> = two::Fp64<Q>;
    let boundary = [0, 1, 2, (P - 1) / 2, P - 2, P - 1];
    for &x in &boundary {
        for &y in &boundary {
            check_fp64_integer_oracle::<P, <F<P> as WithPacking>::Packing>(
                &vec![x; <F<P> as WithPacking>::Packing::WIDTH],
                &vec![y; <F<P> as WithPacking>::Packing::WIDTH],
            );
        }
    }
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    for _ in 0..1024 {
        let lhs: Vec<u64> = (0..<F<P> as WithPacking>::Packing::WIDTH)
            .map(|_| rng.gen::<u64>() % P)
            .collect();
        let rhs: Vec<u64> = (0..<F<P> as WithPacking>::Packing::WIDTH)
            .map(|_| rng.gen::<u64>() % P)
            .collect();
        check_fp64_integer_oracle::<P, <F<P> as WithPacking>::Packing>(&lhs, &rhs);
    }
}

/// `from_fn`/`extract`/`broadcast` and the slice-helper laws.
fn check_lane_laws<PF: Packed>(vals: &[PF::Scalar]) {
    let w = PF::WIDTH;
    assert!(w >= 1);
    let p = PF::from_fn(|i| vals[i % vals.len()]);
    for i in 0..w {
        assert_eq!(
            p.extract(i),
            vals[i % vals.len()],
            "from_fn/extract lane {i}"
        );
    }
    let b = PF::broadcast(vals[0]);
    for i in 0..w {
        assert_eq!(b.extract(i), vals[0], "broadcast lane {i}");
    }
    let len = w * 3 + (w - 1);
    let buf: Vec<_> = (0..len).map(|i| vals[i % vals.len()]).collect();
    let (packed, suffix) = PF::pack_slice_with_suffix(&buf);
    assert_eq!(packed.len(), 3);
    assert_eq!(suffix.len(), w - 1);
    let mut out = PF::unpack_slice(&packed);
    out.extend_from_slice(suffix);
    assert_eq!(out, buf, "pack/unpack roundtrip");
    assert_eq!(PF::pack_slice(&buf[..w * 3]).len(), 3);
}

/// `WithPacking` associated-type sanity: the packing's scalar is the field
/// itself and the lane laws hold.
fn check_with_packing<F: WithPacking>(seed: u64) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let vals: Vec<F> = (0..<F::Packing as Packed>::WIDTH.max(4))
        .map(|_| Field::random(&mut rng))
        .collect();
    check_lane_laws::<F::Packing>(&vals);
}

fn pm(bits: u32, offset: u128) -> u128 {
    pseudo_mersenne_modulus(bits, offset).unwrap()
}

#[test]
fn packed_fp32_matches_scalar() {
    check_prime_field::<<two::Prime24Offset3 as WithPacking>::Packing>(pm(24, 3), 0x2401);
    check_prime_field::<<two::Prime30Offset35 as WithPacking>::Packing>(pm(30, 35), 0x3001);
    check_prime_field::<<two::Prime31Offset19 as WithPacking>::Packing>(pm(31, 19), 0x3101);
    check_prime_field::<<two::Prime32Offset99 as WithPacking>::Packing>(pm(32, 99), 0x3201);
}

#[test]
fn packed_fp64_matches_scalar() {
    check_prime_field::<<two::Prime40Offset195 as WithPacking>::Packing>(pm(40, 195), 0x4001);
    check_prime_field::<<two::Prime48Offset59 as WithPacking>::Packing>(pm(48, 59), 0x4801);
    check_prime_field::<<two::Prime56Offset27 as WithPacking>::Packing>(pm(56, 27), 0x5601);
    check_prime_field::<<two::Prime64Offset59 as WithPacking>::Packing>(pm(64, 59), 0x6401);
}

#[test]
fn packed_fp64_wide_sub_word_matches_integer_reference() {
    check_fp64_wide_sub_word::<{ (1u64 << 63) - 259 }>(0xAA63_0259);
    check_fp64_wide_sub_word::<{ (1u64 << 63) - 25 }>(0xAA63_0025);
}

#[test]
fn packed_fp128_matches_scalar() {
    check_prime_field::<<two::Prime128Offset275 as WithPacking>::Packing>(pm(128, 275), 0x12801);
    check_prime_field::<<two::Prime128OffsetA7F7 as WithPacking>::Packing>(
        pm(128, 0xFFFF_A7F7),
        0x12804,
    );
}

/// Extension boundary lanes: all-max coefficient vectors and mixed extremes.
fn check_ext_boundaries<PF, F>(p: u128)
where
    F: Field + CanonicalEncoding,
    PF: Packed,
    PF::Scalar: ExtField<F>,
{
    let f = |v: u128| F::from_u128_checked(v).unwrap();
    let d = <PF::Scalar as ExtField<F>>::DEGREE;
    let all_max = PF::Scalar::from_base_slice(&vec![f(p - 1); d]);
    let mixed = PF::Scalar::from_base_slice(
        &(0..d)
            .map(|i| f([0, 1, p - 2, p - 1][i % 4]))
            .collect::<Vec<_>>(),
    );
    let w = PF::WIDTH;
    let max_lanes = vec![all_max; w];
    let mixed_lanes: Vec<_> = (0..w)
        .map(|i| if i % 2 == 0 { all_max } else { mixed })
        .collect();
    check_packed_matches_scalar::<PF>(&max_lanes, &max_lanes);
    check_packed_matches_scalar::<PF>(&mixed_lanes, &max_lanes);
    check_packed_matches_scalar::<PF>(&mixed_lanes, &mixed_lanes);
}

/// Packed extension towers vs scalar extension arithmetic.
fn check_ext_field<PF, F>(p: u128, seed: u64)
where
    F: Field + CanonicalEncoding,
    PF: Packed,
    PF::Scalar: ExtField<F>,
{
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let n = PF::WIDTH * 8;
    let lhs: Vec<PF::Scalar> = (0..n).map(|_| Field::random(&mut rng)).collect();
    let rhs: Vec<PF::Scalar> = (0..n).map(|_| Field::random(&mut rng)).collect();
    check_packed_matches_scalar::<PF>(&lhs, &rhs);
    check_ext_boundaries::<PF, F>(p);
}

#[test]
fn packed_ext2_matches_scalar() {
    type F32 = two::Prime32Offset99;
    type E2 = two::Ext2<F32>;
    check_ext_field::<<E2 as WithPacking>::Packing, F32>(pm(32, 99), 0xE201);
    // NegOneNr is a genuine field over p ≡ 3 (mod 4).
    type F251 = two::Fp32<251>;
    type E2Neg = two::FpExt2<F251, two::NegOneNr>;
    check_ext_field::<<E2Neg as WithPacking>::Packing, F251>(251, 0xE202);
    type F64 = two::Prime64Offset59;
    check_ext_field::<<two::Ext2<F64> as WithPacking>::Packing, F64>(pm(64, 59), 0xE203);
    type F128 = two::Prime128Offset275;
    check_ext_field::<<two::Ext2<F128> as WithPacking>::Packing, F128>(pm(128, 275), 0xE204);

    // Fused NR=2 paths: a wide 63-bit base product, a narrow base reducer
    // whose three-product coefficient needs the wide fold, and a large
    // offset that exercises the full-low-word SIMD correction multiply.
    type Wide = two::Fp64<{ (1u64 << 63) - 259 }>;
    check_ext_field::<<two::Ext2<Wide> as WithPacking>::Packing, Wide>((1u128 << 63) - 259, 0xE205);
    type NarrowBase = two::Fp64<{ (1u64 << 58) - 27 }>;
    check_ext_field::<<two::Ext2<NarrowBase> as WithPacking>::Packing, NarrowBase>(
        (1u128 << 58) - 27,
        0xE206,
    );
    type LargeOffset = two::Fp64<{ (1u64 << 63) - 1_500_000_051 }>;
    check_ext_field::<<two::Ext2<LargeOffset> as WithPacking>::Packing, LargeOffset>(
        (1u128 << 63) - 1_500_000_051,
        0xE207,
    );
}

#[test]
fn packed_ext4_matches_scalar() {
    type F32 = two::Prime32Offset99;
    check_ext_field::<<two::FpExt4<F32> as WithPacking>::Packing, F32>(pm(32, 99), 0xE401);
    type F31 = two::Prime31Offset19;
    check_ext_field::<<two::FpExt4<F31> as WithPacking>::Packing, F31>(pm(31, 19), 0xE402);
    type F64 = two::Prime64Offset59;
    check_ext_field::<<two::FpExt4<F64> as WithPacking>::Packing, F64>(pm(64, 59), 0xE403);
    type F128 = two::Prime128OffsetA7F7;
    check_ext_field::<<two::FpExt4<F128> as WithPacking>::Packing, F128>(
        pm(128, 0xFFFF_A7F7),
        0xE404,
    );
}

#[test]
fn packed_ext8_matches_scalar() {
    type F32 = two::Prime32Offset99;
    check_ext_field::<<two::FpExt8<F32> as WithPacking>::Packing, F32>(pm(32, 99), 0xE801);
    type F64 = two::Prime48Offset59;
    check_ext_field::<<two::FpExt8<F64> as WithPacking>::Packing, F64>(pm(48, 59), 0xE802);
    type F128 = two::Prime128Offset275;
    check_ext_field::<<two::FpExt8<F128> as WithPacking>::Packing, F128>(pm(128, 275), 0xE803);
}

#[test]
fn with_packing_associated_types() {
    check_with_packing::<two::Prime24Offset3>(0x5101);
    check_with_packing::<two::Prime30Offset35>(0x5102);
    check_with_packing::<two::Prime31Offset19>(0x5103);
    check_with_packing::<two::Prime32Offset99>(0x5104);
    check_with_packing::<two::Prime40Offset195>(0x5105);
    check_with_packing::<two::Prime48Offset59>(0x5106);
    check_with_packing::<two::Prime56Offset27>(0x5107);
    check_with_packing::<two::Prime64Offset59>(0x5108);
    check_with_packing::<two::Prime128Offset275>(0x5109);
    check_with_packing::<two::Prime128OffsetA7F7>(0x510c);
    check_with_packing::<two::Ext2<two::Prime32Offset99>>(0x510d);
    check_with_packing::<two::FpExt2<two::Fp32<251>, two::NegOneNr>>(0x510e);
    check_with_packing::<two::FpExt4<two::Prime32Offset99>>(0x510f);
    check_with_packing::<two::FpExt4<two::Prime128OffsetA7F7>>(0x5110);
    check_with_packing::<two::FpExt8<two::Prime32Offset99>>(0x5111);
    check_with_packing::<two::FpExt8<two::Prime64Offset59>>(0x5112);
}

#[test]
fn no_packing_equivalence() {
    // A type with no SIMD path: NoPacking over a word field, exercised
    // through the same laws and differentials as the SIMD backends.
    type PF = NoPacking<two::Prime32Offset99>;
    check_prime_field::<PF>(pm(32, 99), 0x0001);
    let mut rng = ChaCha20Rng::seed_from_u64(0x0002);
    let vals: Vec<two::Prime32Offset99> = (0..4).map(|_| Field::random(&mut rng)).collect();
    check_lane_laws::<PF>(&vals);
    assert_eq!(PF::WIDTH, 1);
}

/// Expected NEON lane widths on aarch64 (previously asserted against
/// jolt-field's packed types; the widths are part of the layout contract).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn neon_lane_widths() {
    assert_eq!(<two::Prime32Offset99 as WithPacking>::Packing::WIDTH, 4);
    assert_eq!(<two::Prime64Offset59 as WithPacking>::Packing::WIDTH, 2);
    assert_eq!(<two::Prime128Offset275 as WithPacking>::Packing::WIDTH, 2);
}
