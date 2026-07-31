//! Differential tests for the packed SIMD backends.
//!
//! Packed-vs-scalar equivalence on the native ISA for every width
//! (32/64/128) and every packed extension type, over random inputs and
//! boundary lane patterns (all-max lanes, mixed canonical extremes,
//! single-lane-nonzero); lane-access and slice-helper laws;
//! `WithPacking` associated-type sanity for every field type; `NoPacking`
//! equivalence; and, on aarch64/NEON, lane-exact differentials against
//! jolt-field's packed types (including the packed ext2 kernel hook and
//! the fused degree-4/8 kernels).

#![cfg(feature = "solinas")]
#![expect(clippy::unwrap_used, reason = "test code")]

use jolt_field_two as two;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use two::{
    pseudo_mersenne_modulus, CanonicalEncoding, ExtField, Field, NoPacking, Packed, Ring,
    WithPacking,
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
fn packed_fp128_matches_scalar() {
    check_prime_field::<<two::Prime128Offset275 as WithPacking>::Packing>(pm(128, 275), 0x12801);
    check_prime_field::<<two::Prime128Offset159 as WithPacking>::Packing>(pm(128, 159), 0x12802);
    check_prime_field::<<two::Prime128Offset2355 as WithPacking>::Packing>(pm(128, 2355), 0x12803);
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
    check_with_packing::<two::Prime128Offset159>(0x510a);
    check_with_packing::<two::Prime128Offset2355>(0x510b);
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

/// Lane-exact differentials against jolt-field's packed types on the
/// native NEON backend: same canonical inputs, same lane results.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod baseline_diff {
    use super::*;
    use jolt_field as base;

    use base::packed::{HasPacking, PackedField};
    use base::{CanonicalField, FromPrimitiveInt};
    use rand::Rng;

    /// Random + boundary lane vectors of canonical representatives.
    fn canonical_inputs(p: u128, w: usize, seed: u64) -> Vec<u128> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut out = vec![p - 1; w];
        out.extend((0..w).map(|i| [0, 1, p - 2, p - 1][i % 4]));
        for lane in 0..w {
            out.extend((0..w).map(|i| if i == lane { p - 1 } else { 0 }));
        }
        out.extend((0..w * 16).map(|_| rng.gen::<u128>() % p));
        out
    }

    fn diff_prime<NP, BP>(p: u128, seed: u64)
    where
        NP: Packed,
        NP::Scalar: CanonicalEncoding,
        BP: PackedField,
        BP::Scalar: CanonicalField + FromPrimitiveInt,
    {
        assert_eq!(NP::WIDTH, BP::WIDTH, "lane width mismatch vs baseline");
        let w = NP::WIDTH;
        let lhs = canonical_inputs(p, w, seed);
        let rhs = canonical_inputs(p, w, seed ^ 0xFFFF);
        for (la, ra) in lhs.chunks_exact(w).zip(rhs.chunks_exact(w)) {
            let na = NP::from_fn(|i| <NP::Scalar as Ring>::from_u128(la[i]));
            let nb = NP::from_fn(|i| <NP::Scalar as Ring>::from_u128(ra[i]));
            let ba = BP::from_fn(|i| BP::Scalar::from_u128(la[i]));
            let bb = BP::from_fn(|i| BP::Scalar::from_u128(ra[i]));
            let pairs = [(na + nb, ba + bb), (na - nb, ba - bb), (na * nb, ba * bb)];
            for (op, (n, b)) in ["add", "sub", "mul"].iter().zip(pairs) {
                for i in 0..w {
                    assert_eq!(
                        n.extract(i).to_u128_checked().unwrap(),
                        b.extract(i).to_canonical_u128(),
                        "{op} lane {i} differs from baseline"
                    );
                }
            }
        }
    }

    #[test]
    fn fp32_lanes_match_baseline() {
        assert_eq!(<two::Prime32Offset99 as WithPacking>::Packing::WIDTH, 4);
        diff_prime::<
            <two::Prime24Offset3 as WithPacking>::Packing,
            <base::Prime24Offset3 as HasPacking>::Packing,
        >(pm(24, 3), 0xB3201);
        diff_prime::<
            <two::Prime30Offset35 as WithPacking>::Packing,
            <base::Prime30Offset35 as HasPacking>::Packing,
        >(pm(30, 35), 0xB3202);
        diff_prime::<
            <two::Prime31Offset19 as WithPacking>::Packing,
            <base::Prime31Offset19 as HasPacking>::Packing,
        >(pm(31, 19), 0xB3203);
        diff_prime::<
            <two::Prime32Offset99 as WithPacking>::Packing,
            <base::Prime32Offset99 as HasPacking>::Packing,
        >(pm(32, 99), 0xB3204);
    }

    #[test]
    fn fp64_lanes_match_baseline() {
        assert_eq!(<two::Prime64Offset59 as WithPacking>::Packing::WIDTH, 2);
        diff_prime::<
            <two::Prime40Offset195 as WithPacking>::Packing,
            <base::Prime40Offset195 as HasPacking>::Packing,
        >(pm(40, 195), 0xB6401);
        diff_prime::<
            <two::Prime48Offset59 as WithPacking>::Packing,
            <base::Prime48Offset59 as HasPacking>::Packing,
        >(pm(48, 59), 0xB6402);
        diff_prime::<
            <two::Prime56Offset27 as WithPacking>::Packing,
            <base::Prime56Offset27 as HasPacking>::Packing,
        >(pm(56, 27), 0xB6403);
        diff_prime::<
            <two::Prime64Offset59 as WithPacking>::Packing,
            <base::Prime64Offset59 as HasPacking>::Packing,
        >(pm(64, 59), 0xB6404);
    }

    #[test]
    fn fp128_lanes_match_baseline() {
        assert_eq!(<two::Prime128Offset275 as WithPacking>::Packing::WIDTH, 2);
        diff_prime::<
            <two::Prime128Offset275 as WithPacking>::Packing,
            <base::Prime128Offset275 as HasPacking>::Packing,
        >(pm(128, 275), 0x00B1_2801);
        diff_prime::<
            <two::Prime128OffsetA7F7 as WithPacking>::Packing,
            <base::Prime128OffsetA7F7 as HasPacking>::Packing,
        >(pm(128, 0xFFFF_A7F7), 0x00B1_2802);
    }

    /// Coefficient matrices for extension lanes.
    fn coeff_lanes<const D: usize>(p: u128, w: usize, seed: u64) -> Vec<[u128; D]> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut out = vec![[p - 1; D]; w];
        out.extend(
            (0..w).map(|lane| std::array::from_fn(|j| [0, 1, p - 2, p - 1][(lane + j) % 4])),
        );
        out.extend((0..w * 8).map(|_| std::array::from_fn(|_| rng.gen::<u128>() % p)));
        out
    }

    fn diff_ext<const D: usize, NP, BP>(
        p: u128,
        seed: u64,
        mk_new: impl Fn([u128; D]) -> NP::Scalar,
        mk_base: impl Fn([u128; D]) -> BP::Scalar,
        canon_new: impl Fn(&NP::Scalar) -> Vec<u128>,
        canon_base: impl Fn(&BP::Scalar) -> Vec<u128>,
    ) where
        NP: Packed,
        BP: PackedField,
    {
        assert_eq!(NP::WIDTH, BP::WIDTH, "ext lane width mismatch vs baseline");
        let w = NP::WIDTH;
        let lhs = coeff_lanes::<D>(p, w, seed);
        let rhs = coeff_lanes::<D>(p, w, seed ^ 0xFFFF);
        for (la, ra) in lhs.chunks_exact(w).zip(rhs.chunks_exact(w)) {
            let na = NP::from_fn(|i| mk_new(la[i]));
            let nb = NP::from_fn(|i| mk_new(ra[i]));
            let ba = BP::from_fn(|i| mk_base(la[i]));
            let bb = BP::from_fn(|i| mk_base(ra[i]));
            let pairs = [
                (na + nb, ba + bb),
                (na - nb, ba - bb),
                (na * nb, ba * bb),
                (na.square(), ba.square()),
            ];
            for (op, (n, b)) in ["add", "sub", "mul", "square"].iter().zip(pairs) {
                for i in 0..w {
                    assert_eq!(
                        canon_new(&n.extract(i)),
                        canon_base(&b.extract(i)),
                        "ext {op} lane {i} differs from baseline"
                    );
                }
            }
        }
    }

    /// The packed ext2 kernel hook, differentially vs the baseline hook.
    #[test]
    fn ext2_kernel_matches_baseline() {
        type NF = two::Prime32Offset99;
        type BF = base::Prime32Offset99;
        type NP = <two::Ext2<NF> as WithPacking>::Packing;
        type BP = base::packed::PackedFpExt2<BF, base::TwoNr, <BF as HasPacking>::Packing>;
        diff_ext::<2, NP, BP>(
            pm(32, 99),
            0xBE201,
            |c| two::FpExt2::new(Ring::from_u128(c[0]), Ring::from_u128(c[1])),
            |c| base::FpExt2::new(BF::from_u128(c[0]), BF::from_u128(c[1])),
            |x| {
                x.coeffs
                    .iter()
                    .map(|c| c.to_u128_checked().unwrap())
                    .collect()
            },
            |x| x.coeffs.iter().map(|c| c.to_canonical_u128()).collect(),
        );
        type NF251 = two::Fp32<251>;
        type BF251 = base::Fp32<251>;
        type NPn = <two::FpExt2<NF251, two::NegOneNr> as WithPacking>::Packing;
        type BPn =
            base::packed::PackedFpExt2<BF251, base::NegOneNr, <BF251 as HasPacking>::Packing>;
        diff_ext::<2, NPn, BPn>(
            251,
            0xBE202,
            |c| two::FpExt2::new(Ring::from_u128(c[0]), Ring::from_u128(c[1])),
            |c| base::FpExt2::new(BF251::from_u128(c[0]), BF251::from_u128(c[1])),
            |x| {
                x.coeffs
                    .iter()
                    .map(|c| c.to_u128_checked().unwrap())
                    .collect()
            },
            |x| x.coeffs.iter().map(|c| c.to_canonical_u128()).collect(),
        );
    }

    /// Fused degree-4 kernels (dot products on fp32) vs the baseline NEON
    /// kernels, plus the schedule-default paths on wider bases.
    #[test]
    fn ext4_kernels_match_baseline() {
        macro_rules! diff_ext4 {
            ($nf:ty, $bf:ty, $p:expr, $seed:expr) => {
                diff_ext::<
                    4,
                    <two::FpExt4<$nf> as WithPacking>::Packing,
                    base::packed::PackedFpExt4<$bf, <$bf as HasPacking>::Packing>,
                >(
                    $p,
                    $seed,
                    |c| two::FpExt4::new(c.map(Ring::from_u128)),
                    |c| base::FpExt4::new(c.map(<$bf>::from_u128)),
                    |x| {
                        x.coeffs
                            .iter()
                            .map(|c| c.to_u128_checked().unwrap())
                            .collect()
                    },
                    |x| x.coeffs.iter().map(|c| c.to_canonical_u128()).collect(),
                );
            };
        }
        diff_ext4!(
            two::Prime32Offset99,
            base::Prime32Offset99,
            pm(32, 99),
            0xBE401
        );
        diff_ext4!(
            two::Prime31Offset19,
            base::Prime31Offset19,
            pm(31, 19),
            0xBE402
        );
        diff_ext4!(
            two::Prime64Offset59,
            base::Prime64Offset59,
            pm(64, 59),
            0xBE403
        );
        diff_ext4!(
            two::Prime128Offset275,
            base::Prime128Offset275,
            pm(128, 275),
            0xBE404
        );
    }

    #[test]
    fn ext8_kernels_match_baseline() {
        macro_rules! diff_ext8 {
            ($nf:ty, $bf:ty, $p:expr, $seed:expr) => {
                diff_ext::<
                    8,
                    <two::FpExt8<$nf> as WithPacking>::Packing,
                    base::packed::PackedFpExt8<$bf, <$bf as HasPacking>::Packing>,
                >(
                    $p,
                    $seed,
                    |c| two::FpExt8::new(c.map(Ring::from_u128)),
                    |c| base::FpExt8::new(c.map(<$bf>::from_u128)),
                    |x| {
                        x.coeffs
                            .iter()
                            .map(|c| c.to_u128_checked().unwrap())
                            .collect()
                    },
                    |x| x.coeffs.iter().map(|c| c.to_canonical_u128()).collect(),
                );
            };
        }
        diff_ext8!(
            two::Prime32Offset99,
            base::Prime32Offset99,
            pm(32, 99),
            0xBE801
        );
        diff_ext8!(
            two::Prime64Offset59,
            base::Prime64Offset59,
            pm(64, 59),
            0xBE802
        );
        diff_ext8!(
            two::Prime128Offset275,
            base::Prime128Offset275,
            pm(128, 275),
            0xBE803
        );
    }
}
