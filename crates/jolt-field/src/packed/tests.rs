#![expect(
    clippy::unreadable_literal,
    reason = "packed regression vectors retain their generated decimal form"
)]

use super::{HasPacking, PackedField, PackedValue};
use crate::{
    CanonicalField, FieldCore, Fp32, Prime128Offset275, Prime24Offset3, Prime31Offset19,
    Prime32Offset99, Prime40Offset195, Prime64Offset59,
};
use rand::{rngs::StdRng, RngCore, SeedableRng};

fn rand_u128<R: RngCore>(rng: &mut R) -> u128 {
    let lo = rng.next_u64() as u128;
    let hi = rng.next_u64() as u128;
    lo | (hi << 64)
}

fn check_packed_add_sub_mul<F, PF>(seed: u64)
where
    F: FieldCore + PartialEq + std::fmt::Debug,
    PF: PackedField<Scalar = F> + PackedValue<Value = F>,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let len = PF::WIDTH * 17 + 3;
    let lhs: Vec<F> = (0..len).map(|_| FieldCore::random(&mut rng)).collect();
    let rhs: Vec<F> = (0..len).map(|_| FieldCore::random(&mut rng)).collect();

    let (lhs_p, lhs_s) = PF::pack_slice_with_suffix(&lhs);
    let (rhs_p, rhs_s) = PF::pack_slice_with_suffix(&rhs);

    let add_p: Vec<PF> = lhs_p
        .iter()
        .zip(rhs_p.iter())
        .map(|(&a, &b)| a + b)
        .collect();
    let sub_p: Vec<PF> = lhs_p
        .iter()
        .zip(rhs_p.iter())
        .map(|(&a, &b)| a - b)
        .collect();
    let mul_p: Vec<PF> = lhs_p
        .iter()
        .zip(rhs_p.iter())
        .map(|(&a, &b)| a * b)
        .collect();

    let mut add_out = PF::unpack_slice(&add_p);
    let mut sub_out = PF::unpack_slice(&sub_p);
    let mut mul_out = PF::unpack_slice(&mul_p);

    for (&a, &b) in lhs_s.iter().zip(rhs_s.iter()) {
        add_out.push(a + b);
        sub_out.push(a - b);
        mul_out.push(a * b);
    }

    for i in 0..len {
        assert_eq!(
            add_out[i],
            lhs[i] + rhs[i],
            "packed add mismatch at lane {i}"
        );
        assert_eq!(
            sub_out[i],
            lhs[i] - rhs[i],
            "packed sub mismatch at lane {i}"
        );
        assert_eq!(
            mul_out[i],
            lhs[i] * rhs[i],
            "packed mul mismatch at lane {i}"
        );
    }
}

fn check_broadcast_roundtrip<F, PF>(val: F)
where
    F: FieldCore + PartialEq + std::fmt::Debug,
    PF: PackedField<Scalar = F> + PackedValue<Value = F>,
{
    let p = PF::broadcast(val);
    for lane in 0..PF::WIDTH {
        assert_eq!(p.extract(lane), val);
    }
}

fn check_packed_fp32_edge_lanes<const P: u32, PF>()
where
    PF: PackedField<Scalar = Fp32<P>> + PackedValue<Value = Fp32<P>>,
{
    let p_minus_one = Fp32::<P>::from_canonical_u32(P - 1);
    let p_minus_two = Fp32::<P>::from_canonical_u32(P - 2);
    let values = [
        Fp32::<P>::zero(),
        Fp32::<P>::one(),
        p_minus_two,
        p_minus_one,
    ];
    let a = PF::from_fn(|i| values[i % values.len()]);
    let b = PF::from_fn(|i| values[(i + 1) % values.len()]);

    let add = a + b;
    let sub = a - b;
    let mul = a * b;

    for lane in 0..PF::WIDTH {
        let lhs = values[lane % values.len()];
        let rhs = values[(lane + 1) % values.len()];
        assert_eq!(add.extract(lane), lhs + rhs, "packed add edge lane {lane}");
        assert_eq!(sub.extract(lane), lhs - rhs, "packed sub edge lane {lane}");
        assert_eq!(mul.extract(lane), lhs * rhs, "packed mul edge lane {lane}");
    }
}

#[test]
fn packed_fp128_add_sub_mul_match_scalar() {
    type F = Prime128Offset275;
    type PF = <F as HasPacking>::Packing;

    let mut rng = StdRng::seed_from_u64(0x55aa_4422_1177_0033);
    let len = PF::WIDTH * 17 + 3;
    let lhs: Vec<F> = (0..len)
        .map(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();
    let rhs: Vec<F> = (0..len)
        .map(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();

    let (lhs_p, lhs_s) = PF::pack_slice_with_suffix(&lhs);
    let (rhs_p, rhs_s) = PF::pack_slice_with_suffix(&rhs);

    let add_p: Vec<PF> = lhs_p
        .iter()
        .zip(rhs_p.iter())
        .map(|(&a, &b)| a + b)
        .collect();
    let sub_p: Vec<PF> = lhs_p
        .iter()
        .zip(rhs_p.iter())
        .map(|(&a, &b)| a - b)
        .collect();
    let mul_p: Vec<PF> = lhs_p
        .iter()
        .zip(rhs_p.iter())
        .map(|(&a, &b)| a * b)
        .collect();

    let mut add_out = PF::unpack_slice(&add_p);
    let mut sub_out = PF::unpack_slice(&sub_p);
    let mut mul_out = PF::unpack_slice(&mul_p);

    for (&a, &b) in lhs_s.iter().zip(rhs_s.iter()) {
        add_out.push(a + b);
        sub_out.push(a - b);
        mul_out.push(a * b);
    }

    for i in 0..len {
        assert_eq!(
            add_out[i],
            lhs[i] + rhs[i],
            "packed add mismatch at lane {i}"
        );
        assert_eq!(
            sub_out[i],
            lhs[i] - rhs[i],
            "packed sub mismatch at lane {i}"
        );
        assert_eq!(
            mul_out[i],
            lhs[i] * rhs[i],
            "packed mul mismatch at lane {i}"
        );
    }
}

#[test]
fn fp128_broadcast_and_extract_roundtrip() {
    type F = Prime128Offset275;
    type PF = <F as HasPacking>::Packing;
    check_broadcast_roundtrip::<F, PF>(F::from_u64(42));
}

#[test]
fn packed_fp32_24b_add_sub_mul() {
    type F = Prime24Offset3;
    type PF = <F as HasPacking>::Packing;
    check_packed_add_sub_mul::<F, PF>(0xaa24_bb24_cc24_dd24);
}

#[test]
fn packed_fp32_31b_add_sub_mul() {
    type F = Prime31Offset19;
    type PF = <F as HasPacking>::Packing;
    check_packed_add_sub_mul::<F, PF>(0xaa31_bb31_cc31_dd31);
}

#[test]
fn packed_fp32_31b_edge_lanes() {
    type F = Prime31Offset19;
    type PF = <F as HasPacking>::Packing;
    check_packed_fp32_edge_lanes::<{ crate::prime::pseudo_mersenne::PRIME31_OFFSET19_MODULUS }, PF>(
    );
}

#[test]
fn packed_mersenne31_edge_lanes() {
    type F = Fp32<{ (1u32 << 31) - 1 }>;
    type PF = <F as HasPacking>::Packing;
    check_packed_fp32_edge_lanes::<{ (1u32 << 31) - 1 }, PF>();
}

/// Stress the 31-bit pseudo-Mersenne (`C > 1`) packed multiply against the
/// scalar reference across boundary values and a large random sweep. This
/// confirms (does not justify) the exact correctness proof on
/// `mul_pmersenne31_vec`: the tightest cases are `z = (P-1)^2` and inputs
/// that drive the second fold's `t'` toward `2P`.
#[test]
fn packed_fp32_31b_mul_matches_scalar_stress() {
    type F = Prime31Offset19;
    type PF = <F as HasPacking>::Packing;
    const P: u32 = crate::prime::pseudo_mersenne::PRIME31_OFFSET19_MODULUS;

    let boundary = [
        0u32,
        1,
        2,
        3,
        19,
        1 << 15,
        1 << 30,
        (1 << 30) + 1,
        (P - 1) / 2,
        P - 3,
        P - 2,
        P - 1,
    ];

    let mut inputs: Vec<F> = boundary.iter().map(|&v| F::from_canonical_u32(v)).collect();
    let mut rng = StdRng::seed_from_u64(0x31be_19ca_fe00_1357);
    for _ in 0..(1 << 16) {
        inputs.push(F::from_canonical_u32(rng.next_u32() % P));
    }

    let lhs: Vec<F> = inputs.clone();
    let rhs: Vec<F> = {
        let mut r = inputs.clone();
        r.rotate_left(1);
        r
    };

    let (lhs_p, lhs_s) = PF::pack_slice_with_suffix(&lhs);
    let (rhs_p, rhs_s) = PF::pack_slice_with_suffix(&rhs);
    let mul_p: Vec<PF> = lhs_p
        .iter()
        .zip(rhs_p.iter())
        .map(|(&a, &b)| a * b)
        .collect();
    let mut mul_out = PF::unpack_slice(&mul_p);
    for (&a, &b) in lhs_s.iter().zip(rhs_s.iter()) {
        mul_out.push(a * b);
    }
    for i in 0..lhs.len() {
        assert_eq!(mul_out[i], lhs[i] * rhs[i], "packed mul mismatch at {i}");
    }

    // Full boundary x boundary cross product (every tight combination).
    for &x in &boundary {
        for &y in &boundary {
            let a = PF::broadcast(F::from_canonical_u32(x));
            let b = PF::broadcast(F::from_canonical_u32(y));
            let got = (a * b).extract(0);
            let want = F::from_canonical_u32(x) * F::from_canonical_u32(y);
            assert_eq!(got, want, "boundary mul {x}*{y}");
        }
    }
}

#[test]
fn packed_fp32_32b_add_sub_mul() {
    type F = Prime32Offset99;
    type PF = <F as HasPacking>::Packing;
    check_packed_add_sub_mul::<F, PF>(0xaa32_bb32_cc32_dd32);
}

/// Regression guard for the 32-bit (`BITS == 32`) packed base multiply.
///
/// For these primes the two-fold Solinas residue can land in `[2^32, 2*P)`
/// (up to `2^32 + C^2`). The packed `Mul` recombine must subtract `P` on the
/// full 64-bit lanes before packing; a 32-bit recombine drops bit 32 and
/// returns a result that is `C` too small. The probability of hitting this
/// window with uniform random inputs is `~C/2^32 ≈ 2e-6`, so the random
/// parity sweep misses it; these vectors hit it deterministically. They were
/// found by exhaustively comparing the truncating recombine to the true
/// modular product (all land in the overflow window on `Prime32Offset99`).
#[test]
fn packed_fp32_32b_mul_two_fold_overflow_window() {
    type F = Prime32Offset99;
    type PF = <F as HasPacking>::Packing;
    const VECTORS: [(u32, u32); 7] = [
        (3136721438, 3536064673),
        (2498152412, 1827148629),
        (2062525777, 3207684599),
        (4027016701, 3739597742),
        (2476582663, 3902052967),
        (4161561975, 3109742861),
        (1924659530, 1057556213),
    ];
    for (x, y) in VECTORS {
        let a = F::from_canonical_u32(x);
        let b = F::from_canonical_u32(y);
        let got = (PF::broadcast(a) * PF::broadcast(b)).extract(0);
        assert_eq!(got, a * b, "packed 32b mul mismatch for {x} * {y}");
    }
}

#[test]
fn fp32_broadcast_and_extract_roundtrip() {
    type F = Prime24Offset3;
    type PF = <F as HasPacking>::Packing;
    check_broadcast_roundtrip::<F, PF>(F::from_u64(42));
}

#[test]
fn packed_fp64_40b_add_sub_mul() {
    type F = Prime40Offset195;
    type PF = <F as HasPacking>::Packing;
    check_packed_add_sub_mul::<F, PF>(0xaa40_bb40_cc40_dd40);
}

#[test]
fn packed_fp64_64b_add_sub_mul() {
    type F = Prime64Offset59;
    type PF = <F as HasPacking>::Packing;
    check_packed_add_sub_mul::<F, PF>(0xaa64_bb64_cc64_dd64);
}

#[test]
fn fp64_broadcast_and_extract_roundtrip() {
    type F = Prime40Offset195;
    type PF = <F as HasPacking>::Packing;
    check_broadcast_roundtrip::<F, PF>(F::from_u64(42));
}
