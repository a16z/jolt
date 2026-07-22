#![expect(
    clippy::unreadable_literal,
    clippy::unwrap_used,
    reason = "tests assert field identities and retain copied field constants"
)]

use super::*;
use crate::ext::{Ext2, FpExt2, FpExt4, TwoNr};
use crate::FieldCore;
use crate::Fp32;
use crate::Fp64;
use crate::Prime31Offset19;
use crate::Prime32Offset99;
use crate::Prime64Offset59;
use crate::RingCore;
use rand::rngs::StdRng;
use rand::SeedableRng;

type F = Fp64<4294967197>;
type E2 = Ext2<F>;
type R4 = FpExt4<F>;
type PE2 = PackedFpExt2<F, TwoNr, <F as HasPacking>::Packing>;
type PR4 = PackedFpExt4<F, <F as HasPacking>::Packing>;
type Mersenne31 = Fp32<{ (1u32 << 31) - 1 }>;
type Generic30Offset16397 = Fp32<{ (1u32 << 30) - 16_397 }>;
type Generic31Offset61 = Fp32<{ (1u32 << 31) - 61 }>;
type Generic31Offset32787 = Fp32<{ (1u32 << 31) - 32_787 }>;
type PR4Prime31 = PackedFpExt4<Prime31Offset19, <Prime31Offset19 as HasPacking>::Packing>;
type PR4Mersenne31 = PackedFpExt4<Mersenne31, <Mersenne31 as HasPacking>::Packing>;
type PR4Generic30Offset16397 =
    PackedFpExt4<Generic30Offset16397, <Generic30Offset16397 as HasPacking>::Packing>;
type PR4Generic31Offset61 =
    PackedFpExt4<Generic31Offset61, <Generic31Offset61 as HasPacking>::Packing>;
type PR4Generic31Offset32787 =
    PackedFpExt4<Generic31Offset32787, <Generic31Offset32787 as HasPacking>::Packing>;
type R4Prime32 = FpExt4<Prime32Offset99>;
type PR4Prime32 = PackedFpExt4<Prime32Offset99, <Prime32Offset99 as HasPacking>::Packing>;
type E2Full = FpExt2<Prime64Offset59, TwoNr>;
type PE2Full = PackedFpExt2<Prime64Offset59, TwoNr, <Prime64Offset59 as HasPacking>::Packing>;

fn fp32_ext_edge_values<const P: u32>() -> [Fp32<P>; 4] {
    [
        Fp32::<P>::from_canonical_u32(P - 1),
        Fp32::<P>::from_canonical_u32(P - 2),
        Fp32::<P>::from_canonical_u32((P - 1) / 2),
        Fp32::<P>::one(),
    ]
}

fn check_packed_fp_ext4_edge<const P: u32, PR4>()
where
    PR4: PackedField<Scalar = FpExt4<Fp32<P>>> + PackedValue<Value = FpExt4<Fp32<P>>>,
{
    let values = fp32_ext_edge_values::<P>();
    let elem = |offset: usize| {
        FpExt4::<Fp32<P>>::new(std::array::from_fn(|j| values[(offset + j) % values.len()]))
    };
    let a = PR4::from_fn(elem);
    let b = PR4::from_fn(|i| elem(i + 1));
    let product = a * b;
    let square = a.square();

    for lane in 0..PR4::WIDTH {
        let lhs = elem(lane);
        let rhs = elem(lane + 1);
        assert_eq!(
            product.extract(lane),
            lhs * rhs,
            "packed FpExt4 edge mul mismatch at lane {lane}"
        );
        assert_eq!(
            square.extract(lane),
            lhs.square(),
            "packed FpExt4 edge square mismatch at lane {lane}"
        );
    }
}

#[test]
fn packed_fp_ext2_add() {
    let mut rng = StdRng::seed_from_u64(100);
    let width = <PE2 as PackedValue>::WIDTH;
    let a_elems: Vec<E2> = (0..width).map(|_| E2::random(&mut rng)).collect();
    let b_elems: Vec<E2> = (0..width).map(|_| E2::random(&mut rng)).collect();

    let pa = PE2::from_fn(|i| a_elems[i]);
    let pb = PE2::from_fn(|i| b_elems[i]);
    let pc = pa + pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(pc.extract(i), *a + *b);
    }
}

#[test]
fn packed_fp_ext2_mul() {
    let mut rng = StdRng::seed_from_u64(200);
    let width = <PE2 as PackedValue>::WIDTH;
    let a_elems: Vec<E2> = (0..width).map(|_| E2::random(&mut rng)).collect();
    let b_elems: Vec<E2> = (0..width).map(|_| E2::random(&mut rng)).collect();

    let pa = PE2::from_fn(|i| a_elems[i]);
    let pb = PE2::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "packed FpExt2 mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp_ext2_mul_full_word_fp64() {
    let mut rng = StdRng::seed_from_u64(201);
    let width = <PE2Full as PackedValue>::WIDTH;
    let a_elems: Vec<E2Full> = (0..width).map(|_| E2Full::random(&mut rng)).collect();
    let b_elems: Vec<E2Full> = (0..width).map(|_| E2Full::random(&mut rng)).collect();

    let pa = PE2Full::from_fn(|i| a_elems[i]);
    let pb = PE2Full::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "full-word packed FpExt2 mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp_ext2_broadcast() {
    let val = E2::new(F::from_u64(7), F::from_u64(11));
    let packed = PE2::broadcast(val);
    let width = <PE2 as PackedValue>::WIDTH;
    for i in 0..width {
        assert_eq!(packed.extract(i), val);
    }
}

#[test]
fn packed_fp_ext4_add() {
    let mut rng = StdRng::seed_from_u64(360);
    let width = <PR4 as PackedValue>::WIDTH;
    let a_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();
    let b_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();

    let pa = PR4::from_fn(|i| a_elems[i]);
    let pb = PR4::from_fn(|i| b_elems[i]);
    let pc = pa + pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a + *b,
            "packed FpExt4 add mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp_ext4_sub() {
    let mut rng = StdRng::seed_from_u64(361);
    let width = <PR4 as PackedValue>::WIDTH;
    let a_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();
    let b_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();

    let pa = PR4::from_fn(|i| a_elems[i]);
    let pb = PR4::from_fn(|i| b_elems[i]);
    let pc = pa - pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a - *b,
            "packed FpExt4 sub mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp_ext4_mul() {
    let mut rng = StdRng::seed_from_u64(362);
    let width = <PR4 as PackedValue>::WIDTH;
    let a_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();
    let b_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();

    let pa = PR4::from_fn(|i| a_elems[i]);
    let pb = PR4::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "packed FpExt4 mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp_ext4_mul_prime32() {
    let mut rng = StdRng::seed_from_u64(365);
    let width = <PR4Prime32 as PackedValue>::WIDTH;
    let a_elems: Vec<R4Prime32> = (0..width).map(|_| R4Prime32::random(&mut rng)).collect();
    let b_elems: Vec<R4Prime32> = (0..width).map(|_| R4Prime32::random(&mut rng)).collect();

    let pa = PR4Prime32::from_fn(|i| a_elems[i]);
    let pb = PR4Prime32::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "Prime32 packed FpExt4 mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp_ext4_prime31_edge_lanes() {
    check_packed_fp_ext4_edge::<
        { crate::prime::pseudo_mersenne::PRIME31_OFFSET19_MODULUS },
        PR4Prime31,
    >();
}

#[test]
fn packed_fp_ext4_mersenne31_edge_lanes() {
    check_packed_fp_ext4_edge::<{ (1u32 << 31) - 1 }, PR4Mersenne31>();
}

#[test]
fn packed_fp_ext4_prime32_edge_lanes() {
    check_packed_fp_ext4_edge::<
        { crate::prime::pseudo_mersenne::PRIME32_OFFSET99_MODULUS },
        PR4Prime32,
    >();
}

#[test]
fn packed_fp_ext4_generic31_edge_lanes() {
    check_packed_fp_ext4_edge::<{ (1u32 << 31) - 61 }, PR4Generic31Offset61>();
}

#[test]
fn packed_fp_ext4_large_generic30_edge_lanes() {
    check_packed_fp_ext4_edge::<{ (1u32 << 30) - 16_397 }, PR4Generic30Offset16397>();
}

#[test]
fn packed_fp_ext4_large_generic31_edge_lanes() {
    check_packed_fp_ext4_edge::<{ (1u32 << 31) - 32_787 }, PR4Generic31Offset32787>();
}

#[test]
fn packed_fp_ext4_square() {
    let mut rng = StdRng::seed_from_u64(363);
    let width = <PR4 as PackedValue>::WIDTH;
    let elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();

    let packed = PR4::from_fn(|i| elems[i]);
    let squared = packed.square();

    for (i, elem) in elems.iter().enumerate() {
        assert_eq!(
            squared.extract(i),
            elem.square(),
            "packed FpExt4 square mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp_ext4_square_prime32() {
    let mut rng = StdRng::seed_from_u64(366);
    let width = <PR4Prime32 as PackedValue>::WIDTH;
    let elems: Vec<R4Prime32> = (0..width).map(|_| R4Prime32::random(&mut rng)).collect();

    let packed = PR4Prime32::from_fn(|i| elems[i]);
    let squared = packed.square();

    for (i, elem) in elems.iter().enumerate() {
        assert_eq!(
            squared.extract(i),
            elem.square(),
            "Prime32 packed FpExt4 square mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp_ext4_square_mersenne31() {
    let mut rng = StdRng::seed_from_u64(367);
    type R4M31 = FpExt4<Mersenne31>;
    let width = <PR4Mersenne31 as PackedValue>::WIDTH;
    let elems: Vec<R4M31> = (0..width).map(|_| R4M31::random(&mut rng)).collect();

    let packed = PR4Mersenne31::from_fn(|i| elems[i]);
    let squared = packed.square();

    for (i, elem) in elems.iter().enumerate() {
        assert_eq!(
            squared.extract(i),
            elem.square(),
            "Mersenne31 packed FpExt4 square mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp_ext4_inverse() {
    let mut rng = StdRng::seed_from_u64(367);
    let width = <PR4 as PackedValue>::WIDTH;
    let elems: Vec<R4> = (0..width)
        .map(|_| {
            let x = R4::random(&mut rng);
            if x.is_zero() {
                R4::one()
            } else {
                x
            }
        })
        .collect();

    let packed = PR4::from_fn(|i| elems[i]);
    let inverted = packed.inverse().unwrap();

    for (i, elem) in elems.iter().enumerate() {
        assert_eq!(
            inverted.extract(i),
            elem.inverse().unwrap(),
            "packed FpExt4 inverse mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp_ext4_broadcast() {
    let val = R4::new([
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
        F::from_u64(17),
    ]);
    let packed = PR4::broadcast(val);
    let width = <PR4 as PackedValue>::WIDTH;
    for i in 0..width {
        assert_eq!(packed.extract(i), val);
    }
}

#[test]
fn packed_fp_ext4_pack_unpack() {
    let mut rng = StdRng::seed_from_u64(364);
    let width = <PR4 as PackedValue>::WIDTH;
    let elems: Vec<R4> = (0..width * 3).map(|_| R4::random(&mut rng)).collect();

    let packed = PR4::pack_slice(&elems);
    let unpacked = PR4::unpack_slice(&packed);

    assert_eq!(elems, unpacked);
}

#[test]
fn pack_unpack_roundtrip_fp_ext2() {
    let mut rng = StdRng::seed_from_u64(400);
    let width = <PE2 as PackedValue>::WIDTH;
    let elems: Vec<E2> = (0..width * 3).map(|_| E2::random(&mut rng)).collect();

    let packed = PE2::pack_slice(&elems);
    let unpacked = PE2::unpack_slice(&packed);

    assert_eq!(elems, unpacked);
}

type R8Fp64 = FpExt8<F>;
type PR8Fp64 = PackedFpExt8<F, <F as HasPacking>::Packing>;
type R8Prime31 = FpExt8<Prime31Offset19>;
type PR8Prime31 = PackedFpExt8<Prime31Offset19, <Prime31Offset19 as HasPacking>::Packing>;
type R8Prime32 = FpExt8<Prime32Offset99>;
type PR8Prime32 = PackedFpExt8<Prime32Offset99, <Prime32Offset99 as HasPacking>::Packing>;

#[test]
fn packed_fp_ext8_mul_fp64() {
    let mut rng = StdRng::seed_from_u64(500);
    let width = <PR8Fp64 as PackedValue>::WIDTH;
    let a_elems: Vec<R8Fp64> = (0..width).map(|_| R8Fp64::random(&mut rng)).collect();
    let b_elems: Vec<R8Fp64> = (0..width).map(|_| R8Fp64::random(&mut rng)).collect();

    let pa = PR8Fp64::from_fn(|i| a_elems[i]);
    let pb = PR8Fp64::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "packed FpExt8<Fp64> mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp_ext8_mul_prime31() {
    let mut rng = StdRng::seed_from_u64(501);
    let width = <PR8Prime31 as PackedValue>::WIDTH;
    let a_elems: Vec<R8Prime31> = (0..width).map(|_| R8Prime31::random(&mut rng)).collect();
    let b_elems: Vec<R8Prime31> = (0..width).map(|_| R8Prime31::random(&mut rng)).collect();

    let pa = PR8Prime31::from_fn(|i| a_elems[i]);
    let pb = PR8Prime31::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "packed FpExt8<Prime31> mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp_ext8_mul_prime32() {
    let mut rng = StdRng::seed_from_u64(502);
    let width = <PR8Prime32 as PackedValue>::WIDTH;
    let a_elems: Vec<R8Prime32> = (0..width).map(|_| R8Prime32::random(&mut rng)).collect();
    let b_elems: Vec<R8Prime32> = (0..width).map(|_| R8Prime32::random(&mut rng)).collect();

    let pa = PR8Prime32::from_fn(|i| a_elems[i]);
    let pb = PR8Prime32::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "packed FpExt8<Prime32> mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp_ext8_square() {
    let mut rng = StdRng::seed_from_u64(504);
    let width = <PR8Prime31 as PackedValue>::WIDTH;
    let a_elems: Vec<R8Prime31> = (0..width).map(|_| R8Prime31::random(&mut rng)).collect();

    let pa = PR8Prime31::from_fn(|i| a_elems[i]);
    let sq = pa.square();

    for (i, a) in a_elems.iter().enumerate() {
        assert_eq!(
            sq.extract(i),
            a.square(),
            "packed FpExt8 square mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp_ext8_broadcast() {
    let val = R8Fp64::new([
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
        F::from_u64(5),
        F::from_u64(6),
        F::from_u64(7),
        F::from_u64(8),
    ]);
    let packed = PR8Fp64::broadcast(val);
    let width = <PR8Fp64 as PackedValue>::WIDTH;
    for i in 0..width {
        assert_eq!(packed.extract(i), val);
    }
}
