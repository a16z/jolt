#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;
use alloc::vec::Vec;
use jolt_inlines_secp256k1::{Secp256k1Fq, Secp256k1Point};

// basic version
fn secp256k1_scalar_mul_simple(scalars: [u128; 4], points: [u64; 32]) -> [u64; 8] {
    let points = [
        Secp256k1Point::from_u64_arr_unchecked(&points[0..8].try_into().unwrap()),
        Secp256k1Point::from_u64_arr_unchecked(&points[8..16].try_into().unwrap()),
        Secp256k1Point::from_u64_arr_unchecked(&points[16..24].try_into().unwrap()),
        Secp256k1Point::from_u64_arr_unchecked(&points[24..32].try_into().unwrap()),
    ];
    let mut lookup = Vec::<Secp256k1Point>::with_capacity(16);
    lookup.push(Secp256k1Point::infinity());
    lookup.push(points[0].clone());
    lookup.push(points[1].clone());
    lookup.push(lookup[1].add(&lookup[2]));
    lookup.push(points[2].clone());
    lookup.push(lookup[1].add(&lookup[4]));
    lookup.push(lookup[2].add(&lookup[4]));
    lookup.push(lookup[1].add(&lookup[6]));
    lookup.push(points[3].clone());
    for i in 1..8 {
        lookup.push(lookup[i].add(&lookup[8]));
    }
    let mut res = Secp256k1Point::infinity();
    for i in (0..128).rev() {
        let mut idx = 0;
        for j in 0..4 {
            if (scalars[j] >> i) & 1 == 1 {
                idx |= 1 << j;
            }
        }
        if idx != 0 {
            res = res.double_and_add(&lookup[idx]);
        } else {
            res = res.double();
        }
    }
    res.to_u64_arr()
}

// compute n1*P1 + n2*P2 + n3*P3 + n4*P4 for secp256k1 points P1, P2, P3, P4 and scalars n1, n2, n3, n4
// Note, this function does not check that the points are on the curve
#[jolt::provable(memory_size = 200000, max_trace_length = 4194304)]
fn secp256k1_scalar_mul(scalars: [u128; 4], points: [u64; 32]) -> [u64; 8] {
    secp256k1_scalar_mul_simple(scalars, points)
    // Code below is for getting cycle counts of various operations
    // compute 1000 point additions to stress test
    /*let mut res = Secp256k1Point::infinity();
    let tmp = Secp256k1Point::from_u64_arr_unchecked(&point);
    for _ in 0..1000 {
        res = res.add(&res);
    }
    res.to_u64_arr()*/
    // compute 1000 fused double + adds to stress test
    /*let mut res = Secp256k1Point::infinity();
    let tmp = Secp256k1Point::from_u64_arr_unchecked(&point);
    for _ in 0..1000 {
        res = res.double_and_add(&tmp);
    }
    res.to_u64_arr()*/
    // perform 1000 muls to stress test
    /*let mut x = Secp256k1Fq::seven();
    for _ in 0..1000 {
        x = x.mul(&x);
    }
    let mut out = [0u64; 8];
    out[0] = x.is_zero() as u64;
    out*/
    // perform 1000 squares to stress test
    /*let mut x = Secp256k1Fq::seven();
    for _ in 0..1000 {
        x = x.square();
    }
    let mut out = [0u64; 8];
    out[0] = x.is_zero() as u64;
    out*/
    // perform 1000 inversions to stress test
    /*let mut x = Secp256k1Fq::seven();
    let seven = Secp256k1Fq::seven();
    for _ in 0..1000 {
        x = x.div(&seven);
    }
    let mut out = [0u64; 8];
    out[0] = x.is_zero() as u64;
    out*/
}
