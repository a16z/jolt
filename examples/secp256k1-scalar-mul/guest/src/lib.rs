#![cfg_attr(feature = "guest", no_std)]
use jolt_inlines_secp256k1::{Secp256k1Fq, Secp256k1Point};

// compute n1*P1 + n2*P2 + n3*P3 + n4*P4 for secp256k1 points P1, P2, P3, P4 and scalars n1, n2, n3, n4
// Note, this function does not check that the points are on the curve
/*#[jolt::provable(memory_size = 32768, max_trace_length = 4194304)]
fn secp256k1_4x128bit_scalar_mul(scalars: [u128; 4], points: [[u64; 8]; 4]) -> [u64; 8] {
    // convert points to Secp256k1Point
    let p = [
        Secp256k1Point::from_u64_arr_unchecked(&points[0]),
        Secp256k1Point::from_u64_arr_unchecked(&points[1]),
        Secp256k1Point::from_u64_arr_unchecked(&points[2]),
        Secp256k1Point::from_u64_arr_unchecked(&points[3]),
    ];
    // build lookup table for all combinations of the 4 points
    use core::array;
    let mut lookup: [Secp256k1Point; 16] = array::from_fn(|_| Secp256k1Point::infinity());
    lookup[1] = p[0].clone();
    lookup[2] = p[1].clone();
    lookup[3] = lookup[1].add(&lookup[2]);
    lookup[4] = p[2].clone();
    lookup[5] = lookup[1].add(&lookup[4]);
    lookup[6] = lookup[2].add(&lookup[4]);
    lookup[7] = lookup[1].add(&lookup[6]);
    lookup[8] = p[3].clone();
    for i in 1..8 {
        lookup[8 + i] = lookup[i].add(&lookup[8]);
    }
    // compute the result using a double-and-add algorithm with the lookup table
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
}*/

// Given a 256-bit scalar n and a point P
// Compute nP
// Note, this function does not check that P is on the curve
#[jolt::provable(memory_size = 32768, max_trace_length = 4194304)]
fn secp256k1_scalar_mul(scalar: [u64; 4], point: [u64; 8]) -> [u64; 8] {
    let g = Secp256k1Point::from_u64_arr_unchecked(&point);
    let mut res = Secp256k1Point::infinity();
    for i in (0..4).rev() {
        for j in (0..64).rev() {
            if (scalar[i] >> j) & 1 == 1 {
                res = res.double_and_add(&g);
            } else {
                res = res.double();
            }
        }
    }
    res.to_u64_arr()
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
