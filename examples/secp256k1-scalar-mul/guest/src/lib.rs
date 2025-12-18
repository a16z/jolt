#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;
use alloc::vec::Vec;
use jolt_inlines_secp256k1::{Secp256k1Fq, Secp256k1Point};

#[inline(never)]
fn secp256k1_4x128_scalar_mul(scalars: [u128; 4], points: [Secp256k1Point; 4]) -> [u64; 8] {
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

#[inline(never)]
fn conditional_negate(x: Secp256k1Point, cond: bool) -> Secp256k1Point {
    if cond {
        x.neg()
    } else {
        x
    }
}

// compute uG + vQ for an secp256k1 points Q and scalars u and v
// where G is the generator point of secp256k1
// this is the most expensive part of ECDSA signature verification
// Note, this function does not check that Q is on the curve
#[jolt::provable(memory_size = 300000, max_trace_length = 4194304)]
fn secp256k1_scalar_mul(u: [u64; 4], v: [u64; 4], q: [u64; 8]) -> [u64; 8] {
    // convert u, v, and q to the appropriate types
    let q = Secp256k1Point::from_u64_arr_unchecked(&q);
    let ur = Secp256k1Point::scalar_from_u64_arr(&u);
    let vr = Secp256k1Point::scalar_from_u64_arr(&v);
    // perform the glv scalar decomposition
    let decomp_u = Secp256k1Point::decompose_scalar(&ur);
    let decomp_v = Secp256k1Point::decompose_scalar(&vr);
    // get scalars as a 4x128-bit array
    let scalars = [decomp_u[0].1, decomp_u[1].1, decomp_v[0].1, decomp_v[1].1];
    // get 4 points: G, lambda*G, Q, and lambda*Q
    // appropriately negated to match signs of the decomposed scalars
    let points = [
        conditional_negate(Secp256k1Point::generator(), decomp_u[0].0),
        conditional_negate(Secp256k1Point::generator_w_endomorphism(), decomp_u[1].0),
        conditional_negate(q.clone(), decomp_v[0].0),
        conditional_negate(q.endomorphism(), decomp_v[1].0),
    ];
    // perform the 4x128-bit scalar multiplication
    secp256k1_4x128_scalar_mul(scalars, points)
}
