#![allow(clippy::op_ref)]
use jolt_core::field::tracked_ark::TrackedFr as Fr;
use jolt_core::field::{JoltField, OptimizedMul};
use jolt_core::utils::counters::{
    get_inverse_count, get_mult_count, reset_inverse_count, reset_mult_count,
};
use std::ops::MulAssign;

fn main() {
    reset_mult_count();
    let a = Fr::from_u8(12);
    let b = Fr::from_u8(12);
    let c = a.mul_0_optimized(b);
    let num_mults = get_mult_count();
    println!("After 0-1 optimisation {num_mults}, {c}");

    reset_mult_count();
    let a = Fr::from_u8(12);
    let b = Fr::from_u8(12);
    let c = a * &b;
    let num_mults = get_mult_count();
    println!("{num_mults}, {c}");

    reset_mult_count();
    let a = Fr::from_u8(12);
    let b = Fr::from_u8(12);
    let c = &a * b;
    let num_mults = get_mult_count();
    println!("{num_mults}, {c}");

    reset_mult_count();
    let a = Fr::from_u8(12);
    let b = Fr::from_u8(12);
    let c = &a * &b;
    let num_mults = get_mult_count();
    println!("{num_mults}, {c}");

    reset_mult_count();
    let a = Fr::from_u8(12);
    let b = Fr::from_u8(12);
    let c = a * b;
    let num_mults = get_mult_count();
    println!("{num_mults}, {c}");

    reset_inverse_count();
    let mut a = Fr::from_u8(12);
    let _b = Fr::from_u8(12);
    let c = a.inverse().unwrap();
    let num_mults = get_inverse_count();
    println!("After inverse {num_mults}, {c}");

    reset_mult_count();
    a.mul_assign(b);
    let num_mults = get_mult_count();
    println!("After mul_assign {num_mults}, {a}");

    reset_mult_count();
    let vals = [&a, &b, &c];
    let prod: Fr = vals.into_iter().product();
    let num_mults = get_mult_count();
    println!("Product - this should be 3? {num_mults}, {prod}");
}
