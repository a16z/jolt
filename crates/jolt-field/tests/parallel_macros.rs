//! Serial-vs-parallel equivalence for the `cfg_*!` conditional-parallelism
//! macros (`solinas::parallel`).
//!
//! The macros dispatch on the `parallel` feature at the expansion site, so
//! this suite exercises the sequential expansions when run without the
//! feature and the rayon expansions when run with it (`--all-features`);
//! every assertion compares against an explicitly serial computation, so a
//! green run in both configurations is the equivalence proof.

#![cfg(feature = "solinas")]

use jolt_field::{Prime64Offset59, Ring, Zero};

#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::{
    cfg_chunks, cfg_chunks_mut, cfg_fold_reduce, cfg_into_iter, cfg_iter, cfg_iter_mut, cfg_join,
    cfg_try_fold_reduce,
};

type F = Prime64Offset59;

fn inputs() -> Vec<F> {
    (0..1000u64).map(F::from_u64).collect()
}

fn serial_sum(v: &[F]) -> F {
    v.iter().fold(F::zero(), |acc, x| acc + *x)
}

#[test]
fn iter_and_into_iter_match_serial() {
    let v = inputs();
    let expected = serial_sum(&v);
    let sum: F = cfg_iter!(v).copied().sum();
    assert_eq!(sum, expected);
    let sum: F = cfg_into_iter!(v.clone()).sum();
    assert_eq!(sum, expected);
}

#[test]
fn iter_mut_and_chunks_mut_match_serial() {
    let v = inputs();
    let two = F::from_u64(2);
    let expected: Vec<F> = v.iter().map(|x| *x * two).collect();

    let mut a = v.clone();
    cfg_iter_mut!(a).for_each(|x| *x *= two);
    assert_eq!(a, expected);

    let mut b = v;
    cfg_chunks_mut!(b, 7).for_each(|chunk| {
        for x in chunk {
            *x *= two;
        }
    });
    assert_eq!(b, expected);
}

#[test]
fn chunks_match_serial() {
    let v = inputs();
    let expected: Vec<F> = v.chunks(13).map(serial_sum).collect();
    let sums: Vec<F> = cfg_chunks!(v, 13).map(serial_sum).collect();
    assert_eq!(sums, expected);
}

#[test]
fn join_returns_both_results() {
    let v = inputs();
    let (lo, hi) = v.split_at(v.len() / 2);
    let (a, b) = cfg_join!(|| serial_sum(lo), || serial_sum(hi));
    assert_eq!(a + b, serial_sum(&v));
}

#[test]
fn fold_reduce_matches_serial_sum_of_squares() {
    let v = inputs();
    let expected = v.iter().fold(F::zero(), |acc, x| acc + x.square());
    let got = cfg_fold_reduce!(
        0..v.len(),
        F::zero,
        |acc: F, i: usize| acc + v[i].square(),
        |a: F, b: F| a + b
    );
    assert_eq!(got, expected);
}

#[test]
fn try_fold_reduce_matches_serial_and_propagates_errors() {
    let v = inputs();
    let got: Result<F, &'static str> = cfg_try_fold_reduce!(
        0..v.len(),
        F::zero,
        |acc: F, i: usize| Ok(acc + v[i]),
        |a: F, b: F| Ok(a + b)
    );
    assert_eq!(got, Ok(serial_sum(&v)));

    let failed: Result<F, &'static str> = cfg_try_fold_reduce!(
        0..v.len(),
        F::zero,
        |acc: F, i: usize| {
            if i == 17 {
                Err("sentinel")
            } else {
                Ok(acc + v[i])
            }
        },
        |a: F, b: F| Ok(a + b)
    );
    assert_eq!(failed, Err("sentinel"));
}
