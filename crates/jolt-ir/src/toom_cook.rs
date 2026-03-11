//! Toom-Cook evaluation kernels for products of linear polynomials.
//!
//! Evaluates $P(x) = \prod_{j=0}^{D-1} p_j(x)$ on the grid
//! $U_D = \{1, 2, \ldots, D-1, \infty\}$, producing $D$ evaluations.
//!
//! Each linear polynomial is specified by its even/odd pair:
//! $p_j(x) = p_j(0) + (p_j(1) - p_j(0)) \cdot x$.
//! The point at infinity evaluates the leading term:
//! $P(\infty) = \prod_{j} (p_j(1) - p_j(0))$.
//!
//! # Algorithm
//!
//! For large $D$, balanced binary splitting reduces multiplication count
//! from $O(D^2)$ to $O(D \log D)$:
//!
//! 1. Split $D$ polynomials into two halves of size $D/2$
//! 2. Evaluate each half-product on $D/2 + 1$ points (including $\infty$)
//! 3. Extrapolate both halves to $D - 1$ points using finite-difference formulas
//! 4. Multiply point-wise to get the full product at $D - 1$ points + $\infty$
//!
//! Extrapolation uses the identity: for a degree-$k$ polynomial,
//! the $k$-th forward difference is constant and equals $k! \cdot P(\infty)$.
//!
//! # Specializations
//!
//! | D | Algorithm | Muls | vs naive |
//! |---|-----------|------|----------|
//! | 2 | Direct | 2 | 2 |
//! | 3 | 2+1 split | 4 | 6 |
//! | 4 | 2×2 Toom-Cook | 8 | 12 |
//! | 5 | 2×2+1 | 13 | 20 |
//! | 6 | 4+2 split | ~20 | 30 |
//! | 7 | 4+3 split | ~28 | 42 |
//! | 8 | 4×4 Toom-Cook | ~24 | 56 |
//! | 16 | 8×8 + ex8 expansion | ~64 | 240 |
//! | 32 | 16×16 + ex16 expansion | ~160 | 992 |

use core::{mem::MaybeUninit, ptr};

use jolt_field::Field;

#[inline(always)]
fn dbl<F: Field>(x: F) -> F {
    x + x
}

#[inline(always)]
fn dbl_assign<F: Field>(x: &mut F) {
    *x += *x;
}

/// Extrapolate the next value of a degree-2 polynomial on the natural grid.
///
/// Given `f = [P(n), P(n+1)]` and `f_inf = P(∞)` (leading coefficient),
/// returns `P(n+2) = 2(P(n+1) + P(∞)) - P(n)`.
#[inline(always)]
fn ex2<F: Field>(f: &[F; 2], f_inf: &F) -> F {
    dbl(f[1] + f_inf) - f[0]
}

/// Extrapolate the next value of a degree-4 polynomial on the natural grid.
///
/// Given 4 consecutive evaluations `f = [P(n), ..., P(n+3)]` and
/// `f_inf6 = 6 · P(∞)`, returns `P(n+4)`.
///
/// Uses the 4th-order forward difference identity:
/// `P(n+4) = 4P(n+3) - 6P(n+2) + 4P(n+1) - P(n) + 24·P(∞)`.
/// The `6·P(∞)` pre-scaling avoids a separate `mul_u64(24)` call.
#[inline]
fn ex4<F: Field>(f: &[F; 4], f_inf6: &F) -> F {
    let mut t = *f_inf6;
    t += f[3];
    t -= f[2];
    t += f[1];
    dbl_assign(&mut t);
    t -= f[2];
    dbl_assign(&mut t);
    t -= f[0];
    t
}

/// Extrapolate the next two values of a degree-4 polynomial on the natural grid.
///
/// Returns `(P(n+4), P(n+5))` from `f = [P(n), ..., P(n+3)]` and `f_inf6 = 6·P(∞)`.
#[inline]
fn ex4_2<F: Field>(f: &[F; 4], f_inf6: &F) -> (F, F) {
    let f3m2 = f[3] - f[2];
    let mut f4 = *f_inf6;
    f4 += f3m2;
    f4 += f[1];
    dbl_assign(&mut f4);
    f4 -= f[2];
    dbl_assign(&mut f4);
    f4 -= f[0];

    let mut f5 = f4 - f3m2 + f_inf6;
    dbl_assign(&mut f5);
    f5 -= f[3];
    dbl_assign(&mut f5);
    f5 -= f[1];

    (f4, f5)
}

/// Extrapolate `P(n+8)` from 8 consecutive evaluations of a degree-8 polynomial.
///
/// Uses binomial coefficients with alternating signs:
/// `P(n+8) = Σ (-1)^k C(8,k) P(n+7-k) + 8!·P(∞)`
///
/// Pre-scaled: `f_inf40320 = 40320·P(∞)` where 40320 = 8!.
#[inline(always)]
fn ex8<F: Field>(f: &[F; 8], f_inf40320: F) -> F {
    let t1 = f[1] + f[7];
    let t2 = f[3] + f[5];
    let t3 = f[2] + f[6];

    t1.mul_u64(8) + t2.mul_u64(56) + f_inf40320 - t3.mul_u64(28) - f[4].mul_u64(70) - f[0]
}

/// Extrapolate `P(n+16)` from 16 consecutive evaluations of a degree-16 polynomial.
///
/// Pre-scaled: `f_inf16_fact = 16!·P(∞)`.
#[inline]
fn ex16<F: Field>(f: &[F; 16], f_inf16_fact: F) -> F {
    let s16 = f[1] + f[15];
    let s120 = f[2] + f[14];
    let s560 = f[3] + f[13];
    let s1820 = f[4] + f[12];
    let s4368 = f[5] + f[11];
    let s8008 = f[6] + f[10];
    let s11440 = f[7] + f[9];

    s16.mul_u64(16) + s560.mul_u64(560) + s4368.mul_u64(4368) + s11440.mul_u64(11440) + f_inf16_fact
        - s120.mul_u64(120)
        - s1820.mul_u64(1820)
        - s8008.mul_u64(8008)
        - f[8].mul_u64(12870)
        - f[0]
}

/// Evaluate the product of 2 linear polynomials at points {1, 2, ∞}.
///
/// Returns `(P(1), P(2), P(∞))`.
#[inline(always)]
fn eval_linear_prod_2_internal<F: Field>((p0, p1): (F, F), (q0, q1): (F, F)) -> (F, F, F) {
    let p_inf = p1 - p0;
    let p2 = p_inf + p1;
    let q_inf = q1 - q0;
    let q2 = q_inf + q1;
    let r1 = p1 * q1;
    let r2 = p2 * q2;
    let r_inf = p_inf * q_inf;
    (r1, r2, r_inf)
}

/// Evaluate the product of 4 linear polynomials at points {1, 2, 3, 4, ∞}.
///
/// Uses 2×2 Toom-Cook: split into halves, evaluate each at {1, 2, ∞},
/// extrapolate to {3, 4}, multiply point-wise.
/// Returns `(P(1), P(2), P(3), P(4), P(∞))`.
#[inline]
fn eval_linear_prod_4_internal<F: Field>(p: [(F, F); 4]) -> (F, F, F, F, F) {
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let a4 = ex2(&[a2, a3], &a_inf);
    let (b1, b2, b_inf) = eval_linear_prod_2_internal(p[2], p[3]);
    let b3 = ex2(&[b1, b2], &b_inf);
    let b4 = ex2(&[b2, b3], &b_inf);
    (a1 * b1, a2 * b2, a3 * b3, a4 * b4, a_inf * b_inf)
}

/// Evaluate the product of 8 linear polynomials at points {1, ..., 8, ∞}.
///
/// Uses 4×4 Toom-Cook: split into halves, evaluate each at {1, 2, 3, 4, ∞},
/// extrapolate both to {5, 6, 7, 8} via `ex4_2`/`ex4`, multiply point-wise.
/// Returns 9 values: `[P(1), P(2), ..., P(8), P(∞)]`.
fn eval_linear_prod_8_internal<F: Field>(p: [(F, F); 8]) -> [F; 9] {
    #[inline]
    fn batch_helper<F: Field>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F, F) {
        let f_inf6 = f_inf.mul_u64(6);
        let (f4, f5) = ex4_2(&[f0, f1, f2, f3], &f_inf6);
        let (f6, f7) = ex4_2(&[f2, f3, f4, f5], &f_inf6);
        (f4, f5, f6, f7)
    }

    // SAFETY: `p[0..4]` is a valid 4-element subslice, reinterpreted as `[(F, F); 4]`.
    let (a1, a2, a3, a4, a_inf) =
        eval_linear_prod_4_internal(unsafe { *p[0..4].as_ptr().cast::<[(F, F); 4]>() });
    let (a5, a6, a7, a8) = batch_helper(a1, a2, a3, a4, a_inf);
    // SAFETY: `p[4..8]` is a valid 4-element subslice, reinterpreted as `[(F, F); 4]`.
    let (b1, b2, b3, b4, b_inf) =
        eval_linear_prod_4_internal(unsafe { *p[4..8].as_ptr().cast::<[(F, F); 4]>() });
    let (b5, b6, b7, b8) = batch_helper(b1, b2, b3, b4, b_inf);

    [
        a1 * b1,
        a2 * b2,
        a3 * b3,
        a4 * b4,
        a5 * b5,
        a6 * b6,
        a7 * b7,
        a8 * b8,
        a_inf * b_inf,
    ]
}

/// Expand 9 evaluations ([P(1)..P(8), P(∞)]) to 16 + P(∞) via degree-8 extrapolation.
#[inline]
fn expand8_to16<F: Field>(vals: &[F; 9]) -> ([F; 16], F) {
    let mut f_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();
    let f_ptr = f_mu.as_mut_ptr();
    // SAFETY: Writing to uninitialized storage; all 16 slots are filled before `assume_init`.
    let f_slice_ptr = unsafe { (*f_ptr).as_mut_ptr() };

    // SAFETY: Copying first 8 values from `vals` into slots 0..8.
    unsafe {
        ptr::copy_nonoverlapping(vals.as_ptr(), f_slice_ptr, 8);
    }

    let f_inf = vals[8];
    let f_inf40320 = f_inf.mul_u64(40320);

    for i in 0..8 {
        // SAFETY: Sliding window `[i..i+8]` reads from already-initialized data,
        // result written at slot `8+i`.
        unsafe {
            let win_ptr = f_slice_ptr.add(i) as *const [F; 8];
            let val: F = ex8(&*win_ptr, f_inf40320);
            ptr::write(f_slice_ptr.add(8 + i), val);
        }
    }

    // SAFETY: All 16 slots written (0..8 from copy, 8..16 from loop).
    let f = unsafe { f_mu.assume_init() };
    (f, f_inf)
}

/// Evaluate the product of 16 linear polynomials at points {1..16, ∞} via 8×8 split,
/// returning `([P(1)..P(16)], P(∞))`.
#[inline]
fn eval_half_16_base<F: Field>(p: [(F, F); 16]) -> ([F; 16], F) {
    // SAFETY: `p[0..8]` is a valid 8-element subslice, reinterpreted as `[(F, F); 8]`.
    let a8 = eval_linear_prod_8_internal(unsafe { *p[0..8].as_ptr().cast::<[(F, F); 8]>() });
    // SAFETY: `p[8..16]` is a valid 8-element subslice, reinterpreted as `[(F, F); 8]`.
    let b8 = eval_linear_prod_8_internal(unsafe { *p[8..16].as_ptr().cast::<[(F, F); 8]>() });

    let (a16_vals, a_inf) = expand8_to16::<F>(&a8);
    let (b16_vals, b_inf) = expand8_to16::<F>(&b8);

    let mut base_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();
    let base_ptr = base_mu.as_mut_ptr();
    // SAFETY: Writing to uninitialized storage; all 16 slots filled before `assume_init`.
    let base_slice_ptr = unsafe { (*base_ptr).as_mut_ptr() };

    for i in 0..16 {
        // SAFETY: Writing product at each initialized slot.
        unsafe {
            ptr::write(base_slice_ptr.add(i), a16_vals[i] * b16_vals[i]);
        }
    }

    // SAFETY: All 16 slots written.
    let base = unsafe { base_mu.assume_init() };
    (base, a_inf * b_inf)
}

/// Expand 16 base evaluations + P(∞) to 32 values via degree-16 extrapolation.
#[inline]
fn expand16_to_u32<F: Field>(base16: &[F; 16], inf: F) -> [F; 32] {
    let mut f_mu: MaybeUninit<[F; 32]> = MaybeUninit::uninit();
    let f_ptr = f_mu.as_mut_ptr();
    // SAFETY: Writing to uninitialized storage; all 32 slots filled before `assume_init`.
    let f_slice_ptr = unsafe { (*f_ptr).as_mut_ptr() };

    // SAFETY: Copy first 16 values and pre-write slot 31.
    unsafe {
        ptr::copy_nonoverlapping(base16.as_ptr(), f_slice_ptr, 16);
        ptr::write(f_slice_ptr.add(31), inf);
    }

    let f_inf16_fact = inf.mul_u64(20_922_789_888_000_u64); // 16!

    for i in 0..15 {
        // SAFETY: Sliding window `[i..i+16]` reads from already-initialized data,
        // result written at slot `16+i`.
        unsafe {
            let win_ptr = f_slice_ptr.add(i) as *const [F; 16];
            let val = ex16::<F>(&*win_ptr, f_inf16_fact);
            ptr::write(f_slice_ptr.add(16 + i), val);
        }
    }

    // SAFETY: All 32 slots written (0..16 from copy, 16..31 from loop, 31 pre-written).
    unsafe { f_mu.assume_init() }
}

/// Evaluate the product of $D$ linear polynomials on the Toom-Cook grid.
///
/// Each `pair = (p(0), p(1))` defines a linear polynomial `p(x) = p(0) + (p(1) - p(0))·x`.
/// The output slice has layout `[P(1), P(2), ..., P(D-1), P(∞)]` where
/// $P(x) = \prod_j p_j(x)$.
///
/// Dispatches to D-specific implementations for D ∈ {2, 3, 4, 5, 6, 7, 8, 16, 32},
/// with a naive $O(D^2)$ fallback for other values.
pub fn eval_linear_prod_assign<F: Field>(pairs: &[(F, F)], evals: &mut [F]) {
    debug_assert_eq!(pairs.len(), evals.len());
    match pairs.len() {
        2 => {
            // SAFETY: `pairs` has length 2, layout-compatible with `[(F, F); 2]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 2]>() };
            eval_prod_2_assign(p, evals);
        }
        3 => {
            // SAFETY: `pairs` has length 3, layout-compatible with `[(F, F); 3]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 3]>() };
            eval_prod_3_assign(p, evals);
        }
        4 => {
            // SAFETY: `pairs` has length 4, layout-compatible with `[(F, F); 4]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 4]>() };
            eval_prod_4_assign(p, evals);
        }
        5 => {
            // SAFETY: `pairs` has length 5, layout-compatible with `[(F, F); 5]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 5]>() };
            eval_prod_5_assign(p, evals);
        }
        6 => {
            // SAFETY: `pairs` has length 6, layout-compatible with `[(F, F); 6]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 6]>() };
            eval_prod_6_assign(p, evals);
        }
        7 => {
            // SAFETY: `pairs` has length 7, layout-compatible with `[(F, F); 7]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 7]>() };
            eval_prod_7_assign(p, evals);
        }
        8 => {
            // SAFETY: `pairs` has length 8, layout-compatible with `[(F, F); 8]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 8]>() };
            eval_prod_8_assign(p, evals);
        }
        16 => {
            // SAFETY: `pairs` has length 16, layout-compatible with `[(F, F); 16]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 16]>() };
            eval_prod_16_assign(p, evals);
        }
        32 => {
            // SAFETY: `pairs` has length 32, layout-compatible with `[(F, F); 32]`.
            let p = unsafe { &*pairs.as_ptr().cast::<[(F, F); 32]>() };
            eval_prod_32_assign(p, evals);
        }
        _ => eval_linear_prod_naive_assign(pairs, evals),
    }
}

/// D=2: product of 2 linear polynomials at `{1, ∞}`.
#[inline(always)]
pub fn eval_prod_2_assign<F: Field>(p: &[(F, F); 2], outputs: &mut [F]) {
    outputs[0] = p[0].1 * p[1].1;
    outputs[1] = (p[0].1 - p[0].0) * (p[1].1 - p[1].0);
}

/// D=3: product of 3 linear polynomials at `{1, 2, ∞}`.
#[inline(always)]
pub fn eval_prod_3_assign<F: Field>(pairs: &[(F, F); 3], outputs: &mut [F]) {
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(pairs[0], pairs[1]);
    let (b0, b1) = pairs[2];
    let b_inf = b1 - b0;
    let b2 = b1 + b_inf;
    outputs[0] = a1 * b1;
    outputs[1] = a2 * b2;
    outputs[2] = a_inf * b_inf;
}

/// D=4: product of 4 linear polynomials at `{1, 2, 3, ∞}`.
///
/// 2×2 Toom-Cook with `ex2` extrapolation. 8 field multiplications.
pub fn eval_prod_4_assign<F: Field>(p: &[(F, F); 4], outputs: &mut [F]) {
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let (b1, b2, b_inf) = eval_linear_prod_2_internal(p[2], p[3]);
    let b3 = ex2(&[b1, b2], &b_inf);
    outputs[0] = a1 * b1;
    outputs[1] = a2 * b2;
    outputs[2] = a3 * b3;
    outputs[3] = a_inf * b_inf;
}

/// D=5: product of 5 linear polynomials at `{1, 2, 3, 4, ∞}`.
///
/// 2+2+1 splitting: two 2-products + one linear factor.
pub fn eval_prod_5_assign<F: Field>(p: &[(F, F); 5], outputs: &mut [F]) {
    let (a1, a2, a_inf) = eval_linear_prod_2_internal(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let a4 = ex2(&[a2, a3], &a_inf);

    let (tail1, tail2) = (p[3], p[4]);
    let (r1, r2, r_inf) = eval_linear_prod_2_internal(tail1, tail2);
    let r3 = ex2(&[r1, r2], &r_inf);
    let r4 = ex2(&[r2, r3], &r_inf);

    let (lin0, lin1) = p[2];
    let delta = lin1 - lin0;
    let l1 = lin1;
    let l2 = l1 + delta;
    let l3 = l2 + delta;
    let l4 = l3 + delta;
    let l_inf = delta;

    outputs[0] = a1 * (l1 * r1);
    outputs[1] = a2 * (l2 * r2);
    outputs[2] = a3 * (l3 * r3);
    outputs[3] = a4 * (l4 * r4);
    outputs[4] = a_inf * (l_inf * r_inf);
}

/// D=6: product of 6 linear polynomials at `{1, 2, 3, 4, 5, ∞}`.
///
/// 4+2 balanced split: A = product of first 4 (degree 4), B = product of
/// last 2 (degree 2). Evaluate A at `{1..5, ∞}` via `eval_linear_prod_4_internal`
/// \+ `ex4`, B at `{1..5, ∞}` via `eval_linear_prod_2_internal` + `ex2`,
/// then pointwise multiply. \~20 field multiplications vs 30 naive.
pub fn eval_prod_6_assign<F: Field>(p: &[(F, F); 6], outputs: &mut [F]) {
    // SAFETY: `p[0..4]` is a valid 4-element subslice.
    let (a1, a2, a3, a4, a_inf) =
        eval_linear_prod_4_internal(unsafe { *p[0..4].as_ptr().cast::<[(F, F); 4]>() });
    let a_inf6 = a_inf.mul_u64(6);
    let a5 = ex4(&[a1, a2, a3, a4], &a_inf6);

    let (b1, b2, b_inf) = eval_linear_prod_2_internal(p[4], p[5]);
    let b3 = ex2(&[b1, b2], &b_inf);
    let b4 = ex2(&[b2, b3], &b_inf);
    let b5 = ex2(&[b3, b4], &b_inf);

    outputs[0] = a1 * b1;
    outputs[1] = a2 * b2;
    outputs[2] = a3 * b3;
    outputs[3] = a4 * b4;
    outputs[4] = a5 * b5;
    outputs[5] = a_inf * b_inf;
}

/// D=7: product of 7 linear polynomials at `{1, 2, 3, 4, 5, 6, ∞}`.
///
/// 4+3 balanced split: A = product of first 4 (degree 4), B = product of
/// last 3 (degree 3). Evaluate A at `{1..6, ∞}` via `eval_linear_prod_4_internal`
/// \+ `ex4_2`, B at `{1..6, ∞}` via 2+1 sub-split with `ex2` extrapolation,
/// then pointwise multiply. \~28 field multiplications vs 42 naive.
pub fn eval_prod_7_assign<F: Field>(p: &[(F, F); 7], outputs: &mut [F]) {
    // SAFETY: `p[0..4]` is a valid 4-element subslice.
    let (a1, a2, a3, a4, a_inf) =
        eval_linear_prod_4_internal(unsafe { *p[0..4].as_ptr().cast::<[(F, F); 4]>() });
    let a_inf6 = a_inf.mul_u64(6);
    let (a5, a6) = ex4_2(&[a1, a2, a3, a4], &a_inf6);

    // B = product of last 3 linears, evaluated at {1..6, ∞}.
    // Sub-split: R = product of p[4], p[5] (degree 2), then multiply by p[6].
    let (r1, r2, r_inf) = eval_linear_prod_2_internal(p[4], p[5]);
    let r3 = ex2(&[r1, r2], &r_inf);
    let r4 = ex2(&[r2, r3], &r_inf);
    let r5 = ex2(&[r3, r4], &r_inf);
    let r6 = ex2(&[r4, r5], &r_inf);

    let (l0, l1) = p[6];
    let l_delta = l1 - l0;
    let l_inf = l_delta;
    // l(t) = l0 + t * l_delta, but l(1) = l1
    let mut l_val = l1;

    outputs[0] = r1 * (l_val) * a1;
    l_val += l_delta;
    outputs[1] = r2 * (l_val) * a2;
    l_val += l_delta;
    outputs[2] = r3 * (l_val) * a3;
    l_val += l_delta;
    outputs[3] = r4 * (l_val) * a4;
    l_val += l_delta;
    outputs[4] = r5 * (l_val) * a5;
    l_val += l_delta;
    outputs[5] = r6 * (l_val) * a6;
    outputs[6] = r_inf * l_inf * a_inf;
}

/// D=8: product of 8 linear polynomials at `{1, 2, ..., 7, ∞}`.
///
/// 4×4 Toom-Cook with `ex4_2`/`ex4` extrapolation.
pub fn eval_prod_8_assign<F: Field>(p: &[(F, F); 8], outputs: &mut [F]) {
    #[inline]
    fn batch_helper<F: Field>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F) {
        let f_inf6 = f_inf.mul_u64(6);
        let (f4, f5) = ex4_2(&[f0, f1, f2, f3], &f_inf6);
        let f6 = ex4(&[f2, f3, f4, f5], &f_inf6);
        (f4, f5, f6)
    }

    // SAFETY: `p[0..4]` is a valid 4-element subslice, reinterpreted as `[(F, F); 4]`.
    let (a1, a2, a3, a4, a_inf) =
        eval_linear_prod_4_internal(unsafe { *p[0..4].as_ptr().cast::<[(F, F); 4]>() });
    let (a5, a6, a7) = batch_helper(a1, a2, a3, a4, a_inf);
    // SAFETY: `p[4..8]` is a valid 4-element subslice, reinterpreted as `[(F, F); 4]`.
    let (b1, b2, b3, b4, b_inf) =
        eval_linear_prod_4_internal(unsafe { *p[4..8].as_ptr().cast::<[(F, F); 4]>() });
    let (b5, b6, b7) = batch_helper(b1, b2, b3, b4, b_inf);

    outputs[0] = a1 * b1;
    outputs[1] = a2 * b2;
    outputs[2] = a3 * b3;
    outputs[3] = a4 * b4;
    outputs[4] = a5 * b5;
    outputs[5] = a6 * b6;
    outputs[6] = a7 * b7;
    outputs[7] = a_inf * b_inf;
}

/// D=16: product of 16 linear polynomials at `{1, 2, ..., 15, ∞}`.
///
/// 8×8 Toom-Cook with sliding-window `ex8` extrapolation.
pub fn eval_prod_16_assign<F: Field>(p: &[(F, F); 16], outputs: &mut [F]) {
    debug_assert!(outputs.len() >= 16);

    // SAFETY: `p[0..8]` is a valid 8-element subslice, reinterpreted as `[(F, F); 8]`.
    let a = eval_linear_prod_8_internal(unsafe { *p[0..8].as_ptr().cast::<[(F, F); 8]>() });
    // SAFETY: `p[8..16]` is a valid 8-element subslice, reinterpreted as `[(F, F); 8]`.
    let b = eval_linear_prod_8_internal(unsafe { *p[8..16].as_ptr().cast::<[(F, F); 8]>() });

    for i in 0..8 {
        outputs[i] = a[i] * b[i];
    }

    let a_inf40320 = a[8].mul_u64(40320);
    let b_inf40320 = b[8].mul_u64(40320);

    let mut aw_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();
    let mut bw_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();

    let aw_ptr = aw_mu.as_mut_ptr();
    let bw_ptr = bw_mu.as_mut_ptr();

    // SAFETY: Writing to uninitialized storage; all read indices are written first.
    let aw_slice_ptr = unsafe { (*aw_ptr).as_mut_ptr() };
    // SAFETY: Same as above for `bw`.
    let bw_slice_ptr = unsafe { (*bw_ptr).as_mut_ptr() };

    // SAFETY: Copying 8 values from `a` and `b` into the first 8 slots,
    // and writing `a[8]`/`b[8]` at slot 15.
    unsafe {
        ptr::copy_nonoverlapping(a.as_ptr(), aw_slice_ptr, 8);
        ptr::write(aw_slice_ptr.add(15), a[8]);
        ptr::copy_nonoverlapping(b.as_ptr(), bw_slice_ptr, 8);
        ptr::write(bw_slice_ptr.add(15), b[8]);
    }

    for i in 0..7 {
        // SAFETY: Sliding window `[i..i+8]` over initialized data.
        let na = unsafe {
            let win_a_ptr = aw_slice_ptr.add(i) as *const [F; 8];
            ex8::<F>(&*win_a_ptr, a_inf40320)
        };
        // SAFETY: Same sliding window for `b`.
        let nb = unsafe {
            let win_b_ptr = bw_slice_ptr.add(i) as *const [F; 8];
            ex8::<F>(&*win_b_ptr, b_inf40320)
        };

        outputs[8 + i] = na * nb;

        // SAFETY: Writing newly computed values into slots 8..15.
        unsafe {
            ptr::write(aw_slice_ptr.add(8 + i), na);
            ptr::write(bw_slice_ptr.add(8 + i), nb);
        }
    }

    outputs[15] = a[8] * b[8];
}

/// D=32: product of 32 linear polynomials at `{1, 2, ..., 31, ∞}`.
///
/// 16×16 Toom-Cook with `ex16` expansion.
pub fn eval_prod_32_assign<F: Field>(p: &[(F, F); 32], outputs: &mut [F]) {
    // SAFETY: `p[0..16]` is a valid 16-element subslice, reinterpreted as `[(F, F); 16]`.
    let (a16_base, a_inf) =
        eval_half_16_base::<F>(unsafe { *p[0..16].as_ptr().cast::<[(F, F); 16]>() });
    let a_full = expand16_to_u32::<F>(&a16_base, a_inf);

    // SAFETY: `p[16..32]` is a valid 16-element subslice, reinterpreted as `[(F, F); 16]`.
    let (b16_base, b_inf) =
        eval_half_16_base::<F>(unsafe { *p[16..32].as_ptr().cast::<[(F, F); 16]>() });
    let b_full = expand16_to_u32::<F>(&b16_base, b_inf);

    for i in 0..32 {
        outputs[i] = a_full[i] * b_full[i];
    }
}

/// Naive $O(D^2)$ fallback for products of $D$ linear polynomials.
///
/// Evaluates at each grid point independently. Used for $D$ values that
/// lack specialized Toom-Cook implementations.
pub fn eval_linear_prod_naive_assign<F: Field>(pairs: &[(F, F)], evals: &mut [F]) {
    let d = pairs.len();
    debug_assert_eq!(evals.len(), d);
    if d == 0 {
        return;
    }

    // Initialize: cur_val[j] = p_j(1), pinf[j] = p_j(1) - p_j(0) (the slope)
    let mut cur_vals_pinfs: Vec<(F, F)> = Vec::with_capacity(d);
    for &(p0, p1) in pairs {
        cur_vals_pinfs.push((p1, p1 - p0));
    }

    // Evaluate at t = 1, 2, ..., D-1 by stepping cur_val += pinf each iteration
    for output in &mut evals[..(d - 1)] {
        let mut iter = cur_vals_pinfs.iter();
        let (first_val, _) = iter.next().expect("d > 0");
        let mut acc = *first_val;
        for (cur_val, _) in iter {
            acc *= *cur_val;
        }
        *output = acc;

        for (cur_val, pinf) in &mut cur_vals_pinfs {
            *cur_val += *pinf;
        }
    }

    // Evaluate at ∞: product of slopes
    let mut iter = cur_vals_pinfs.iter();
    let (_, first_pinf) = iter.next().expect("d > 0");
    let mut acc_inf = *first_pinf;
    for (_, pinf) in iter {
        acc_inf *= *pinf;
    }
    evals[d - 1] = acc_inf;
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// Naive reference: evaluate Π p_j(t) at each point in the Toom-Cook grid.
    fn reference_eval(pairs: &[(Fr, Fr)]) -> Vec<Fr> {
        let d = pairs.len();
        let mut evals = vec![Fr::zero(); d];

        // t = 1, 2, ..., D-1
        for t in 1..d {
            let t_f = Fr::from_u64(t as u64);
            let mut prod = Fr::one();
            for &(p0, p1) in pairs {
                let delta = p1 - p0;
                prod *= p0 + t_f * delta;
            }
            evals[t - 1] = prod;
        }

        // t = ∞: product of slopes
        let mut prod_inf = Fr::one();
        for &(p0, p1) in pairs {
            prod_inf *= p1 - p0;
        }
        evals[d - 1] = prod_inf;

        evals
    }

    fn random_pairs(d: usize, seed: u64) -> Vec<(Fr, Fr)> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..d)
            .map(|_| (Fr::random(&mut rng), Fr::random(&mut rng)))
            .collect()
    }

    #[test]
    fn eval_prod_2_matches_reference() {
        let pairs = random_pairs(2, 100);
        let expected = reference_eval(&pairs);
        let mut actual = vec![Fr::zero(); 2];
        let p: [(Fr, Fr); 2] = [pairs[0], pairs[1]];
        eval_prod_2_assign(&p, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn eval_prod_3_matches_reference() {
        let pairs = random_pairs(3, 200);
        let expected = reference_eval(&pairs);
        let mut actual = vec![Fr::zero(); 3];
        let p: [(Fr, Fr); 3] = [pairs[0], pairs[1], pairs[2]];
        eval_prod_3_assign(&p, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn eval_prod_4_matches_reference() {
        let pairs = random_pairs(4, 300);
        let expected = reference_eval(&pairs);
        let mut actual = vec![Fr::zero(); 4];
        let p: [(Fr, Fr); 4] = core::array::from_fn(|i| pairs[i]);
        eval_prod_4_assign(&p, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn eval_prod_5_matches_reference() {
        let pairs = random_pairs(5, 400);
        let expected = reference_eval(&pairs);
        let mut actual = vec![Fr::zero(); 5];
        let p: [(Fr, Fr); 5] = core::array::from_fn(|i| pairs[i]);
        eval_prod_5_assign(&p, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn eval_prod_6_matches_reference() {
        let pairs = random_pairs(6, 500);
        let expected = reference_eval(&pairs);
        let mut actual = vec![Fr::zero(); 6];
        let p: [(Fr, Fr); 6] = core::array::from_fn(|i| pairs[i]);
        eval_prod_6_assign(&p, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn eval_prod_7_matches_reference() {
        let pairs = random_pairs(7, 600);
        let expected = reference_eval(&pairs);
        let mut actual = vec![Fr::zero(); 7];
        let p: [(Fr, Fr); 7] = core::array::from_fn(|i| pairs[i]);
        eval_prod_7_assign(&p, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn eval_prod_8_matches_reference() {
        let pairs = random_pairs(8, 700);
        let expected = reference_eval(&pairs);
        let mut actual = vec![Fr::zero(); 8];
        let p: [(Fr, Fr); 8] = core::array::from_fn(|i| pairs[i]);
        eval_prod_8_assign(&p, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn eval_prod_16_matches_reference() {
        let pairs = random_pairs(16, 800);
        let expected = reference_eval(&pairs);
        let mut actual = vec![Fr::zero(); 16];
        let p: [(Fr, Fr); 16] = core::array::from_fn(|i| pairs[i]);
        eval_prod_16_assign(&p, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn eval_prod_32_matches_reference() {
        let pairs = random_pairs(32, 900);
        let expected = reference_eval(&pairs);
        let mut actual = vec![Fr::zero(); 32];
        let p: [(Fr, Fr); 32] = core::array::from_fn(|i| pairs[i]);
        eval_prod_32_assign(&p, &mut actual);
        assert_eq!(actual, expected);
    }

    #[test]
    fn dispatcher_routes_correctly() {
        for d in [2, 3, 4, 5, 6, 7, 8, 16, 32] {
            let pairs = random_pairs(d, 1000 + d as u64);
            let expected = reference_eval(&pairs);
            let mut actual = vec![Fr::zero(); d];
            eval_linear_prod_assign(&pairs, &mut actual);
            assert_eq!(actual, expected, "mismatch for D={d}");
        }
    }

    #[test]
    fn dispatcher_naive_fallback() {
        for d in [9, 10, 11, 13, 15, 20] {
            let pairs = random_pairs(d, 2000 + d as u64);
            let expected = reference_eval(&pairs);
            let mut actual = vec![Fr::zero(); d];
            eval_linear_prod_assign(&pairs, &mut actual);
            assert_eq!(actual, expected, "mismatch for D={d}");
        }
    }

    #[test]
    fn naive_matches_reference() {
        for d in [2, 3, 4, 5, 8, 16] {
            let pairs = random_pairs(d, 3000 + d as u64);
            let expected = reference_eval(&pairs);
            let mut actual = vec![Fr::zero(); d];
            eval_linear_prod_naive_assign(&pairs, &mut actual);
            assert_eq!(actual, expected, "naive mismatch for D={d}");
        }
    }

    #[test]
    fn d2_known_values() {
        // p0(x) = 2 + 3x, p1(x) = 5 + 7x
        let pairs = [
            (Fr::from_u64(2), Fr::from_u64(5)),
            (Fr::from_u64(5), Fr::from_u64(12)),
        ];
        let mut evals = [Fr::zero(); 2];
        eval_prod_2_assign(&pairs, &mut evals);

        // P(1) = 5 * 12 = 60
        assert_eq!(evals[0], Fr::from_u64(60));
        // P(∞) = (5-2) * (12-5) = 3 * 7 = 21
        assert_eq!(evals[1], Fr::from_u64(21));
    }

    #[test]
    fn d4_known_values() {
        // All p_j(x) = 1 + x (i.e., p_j(0)=1, p_j(1)=2)
        let p: [(Fr, Fr); 4] = [(Fr::one(), Fr::from_u64(2)); 4];
        let mut evals = [Fr::zero(); 4];
        eval_prod_4_assign(&p, &mut evals);

        // P(t) = (1+t)^4
        // P(1) = 2^4 = 16
        assert_eq!(evals[0], Fr::from_u64(16));
        // P(2) = 3^4 = 81
        assert_eq!(evals[1], Fr::from_u64(81));
        // P(3) = 4^4 = 256
        assert_eq!(evals[2], Fr::from_u64(256));
        // P(∞) = 1^4 = 1 (slopes are all 1)
        assert_eq!(evals[3], Fr::one());
    }

    #[test]
    fn d8_known_values() {
        let p: [(Fr, Fr); 8] = [(Fr::one(), Fr::from_u64(2)); 8];
        let mut evals = [Fr::zero(); 8];
        eval_prod_8_assign(&p, &mut evals);

        // P(t) = (1+t)^8
        assert_eq!(evals[0], Fr::from_u64(2u64.pow(8))); // P(1) = 256
        assert_eq!(evals[1], Fr::from_u64(3u64.pow(8))); // P(2) = 6561
        assert_eq!(evals[6], Fr::from_u64(8u64.pow(8))); // P(7) = 8^8 = 16777216
        assert_eq!(evals[7], Fr::one());
    }

    #[test]
    fn extrapolation_ex2_correctness() {
        // f(x) = 3x^2 + 2x + 1 => f(∞) = 3 (leading coeff)
        // f(1) = 6, f(2) = 17
        // f(3) = 34
        let f = [Fr::from_u64(6), Fr::from_u64(17)];
        let f_inf = Fr::from_u64(3);
        let result = ex2(&f, &f_inf);
        assert_eq!(result, Fr::from_u64(34));
    }

    #[test]
    fn eval_prod_multiple_random_seeds() {
        for seed in 0..10 {
            for d in [4, 8, 16] {
                let pairs = random_pairs(d, seed * 100 + d as u64);
                let expected = reference_eval(&pairs);
                let mut actual = vec![Fr::zero(); d];
                eval_linear_prod_assign(&pairs, &mut actual);
                assert_eq!(actual, expected, "seed={seed}, D={d}");
            }
        }
    }
}
