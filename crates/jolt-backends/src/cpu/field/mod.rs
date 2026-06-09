use std::{mem::MaybeUninit, ptr};

use jolt_field::{
    AdditiveAccumulator, Field, RingAccumulator, SignedScalarAccumulator, WithAccumulator,
    WithSmallScalarAccumulator,
};

pub fn eval_linear_product_d2_assign<F: Field>(pairs: &[(F, F); 2], outputs: &mut [F]) {
    assert!(
        outputs.len() >= 2,
        "degree-2 linear product output must have at least 2 entries"
    );
    let (v1, _v2, v_inf) = eval_linear_product_2_internal(pairs[0], pairs[1]);
    outputs[0] = v1;
    outputs[1] = v_inf;
}

pub fn eval_linear_product_d3_assign<F: Field>(pairs: &[(F, F); 3], outputs: &mut [F]) {
    assert!(
        outputs.len() >= 3,
        "degree-3 linear product output must have at least 3 entries"
    );
    let (a1, a2, a_inf) = eval_linear_product_2_internal(pairs[0], pairs[1]);
    let (b0, b1) = pairs[2];
    let b_inf = b1 - b0;
    let b2 = b1 + b_inf;
    outputs[0] = a1 * b1;
    outputs[1] = a2 * b2;
    outputs[2] = a_inf * b_inf;
}

#[inline(always)]
pub fn eval_linear_product_d4_assign<F: Field>(pairs: &[(F, F); 4], outputs: &mut [F]) {
    debug_assert!(
        outputs.len() >= 4,
        "degree-4 linear product output must have at least 4 entries"
    );
    let (a1, a2, a_inf) = eval_linear_product_2_internal(pairs[0], pairs[1]);
    let a3 = extrapolate_quadratic(&[a1, a2], a_inf);
    let (b1, b2, b_inf) = eval_linear_product_2_internal(pairs[2], pairs[3]);
    let b3 = extrapolate_quadratic(&[b1, b2], b_inf);
    outputs[0] = a1 * b1;
    outputs[1] = a2 * b2;
    outputs[2] = a3 * b3;
    outputs[3] = a_inf * b_inf;
}

pub fn eval_linear_product_d5_assign<F: Field>(pairs: &[(F, F); 5], outputs: &mut [F]) {
    assert!(
        outputs.len() >= 5,
        "degree-5 linear product output must have at least 5 entries"
    );
    let (a1, a2, a_inf) = eval_linear_product_2_internal(pairs[0], pairs[1]);
    let a3 = extrapolate_quadratic(&[a1, a2], a_inf);
    let a4 = extrapolate_quadratic(&[a2, a3], a_inf);

    let (tail1, tail2) = (pairs[3], pairs[4]);
    let (r1, r2, r_inf) = eval_linear_product_2_internal(tail1, tail2);
    let r3 = extrapolate_quadratic(&[r1, r2], r_inf);
    let r4 = extrapolate_quadratic(&[r2, r3], r_inf);

    let (lin0, lin1) = pairs[2];
    let delta = lin1 - lin0;
    let l1 = lin1;
    let l2 = l1 + delta;
    let l3 = l2 + delta;
    let l4 = l3 + delta;

    outputs[0] = a1 * l1 * r1;
    outputs[1] = a2 * l2 * r2;
    outputs[2] = a3 * l3 * r3;
    outputs[3] = a4 * l4 * r4;
    outputs[4] = a_inf * delta * r_inf;
}

pub fn eval_linear_product_d6_assign<F: Field>(pairs: &[(F, F); 6], outputs: &mut [F]) {
    assert!(
        outputs.len() >= 6,
        "degree-6 linear product output must have at least 6 entries"
    );
    eval_linear_product_sliding_assign(pairs, outputs);
}

pub fn eval_linear_product_d7_assign<F: Field>(pairs: &[(F, F); 7], outputs: &mut [F]) {
    assert!(
        outputs.len() >= 7,
        "degree-7 linear product output must have at least 7 entries"
    );
    eval_linear_product_sliding_assign(pairs, outputs);
}

pub fn eval_linear_product_d8_assign<F: Field>(pairs: &[(F, F); 8], outputs: &mut [F]) {
    assert!(
        outputs.len() >= 8,
        "degree-8 linear product output must have at least 8 entries"
    );

    // SAFETY: both sub-slices are length 4, correctly aligned, and disjoint.
    let (a1, a2, a3, a4, a_inf) =
        eval_linear_product_4_internal(unsafe { *pairs[0..4].as_ptr().cast::<[(F, F); 4]>() });
    let (a5, a6, a7) = extrapolate_quartic_next_three([a1, a2, a3, a4], a_inf);

    // SAFETY: both sub-slices are length 4, correctly aligned, and disjoint.
    let (b1, b2, b3, b4, b_inf) =
        eval_linear_product_4_internal(unsafe { *pairs[4..8].as_ptr().cast::<[(F, F); 4]>() });
    let (b5, b6, b7) = extrapolate_quartic_next_three([b1, b2, b3, b4], b_inf);

    outputs[0] = a1 * b1;
    outputs[1] = a2 * b2;
    outputs[2] = a3 * b3;
    outputs[3] = a4 * b4;
    outputs[4] = a5 * b5;
    outputs[5] = a6 * b6;
    outputs[6] = a7 * b7;
    outputs[7] = a_inf * b_inf;
}

pub fn eval_linear_product_d9_assign<F: Field>(pairs: &[(F, F); 9], outputs: &mut [F]) {
    assert!(
        outputs.len() >= 9,
        "degree-9 linear product output must have at least 9 entries"
    );

    // SAFETY: the prefix sub-slice has length 8, is correctly aligned, and is
    // within the fixed-size degree-9 input.
    let prefix =
        eval_linear_product_8_internal(unsafe { *pairs[0..8].as_ptr().cast::<[(F, F); 8]>() });

    let (lin_0, lin_1) = pairs[8];
    let delta = lin_1 - lin_0;
    let mut linear = lin_1;
    for (output, prefix_eval) in outputs.iter_mut().zip(prefix.iter()).take(8) {
        *output = *prefix_eval * linear;
        linear += delta;
    }
    outputs[8] = prefix[8] * delta;
}

pub fn eval_linear_product_d9_accumulate<F>(
    pairs: &[(F, F); 9],
    outputs: &mut [<F as WithAccumulator>::Accumulator],
) where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    assert!(
        outputs.len() >= 9,
        "degree-9 linear product output must have at least 9 entries"
    );

    // SAFETY: the prefix sub-slice has length 8, is correctly aligned, and is
    // within the fixed-size degree-9 input.
    let prefix =
        eval_linear_product_8_internal(unsafe { *pairs[0..8].as_ptr().cast::<[(F, F); 8]>() });

    let (lin_0, lin_1) = pairs[8];
    let delta = lin_1 - lin_0;
    let mut linear = lin_1;
    for (output, prefix_eval) in outputs.iter_mut().zip(prefix.iter()).take(8) {
        output.fmadd(*prefix_eval, linear);
        linear += delta;
    }
    outputs[8].fmadd(prefix[8], delta);
}

pub fn eval_linear_product_d16_assign<F: Field>(pairs: &[(F, F); 16], outputs: &mut [F]) {
    assert!(
        outputs.len() >= 16,
        "degree-16 linear product output must have at least 16 entries"
    );

    // SAFETY: both sub-slices are length 8, correctly aligned, and disjoint.
    let a = eval_linear_product_8_internal(unsafe { *pairs[0..8].as_ptr().cast::<[(F, F); 8]>() });
    // SAFETY: both sub-slices are length 8, correctly aligned, and disjoint.
    let b = eval_linear_product_8_internal(unsafe { *pairs[8..16].as_ptr().cast::<[(F, F); 8]>() });

    for i in 0..8 {
        outputs[i] = a[i] * b[i];
    }

    let a_inf40320 = a[8].mul_u64(40320);
    let b_inf40320 = b[8].mul_u64(40320);
    let mut a_window = MaybeUninit::<[F; 16]>::uninit();
    let mut b_window = MaybeUninit::<[F; 16]>::uninit();
    let a_window_ptr = a_window.as_mut_ptr().cast::<F>();
    let b_window_ptr = b_window.as_mut_ptr().cast::<F>();
    // SAFETY: the pointers reference uninitialized `[F; 16]` storage. We write
    // the initial windows and the final infinity slots before reading them.
    unsafe {
        ptr::copy_nonoverlapping(a.as_ptr(), a_window_ptr, 8);
        ptr::copy_nonoverlapping(b.as_ptr(), b_window_ptr, 8);
        ptr::write(a_window_ptr.add(15), a[8]);
        ptr::write(b_window_ptr.add(15), b[8]);
    }

    for i in 0..7 {
        // SAFETY: `i` ranges 0..7, so each window `i..i+8` stays inside
        // the initialized prefix grown by prior loop iterations.
        let next_a = extrapolate_octic_next_one(
            unsafe { &*a_window_ptr.add(i).cast::<[F; 8]>() },
            a_inf40320,
        );
        // SAFETY: same window invariant as above, for `b_window`.
        let next_b = extrapolate_octic_next_one(
            unsafe { &*b_window_ptr.add(i).cast::<[F; 8]>() },
            b_inf40320,
        );
        outputs[8 + i] = next_a * next_b;
        // SAFETY: `8 + i` ranges 8..15 and appends the next initialized value
        // needed by later sliding windows.
        unsafe {
            ptr::write(a_window_ptr.add(8 + i), next_a);
            ptr::write(b_window_ptr.add(8 + i), next_b);
        }
    }

    outputs[15] = a[8] * b[8];
}

pub fn eval_linear_product_d32_assign<F: Field>(pairs: &[(F, F); 32], outputs: &mut [F]) {
    assert!(
        outputs.len() >= 32,
        "degree-32 linear product output must have at least 32 entries"
    );

    // SAFETY: both sub-slices are length 16, correctly aligned, and disjoint.
    let (a16_base, a_inf) = eval_linear_product_16_base_internal(unsafe {
        *pairs[0..16].as_ptr().cast::<[(F, F); 16]>()
    });
    let a_full = expand16_to32(&a16_base, a_inf);

    // SAFETY: both sub-slices are length 16, correctly aligned, and disjoint.
    let (b16_base, b_inf) = eval_linear_product_16_base_internal(unsafe {
        *pairs[16..32].as_ptr().cast::<[(F, F); 16]>()
    });
    let b_full = expand16_to32(&b16_base, b_inf);

    for i in 0..32 {
        outputs[i] = a_full[i] * b_full[i];
    }
}

fn eval_linear_product_4_internal<F: Field>(pairs: [(F, F); 4]) -> (F, F, F, F, F) {
    let (a1, a2, a_inf) = eval_linear_product_2_internal(pairs[0], pairs[1]);
    let a3 = extrapolate_quadratic(&[a1, a2], a_inf);
    let a4 = extrapolate_quadratic(&[a2, a3], a_inf);
    let (b1, b2, b_inf) = eval_linear_product_2_internal(pairs[2], pairs[3]);
    let b3 = extrapolate_quadratic(&[b1, b2], b_inf);
    let b4 = extrapolate_quadratic(&[b2, b3], b_inf);
    (a1 * b1, a2 * b2, a3 * b3, a4 * b4, a_inf * b_inf)
}

fn eval_linear_product_8_internal<F: Field>(pairs: [(F, F); 8]) -> [F; 9] {
    // SAFETY: both sub-slices are length 4, correctly aligned, and disjoint.
    let (a1, a2, a3, a4, a_inf) =
        eval_linear_product_4_internal(unsafe { *pairs[0..4].as_ptr().cast::<[(F, F); 4]>() });
    let (a5, a6, a7, a8) = extrapolate_quartic_next_four([a1, a2, a3, a4], a_inf);

    // SAFETY: both sub-slices are length 4, correctly aligned, and disjoint.
    let (b1, b2, b3, b4, b_inf) =
        eval_linear_product_4_internal(unsafe { *pairs[4..8].as_ptr().cast::<[(F, F); 4]>() });
    let (b5, b6, b7, b8) = extrapolate_quartic_next_four([b1, b2, b3, b4], b_inf);

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

fn eval_linear_product_16_base_internal<F: Field>(pairs: [(F, F); 16]) -> ([F; 16], F) {
    // SAFETY: both sub-slices are length 8, correctly aligned, and disjoint.
    let a8 = eval_linear_product_8_internal(unsafe { *pairs[0..8].as_ptr().cast::<[(F, F); 8]>() });
    // SAFETY: both sub-slices are length 8, correctly aligned, and disjoint.
    let b8 =
        eval_linear_product_8_internal(unsafe { *pairs[8..16].as_ptr().cast::<[(F, F); 8]>() });

    let (a16_vals, a_inf) = expand8_to16(&a8);
    let (b16_vals, b_inf) = expand8_to16(&b8);

    (
        core::array::from_fn(|i| a16_vals[i] * b16_vals[i]),
        a_inf * b_inf,
    )
}

fn eval_linear_product_sliding_assign<F: Field, const D: usize>(
    pairs: &[(F, F); D],
    outputs: &mut [F],
) {
    let mut current_and_inf = [(F::zero(), F::zero()); D];
    for (slot, &(p0, p1)) in current_and_inf.iter_mut().zip(pairs.iter()) {
        let p_inf = p1 - p0;
        *slot = (p1, p_inf);
    }

    for output in outputs.iter_mut().take(D - 1) {
        let mut product = current_and_inf[0].0;
        for &(value, _) in current_and_inf.iter().skip(1) {
            product *= value;
        }
        *output = product;

        for (value, p_inf) in &mut current_and_inf {
            *value += *p_inf;
        }
    }

    let mut product_inf = current_and_inf[0].1;
    for &(_, p_inf) in current_and_inf.iter().skip(1) {
        product_inf *= p_inf;
    }
    outputs[D - 1] = product_inf;
}

pub fn accumulate_linear_product_d4<F>(products: &[[(F, F); 4]]) -> [F; 4]
where
    F: Field,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let mut outputs = [<F as WithAccumulator>::Accumulator::default(); 4];
    for pairs in products {
        let (a1, a2, a_inf) = eval_linear_product_2_internal(pairs[0], pairs[1]);
        let a3 = extrapolate_quadratic(&[a1, a2], a_inf);
        let (b1, b2, b_inf) = eval_linear_product_2_internal(pairs[2], pairs[3]);
        let b3 = extrapolate_quadratic(&[b1, b2], b_inf);
        outputs[0].fmadd(a1, b1);
        outputs[1].fmadd(a2, b2);
        outputs[2].fmadd(a3, b3);
        outputs[3].fmadd(a_inf, b_inf);
    }
    outputs.map(AdditiveAccumulator::reduce)
}

#[inline(always)]
fn eval_linear_product_2_internal<F: Field>((p0, p1): (F, F), (q0, q1): (F, F)) -> (F, F, F) {
    let p_inf = p1 - p0;
    let p2 = p_inf + p1;
    let q_inf = q1 - q0;
    let q2 = q_inf + q1;
    let r1 = p1 * q1;
    let r2 = p2 * q2;
    let r_inf = p_inf * q_inf;
    (r1, r2, r_inf)
}

#[inline]
fn extrapolate_quadratic<F: Field>(evals: &[F; 2], e_inf: F) -> F {
    evals[1] + (evals[1] - evals[0]) + e_inf + e_inf
}

#[inline]
fn extrapolate_quartic_next_three<F: Field>(evals: [F; 4], e_inf: F) -> (F, F, F) {
    let e_inf_times_six = e_inf.mul_u64(6);
    let (e4, e5) = extrapolate_quartic_next_two(&evals, e_inf_times_six);
    let e6 = extrapolate_quartic_next_one(&[evals[2], evals[3], e4, e5], e_inf_times_six);
    (e4, e5, e6)
}

#[inline]
fn extrapolate_quartic_next_four<F: Field>(evals: [F; 4], e_inf: F) -> (F, F, F, F) {
    let e_inf_times_six = e_inf.mul_u64(6);
    let (e4, e5) = extrapolate_quartic_next_two(&evals, e_inf_times_six);
    let (e6, e7) = extrapolate_quartic_next_two(&[evals[2], evals[3], e4, e5], e_inf_times_six);
    (e4, e5, e6, e7)
}

#[inline]
fn extrapolate_quartic_next_two<F: Field>(evals: &[F; 4], e_inf_times_six: F) -> (F, F) {
    let f3_minus_f2 = evals[3] - evals[2];
    let mut f4 = e_inf_times_six;
    f4 += f3_minus_f2;
    f4 += evals[1];
    f4 = f4 + f4;
    f4 -= evals[2];
    f4 = f4 + f4;
    f4 -= evals[0];

    let mut f5 = f4 - f3_minus_f2 + e_inf_times_six;
    f5 = f5 + f5;
    f5 -= evals[3];
    f5 = f5 + f5;
    f5 -= evals[1];

    (f4, f5)
}

#[inline]
fn extrapolate_quartic_next_one<F: Field>(evals: &[F; 4], e_inf_times_six: F) -> F {
    let mut next = e_inf_times_six;
    next += evals[3];
    next -= evals[2];
    next += evals[1];
    next = next + next;
    next -= evals[2];
    next = next + next;
    next -= evals[0];
    next
}

#[inline(always)]
fn extrapolate_octic_next_one<F>(evals: &[F; 8], e_inf_times_40320: F) -> F
where
    F: Field,
{
    let mut acc = <F as WithSmallScalarAccumulator>::SmallScalarAccumulator::default();
    acc.fmadd_u64(evals[1] + evals[7], 8);
    acc.fmadd_u64(evals[3] + evals[5], 56);
    acc.add(e_inf_times_40320);
    acc.fmadd_i64(evals[2] + evals[6], -28);
    acc.fmadd_i64(evals[4], -70);
    acc.fmadd_i64(evals[0], -1);
    acc.reduce()
}

fn expand8_to16<F: Field>(values: &[F; 9]) -> ([F; 16], F) {
    let mut expanded = MaybeUninit::<[F; 16]>::uninit();
    let expanded_ptr = expanded.as_mut_ptr().cast::<F>();

    // SAFETY: `expanded_ptr` references uninitialized `[F; 16]` storage. The
    // first 8 entries are copied before they can be read by sliding windows.
    unsafe {
        ptr::copy_nonoverlapping(values.as_ptr(), expanded_ptr, 8);
    }

    let value_inf = values[8];
    let value_inf40320 = value_inf.mul_u64(40320);

    for i in 0..8 {
        // SAFETY: `i` ranges 0..8, so each `i..i+8` window has been initialized
        // by the initial copy and prior loop writes. `8 + i` is then written once.
        unsafe {
            let window = &*expanded_ptr.add(i).cast::<[F; 8]>();
            let next = extrapolate_octic_next_one(window, value_inf40320);
            ptr::write(expanded_ptr.add(8 + i), next);
        }
    }

    // SAFETY: all 16 entries were initialized by the copy plus loop writes.
    (unsafe { expanded.assume_init() }, value_inf)
}

fn expand16_to32<F: Field>(base: &[F; 16], value_inf: F) -> [F; 32] {
    let mut expanded = MaybeUninit::<[F; 32]>::uninit();
    let expanded_ptr = expanded.as_mut_ptr().cast::<F>();

    // SAFETY: `expanded_ptr` references uninitialized `[F; 32]` storage. The
    // first 16 entries and final infinity slot are initialized before output.
    unsafe {
        ptr::copy_nonoverlapping(base.as_ptr(), expanded_ptr, 16);
        ptr::write(expanded_ptr.add(31), value_inf);
    }

    let value_inf16_fact = value_inf.mul_u64(20_922_789_888_000);

    for i in 0..15 {
        // SAFETY: `i` ranges 0..15, so each `i..i+16` window has been
        // initialized by the initial copy and prior loop writes. `16 + i`
        // ranges 16..31 and is written once before any later read.
        unsafe {
            let window = &*expanded_ptr.add(i).cast::<[F; 16]>();
            let next = extrapolate_hexadecic_next_one(window, value_inf16_fact);
            ptr::write(expanded_ptr.add(16 + i), next);
        }
    }

    // SAFETY: indices 0..15 were copied, 16..30 were written in the loop, and
    // 31 was written with the infinity value before initialization.
    unsafe { expanded.assume_init() }
}

#[inline(always)]
fn extrapolate_hexadecic_next_one<F>(evals: &[F; 16], e_inf_times_16_factorial: F) -> F
where
    F: Field,
{
    let mut acc = <F as WithSmallScalarAccumulator>::SmallScalarAccumulator::default();
    acc.fmadd_u64(evals[1] + evals[15], 16);
    acc.fmadd_i64(evals[2] + evals[14], -120);
    acc.fmadd_u64(evals[3] + evals[13], 560);
    acc.fmadd_i64(evals[4] + evals[12], -1820);
    acc.fmadd_u64(evals[5] + evals[11], 4368);
    acc.fmadd_i64(evals[6] + evals[10], -8008);
    acc.fmadd_u64(evals[7] + evals[9], 11440);
    acc.fmadd_i64(evals[8], -12870);
    acc.fmadd_i64(evals[0], -1);
    acc.add(e_inf_times_16_factorial);
    acc.reduce()
}
