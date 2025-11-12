use crate::{
    field::{BarrettReduce, FMAdd, JoltField},
    poly::{
        eq_poly::EqPolynomial, ra_poly::RaPolynomial, split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    utils::accumulation::Acc5S,
};
use num_traits::Zero;
use core::{mem::MaybeUninit, ptr};

/// Computes the univariate polynomial `g(X) = sum_j eq((r', X, j), r) * prod_i mle_i(X, j)`.
///
/// Note `claim` should equal `g(0) + g(1)`.
pub fn compute_mles_product_sum<F: JoltField>(
    mles: &[RaPolynomial<u8, F>],
    claim: F,
    eq_poly: &GruenSplitEqPolynomial<F>,
) -> UniPoly<F> {
    // Evaluate g(X) / eq(X, r[round]) at [1, 2, ..., |mles| - 1, ∞] using split-eq fold.
    let d = mles.len();
    let current_scalar = eq_poly.get_current_scalar();
    let sum_evals: Vec<F> = eq_poly
        .par_fold_out_in(
            || vec![F::Unreduced::<9>::zero(); d],
            |inner, g, _x_in, e_in| {
                // Build per-g pairs [(p0, p1); D]
                let mut pairs: Vec<(F, F)> = Vec::with_capacity(d);
                for mle in mles.iter() {
                    let p0 = mle.get_bound_coeff(2 * g);
                    let p1 = mle.get_bound_coeff(2 * g + 1);
                    pairs.push((p0, p1));
                }
                // Compute endpoints on U_D into a small Vec<F>
                let mut endpoints = vec![F::zero(); d];
                product_eval_univariate_assign(&pairs, &mut endpoints);
                // Accumulate with unreduced arithmetic
                for k in 0..d {
                    inner[k] += e_in.mul_unreduced::<9>(endpoints[k]);
                }
            },
            |_x_out, e_out, inner| {
                // Reduce inner lanes, scale by e_out (unreduced), return outer acc vector
                let mut out = vec![F::Unreduced::<9>::zero(); d];
                for k in 0..d {
                    let reduced_k = F::from_montgomery_reduce::<9>(inner[k]);
                    out[k] = e_out.mul_unreduced::<9>(reduced_k);
                }
                out
            },
            |mut a, b| {
                for k in 0..d {
                    a[k] += b[k];
                }
                a
            },
        )
        .into_iter()
        .map(|x| F::from_montgomery_reduce::<9>(x) * current_scalar)
        .collect();

    // Get r[round] from the eq polynomial
    let r_round = eq_poly.get_current_w();
    let eq_eval_at_0 = EqPolynomial::mle(&[F::zero()], &[r_round]);
    let eq_eval_at_1 = EqPolynomial::mle(&[F::one()], &[r_round]);

    // Obtain the eval at 0 from the claim.
    let eval_at_1 = sum_evals[0];
    let eval_at_0 = (claim - eq_eval_at_1 * eval_at_1) / eq_eval_at_0;

    // Interpolate the intermediate polynomial.
    let toom_evals = [&[eval_at_0], &*sum_evals].concat();
    let tmp_coeffs = UniPoly::from_evals_toom(&toom_evals).coeffs;

    // Add in the missing eq(X, r[round]) factor.
    // Note eq(X, r[round]) = (1 - r[round]) + (2r[round] - 1)X.
    let constant_coeff = F::one() - r_round;
    let x_coeff = r_round + r_round - F::one();
    let mut coeffs = vec![F::zero(); tmp_coeffs.len() + 1];
    for (i, coeff) in tmp_coeffs.into_iter().enumerate() {
        coeffs[i] += coeff * constant_coeff;
        coeffs[i + 1] += coeff * x_coeff;
    }

    UniPoly::from_coeff(coeffs)
}

/// Computes the product of `D` linear polynomials on `U_D = [1, 2, ..., D - 1, ∞]`.
///
/// The evaluations on `U_D` are assigned to `evals`.
///
/// Inputs:
/// - `pairs[j] = (p_j(0), p_j(1))`
/// - `evals`: output slice with layout `[1, 2, ..., D - 1, ∞]`
pub fn product_eval_univariate_assign<F: JoltField>(pairs: &[(F, F)], evals: &mut [F]) {
    match pairs.len() {
        2 => eval_inter2_final_op(pairs.try_into().unwrap(), evals, assign),
        3 => eval_inter3_final_op(pairs.try_into().unwrap(), evals, assign),
        4 => eval_inter4_final_op(pairs.try_into().unwrap(), evals, assign),
        8 => eval_inter8_final_op(pairs.try_into().unwrap(), evals, assign),
        16 => eval_inter16_final_op(pairs.try_into().unwrap(), evals, assign),
        32 => eval_inter32_final_op(pairs.try_into().unwrap(), evals, assign),
        _ => product_eval_univariate_naive_assign(pairs, evals),
    }
}

fn eval_inter2<F: JoltField>((p0, p1): (F, F), (q0, q1): (F, F)) -> (F, F, F) {
    let p_inf = p1 - p0;
    let p2 = p_inf + p1;
    let q_inf = q1 - q0;
    let q2 = q_inf + q1;
    let r1 = p1 * q1;
    let r2 = p2 * q2;
    let r_inf = p_inf * q_inf;
    (r1, r2, r_inf)
}

fn eval_inter2_final_op<F: JoltField>(p: &[(F, F); 2], outputs: &mut [F], op: impl Fn(&mut F, F)) {
    op(&mut outputs[0], p[0].1 * p[1].1); // 1
    op(&mut outputs[1], (p[0].1 - p[0].0) * (p[1].1 - p[1].0)); // ∞
}

fn eval_inter3_final_op<F: JoltField>(
    pairs: &[(F, F); 3],
    outputs: &mut [F],
    op: impl Fn(&mut F, F),
) {
    let (a1, a2, a_inf) = eval_inter2(pairs[0], pairs[1]);
    let (b0, b1) = pairs[2];
    let b_inf = b1 - b0;
    let b2 = b1 + b_inf;
    op(&mut outputs[0], a1 * b1);
    op(&mut outputs[1], a2 * b2);
    op(&mut outputs[2], a_inf * b_inf);
}

fn eval_inter4<F: JoltField>(p: [(F, F); 4]) -> (F, F, F, F, F) {
    let (a1, a2, a_inf) = eval_inter2(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let a4 = ex2(&[a2, a3], &a_inf);
    let (b1, b2, b_inf) = eval_inter2(p[2], p[3]);
    let b3 = ex2(&[b1, b2], &b_inf);
    let b4 = ex2(&[b2, b3], &b_inf);
    (a1 * b1, a2 * b2, a3 * b3, a4 * b4, a_inf * b_inf)
}

fn eval_inter4_final_op<F: JoltField>(p: &[(F, F); 4], outputs: &mut [F], op: impl Fn(&mut F, F)) {
    let (a1, a2, a_inf) = eval_inter2(p[0], p[1]);
    let a3 = ex2(&[a1, a2], &a_inf);
    let (b1, b2, b_inf) = eval_inter2(p[2], p[3]);
    let b3 = ex2(&[b1, b2], &b_inf);
    op(&mut outputs[0], a1 * b1); // 1
    op(&mut outputs[1], a2 * b2); // 2
    op(&mut outputs[2], a3 * b3); // 3
    op(&mut outputs[3], a_inf * b_inf); // ∞
}

fn eval_inter8<F: JoltField>(p: [(F, F); 8]) -> [F; 9] {
    #[inline]
    fn batch_helper<F: JoltField>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F, F) {
        let f_inf6 = f_inf.mul_u64(6);
        let (f4, f5) = ex4_2(&[f0, f1, f2, f3], &f_inf6);
        let (f6, f7) = ex4_2(&[f2, f3, f4, f5], &f_inf6);
        (f4, f5, f6, f7)
    }
    let (a1, a2, a3, a4, a_inf) = eval_inter4(unsafe { *(p[0..4].as_ptr() as *const [(F, F); 4]) });
    let (a5, a6, a7, a8) = batch_helper(a1, a2, a3, a4, a_inf);
    let (b1, b2, b3, b4, b_inf) = eval_inter4(unsafe { *(p[4..8].as_ptr() as *const [(F, F); 4]) });
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

fn eval_inter8_final_op<F: JoltField>(p: &[(F, F); 8], outputs: &mut [F], op: impl Fn(&mut F, F)) {
    #[inline]
    fn batch_helper<F: JoltField>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F) {
        let f_inf6 = f_inf.mul_u64(6);
        let (f4, f5) = ex4_2(&[f0, f1, f2, f3], &f_inf6);
        let f6 = ex4(&[f2, f3, f4, f5], &f_inf6);
        (f4, f5, f6)
    }
    let (a1, a2, a3, a4, a_inf) = eval_inter4(unsafe { *(p[0..4].as_ptr() as *const [(F, F); 4]) });
    let (a5, a6, a7) = batch_helper(a1, a2, a3, a4, a_inf);
    let (b1, b2, b3, b4, b_inf) = eval_inter4(unsafe { *(p[4..8].as_ptr() as *const [(F, F); 4]) });
    let (b5, b6, b7) = batch_helper(b1, b2, b3, b4, b_inf);

    op(&mut outputs[0], a1 * b1);
    op(&mut outputs[1], a2 * b2);
    op(&mut outputs[2], a3 * b3);
    op(&mut outputs[3], a4 * b4);
    op(&mut outputs[4], a5 * b5);
    op(&mut outputs[5], a6 * b6);
    op(&mut outputs[6], a7 * b7);
    op(&mut outputs[7], a_inf * b_inf);
}

fn eval_inter16_final_op<F: JoltField>(
    p: &[(F, F); 16],
    outputs: &mut [F],
    op: impl Fn(&mut F, F),
) {
    debug_assert!(outputs.len() >= 16);
    let a = eval_inter8(unsafe { *(p[0..8].as_ptr() as *const [(F, F); 8]) });
    let b = eval_inter8(unsafe { *(p[8..16].as_ptr() as *const [(F, F); 8]) });
    // Emit indices 1..8 directly
    for i in 0..8 {
        let v = a[i] * b[i];
        op(&mut outputs[i], v);
    }
    // Slide both 8-wide windows using pointer windows over a scratch buffer (no per-iter shifts)
    let a_inf40320 = a[8].mul_u64(40320);
    let b_inf40320 = b[8].mul_u64(40320);
    // Scratch buffers: seed first 8, prewrite slot 15 with inf for the final window
    let mut aw_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();
    let mut bw_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();
    let aw_ptr = aw_mu.as_mut_ptr();
    let bw_ptr = bw_mu.as_mut_ptr();
    let aw_slice_ptr = unsafe { (*aw_ptr).as_mut_ptr() };
    let bw_slice_ptr = unsafe { (*bw_ptr).as_mut_ptr() };
    unsafe {
        ptr::copy_nonoverlapping(a.as_ptr(), aw_slice_ptr, 8);
        ptr::write(aw_slice_ptr.add(15), a[8]);
        ptr::copy_nonoverlapping(b.as_ptr(), bw_slice_ptr, 8);
        ptr::write(bw_slice_ptr.add(15), b[8]);
    }
    for i in 0..7 {
        // Window over aw[i..i+8] and bw[i..i+8] without bounds checks
        let na = unsafe {
            let win_a_ptr = aw_slice_ptr.add(i) as *const [F; 8];
            ex8::<F>(&*win_a_ptr, a_inf40320)
        };
        let nb = unsafe {
            let win_b_ptr = bw_slice_ptr.add(i) as *const [F; 8];
            ex8::<F>(&*win_b_ptr, b_inf40320)
        };
        let v = na * nb;
        op(&mut outputs[8 + i], v);
        // Append newly computed elements for subsequent windows
        unsafe {
            ptr::write(aw_slice_ptr.add(8 + i), na);
            ptr::write(bw_slice_ptr.add(8 + i), nb);
        }
    }
    // Write the inf slot at index 15
    let v_inf = a[8] * b[8];
    op(&mut outputs[15], v_inf);
}

#[inline(always)]
fn ex2<F: JoltField>(f: &[F; 2], f_inf: &F) -> F {
    dbl(f[1] + f_inf) - f[0]
}

#[inline]
fn ex4<F: JoltField>(f: &[F; 4], f_inf6: &F) -> F {
    // Natural-grid coeffs for target x+4: [1, -4, 6, -4] and 4!*a4 = 24*a4.
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

#[inline]
fn ex4_2<F: JoltField>(f: &[F; 4], f_inf6: &F) -> (F, F) {
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

#[inline(always)]
fn ex8<F: JoltField>(f: &[F; 8], f_inf40320: F) -> F {
    // P(9) from f[i]=P(i+1): 8(f[1]+f[7]) + 56(f[3]+f[5]) - 28(f[2]+f[6]) - 70 f[4] - f[0] + f_inf40320
    // Use signed accumulator to reduce only once.
    let mut acc: Acc5S<F> = Acc5S::zero();
    let t1 = f[1] + f[7];
    acc.fmadd(&t1, &8u64);
    let t2 = f[3] + f[5];
    acc.fmadd(&t2, &56u64);
    acc.fmadd(&f_inf40320, &1u64);

    let t3 = f[2] + f[6];
    acc.fmadd(&t3, &(-28i64));
    acc.fmadd(&f[4], &(-70i64));
    acc.fmadd(&f[0], &(-1i64));

    acc.barrett_reduce()
}

#[inline]
fn ex16<F: JoltField>(f: &[F; 16], f_inf16_fact: F) -> F {
    // P(17) from f[i]=P(i+1), i=0..15, using 16th-row binomial weights with alternating signs:
    // Coeffs on f[0..15]: [-1, +16, -120, +560, -1820, +4368, -8008, +11440, -12870, +11440, -8008, +4368, -1820, +560, -120, +16]
    // Plus + 16! * a16 (passed as f_inf16_fact).
    //
    // Group symmetric terms with equal coefficients and signs to minimize fmadd calls:
    // +16  : (f[1] + f[15])
    // -120 : (f[2] + f[14])
    // +560 : (f[3] + f[13])
    // -1820: (f[4] + f[12])
    // +4368: (f[5] + f[11])
    // -8008: (f[6] + f[10])
    // +11440: (f[7] + f[9])
    // Center and edges:
    // -12870 f[8], -1 f[0], + f_inf16_fact
    let mut acc: Acc5S<F> = Acc5S::zero();
    let s16 = f[1] + f[15];
    acc.fmadd(&s16, &16u64);
    let s120 = f[2] + f[14];
    acc.fmadd(&s120, &(-120i64));
    let s560 = f[3] + f[13];
    acc.fmadd(&s560, &560u64);
    let s1820 = f[4] + f[12];
    acc.fmadd(&s1820, &(-1820i64));
    let s4368 = f[5] + f[11];
    acc.fmadd(&s4368, &4368u64);
    let s8008 = f[6] + f[10];
    acc.fmadd(&s8008, &(-8008i64));
    let s11440 = f[7] + f[9];
    acc.fmadd(&s11440, &11440u64);
    acc.fmadd(&f[8], &(-12870i64));
    acc.fmadd(&f[0], &(-1i64));
    acc.fmadd(&f_inf16_fact, &1u64);
    acc.barrett_reduce()
}

fn eval_inter32_final_op<F: JoltField>(
    p: &[(F, F); 32],
    outputs: &mut [F],
    op: impl Fn(&mut F, F),
) {
    #[inline]
    fn eval_half_16_base<F: JoltField>(p: [(F, F); 16]) -> ([F; 16], F) {
        // Compute two 8-sized halves
        let a8 = eval_inter8(unsafe { *(p[0..8].as_ptr() as *const [(F, F); 8]) });
        let b8 = eval_inter8(unsafe { *(p[8..16].as_ptr() as *const [(F, F); 8]) });
        // Expand each 8 to 16 using ex8 sliding (compute 9..16)
        #[inline]
        fn expand8_to16<F: JoltField>(vals: &[F; 9]) -> ([F; 16], F) {
            // Build f[1..16] without zero-initialization; return also inf
            let mut f_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();
            let f_ptr = f_mu.as_mut_ptr();
            let f_slice_ptr = unsafe { (*f_ptr).as_mut_ptr() };
            // First 8 from vals
            unsafe {
                ptr::copy_nonoverlapping(vals.as_ptr(), f_slice_ptr, 8);
            }
            let f_inf = vals[8];
            let f_inf40320 = f_inf.mul_u64(40320);
            // Compute positions 9..16 (indices 8..15)
            for i in 0..8 {
                unsafe {
                    let win_ptr = f_slice_ptr.add(i) as *const [F; 8];
                    let win_ref: &[F; 8] = &*win_ptr;
                    let val: F = ex8(win_ref, f_inf40320);
                    ptr::write(f_slice_ptr.add(8 + i), val);
                }
            }
            let f = unsafe { f_mu.assume_init() };
            (f, f_inf)
        }
        let (a16_vals, a_inf) = expand8_to16::<F>(&a8);
        let (b16_vals, b_inf) = expand8_to16::<F>(&b8);
        // Pointwise product to get the 16-base for the half and its inf without zero-initialization
        let mut base_mu: MaybeUninit<[F; 16]> = MaybeUninit::uninit();
        let base_ptr = base_mu.as_mut_ptr();
        let base_slice_ptr = unsafe { (*base_ptr).as_mut_ptr() };
        for i in 0..16 {
            unsafe {
                ptr::write(base_slice_ptr.add(i), a16_vals[i] * b16_vals[i]);
            }
        }
        let base = unsafe { base_mu.assume_init() };
        (base, a_inf * b_inf)
    }
    #[inline]
    fn expand16_to_u32<F: JoltField>(base16: &[F; 16], inf: F) -> [F; 32] {
        // Build [1..31, inf] for a degree-16 product using ex16 sliding without zero-initialization
        let mut f_mu: MaybeUninit<[F; 32]> = MaybeUninit::uninit();
        let f_ptr = f_mu.as_mut_ptr();
        let f_slice_ptr = unsafe { (*f_ptr).as_mut_ptr() };
        // Initialize first 16 with base16
        unsafe {
            ptr::copy_nonoverlapping(base16.as_ptr(), f_slice_ptr, 16);
        }
        // Write inf at position 31 upfront (needed by the last window)
        unsafe {
            ptr::write(f_slice_ptr.add(31), inf);
        }
        let f_inf16_fact = inf.mul_u64(20922789888000u64); // 16!
        // Compute entries 17..31 (indices 16..30)
        for i in 0..15 {
            unsafe {
                let win_ptr = f_slice_ptr.add(i) as *const [F; 16];
                let win_ref: &[F; 16] = &*win_ptr;
                let val = ex16::<F>(win_ref, f_inf16_fact);
                ptr::write(f_slice_ptr.add(16 + i), val);
            }
        }
        unsafe { f_mu.assume_init() }
    }
    // First 16 polynomials → half A
    let (a16_base, a_inf) =
        eval_half_16_base::<F>(unsafe { *(p[0..16].as_ptr() as *const [(F, F); 16]) });
    let a_full = expand16_to_u32::<F>(&a16_base, a_inf);
    // Second 16 polynomials → half B
    let (b16_base, b_inf) =
        eval_half_16_base::<F>(unsafe { *(p[16..32].as_ptr() as *const [(F, F); 16]) });
    let b_full = expand16_to_u32::<F>(&b16_base, b_inf);
    // Combine
    for i in 0..32 {
        let mut v = a_full[i];
        v *= b_full[i];
        op(&mut outputs[i], v);
    }
}

#[inline(always)]
fn dbl<F: JoltField>(x: F) -> F {
    x + x
}

#[inline(always)]
fn dbl_assign<F: JoltField>(x: &mut F) {
    *x += *x;
}

#[inline(always)]
fn assign<T: Sized>(dst: &mut T, src: T) {
    *dst = src;
}

/// Naive evaluator for the product of `D` linear polynomials on `U_D = [1, 2, ..., D - 1, ∞]`.
///
/// The evaluations on `U_D` are accumulated into `sums`.
///
/// Inputs:
/// - `pairs[j] = (p_j(0), p_j(1))`
/// - `sums`: accumulator with layout `[1, 2, ..., D - 1, ∞]`
#[allow(dead_code)]
fn product_eval_univariate_naive_accumulate<F: JoltField>(pairs: &[(F, F)], sums: &mut [F]) {
    let d = pairs.len();
    debug_assert_eq!(sums.len(), d);
    if d == 0 {
        return;
    }
    // Memoize p(1)=p1, then p(2)=p(1)+pinf, p(3)=p(2)+pinf, ...
    let mut cur_vals = Vec::with_capacity(d);
    let mut pinfs = Vec::with_capacity(d);
    for &(p0, p1) in pairs.iter() {
        let pinf = p1 - p0;
        cur_vals.push(p1);
        pinfs.push(pinf);
    }
    // Evaluate at x = 1..(d-1)
    for idx in 0..(d - 1) {
        let mut acc = F::one();
        for v in cur_vals.iter() {
            acc *= *v;
        }
        sums[idx] += acc;
        // advance all to next x
        for i in 0..d {
            cur_vals[i] += pinfs[i];
        }
    }
    // Evaluate at infinity (product of leading coefficients)
    let mut acc_inf = F::one();
    for pinf in pinfs.iter() {
        acc_inf *= *pinf;
    }
    sums[d - 1] += acc_inf;
}

/// Naive evaluator for the product of `D` linear polynomials on `U_D = [1, 2, ..., D - 1, ∞]`.
///
/// The evaluations on `U_D` are assigned to `evals`.
///
/// Inputs:
/// - `pairs[j] = (p_j(0), p_j(1))`
/// - `evals`: output slice with layout `[1, 2, ..., D - 1, ∞]`
fn product_eval_univariate_naive_assign<F: JoltField>(pairs: &[(F, F)], evals: &mut [F]) {
    let d = pairs.len();
    debug_assert_eq!(evals.len(), d);
    if d == 0 {
        return;
    }
    let mut cur_vals = Vec::with_capacity(d);
    let mut pinfs = Vec::with_capacity(d);
    for &(p0, p1) in pairs.iter() {
        let pinf = p1 - p0;
        cur_vals.push(p1);
        pinfs.push(pinf);
    }
    for idx in 0..(d - 1) {
        let mut acc = F::one();
        for v in cur_vals.iter() {
            acc *= *v;
        }
        evals[idx] = acc;
        for i in 0..d {
            cur_vals[i] += pinfs[i];
        }
    }
    let mut acc_inf = F::one();
    for pinf in pinfs.iter() {
        acc_inf *= *pinf;
    }
    evals[d - 1] = acc_inf;
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::UniformRand;
    use dory::curve::test_rng;
    use rand::rngs::StdRng;
    use std::array::from_fn;

    use crate::{
        field::JoltField,
        poly::{
            dense_mlpoly::DensePolynomial,
            eq_poly::EqPolynomial,
            multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialEvaluation},
            ra_poly::RaPolynomial,
            split_eq_poly::GruenSplitEqPolynomial,
        },
        subprotocols::mles_product_sum::compute_mles_product_sum,
    };

    #[test]
    fn test_naive_eval_matches_optimized_with_4_mles() {
        const N_MLE: usize = 4;
        let mut rng = &mut test_rng();
        let r_whole = [<Fr as JoltField>::Challenge::rand(&mut rng)];
        let r: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let base_mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let claim = gen_product_mle(&base_mles).evaluate(r);
        let challenge_whole = [<Fr as JoltField>::Challenge::rand(&mut rng)];
        let challenge: &[<Fr as JoltField>::Challenge; 1] = &challenge_whole;
        // Direct definition computed before consuming base_mles
        let mle_challenge_product = base_mles
            .iter()
            .map(|p| p.evaluate(challenge))
            .product::<Fr>();
        let rhs = EqPolynomial::mle(challenge, r) * mle_challenge_product;
        let mles = base_mles.map(RaPolynomial::RoundN);

        let eq_poly = GruenSplitEqPolynomial::new(r, BindingOrder::LowToHigh);
        let sum_poly = compute_mles_product_sum(&mles, claim, &eq_poly);
        let lhs = sum_poly.evaluate(&challenge[0]);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_naive_eval_matches_optimized_with_8_mles() {
        const N_MLE: usize = 8;
        let mut rng = &mut test_rng();
        let r_whole = [<Fr as JoltField>::Challenge::rand(&mut rng)];
        let r: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let base_mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let claim = gen_product_mle(&base_mles).evaluate(r);
        let challenge_whole = [<Fr as JoltField>::Challenge::rand(&mut rng)];
        let challenge: &[<Fr as JoltField>::Challenge; 1] = &challenge_whole;
        // Direct definition computed before consuming base_mles
        let mle_challenge_product = base_mles
            .iter()
            .map(|p| p.evaluate(challenge))
            .product::<Fr>();
        let rhs = EqPolynomial::mle(challenge, r) * mle_challenge_product;
        let mles = base_mles.map(RaPolynomial::RoundN);

        let eq_poly = GruenSplitEqPolynomial::new(r, BindingOrder::LowToHigh);
        let sum_poly = compute_mles_product_sum(&mles, claim, &eq_poly);
        let lhs = sum_poly.evaluate(&challenge[0]);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_compute_mles_product_sum_with_2_mles() {
        const N_MLE: usize = 2;
        let mut rng = &mut test_rng();
        let r_whole = [<Fr as JoltField>::Challenge::rand(&mut rng)];
        let r: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let claim = gen_product_mle(&mles).evaluate(r);

        let r_whole = [<Fr as JoltField>::Challenge::rand(&mut rng)];
        let challenge: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mle_challenge_product = mles.iter().map(|p| p.evaluate(challenge)).product::<Fr>();
        let eval = EqPolynomial::mle(challenge, r) * mle_challenge_product;
        let mles = mles.map(RaPolynomial::RoundN);

        let eq_poly = GruenSplitEqPolynomial::new(r, BindingOrder::LowToHigh);
        let sum_poly = compute_mles_product_sum(&mles, claim, &eq_poly);

        assert_eq!(eval, sum_poly.evaluate(&challenge[0]));
    }

    #[test]
    fn test_compute_mles_product_sum_with_4_mles() {
        const N_MLE: usize = 4;
        let mut rng = &mut test_rng();
        let r_whole = [<Fr as JoltField>::Challenge::random(&mut rng)];
        let r: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let claim = gen_product_mle(&mles).evaluate(r);
        let r_whole = [<Fr as JoltField>::Challenge::rand(&mut rng)];
        let challenge: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mle_challenge_product = mles.iter().map(|p| p.evaluate(challenge)).product::<Fr>();
        let eval = EqPolynomial::mle(challenge, r) * mle_challenge_product;
        let mles = mles.map(RaPolynomial::RoundN);

        let eq_poly = GruenSplitEqPolynomial::new(r, BindingOrder::LowToHigh);
        let sum_poly = compute_mles_product_sum(&mles, claim, &eq_poly);

        assert_eq!(eval, sum_poly.evaluate(&challenge[0]));
    }

    #[test]
    fn test_compute_mles_product_sum_with_8_mles() {
        const N_MLE: usize = 8;
        let mut rng = &mut test_rng();
        let r_whole = [<Fr as JoltField>::Challenge::random(&mut rng)];
        let r: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let claim = gen_product_mle(&mles).evaluate(r);
        let r_whole = [<Fr as JoltField>::Challenge::rand(&mut rng)];
        let challenge: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mle_challenge_product = mles.iter().map(|p| p.evaluate(challenge)).product::<Fr>();
        let eval = EqPolynomial::mle(challenge, r) * mle_challenge_product;
        let mles = mles.map(RaPolynomial::RoundN);

        let eq_poly = GruenSplitEqPolynomial::new(r, BindingOrder::LowToHigh);
        let sum_poly = compute_mles_product_sum(&mles, claim, &eq_poly);

        assert_eq!(eval, sum_poly.evaluate(&challenge[0]));
    }

    #[test]
    fn test_compute_mles_product_sum_with_16_mles() {
        const N_MLE: usize = 16;
        let mut rng = &mut test_rng();
        let r_whole = [<Fr as JoltField>::Challenge::random(&mut rng)];
        let r: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let claim = gen_product_mle(&mles).evaluate(r);
        let r_whole = [<Fr as JoltField>::Challenge::random(&mut rng)];
        let challenge: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mle_challenge_product = mles.iter().map(|p| p.evaluate(challenge)).product::<Fr>();
        let eval = EqPolynomial::mle(challenge, r) * mle_challenge_product;
        let mles = mles.map(RaPolynomial::RoundN);

        let eq_poly = GruenSplitEqPolynomial::new(r, BindingOrder::LowToHigh);
        let sum_poly = compute_mles_product_sum(&mles, claim, &eq_poly);

        assert_eq!(eval, sum_poly.evaluate(&challenge[0]));
    }

    #[test]
    fn test_compute_mles_product_sum_with_32_mles() {
        const N_MLE: usize = 32;
        let mut rng = &mut test_rng();
        let r_whole = [<Fr as JoltField>::Challenge::random(&mut rng)];
        let r: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mles: [_; N_MLE] = from_fn(|_| random_mle(1, rng));
        let claim = gen_product_mle(&mles).evaluate(r);
        let r_whole = [<Fr as JoltField>::Challenge::random(&mut rng)];
        let challenge: &[<Fr as JoltField>::Challenge; 1] = &r_whole;
        let mle_challenge_product = mles.iter().map(|p| p.evaluate(challenge)).product::<Fr>();
        let eval = EqPolynomial::mle(challenge, r) * mle_challenge_product;
        let mles = mles.map(RaPolynomial::RoundN);

        let eq_poly = GruenSplitEqPolynomial::new(r, BindingOrder::LowToHigh);
        let sum_poly = compute_mles_product_sum(&mles, claim, &eq_poly);

        assert_eq!(eval, sum_poly.evaluate(&challenge[0]));
    }

    fn random_mle(n_vars: usize, rng: &mut StdRng) -> MultilinearPolynomial<Fr> {
        MultilinearPolynomial::LargeScalars(DensePolynomial::random(n_vars, rng))
    }

    /// Generates MLE `p(x) = sum_j eq(j, x) * prod_i mle_i(j)`.
    fn gen_product_mle(mles: &[MultilinearPolynomial<Fr>]) -> MultilinearPolynomial<Fr> {
        let n_vars = mles[0].get_num_vars();
        assert!(mles.iter().all(|mle| mle.get_num_vars() == n_vars));
        let res = (0..1 << n_vars)
            .map(|i| mles.iter().map(|mle| mle.get_bound_coeff(i)).product())
            .collect::<Vec<Fr>>();
        res.into()
    }
}
