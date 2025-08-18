use crate::field::JoltField;

// the main functions to look at are of the form coeff_kara_n
// they take a vector of n linear polynomials in coefficient form
// constant, slope, constant, slope, ...
// and return a vector containing n + 1 elements, corresponding to the coefficients
// of the polynomial p(x) = prod_i p_i(x) from lowest to highest degree

#[inline(always)]
fn double<F: JoltField>(a: &F) -> F {
    *a + *a
}

// given a linear polynomial (p(0), p(1)),
// return a vector containing the evaluations of p(x) at x = 0, 1, ..., n
#[inline(always)]
fn eval_linear<F: JoltField>(poly: &(F, F), n: usize) -> Vec<F> {
    assert!(n >= 2, "n must be at least 2");
    let p0 = poly.0;
    let p1 = poly.1;
    let diff = p1 - p0;
    let mut res = Vec::with_capacity(n + 1);
    res.push(p0);
    res.push(p1);
    for i in 2..=n {
        res.push(res[i - 1] + diff);
    }
    res
}

// given a vector of n linear polynomials
// [(p_1(0), p_1(1)), ..., (p_n(0), p_n(1))],
// return a vector containing n + 1 elements corresponding to the evaluation of
// prod_i p_i(x) at x = 0, 1, ..., n
pub fn naive_eval<F: JoltField>(polys: &[(F, F)]) -> Vec<F> {
    // legacy version on points 0..n
    // kept for reference
    let n = polys.len();
    let mut res = eval_linear(&polys[0], n);
    for i in 1..n {
        let next = eval_linear(&polys[i], n);
        for j in 0..=n {
            res[j] *= next[j];
        }
    }
    res
}

/// Evaluate product of linear polynomials on the "balanced" grid
/// {0, 1, -1, 2, -2, …, ⌊n/2⌋, -(n/2) (if n odd), ∞}  (total n+1 points).
/// Each input polynomial is given as (p(0), p(1)).
/// Infinity point means the forward difference p(1)-p(0).
fn balanced_grid<F>(n: usize) -> Vec<Option<F>>
where
    F: From<u64> + core::ops::Neg<Output = F> + JoltField,
{
    let mut xs = Vec::with_capacity(n + 1);
    xs.push(Some(F::zero())); // 0
    if n == 0 {
        xs.push(None);
        return xs;
    }
    xs.push(Some(F::one())); // 1
    let mut k = 1u64;
    while xs.len() < n {
        k += 1;
        xs.push(Some(F::from(k))); // +k
        if xs.len() < n {
            xs.push(Some(-F::from(k))); // -k
        }
    }
    xs.push(None); // ∞
    xs
}

fn eval_linear_balanced<F>(poly: &(F, F), xs: &[Option<F>]) -> Vec<F>
where
    F: JoltField,
{
    let diff = poly.1 - poly.0;
    xs.iter()
        .map(|&x| match x {
            Some(val) => poly.0 + diff * val,
            None => diff,
        })
        .collect()
}

/// Evaluate product of linear polynomials on the balanced grid {0,1,-1,2,-2,…,∞}
pub fn naive_eval_balanced<F>(polys: &[(F, F)]) -> Vec<F>
where
    F: JoltField + From<u64> + core::ops::Neg<Output = F>,
{
    let n = polys.len();
    if n == 0 {
        return vec![];
    }
    let xs = balanced_grid::<F>(n);
    let m = xs.len();
    let mut res = vec![F::one(); m];
    for poly in polys {
        let vals = eval_linear_balanced(poly, &xs);
        for j in 0..m {
            res[j] *= vals[j];
        }
    }
    res
}

// naive polynomial multiplication algorithm
// present here to test faster algorithms for correctness
#[inline(always)]
pub fn poly_mul<F: JoltField>(a: &[F], b: &[F]) -> Vec<F> {
    let n = a.len();
    let m = b.len();
    let mut res = vec![F::zero(); n + m - 1];
    for i in 0..n {
        for j in 0..m {
            res[i + j] += a[i] * b[j];
        }
    }
    res
}

// given a vector of n linear polynomials in coefficient form
// pairs are: (constant, slope)
// this is a naive algorithm for getting the coefficients of their product
// this is present for correctness testing
pub fn coeff_naive<F: JoltField>(polys: &[Vec<F>]) -> Vec<F> {
    let n = polys.len();
    let mut res = polys[0].clone();
    for i in 1..n {
        res = poly_mul(&res, &polys[i]);
    }
    res
}

// given a vector of n linear polynomials in coefficient form
// pairs are: (constant, slope)
// return a vector containing n + 1 elements, corresponding to the coefficients
// of the polynomial p(x) = prod_i p_i(x) from lowest to highest degree
// here we have variants specialized for 2, 4, 8, 16, and 32 linear polynomials

// this is the base case, there are two linear polynomials so they
// can be multiplied directly
#[inline(always)]
pub fn coeff_kara_2<F: JoltField>(left: &[F; 2], right: &[F; 2]) -> [F; 3] {
    kara_2(left, right)
}

#[inline(always)]
pub fn coeff_kara_3<F: JoltField>(ps: &[F; 6]) -> [F; 4] {
    // 2+1 split: first 2 polynomials vs last 1 polynomial
    let first_two_result =
        coeff_kara_2(&ps[0..2].try_into().unwrap(), &ps[2..4].try_into().unwrap()); // 3 mults
    let last_one = &ps[4..6]; // 3rd polynomial coefficients

    // Multiply degree-2 result × degree-1 polynomial (5 mults via kara_2x1)
    kara_2x1(&first_two_result, &last_one.try_into().unwrap())
    // Total: 3 + 5 = 8 multiplications
}

#[inline(always)]
pub fn coeff_kara_5<F: JoltField>(ps: &[F; 10]) -> [F; 6] {
    // 2+3 split: first 2 polynomials vs last 3 polynomials
    let first_two_result =
        coeff_kara_2(&ps[0..2].try_into().unwrap(), &ps[2..4].try_into().unwrap()); // 3 mults
    let last_three_result = coeff_kara_3(&ps[4..10].try_into().unwrap()); // 8 mults

    // Multiply degree-2 result × degree-3 result (12 mults via kara_3x2)
    kara_3x2(&last_three_result, &first_two_result)
    // Total: 3 + 8 + 12 = 23 multiplications
}

// note how this splits the list of polynomials into two halves
// and then makes a recursive call to get the polynomial coefficients
// corresponding to the two halves and then multiplies them together
#[inline(always)]
pub fn coeff_kara_4<F: JoltField>(left: &[F; 4], right: &[F; 4]) -> [F; 5] {
    kara_3(
        &coeff_kara_2(
            &left[0..2].try_into().unwrap(),
            &left[2..4].try_into().unwrap(),
        ),
        &coeff_kara_2(
            &right[0..2].try_into().unwrap(),
            &right[2..4].try_into().unwrap(),
        ),
    )
}

#[inline(always)]
pub fn coeff_kara_8<F: JoltField>(left: &[F; 8], right: &[F; 8]) -> [F; 9] {
    kara_5(
        &coeff_kara_4(
            &left[0..4].try_into().unwrap(),
            &left[4..].try_into().unwrap(),
        ),
        &coeff_kara_4(
            &right[..4].try_into().unwrap(),
            &right[4..8].try_into().unwrap(),
        ),
    )
}

#[inline(always)]
pub fn coeff_kara_16<F: JoltField>(left: &[F; 16], right: &[F; 16]) -> [F; 17] {
    kara_9(
        &coeff_kara_8(
            &left[..8].try_into().unwrap(),
            &left[8..].try_into().unwrap(),
        ),
        &coeff_kara_8(
            &right[..8].try_into().unwrap(),
            &right[8..].try_into().unwrap(),
        ),
    )
}

pub fn coeff_kara_32<F: JoltField>(left: &[F; 32], right: &[F; 32]) -> [F; 33] {
    kara_17(
        &coeff_kara_16(
            &left[..16].try_into().unwrap(),
            &left[16..].try_into().unwrap(),
        ),
        &coeff_kara_16(
            &right[..16].try_into().unwrap(),
            &right[16..].try_into().unwrap(),
        ),
    )
}

// add two degree 2 polynomials
#[inline(always)]
fn add_3<F: JoltField>(a: &[F; 3], b: &[F; 3]) -> [F; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

// take the difference of two degree 2 polynomials
#[inline(always)]
fn sub_3<F: JoltField>(a: &[F; 3], b: &[F; 3]) -> [F; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

// double a degree 2 polynomial
#[inline(always)]
fn double_3<F: JoltField>(a: &[F; 3]) -> [F; 3] {
    [double(&a[0]), double(&a[1]), double(&a[2])]
}

// toom-3 multiplication of quadratic polynomials
// given p(x) = p[0] + p[1]x + p[2]x^2
// and   q(x) = q[0] + q[1]x + q[2]x^2
// get p(0)q(0), p(1)q(1), p(-1)q(-1), p(2)q(2), p(inf)q(inf)
#[inline(always)]
pub fn toom_3<F: JoltField>(p: &[F; 3], q: &[F; 3]) -> [F; 5] {
    let e0 = p[0] * q[0];
    let ps = p[0] + p[2];
    let qs = q[0] + q[2];
    let e1 = (ps + p[1]) * (qs + q[1]);
    let en1 = (ps - p[1]) * (qs - q[1]);
    let e2 = (double(&(double(&p[2]) + p[1])) + p[0]) * (double(&(double(&q[2]) + q[1])) + q[0]);
    let einf = p[2] * q[2];
    [e0, e1, en1, e2, einf]
}

// d = 16, relies on coeff_kara_8
// given a list of 16 linear polynomials, get their product
// the result is 25 values, a linear transformation of which form the coefficients of the product polynomial
// for the purposes of sum-check, these 5x5 matrices can be added directly
// and then the expensive step of converting to coefficients can be done at the end
pub fn kara_toom_16<F: JoltField>(ps: &[F; 32]) -> [[F; 5]; 5] {
    let p = coeff_kara_8(
        &ps[0..8].try_into().unwrap(),
        &ps[8..16].try_into().unwrap(),
    );
    let q = coeff_kara_8(
        &ps[16..24].try_into().unwrap(),
        &ps[24..32].try_into().unwrap(),
    );
    let p0 = [p[0], p[1], p[2]];
    let p1 = [p[3], p[4], p[5]];
    let p2 = [p[6], p[7], p[8]];
    let q0 = [q[0], q[1], q[2]];
    let q1 = [q[3], q[4], q[5]];
    let q2 = [q[6], q[7], q[8]];
    let e0 = toom_3(&p0, &q0);
    let p02 = add_3(&p0, &p2);
    let q02 = add_3(&q0, &q2);
    let e1 = toom_3(&add_3(&p1, &p02), &add_3(&q1, &q02));
    let en1 = toom_3(&sub_3(&p02, &p1), &sub_3(&q02, &q1));
    let e2 = toom_3(
        &add_3(&double_3(&add_3(&double_3(&p2), &p1)), &p0),
        &add_3(&double_3(&add_3(&double_3(&q2), &q1)), &q0),
    );
    let einf = toom_3(&p2, &q2);
    [e0, e1, en1, e2, einf]
}

// takes the output of toom_3 and interpolates it to get the coefficients of the polynomial
pub fn toom_3_inter<F: JoltField>(p: &[F; 5]) -> [F; 5] {
    let c0 = p[0];
    let c1 = -p[0] / F::from_u16(2) + p[1] - p[2] / F::from_u16(3) - p[3] / F::from_u16(6)
        + p[4] * F::from_u16(2);
    let c2 = -p[0] + p[1] / F::from_u16(2) + p[2] / F::from_u16(2) - p[4];
    let c3 = p[0] / F::from_u16(2) - p[1] / F::from_u16(2) - p[2] / F::from_u16(6)
        + p[3] / F::from_u16(6)
        - p[4] * F::from_u16(2);
    let c4 = p[4];
    [c0, c1, c2, c3, c4]
}

// takes the output of kara_toom_16 and converts it to coefficients of the product polynomial
pub fn toom_16_to_coeff<F: JoltField>(toom: &[[F; 5]; 5]) -> [F; 17] {
    let t0 = toom_3_inter(&toom[0]); // p(0)
    let t1 = toom_3_inter(&toom[1]); // p(1)
    let t2 = toom_3_inter(&toom[2]); // p(-1)
    let t3 = toom_3_inter(&toom[3]); // p(2)
    let t4 = toom_3_inter(&toom[4]); // p(inf)
    let s0 = toom_3_inter(&[t0[0], t1[0], t2[0], t3[0], t4[0]]);
    let s1 = toom_3_inter(&[t0[1], t1[1], t2[1], t3[1], t4[1]]);
    let s2 = toom_3_inter(&[t0[2], t1[2], t2[2], t3[2], t4[2]]);
    let s3 = toom_3_inter(&[t0[3], t1[3], t2[3], t3[3], t4[3]]);
    let s4 = toom_3_inter(&[t0[4], t1[4], t2[4], t3[4], t4[4]]);
    let mut res = [F::zero(); 17];
    for i in 0..5 {
        let j = i * 3;
        res[j] += s0[i];
        res[j + 1] += s1[i];
        res[j + 2] += s2[i];
        res[j + 3] += s3[i];
        res[j + 4] += s4[i];
    }
    res
}

// fast Karatsuba multiplication for various polynomial sizes.
// the functions below take two forms
// the first are called kara_n and are somewhat generic Karatsuba multiplications
// the second are called kara_n_w_top and are specialized for the case where the
// highest degree coefficient was already computed

// (p[0] + p[1]x) * (q[0] + q[1]x)
#[inline(always)]
pub fn kara_2<F: JoltField>(p: &[F; 2], q: &[F; 2]) -> [F; 3] {
    let l = p[0] * q[0];
    let u = p[1] * q[1];
    let m = (p[0] + p[1]) * (q[0] + q[1]) - u - l;
    [l, m, u]
}

// (p[0] + p[1]x) * (q[0] + q[1]x) where u = p[1] * q[1] is already computed
#[inline(always)]
pub fn kara_2_w_top<F: JoltField>(p: &[F; 2], q: &[F; 2], u: F) -> [F; 2] {
    let l = p[0] * q[0];
    let m = (p[0] + p[1]) * (q[0] + q[1]) - u - l;
    [l, m]
}

// Helper: multiply degree-2 polynomial by degree-1 polynomial (optimized)
#[inline(always)]
fn kara_2x1<F: JoltField>(p: &[F; 3], q: &[F; 2]) -> [F; 4] {
    // p(x) = p₀ + p₁x + p₂x², q(x) = q₀ + q₁x
    // Split: p(x) = p_low(x) + x²p_high(x) where p_low = p₀ + p₁x, p_high = p₂

    // Step 1: p_low × q using kara_2 (3 mults)
    let p_low = [p[0], p[1]];
    let low_result = kara_2(&p_low, q); // [r₀, r₁, r₂]

    // Step 2: p_high × q = p₂ × (q₀ + q₁x) (2 mults)
    let high_q0 = p[2] * q[0];
    let high_q1 = p[2] * q[1];

    // Combine: result = low_result + x²(high_q0 + high_q1·x)
    [
        low_result[0],           // x⁰ term
        low_result[1],           // x¹ term
        low_result[2] + high_q0, // x² term
        high_q1,                 // x³ term
    ]
}

// Helper: multiply degree-3 polynomial by degree-2 polynomial (naive approach)
#[inline(always)]
fn kara_3x2<F: JoltField>(p: &[F; 4], q: &[F; 3]) -> [F; 6] {
    // p(x) = p₀ + p₁x + p₂x² + p₃x³, q(x) = q₀ + q₁x + q₂x²
    // Direct multiplication: (4 × 3 = 12 multiplications)

    [
        p[0] * q[0],                             // x⁰
        p[0] * q[1] + p[1] * q[0],               // x¹
        p[0] * q[2] + p[1] * q[1] + p[2] * q[0], // x²
        p[1] * q[2] + p[2] * q[1] + p[3] * q[0], // x³
        p[2] * q[2] + p[3] * q[1],               // x⁴
        p[3] * q[2],                             // x⁵
    ]
}

// etc.
#[inline(always)]
pub fn kara_3<F: JoltField>(p: &[F; 3], q: &[F; 3]) -> [F; 5] {
    let l = [p[0] * q[0]];
    let u = kara_2(p[1..].try_into().unwrap(), q[1..].try_into().unwrap());
    let mut m = kara_2_w_top(&[p[0] + p[1], p[2]], &[q[0] + q[1], q[2]], u[2]);
    m[0] -= l[0] + u[0];
    m[1] -= u[1];
    [l[0], m[0], m[1] + u[0], u[1], u[2]]
}

#[inline(always)]
pub fn kara_3_w_top<F: JoltField>(p: &[F; 3], q: &[F; 3], uu: F) -> [F; 4] {
    let l = [p[0] * q[0]];
    let u = kara_2_w_top(p[1..].try_into().unwrap(), q[1..].try_into().unwrap(), uu);
    let mut m = kara_2_w_top(&[p[0] + p[1], p[2]], &[q[0] + q[1], q[2]], uu);
    m[0] -= l[0] + u[0];
    m[1] -= u[1];
    [l[0], m[0], m[1] + u[0], u[1]]
}

#[inline(always)]
pub fn kara_4<F: JoltField>(p: &[F; 4], q: &[F; 4]) -> [F; 7] {
    let l = kara_2(p[0..2].try_into().unwrap(), q[0..2].try_into().unwrap());
    let u = kara_2(p[2..4].try_into().unwrap(), q[2..4].try_into().unwrap());
    let mut m = kara_2(&[p[0] + p[2], p[1] + p[3]], &[q[0] + q[2], q[1] + q[3]]);
    m[0] -= l[0] + u[0];
    m[1] -= l[1] + u[1];
    m[2] -= l[2] + u[2];
    [l[0], l[1], l[2] + m[0], m[1], m[2] + u[0], u[1], u[2]]
}

#[inline(always)]
pub fn kara_5<F: JoltField>(p: &[F; 5], q: &[F; 5]) -> [F; 9] {
    let l = kara_2(p[0..2].try_into().unwrap(), q[0..2].try_into().unwrap());
    let u = kara_3(p[2..5].try_into().unwrap(), q[2..5].try_into().unwrap());
    let mut m = kara_3_w_top(
        &[p[0] + p[2], p[1] + p[3], p[4]],
        &[q[0] + q[2], q[1] + q[3], q[4]],
        u[4],
    );
    m[0] -= l[0] + u[0];
    m[1] -= l[1] + u[1];
    m[2] -= l[2] + u[2];
    m[3] -= u[3];
    [
        l[0],
        l[1],
        l[2] + m[0],
        m[1],
        m[2] + u[0],
        m[3] + u[1],
        u[2],
        u[3],
        u[4],
    ]
}

#[inline(always)]
pub fn kara_5_w_top<F: JoltField>(p: &[F; 5], q: &[F; 5], uu: F) -> [F; 8] {
    let l = kara_2(p[0..2].try_into().unwrap(), q[0..2].try_into().unwrap());
    let u = kara_3_w_top(p[2..5].try_into().unwrap(), q[2..5].try_into().unwrap(), uu);
    let mut m = kara_3_w_top(
        &[p[0] + p[2], p[1] + p[3], p[4]],
        &[q[0] + q[2], q[1] + q[3], q[4]],
        uu,
    );
    m[0] -= l[0] + u[0];
    m[1] -= l[1] + u[1];
    m[2] -= l[2] + u[2];
    m[3] -= u[3];
    [
        l[0],
        l[1],
        l[2] + m[0],
        m[1],
        m[2] + u[0],
        m[3] + u[1],
        u[2],
        u[3],
    ]
}

// see the following two functions for an idea of how to extend this to arbitrary polynomial degrees
// in the even case, there are three calls to kara_N/2
// in the odd case, there is one call to kara_N/2, one call to kara_N/2 + 1, and one call to kara_N/2 + 1_w_top
// let k = floor(N / 2)
// given p and q, split them into p[0..k], p[k..N], q[0..k], q[k..N]
// then compute l = p[0..k] * q[0..k] and u = p[k..N] * q[k..N]
// then compute m = (p[0..k] + p[k..N]) * (q[0..k] + q[k..N]) - l - u
// then return l + x^k * m + x^{2k} * u

#[inline(always)]
pub fn kara_8<F: JoltField>(p: &[F; 8], q: &[F; 8]) -> [F; 15] {
    let l = kara_4(p[0..4].try_into().unwrap(), q[0..4].try_into().unwrap());
    let u = kara_4(p[4..8].try_into().unwrap(), q[4..8].try_into().unwrap());
    let mut ip = [F::zero(); 4];
    let mut iq = [F::zero(); 4];
    for i in 0..4 {
        ip[i] = p[i] + p[i + 4];
        iq[i] = q[i] + q[i + 4];
    }
    let mut m = kara_4(&ip, &iq);
    for i in 0..7 {
        m[i] -= l[i] + u[i];
    }
    let mut res = [F::zero(); 15];
    res[..l.len()].copy_from_slice(&l);
    res[8..].copy_from_slice(&u);
    for i in 0..m.len() {
        res[i + 4] += m[i];
    }
    res
}

#[inline(always)]
pub fn kara_9<F: JoltField>(p: &[F; 9], q: &[F; 9]) -> [F; 17] {
    let l = kara_4(p[0..4].try_into().unwrap(), q[0..4].try_into().unwrap());
    let u = kara_5(p[4..9].try_into().unwrap(), q[4..9].try_into().unwrap());
    let mut ip = [F::zero(); 5];
    let mut iq = [F::zero(); 5];
    for i in 0..4 {
        ip[i] = p[i] + p[i + 4];
        iq[i] = q[i] + q[i + 4];
    }
    ip[4] = p[8];
    iq[4] = q[8];
    let mut m = kara_5_w_top(&ip, &iq, u[8]);
    for i in 0..7 {
        m[i] -= l[i] + u[i];
    }
    m[7] -= u[7];
    let mut res = [F::zero(); 17];
    res[..l.len()].copy_from_slice(&l);
    res[8..].copy_from_slice(&u);

    for i in 0..m.len() {
        res[i + 4] += m[i];
    }
    res
}

#[inline(always)]
pub fn kara_9_w_top<F: JoltField>(p: &[F; 9], q: &[F; 9], uu: F) -> [F; 16] {
    let l = kara_4(p[0..4].try_into().unwrap(), q[0..4].try_into().unwrap());
    let u = kara_5_w_top(p[4..9].try_into().unwrap(), q[4..9].try_into().unwrap(), uu);
    let mut ip = [F::zero(); 5];
    let mut iq = [F::zero(); 5];
    for i in 0..4 {
        ip[i] = p[i] + p[i + 4];
        iq[i] = q[i] + q[i + 4];
    }
    ip[4] = p[8];
    iq[4] = q[8];
    let mut m = kara_5_w_top(&ip, &iq, uu);
    for i in 0..7 {
        m[i] -= l[i] + u[i];
    }
    m[7] -= u[7];
    let mut res = [F::zero(); 16];
    res[..l.len()].copy_from_slice(&l);
    res[8..].copy_from_slice(&u);

    for i in 0..m.len() {
        res[i + 4] += m[i];
    }
    res
}

#[inline(always)]
pub fn kara_17<F: JoltField>(p: &[F; 17], q: &[F; 17]) -> [F; 33] {
    let l = kara_8(p[0..8].try_into().unwrap(), q[0..8].try_into().unwrap());
    let u = kara_9(p[8..17].try_into().unwrap(), q[8..17].try_into().unwrap());
    let mut ip = [F::zero(); 9];
    let mut iq = [F::zero(); 9];
    for i in 0..8 {
        ip[i] = p[i] + p[i + 8];
        iq[i] = q[i] + q[i + 8];
    }
    ip[8] = p[16];
    iq[8] = q[16];
    let mut m = kara_9_w_top(&ip, &iq, u[16]);
    for i in 0..15 {
        m[i] -= l[i] + u[i];
    }
    m[15] -= u[15];
    let mut res = [F::zero(); 33];
    res[..l.len()].copy_from_slice(&l);
    res[16..].copy_from_slice(&u);

    for i in 0..m.len() {
        res[i + 8] += m[i];
    }
    res
}

// test
#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::UniformRand;
    use ark_std::rand::SeedableRng;

    use crate::subprotocols::karatsuba::{
        coeff_kara_16, coeff_kara_2, coeff_kara_32, coeff_kara_4, coeff_kara_8, coeff_naive,
    };

    // verify that eval_naive and eval_karatsuba give the same result
    #[test]
    fn test_eval() {
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0);
        let polys: Vec<Vec<Fr>> = (0..32)
            .map(|_| vec![Fr::rand(&mut rng), Fr::rand(&mut rng)])
            .collect();
        let flat: [Fr; 64] = polys
            .iter()
            .flat_map(|p| [p[0], p[1]])
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let naive_res = coeff_naive(&polys);
        let kara_res = coeff_kara_32(
            &flat[..32].try_into().unwrap(),
            &flat[32..].try_into().unwrap(),
        );
        assert_eq!(naive_res.len(), kara_res.len());
        for i in 0..naive_res.len() {
            assert_eq!(naive_res[i], kara_res[i]);
        }
    }
    // not really a test, just a way to see how many multiplications are done
    #[test]
    fn test_count() {
        let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(0);
        let p: [Fr; 64] = (0..64)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        coeff_kara_2(&p[0..2].try_into().unwrap(), &p[2..4].try_into().unwrap());
        coeff_kara_4(&p[0..4].try_into().unwrap(), &p[4..8].try_into().unwrap());
        coeff_kara_8(&p[0..8].try_into().unwrap(), &p[8..16].try_into().unwrap());
        coeff_kara_16(
            &p[0..16].try_into().unwrap(),
            &p[16..32].try_into().unwrap(),
        );
        coeff_kara_32(
            &p[0..32].try_into().unwrap(),
            &p[32..64].try_into().unwrap(),
        );
    }
}
