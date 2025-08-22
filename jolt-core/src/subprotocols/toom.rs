// Minimal multivariate Toom–Cook / Karatsuba helpers operating on lists of
// linear polynomials (p(0), p(1)).
//
// We keep them separate from the existing coefficient-oriented implementation
// so benchmarks can compare the evaluation-based approach against the older
// coefficient algorithms.
//
// No Counters instrumentation for now – we only care about functional
// correctness and speed.

use crate::field::JoltField;

#[inline(always)]
fn double<F: JoltField>(a: &F) -> F {
    *a + *a
}

#[inline(always)]
pub fn toom_eval_points<F: JoltField>(n: usize) -> Vec<F> {
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        points.push(F::from_u64(i as u64));
    }
    points
}

/// Extension trait exposing small integer multiplication for field elements.
/// Implement only for concrete field types that have an efficient intrinsic.
/// Default (blanket) implementation deliberately omitted so that code that
/// relies on these functions only compiles when the type truly supports it.
pub trait FieldMulSmall: JoltField {
    /// Multiply by an unsigned 64-bit constant efficiently.
    fn mul_u64(self, n: u64) -> Self;

    /// Multiply by an unsigned 128-bit constant efficiently.
    fn mul_u128(self, n: u128) -> Self;
}

// Efficient backend in arkworks for BN254.
impl FieldMulSmall for ark_bn254::Fr {
    #[inline(always)]
    fn mul_u64(self, n: u64) -> Self {
        // `Fp` impl is in ark-ff; call through via inherent method.
        ark_ff::Fp::mul_u64::<5>(self, n)
    }

    #[inline(always)]
    fn mul_u128(self, n: u128) -> Self {
        // `Fp` impl is in ark-ff; call through via inherent method.
        ark_ff::Fp::mul_u128::<5, 6>(self, n)
    }
}

/// doubling & tripling helpers (cheaper than generic scalar mul)
#[inline]
fn dbl<F: FieldMulSmall>(x: F) -> F {
    double(&x)
}
#[allow(dead_code)]
#[inline]
fn tpl<F: FieldMulSmall>(x: F) -> F {
    double(&x) + x
}

// -----------------------------------------------------------------------------
//  d = 2  (Karatsuba = 2-way Toom–Cook)
//  • inputs:  (p0,p1) and (q0,q1)
//  • outputs: product on  {0, 1, ∞}  as  (r0, r1, rInf)
//  total: 3 mults
// -----------------------------------------------------------------------------
#[inline]
pub fn eval_toom2<F: FieldMulSmall>((p0, p1): (F, F), (q0, q1): (F, F)) -> (F, F, F) {
    // 3 field multiplications
    let r0 = p0 * q0; // p(0) q(0)
    let r1 = p1 * q1; // p(1) q(1)
    let r_inf = (p1 - p0) * (q1 - q0); // top-coeff product
    (r0, r1, r_inf)
}

// -----------------------------------------------------------------------------
//  d = 2, n = 2  (two multilinear polynomials in two vars)
//  • inputs:  p, q given by their 4 values on {0,1}², order (00,10,01,11)
//  • outputs: product on {0,1,∞}² in the order
//      (0,0),(1,0),(0,1),(1,1),(∞,0),(∞,1),(0,∞),(1,∞),(∞,∞)
//  total: 9 mults
// -----------------------------------------------------------------------------
#[inline]
pub fn eval_toom2_2<F: FieldMulSmall>(p: [F; 4], q: [F; 4]) -> [F; 9] {
    // ---- precompute forward differences for p ----
    let p_x0 = p[1] - p[0]; // x-slope at y = 0
    let p_x1 = p[3] - p[2]; // x-slope at y = 1
    let p_y0 = p[2] - p[0]; // y-slope at x = 0
    let p_y1 = p[3] - p[1]; // y-slope at x = 1
    let p_xy = p_x1 - p_x0; // mixed term = p11

    // ---- same for q ----
    let q_x0 = q[1] - q[0];
    let q_x1 = q[3] - q[2];
    let q_y0 = q[2] - q[0];
    let q_y1 = q[3] - q[1];
    let q_xy = q_x1 - q_x0;

    // ---- point-wise products (9 mults) ----
    [
        // y = 0 row
        p[0] * q[0], // (0,0)
        p[1] * q[1], // (1,0)
        p_x0 * q_x0, // (∞,0)
        // y = 1 row
        p[2] * q[2], // (0,1)
        p[3] * q[3], // (1,1)
        p_x1 * q_x1, // (∞,1)
        // y = ∞ row
        p_y0 * q_y0, // (0,∞)
        p_y1 * q_y1, // (1,∞)
        p_xy * q_xy, // (∞,∞)
    ]
}

// -----------------------------------------------------------------------------
//  d = 3, n = 2  (three multilinear polynomials in 2 variables)
//  • inputs:  three arrays length 4 each with order (00,10,01,11)
//  • outputs: product on {0,1,−1,∞}² (16 points), row-major x-order [0,1,−1,∞]
// -----------------------------------------------------------------------------
#[inline]
pub fn eval_toom3_2<F: FieldMulSmall>(polys: [[F; 4]; 3]) -> [F; 16] {
    // Step 1: quadratic grid from first two polys
    let a = eval_toom2_2(polys[0], polys[1]);
    let idx3 = |x: usize, y: usize| -> usize { y * 3 + x };
    // Build 4×4 grid for A
    let mut a_grid = [[F::zero(); 4]; 4];
    // helper closure for x-extension
    #[inline]
    fn quad_x_m1<F: FieldMulSmall>(a0: F, a1: F, a_inf: F) -> F {
        dbl(a0) - a1 + dbl(a_inf)
    }
    // rows y=0,1,∞
    for (src_y, dst_y) in [(0usize, 0usize), (1, 1), (2, 3)] {
        let a0 = a[idx3(0, src_y)];
        let a1 = a[idx3(1, src_y)];
        let a_inf = a[idx3(2, src_y)];
        let a_m1 = quad_x_m1(a0, a1, a_inf);
        a_grid[dst_y] = [a0, a1, a_m1, a_inf];
    }
    // y = −1 row (dst 2)
    for x in 0..4 {
        let a0 = a_grid[0][x];
        let a1 = a_grid[1][x];
        let a_inf = a_grid[3][x];
        a_grid[2][x] = quad_x_m1(a0, a1, a_inf);
    }

    // Step 2: third linear poly values on grid
    let p = polys[2];
    let a0 = p[0];
    let bx = p[1] - p[0];
    let by = p[2] - p[0];
    let bxy = p[3] - p[1] - p[2] + p[0];
    let xs = [Some(F::zero()), Some(F::one()), Some(-F::one()), None];
    let ys = xs;
    let mut b_grid = [[F::zero(); 4]; 4];

    // Not unrolling for optimization as the function is only used in a test currently.
    for (i_y, y_opt) in ys.iter().enumerate() {
        for (i_x, x_opt) in xs.iter().enumerate() {
            b_grid[i_y][i_x] = match (x_opt, y_opt) {
                (Some(xv), Some(yv)) => a0 + bx * (*xv) + by * (*yv) + bxy * (*xv) * (*yv),
                (None, Some(yv)) => bx + bxy * (*yv),
                (Some(xv), None) => by + bxy * (*xv),
                (None, None) => bxy,
            };
        }
    }

    // Step 3: point-wise product
    let mut out = [F::zero(); 16];
    let mut k = 0usize;
    for y in 0..4 {
        for x in 0..4 {
            out[k] = a_grid[y][x] * b_grid[y][x];
            k += 1;
        }
    }
    out
}

// -----------------------------------------------------------------------------
//  d = 2, generic n  (two multilinear polynomials in n variables)
//  • inputs:  p_vals, q_vals – slices of length 2^n containing evaluations on
//              the boolean hypercube {0,1}^n.
//      Ordering assumption for {0,1}^n: lexicographic with variable 0 (least–
//      significant bit) fastest.  Concretely, index = Σ bit_i * 2^i.
//  • output: Vec of length 3^n holding evaluations of the product on the grid
//      {0,1,∞}^n, arranged in row-major base-3 order with the same convention
//      (trit 0,1,2=∞; variable 0 fastest).
//
//  Algorithm: recursively extend each polynomial from 2^n to 3^n by forward
//  differences along one variable at a time (variable n-1 outermost).  The
//  extension for a dimension stacks three (n-1)-dimensional blocks:
//      f(x_i=0, *)
//      f(x_i=1, *)
//      f(x_i=∞, *) = f(1,*) − f(0,*)
//  Complexity:  n·3^n additions, 3^n multiplications.
// -----------------------------------------------------------------------------
#[inline]
pub fn eval_toom2_n<F: FieldMulSmall>(p_vals: &[F], q_vals: &[F]) -> Vec<F> {
    assert_eq!(p_vals.len(), q_vals.len(), "p and q must have same length");
    let len2 = p_vals.len();
    assert!(len2.is_power_of_two(), "input length must be 2^n");
    // ---- derive n and pre-compute powers of 3 ------------------------------
    let n = len2.trailing_zeros() as usize; // since len2 = 2^n
    let mut pow3: Vec<usize> = Vec::with_capacity(n + 1);
    pow3.push(1);
    for i in 1..=n {
        pow3.push(pow3[i - 1] * 3);
    }
    let len3 = pow3[n];

    // ---- embed the {0,1}^n table into a {0,1,∞}^n table -------------------
    let mut embed = vec![F::zero(); len3];
    for (idx_bin, &val) in p_vals.iter().enumerate() {
        // convert binary index to base-3 index where bit → trit (0/1)
        let mut idx3 = 0usize;
        let mut t = idx_bin;
        for dim in 0..n {
            if t & 1 == 1 {
                idx3 += pow3[dim]; // trit 1 contributes 1·3^dim
            }
            t >>= 1;
        }
        embed[idx3] = val;
    }

    // same for q
    let mut embed_q = vec![F::zero(); len3];
    for (idx_bin, &val) in q_vals.iter().enumerate() {
        let mut idx3 = 0usize;
        let mut t = idx_bin;
        for dim in 0..n {
            if t & 1 == 1 {
                idx3 += pow3[dim];
            }
            t >>= 1;
        }
        embed_q[idx3] = val;
    }

    // ---- dimension sweep: fill ∞ trits in place ----------------------------
    // process dimensions from high to low so sources are ready
    for dim_rev in (0..n).rev() {
        let block = pow3[dim_rev]; // 3^{dim}
        let step = block * 3; // size of one complete slice along this dim
        let mut base = 0usize;
        while base < len3 {
            let left = base; // trit 0 block
            let right = base + block; // trit 1 block
            let inf = base + 2 * block; // trit 2 block (to fill)
            for i in 0..block {
                embed[inf + i] = embed[right + i] - embed[left + i];
                embed_q[inf + i] = embed_q[right + i] - embed_q[left + i];
            }
            base += step;
        }
    }

    // ---- point-wise product -------------------------------------------------
    embed.into_iter().zip(embed_q).map(|(a, b)| a * b).collect()
}

// -----------------------------------------------------------------------------
//  d = 3  (quadratic × linear approach)
//  • first: combine 2 linear terms via prod2 → quadratic on {0,1,∞}
//  • then: multiply quadratic × remaining linear → 4 coefficients
//  total: 3 mults (prod2) + 4 mults (evals) = 7 mults
// -----------------------------------------------------------------------------
#[inline]
pub fn eval_toom3<F: FieldMulSmall>(p: [(F, F); 3]) -> (F, F, F, F) {
    // ---- step 1: combine first two linear terms into quadratic ----
    let (a0, a1, a_inf) = eval_toom2(p[0], p[1]); // quadratic A on {0,1,∞}

    // ---- step 2: extend quadratic A to point -1 with caching ----
    // A(x) = a_0 + (a_1 - a_0)*x + a_inf*x*(x-1)
    // A(-1) = a_0 + (a_1 - a_0)*(-1) + a_inf*(-1)*(-1-1) = a_0 - (a_1 - a_0) + a_inf*(-1)*(-2)
    // A(-1) = a_0 - a_1 + a_0 + 2*a_inf = 2*a_0 - a_1 + 2*a_inf
    let a0_2 = dbl(a0); // Cache 2*a_0
    let a_inf_2 = dbl(a_inf); // Cache 2*a_inf
    let a_m1 = a0_2 - a1 + a_inf_2;

    // ---- step 3: evaluate remaining linear term p[2] at points ----
    let (c0, c1) = p[2];
    let c_inf = c1 - c0; // p[2] at ∞
    let c_m1 = c0 - c_inf; // p[2] at -1: (1 - (-1)) * c0 + (-1) * c1 = 2 * c0 - c1 = c0 - c_inf

    // ---- step 4: multiply quadratic × linear at 4 points (4 mults) ----
    let r0 = a0 * c0; // A(0) * p[2](0)
    let r1 = a1 * c1; // A(1) * p[2](1)
    let r_m1 = a_m1 * c_m1; // A(-1) * p[2](-1)
    let r_inf = a_inf * c_inf; // A(∞) * p[2](∞)

    // Total: 3 mults (prod2) + 4 mults (above) = 7 mults
    // For degree 3 → degree 4 result, we only need 4 points: {0, 1, -1, ∞}
    (r0, r1, r_m1, r_inf) // Return evaluations at {0, 1, -1, ∞}
}

// -----------------------------------------------------------------------------
//  d = 4  (binary split, then quadratic × quadratic)
//  • each quadratic A,B known on {0,1,∞} (3 * 2 mults)
//  • need them on {–1, 2}  via    A(X) = (1−X)A0 + X A1 + (X²−X)A∞
//  • multiply A,B on {0,1,–1,2,∞} (5 mults)
//  total: 3 * 2 + 5 = 11 mults
// -----------------------------------------------------------------------------
#[inline]
pub fn eval_toom4<F: FieldMulSmall>(p: [(F, F); 4]) -> [F; 5] {
    // ---- first level:  two independent quadratic products ----
    let (a0, a1, a_inf) = eval_toom2(p[0], p[1]); // A on {0,1,∞}
    let (b0, b1, b_inf) = eval_toom2(p[2], p[3]); // B on {0,1,∞}

    // ---- extend A to –1 and 2 (6 adds) ----
    let s0 = a0 + a_inf; // S0 = A0 + A∞
    let s1 = a1 + a_inf; // S1 = A1 + A∞

    let d0 = dbl(s0); // D0 = 2·S0
    let d1 = dbl(s1); // D1 = 2·S1

    let a_m1 = d0 - a1; // A(-1)
    let a_2 = d1 - a0; // A(2)

    // ---- same for B (6 adds) ----
    let t0 = b0 + b_inf;
    let t1 = b1 + b_inf;
    let e0 = dbl(t0);
    let e1 = dbl(t1);
    let b_m1 = e0 - b1;
    let b_2 = e1 - b0;

    // ---- final quadratic × quadratic at 5 points (5 muls) ----
    let r0 = a0 * b0;
    let r1 = a1 * b1;
    let r_m1 = a_m1 * b_m1;
    let r2 = a_2 * b_2;
    let r_inf = a_inf * b_inf;

    [r0, r1, r_m1, r2, r_inf]
}

// -----------------------------------------------------------------------------
//  d = 4, n = 2  (four multilinear polynomials in 2 variables)
//  • inputs:  p,q,r,s each given by 4 points on {0,1}², order (00,10,01,11)
//  • outputs: product on {0,1,−1,2,∞}²  (25 points),
//      returned row-major with x-order [0,1,−1,2,∞] and same for y.
//  Strategy:
//      1. Compute A = p·q and B = r·s on {0,1,∞}² using prod2_2   (18 mults).
//      2. Extend each quadratic grid from 3×3 to 5×5 using only additions.
//      3. Point-wise multiply A,B on 25 points (25 mults).
//  Total: 18 + 25 = 43 mults (vs naïve 75).
// -----------------------------------------------------------------------------
#[inline]
pub fn eval_toom4_2<F: FieldMulSmall>(polys: [[F; 4]; 4]) -> [F; 25] {
    // helper closure: extend one quadratic row (x-dimension)
    #[inline]
    fn quad_extend_row<F: FieldMulSmall>(a0: F, a1: F, a_inf: F) -> [F; 5] {
        let a_m1 = dbl(a0) - a1 + dbl(a_inf); // X = −1
        let a_2 = dbl(a1 + a_inf) - a0; // X = 2
        [a0, a1, a_m1, a_2, a_inf]
    }

    // ---- step 1: two sub-products on 3×3 grid --------------------------------
    let a = eval_toom2_2(polys[0], polys[1]); // A(x,y) values (9)
    let b = eval_toom2_2(polys[2], polys[3]); // B(x,y) values (9)

    // index helper to access 3×3 arrays (x in {0,1,∞}, y in {0,1,∞})
    let idx3 = |x: usize, y: usize| -> usize { y * 3 + x };

    // ---- step 2a: extend along X for existing Y = 0,1,∞ ----------------------
    // grids: rows[y][x]
    let mut a_grid = [[F::zero(); 5]; 5];
    let mut b_grid = [[F::zero(); 5]; 5];

    // y = 0 ------------------------------------------------
    let (a00, a10, a_inf0) = (a[idx3(0, 0)], a[idx3(1, 0)], a[idx3(2, 0)]);
    let (b00, b10, b_inf0) = (b[idx3(0, 0)], b[idx3(1, 0)], b[idx3(2, 0)]);
    a_grid[0] = quad_extend_row(a00, a10, a_inf0);
    b_grid[0] = quad_extend_row(b00, b10, b_inf0);

    // y = 1 ------------------------------------------------
    let (a01, a11, a_inf1) = (a[idx3(0, 1)], a[idx3(1, 1)], a[idx3(2, 1)]);
    let (b01, b11, b_inf1) = (b[idx3(0, 1)], b[idx3(1, 1)], b[idx3(2, 1)]);
    a_grid[1] = quad_extend_row(a01, a11, a_inf1);
    b_grid[1] = quad_extend_row(b01, b11, b_inf1);

    // y = ∞ (mapped to index 4) ----------------------------
    let (a0i, a1i, a_infi) = (a[idx3(0, 2)], a[idx3(1, 2)], a[idx3(2, 2)]);
    let (b0i, b1i, b_infi) = (b[idx3(0, 2)], b[idx3(1, 2)], b[idx3(2, 2)]);
    a_grid[4] = quad_extend_row(a0i, a1i, a_infi);
    b_grid[4] = quad_extend_row(b0i, b1i, b_infi);

    // ---- step 2b: extend along Y for all X -----------------------------------
    for x in 0..5 {
        let a0 = a_grid[0][x];
        let a1 = a_grid[1][x];
        let a_inf = a_grid[4][x];
        let a_m1 = dbl(a0) - a1 + dbl(a_inf); // Y = −1 (row 2)
        let a_2 = dbl(a1 + a_inf) - a0; // Y = 2  (row 3)
        a_grid[2][x] = a_m1;
        a_grid[3][x] = a_2;

        let b0 = b_grid[0][x];
        let b1 = b_grid[1][x];
        let b_inf = b_grid[4][x];
        let b_m1 = dbl(b0) - b1 + dbl(b_inf);
        let b_2 = dbl(b1 + b_inf) - b0;
        b_grid[2][x] = b_m1;
        b_grid[3][x] = b_2;
    }

    // ---- step 3: point-wise multiply on 5×5 grid ------------------------------
    let mut r = [F::zero(); 25];
    let mut k = 0usize;
    for y in 0..5 {
        for x in 0..5 {
            r[k] = a_grid[y][x] * b_grid[y][x];
            k += 1;
        }
    }
    r
}

// -----------------------------------------------------------------------------
//  d = 5  (quadratic × cubic via 2+3 split)
//  • quadratic A on {0,1,∞} from first 2 polynomials (3 mults)
//  • cubic B on {0,1,-1,∞} from next 3 polynomials (7 mults)
//  • extend both to {0,1,-1,2,-2,∞} and multiply pointwise (6 mults)
// total: 3 + 7 + 6 = 16 mults
// -----------------------------------------------------------------------------
#[inline]
pub fn eval_toom5<F: FieldMulSmall>(p: [(F, F); 5]) -> (F, F, F, F, F, F) {
    // ---- step 1: compute sub-products ----
    let (a0, a1, a_inf) = eval_toom2(p[0], p[1]); // quadratic A on {0,1,∞}
    let (b0, b1, b_m1, b_inf) = eval_toom3([p[2], p[3], p[4]]); // cubic B on {0,1,-1,∞}

    // ---- step 2: extend quadratic A to points {-1, 2, -2} ----
    // Using optimized formulas from prod4
    let s0 = a0 + a_inf;
    let s1 = a1 + a_inf;
    let d0 = dbl(s0);
    let d1 = dbl(s1);
    let a_m1 = d0 - a1; // A(-1)
    let a_2 = d1 - a0; // A(2)
    let a_m2 = tpl(a0) - dbl(a1) + a_inf.mul_u64(6); // A(-2) = 3a₀ - 2a₁ + 6a∞

    // ---- step 3: extend cubic B to points {2, -2} ----
    // Derived from Lagrange interpolation on {0,1,-1,∞}:
    // B(2) = -3B(0) + 3B(1) + B(-1) + 6B(∞)
    // B(-2) = -3B(0) + B(1) + 3B(-1) - 6B(∞)
    let b0_3 = tpl(b0); // Cache 3*B(0)
    let b_inf_6 = b_inf.mul_u64(6); // Cache 6*B(∞)

    let b_2 = tpl(b1) + b_m1 + b_inf_6 - b0_3; // B(2)
    let b_m2 = b1 + tpl(b_m1) - b0_3 - b_inf_6; // B(-2)

    // ---- step 4: pointwise multiplication at 6 points (6 mults) ----
    let r0 = a0 * b0; // X = 0
    let r1 = a1 * b1; // X = 1
    let r_m1 = a_m1 * b_m1; // X = -1
    let r_2 = a_2 * b_2; // X = 2
    let r_m2 = a_m2 * b_m2; // X = -2
    let r_inf = a_inf * b_inf; // X = ∞

    // Total: 3 (prod2) + 7 (prod3) + 6 (pointwise) = 16 mults
    (r0, r1, r_m1, r_2, r_m2, r_inf)
}

// -----------------------------------------------------------------------------
//  d = 8  (two quartics C,D;  5-point grid becomes 9-point grid)
// -----------------------------------------------------------------------------
#[inline]
pub fn eval_toom8<F: FieldMulSmall>(p: [(F, F); 8]) -> [F; 9] {
    // ---- first level: two quartic blocks ----------------------------------
    let [c0, c1, c_m1, c2, c_inf] = eval_toom4(p[0..4].try_into().unwrap());
    let [d0, d1, d_m1, d2, d_inf] = eval_toom4(p[4..8].try_into().unwrap());

    // ---- compute C(x) & D(x) on the four extra x values -------------------
    // Optimized: batch compute all 4 extra points for both C and D to avoid redundant computation
    #[inline]
    fn quartic_eval_batch<F: FieldMulSmall>(
        f0: F,
        f1: F,
        f_m1: F,
        f2: F,
        f_inf: F,
    ) -> (F, F, F, F) {
        // shared work for everything
        let f_inf6 = f_inf.mul_u64(6);
        // shared work for m2 and m3
        let a = f_inf6 - f0;
        let t = f1 + f_m1;
        // m2 specific work
        let x_m2 = dbl(dbl(a + t) - f0) - f2;
        // m3 specific work
        //let x_m3 = (dbl(dbl(a) + t) + f1).mul_u64(5) - f2.mul_u64(4);
        let x_m3 = (dbl(dbl(a) + t) + f1 - f2).mul_u64(5) + f2;
        // shared work for 3 and 4
        let b = f_inf6 - f1;
        let s = f0 + f2;
        // 3 specific work
        let x_3 = dbl(dbl(b + s) - f1) - f_m1;
        // 4 specific work
        //let x_4 = (dbl(dbl(b) + s) + f0).mul_u64(5) - f_m1.mul_u64(4); // this is slightly slower on my machine
        let x_4 = (dbl(dbl(b) + s) + f0 - f_m1).mul_u64(5) + f_m1;
        // return (m2, 3, m3, 4);
        (x_m2, x_3, x_m3, x_4)
    }

    let (c_m2, c3, c_m3, c4) = quartic_eval_batch(c0, c1, c_m1, c2, c_inf);
    let (d_m2, d3, d_m3, d4) = quartic_eval_batch(d0, d1, d_m1, d2, d_inf);

    // ---- final point-wise products (9 muls) -------------------------------
    [
        c0 * d0,       // X = 0
        c1 * d1,       // X = 1
        c_m1 * d_m1,   // X = –1
        c2 * d2,       // X = 2
        c_m2 * d_m2,   // X = –2
        c3 * d3,       // X = 3
        c_m3 * d_m3,   // X = –3
        c4 * d4,       // X = 4
        c_inf * d_inf, // X = ∞
    ]
}

// -----------------------------------------------------------------------------
//  d = 16  (two octics C,D;  9-point grid becomes 17-point grid)
// -----------------------------------------------------------------------------
#[inline]
pub fn eval_toom16<F: FieldMulSmall>(p: [(F, F); 16]) -> [F; 17] {
    // ---- first level: two octic blocks ------------------------------------
    let c = eval_toom8(p[0..8].try_into().unwrap()); // C(x) on 9 points
    let d = eval_toom8(p[8..16].try_into().unwrap()); // D(x) on 9 points

    // ---- compute C(x) & D(x) on the eight extra x values ------------------
    // Optimized: batch compute all 8 extra points for both C and D to avoid redundant computation
    #[inline]
    fn octic_eval_batch<F: FieldMulSmall>(vals: &[F; 9]) -> [F; 8] {
        // unpack for readability: [0,1,−1,2,−2,3,−3,4,∞]
        let f0 = vals[0];
        let f1 = vals[1];
        let f_m1 = vals[2];
        let f2 = vals[3];
        let f_m2 = vals[4];
        let f3 = vals[5];
        let f_m3 = vals[6];
        let f4 = vals[7];
        let f_inf = vals[8];

        // Pre-shared computation once per polynomial
        let f_m3_8 = f_m3.mul_u64(8);
        let f_m3_36 = f_m3.mul_u64(36);
        let f_m3_120 = f_m3.mul_u64(120);
        let f4_8 = f4.mul_u64(8);
        let f4_36 = f4.mul_u64(36);
        let f4_120 = f4.mul_u64(120);

        // pre-computed multiples of f_inf reused across several points
        let f_inf_40320 = f_inf.mul_u64(40320);
        let f_inf_362880 = f_inf.mul_u64(362_880);
        let f_inf_1814400 = f_inf.mul_u64(1_814_400);
        let f_inf_6652800 = f_inf.mul_u64(6_652_800);

        // Compute all 8 extra points at once (start with positive f_inf terms)
        [
            // X = −4
            f_inf_40320 + f1.mul_u64(56) + f_m1.mul_u64(56) + f3.mul_u64(8) + f_m3_8
                - f0.mul_u64(70)
                - f2.mul_u64(28)
                - f_m2.mul_u64(28)
                - f4,
            // X = 5
            f_inf_40320 + f0.mul_u64(56) + f2.mul_u64(56) + f_m2.mul_u64(8) + f4_8
                - f1.mul_u64(70)
                - f_m1.mul_u64(28)
                - f3.mul_u64(28)
                - f_m3,
            // X = −5
            f_inf_362880 + f1.mul_u64(420) + f_m1.mul_u64(378) + f3.mul_u64(63) + f_m3_36
                - f0.mul_u64(504)
                - f2.mul_u64(216)
                - f_m2.mul_u64(168)
                - f4_8,
            // X = 6
            f_inf_362880 + f0.mul_u64(420) + f2.mul_u64(378) + f_m2.mul_u64(63) + f4_36
                - f1.mul_u64(504)
                - f_m1.mul_u64(216)
                - f3.mul_u64(168)
                - f_m3_8,
            // X = −6
            f_inf_1814400 + f1.mul_u64(1800) + f_m1.mul_u64(1512) + f3.mul_u64(280) + f_m3_120
                - f0.mul_u64(2100)
                - f2.mul_u64(945)
                - f_m2.mul_u64(630)
                - f4_36,
            // X = 7
            f_inf_1814400 + f0.mul_u64(1800) + f2.mul_u64(1512) + f_m2.mul_u64(280) + f4_120
                - f1.mul_u64(2100)
                - f_m1.mul_u64(945)
                - f3.mul_u64(630)
                - f_m3_36,
            // X = −7
            f_inf_6652800
                + f1.mul_u64(5775)
                + f_m1.mul_u64(4620)
                + f3.mul_u64(924)
                + f_m3.mul_u64(330)
                - f0.mul_u64(6600)
                - f2.mul_u64(3080)
                - f_m2.mul_u64(1848)
                - f4_120,
            // X = 8
            f_inf_6652800
                + f0.mul_u64(5775)
                + f2.mul_u64(4620)
                + f_m2.mul_u64(924)
                + f4.mul_u64(330)
                - f1.mul_u64(6600)
                - f_m1.mul_u64(3080)
                - f3.mul_u64(1848)
                - f_m3_120,
        ]
    }

    let c_ext = octic_eval_batch(&c);
    let d_ext = octic_eval_batch(&d);

    // ---- final point-wise products on 17 points ---------------------------
    let mut r = [F::zero(); 17];
    // first 8 points from prod8: [0, 1, −1, 2, −2, 3, −3, 4]
    for i in 0..8 {
        r[i] = c[i] * d[i];
    }
    // next 8 points: X = −4,5,−5,6,−6,7,−7,8
    for i in 0..8 {
        r[8 + i] = c_ext[i] * d_ext[i];
    }
    // ∞ point goes last
    r[16] = c[8] * d[8];
    r
}

#[cfg(test)]
mod tests {
    use crate::{poly::unipoly::UniPoly, subprotocols::karatsuba::coeff_kara_16};

    use super::*;
    use ark_bn254::Fr;
    use ark_ff::{One, UniformRand, Zero};
    use ark_std::test_rng;

    fn eval_product(polys: &[(Fr, Fr)], x: Fr) -> Fr {
        polys
            .iter()
            .map(|(e0, e1)| {
                let slope = *e1 - *e0;
                *e0 + slope * x
            })
            .fold(Fr::one(), |acc, v| acc * v)
    }

    #[test]
    fn toom2_correct() {
        let mut rng = test_rng();
        let p = (Fr::rand(&mut rng), Fr::rand(&mut rng));
        let q = (Fr::rand(&mut rng), Fr::rand(&mut rng));
        let (r0, r1, r_inf) = eval_toom2(p, q);

        assert_eq!(r0, eval_product(&[p, q], Fr::zero()));
        assert_eq!(r1, eval_product(&[p, q], Fr::one()));
        let diff_p = p.1 - p.0;
        let diff_q = q.1 - q.0;
        assert_eq!(r_inf, diff_p * diff_q);
    }

    #[test]
    fn toom2_2_correct() {
        let mut rng = test_rng();
        let p: [Fr; 4] = core::array::from_fn(|_| Fr::rand(&mut rng));
        let q: [Fr; 4] = core::array::from_fn(|_| Fr::rand(&mut rng));
        let r = eval_toom2_2(p, q);

        // helper to compute the 9 evaluation points for a single polynomial
        fn eval_table(poly: &[Fr; 4]) -> [Fr; 9] {
            let p00 = poly[0];
            let p_x0 = poly[1] - p00; // p10
            let p_y0 = poly[2] - p00; // p01
            let p_x1 = poly[3] - poly[2]; // p10 + p11
            let p_y1 = poly[3] - poly[1]; // p01 + p11
            let p_xy = p_x1 - p_x0; // p11

            [
                // y = 0 row
                p00,     // (0,0)
                poly[1], // (1,0)
                p_x0,    // (∞,0)
                // y = 1 row
                poly[2], // (0,1)
                poly[3], // (1,1)
                p_x1,    // (∞,1)
                // y = ∞ row
                p_y0, // (0,∞)
                p_y1, // (1,∞)
                p_xy, // (∞,∞)
            ]
        }

        let pv = eval_table(&p);
        let qv = eval_table(&q);

        for i in 0..9 {
            assert_eq!(r[i], pv[i] * qv[i]);
        }
    }

    #[test]
    fn toom3_2_correct() {
        let mut rng = test_rng();
        let polys: [[Fr; 4]; 3] =
            core::array::from_fn(|_| core::array::from_fn(|_| Fr::rand(&mut rng)));
        let r = eval_toom3_2(polys);
        // helper eval
        fn eval_poly_opt(p: &[Fr; 4], x: Option<Fr>, y: Option<Fr>) -> Fr {
            let a = p[0];
            let b = p[1] - p[0];
            let c = p[2] - p[0];
            let d = p[3] - p[1] - p[2] + p[0];
            match (x, y) {
                (Some(xv), Some(yv)) => a + b * xv + c * yv + d * xv * yv,
                (None, Some(yv)) => b + d * yv,
                (Some(xv), None) => c + d * xv,
                (None, None) => d,
            }
        }
        let vals = [Some(Fr::zero()), Some(Fr::one()), Some(-Fr::one()), None];
        let mut idx = 0usize;
        for &y in &vals {
            for &x in &vals {
                let mut prod = Fr::one();
                for p in &polys {
                    prod *= eval_poly_opt(p, x, y);
                }
                assert_eq!(r[idx], prod);
                idx += 1;
            }
        }
    }

    #[test]
    fn toom4_2_correct() {
        let mut rng = test_rng();
        let polys: [[Fr; 4]; 4] =
            core::array::from_fn(|_| core::array::from_fn(|_| Fr::rand(&mut rng)));
        let r = eval_toom4_2(polys);

        // helper: evaluate multilinear poly at (x,y) where coordinate may be finite or ∞ (None)
        fn eval_poly_opt(p: &[Fr; 4], x: Option<Fr>, y: Option<Fr>) -> Fr {
            let a = p[0];
            let b = p[1] - p[0];
            let c = p[2] - p[0];
            let d = p[3] - p[1] - p[2] + p[0];
            match (x, y) {
                (Some(xv), Some(yv)) => a + b * xv + c * yv + d * xv * yv,
                (None, Some(yv)) => b + d * yv,
                (Some(xv), None) => c + d * xv,
                (None, None) => d,
            }
        }

        let zero = Fr::zero();
        let one = Fr::one();
        let neg_one = -Fr::one();
        let two = Fr::from(2u8);

        let coord_vals = [Some(zero), Some(one), Some(neg_one), Some(two), None];

        // verify all 25 grid points
        let mut idx = 0usize;
        for &y in &coord_vals {
            for &x in &coord_vals {
                let mut prod = Fr::one();
                for p in &polys {
                    prod *= eval_poly_opt(p, x, y);
                }
                assert_eq!(r[idx], prod);
                idx += 1;
            }
        }
    }

    #[test]
    fn toom4_correct() {
        let mut rng = test_rng();
        let polys: [(Fr, Fr); 4] =
            core::array::from_fn(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)));
        let r = eval_toom4(polys);

        let pts = [
            (Fr::from(0u8), r[0]),
            (Fr::from(1u8), r[1]),
            (-Fr::one(), r[2]),
            (Fr::from(2u8), r[3]),
        ];
        for (x, want) in pts {
            assert_eq!(want, eval_product(&polys, x));
        }
        // ∞
        assert_eq!(r[4], polys.iter().map(|p| p.1 - p.0).product::<Fr>());
    }

    #[test]
    fn toom8_correct() {
        let mut rng = test_rng();
        let polys: [(Fr, Fr); 8] =
            core::array::from_fn(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)));
        let r = eval_toom8(polys);

        let xs = [
            Fr::from(0u8),
            Fr::from(1u8),
            -Fr::one(),
            Fr::from(2u8),
            -Fr::from(2u8),
            Fr::from(3u8),
            -Fr::from(3u8),
            Fr::from(4u8),
        ];
        for (i, &x) in xs.iter().enumerate() {
            assert_eq!(r[i], eval_product(&polys, x));
        }
        assert_eq!(r[8], polys.iter().map(|p| p.1 - p.0).product::<Fr>());
    }

    #[test]
    fn toom3_correct() {
        let mut rng = test_rng();
        let polys: [(Fr, Fr); 3] =
            core::array::from_fn(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)));
        let r = eval_toom3(polys);

        let xs = [
            Fr::from(0u8), // r.0 should equal eval_product at x=0
            Fr::from(1u8), // r.1 should equal eval_product at x=1
            -Fr::one(),    // r.2 should equal eval_product at x=-1
        ];

        // Test the evaluation points we computed
        assert_eq!(r.0, eval_product(&polys, xs[0]));
        assert_eq!(r.1, eval_product(&polys, xs[1]));
        assert_eq!(r.2, eval_product(&polys, xs[2]));

        // Test ∞ case
        assert_eq!(r.3, polys.iter().map(|p| p.1 - p.0).product::<Fr>());
    }

    #[test]
    fn toom16_correct() {
        let mut rng = test_rng();
        let polys: [(Fr, Fr); 16] =
            core::array::from_fn(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)));
        let r = eval_toom16(polys);

        let xs = [
            Fr::from(0u8),  // 0
            Fr::from(1u8),  // 1
            -Fr::one(),     // −1
            Fr::from(2u8),  // 2
            -Fr::from(2u8), // −2
            Fr::from(3u8),  // 3
            -Fr::from(3u8), // −3
            Fr::from(4u8),  // 4
            -Fr::from(4u8), // −4
            Fr::from(5u8),  // 5
            -Fr::from(5u8), // −5
            Fr::from(6u8),  // 6
            -Fr::from(6u8), // −6
            Fr::from(7u8),  // 7
            -Fr::from(7u8), // −7
            Fr::from(8u8),  // 8
        ];
        for (i, &x) in xs.iter().enumerate() {
            assert_eq!(r[i], eval_product(&polys, x));
        }
        // ∞
        assert_eq!(r[16], polys.iter().map(|p| p.1 - p.0).product::<Fr>());

        // Compare the result with Karatsuba
        let left: [Fr; 16] = core::array::from_fn(|i| {
            if i % 2 == 0 {
                polys[i / 2].0
            } else {
                polys[i / 2].1 - polys[i / 2].0
            }
        });

        let right: [Fr; 16] = core::array::from_fn(|i| {
            if i % 2 == 0 {
                polys[i / 2 + 8].0
            } else {
                polys[i / 2 + 8].1 - polys[i / 2 + 8].0
            }
        });

        let kara_res = coeff_kara_16(&left, &right);
        let univariate_poly = UniPoly::from_evals_toom(&r);
        assert_eq!(univariate_poly.coeffs, kara_res);
    }
}
