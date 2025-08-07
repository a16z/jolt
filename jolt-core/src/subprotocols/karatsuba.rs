// Zach's Karatsuba algorithm.
use crate::field::JoltField;

// the main functions to look at are of the form coeff_kara_n
// they take a vector of n linear polynomials in coefficient form
// constant, slope, constant, slope, ...
// and return a vector containing n + 1 elements, corresponding to the coefficients
// of the polynomial p(x) = prod_i p_i(x) from lowest to highest degree

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

// note how this splits the list of polynomials into two halves
// and then makes a recursive call to get the polynomial coefficients
// corresponding to the two halves and then multiplies them together
#[inline(always)]
pub fn coeff_kara_4<F: JoltField>(left: &[F; 4], right: &[F; 4]) -> [F; 5] {
    kara_3(
        &coeff_kara_2(
            &left[..2].try_into().unwrap(),
            &left[2..].try_into().unwrap(),
        ),
        &coeff_kara_2(
            &right[..2].try_into().unwrap(),
            &right[2..].try_into().unwrap(),
        ),
    )
}

#[inline(always)]
pub fn coeff_kara_8<F: JoltField>(left: &[F; 8], right: &[F; 8]) -> [F; 9] {
    kara_5(
        &coeff_kara_4(
            &left[..4].try_into().unwrap(),
            &left[4..].try_into().unwrap(),
        ),
        &coeff_kara_4(
            &right[..4].try_into().unwrap(),
            &right[4..].try_into().unwrap(),
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
    res[..l.len()].copy_from_slice(&l[..]);
    res[8..(u.len() + 8)].copy_from_slice(&u[..]);

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
    res[..l.len()].copy_from_slice(&l[..]);
    res[8..(u.len() + 8)].copy_from_slice(&u[..]);
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
    res[..l.len()].copy_from_slice(&l[..]);
    res[8..(u.len() + 8)].copy_from_slice(&u[..]);
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
    res[..l.len()].copy_from_slice(&l[..]);
    res[16..(u.len() + 16)].copy_from_slice(&u[..]);
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
