//! Hand-coded kernels for common small compositions.
//!
//! These eliminate the stack-VM dispatch overhead of [`Custom`](crate::custom)
//! kernels for the two most frequently used compositions:
//!
//! - [`eq_product`]: `opening(0) * opening(1)` — degree 2, 3 evaluations
//! - [`hamming_booleanity`]: `opening(0) * opening(1) * (opening(1) - 1)` —
//!   degree 3, 4 evaluations

use crate::CpuKernel;
use jolt_field::Field;

/// Hand-coded `eq · g` kernel on grid `{0, 2}` (skipping `t=1`).
///
/// Evaluates `f(t) = a(t) · b(t)` where `a` and `b` are linear interpolants:
/// - `f(0) = lo[0] · lo[1]`
/// - `f(2) = (2·hi[0] − lo[0]) · (2·hi[1] − lo[1])`
///
/// `f(1) = hi[0] · hi[1]` is derived from the sumcheck claim externally.
/// 2 field multiplications, zero branching.
#[inline]
pub fn eq_product<F: Field>() -> CpuKernel<F> {
    CpuKernel::new(|lo: &[F], hi: &[F], _challenges: &[F], out: &mut [F]| {
        out[0] = lo[0] * lo[1];
        let a2 = hi[0] + hi[0] - lo[0];
        let b2 = hi[1] + hi[1] - lo[1];
        out[1] = a2 * b2;
    })
}

/// Hand-coded `eq · h · (h − 1)` kernel on grid `{0, 2, 3}` (skipping `t=1`).
///
/// Evaluates `f(t) = eq(t) · h(t) · (h(t) − 1)` where `eq` and `h` are
/// linear interpolants. `f(1)` is derived from the sumcheck claim externally.
///
/// 6 field multiplications (2 per grid point), zero branching.
#[inline]
pub fn hamming_booleanity<F: Field>() -> CpuKernel<F> {
    CpuKernel::new(|lo: &[F], hi: &[F], _challenges: &[F], out: &mut [F]| {
        let one = F::one();
        let d_eq = hi[0] - lo[0];
        let d_h = hi[1] - lo[1];

        // t=0: use lo values directly
        out[0] = lo[0] * lo[1] * (lo[1] - one);

        // t=2, 3: start from hi + delta (skipping t=1)
        let mut eq_val = hi[0] + d_eq;
        let mut h_val = hi[1] + d_h;

        for slot in &mut out[1..] {
            *slot = eq_val * h_val * (h_val - one);
            eq_val += d_eq;
            h_val += d_h;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use num_traits::{One, Zero};

    fn eval_kernel(kernel: &CpuKernel<Fr>, lo: &[Fr], hi: &[Fr], n: usize) -> Vec<Fr> {
        let mut out = vec![Fr::zero(); n];
        kernel.evaluate(lo, hi, &[], &mut out);
        out
    }

    #[test]
    fn eq_product_known_values() {
        let kernel = eq_product::<Fr>();
        let lo = vec![Fr::from_u64(3), Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(7), Fr::from_u64(11)];
        // Grid: {0, 2} — 2 evals, t=1 derived from claim
        let result = eval_kernel(&kernel, &lo, &hi, 2);

        // f(0) = 3*5 = 15
        assert_eq!(result[0], Fr::from_u64(15));
        // f(2) = (2*7-3)*(2*11-5) = 11*17 = 187
        assert_eq!(result[1], Fr::from_u64(187));
    }

    #[test]
    fn eq_product_matches_generic() {
        use jolt_compiler::{Factor, Formula, ProductTerm};
        // Input(0) * Input(1)
        let formula = Formula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(1)],
        }]);
        let generic_kernel = crate::formula::compile::<Fr>(&formula);
        let specialized_kernel = eq_product::<Fr>();

        let lo = vec![Fr::from_u64(42), Fr::from_u64(99)];
        let hi = vec![Fr::from_u64(17), Fr::from_u64(53)];
        assert_eq!(
            eval_kernel(&generic_kernel, &lo, &hi, 2),
            eval_kernel(&specialized_kernel, &lo, &hi, 2),
        );
    }

    #[test]
    fn hamming_booleanity_known_values() {
        let kernel = hamming_booleanity::<Fr>();
        // h=0 → h*(h-1)=0, h=1 → h*(h-1)=0, h=2 → h*(h-1)=2
        let lo = vec![Fr::from_u64(1), Fr::from_u64(0)]; // eq(0)=1, h(0)=0
        let hi = vec![Fr::from_u64(1), Fr::from_u64(1)]; // eq(1)=1, h(1)=1
                                                         // Grid: {0, 2, 3} — 3 evals, t=1 derived from claim
        let result = eval_kernel(&kernel, &lo, &hi, 3);

        let one = Fr::one();
        // f(0) = 1 * 0 * (0-1) = 0
        assert_eq!(result[0], Fr::zero());
        // f(2) = 1 * 2 * (2-1) = 2
        assert_eq!(result[1], one + one);
        // f(3) = 1 * 3 * (3-1) = 6
        assert_eq!(result[2], Fr::from_u64(6));
    }

    #[test]
    fn hamming_booleanity_matches_generic() {
        use jolt_compiler::{Factor, Formula, ProductTerm};
        // eq * h * (h - 1) = eq*h*h - eq*h
        // = [coeff:1 Input(0)*Input(1)*Input(1)] + [coeff:-1 Input(0)*Input(1)]
        let formula = Formula::from_terms(vec![
            ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(1)],
            },
            ProductTerm {
                coefficient: -1,
                factors: vec![Factor::Input(0), Factor::Input(1)],
            },
        ]);
        let generic_kernel = crate::formula::compile::<Fr>(&formula);
        let specialized_kernel = hamming_booleanity::<Fr>();

        let lo = vec![Fr::from_u64(42), Fr::from_u64(7)];
        let hi = vec![Fr::from_u64(17), Fr::from_u64(13)];
        assert_eq!(
            eval_kernel(&generic_kernel, &lo, &hi, 3),
            eval_kernel(&specialized_kernel, &lo, &hi, 3),
        );
    }
}
