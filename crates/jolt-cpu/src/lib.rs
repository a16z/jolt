//! CPU compute backend and kernel compiler for the Jolt zkVM.
//!
//! Provides [`CpuBackend`], the reference [`ComputeBackend`](jolt_compute::ComputeBackend)
//! implementation using `Vec<T>` buffers and Rayon parallelism. Also provides
//! kernel compilation from [`Formula`]s into [`CpuKernel`]s.
//!
//! # Kernel compilation strategies
//!
//! - **ProductSum D∈{4,8,16,32}** — hand-optimized closures with fully unrolled
//!   product evaluation. These cover ~80% of prover time (instruction RA
//!   sumchecks and claim reductions). Detected via
//!   [`Formula::as_product_sum()`].
//!
//! - **ProductSum generic** — loop-based fallback for other D values.
//!
//! - **Hamming booleanity** — hand-coded `eq · h · (h − 1)` kernel. Detected
//!   via [`Formula::is_hamming_booleanity()`].
//!
//! - **Eq product** — hand-coded `a · b` kernel for 2-input degree-2 products.
//!
//! - **Generic** — direct sum-of-products evaluator from the normalized formula.

mod backend;
mod formula;
mod product_sum;
mod specialized;
pub mod toom_cook;

pub use backend::{CpuBackend, CpuKernel};

use jolt_compiler::Formula;
use jolt_field::Field;

/// Compile a [`Formula`] into a CPU kernel.
///
/// Dispatches to specialized implementations based on the formula's structure:
/// 1. Eq-product / Hamming booleanity → hand-coded standard-grid kernels
/// 2. Product-sum with D∈{4,8,16,32} → Toom-Cook grid kernels
/// 3. All other formulas → generic standard-grid evaluator
///
/// Challenge values are resolved at dispatch time via
/// [`CpuKernel::evaluate`], not at compilation.
pub fn compile<F: Field>(formula: &Formula) -> CpuKernel<F> {
    // Standard-grid specializations must be checked first — these patterns
    // also match as_product_sum() but require standard grid {0,2,3,...} output,
    // not the Toom-Cook grid {1,...,D-1,∞}.

    // Fast path: simple 2-input product (eq·g pattern)
    if formula.is_eq_product() {
        return specialized::eq_product::<F>();
    }

    // Fast path: Hamming booleanity → hand-coded kernel
    if formula.is_hamming_booleanity() {
        return specialized::hamming_booleanity::<F>();
    }

    // Fast path: pure product-sum → Toom-Cook kernel
    if let Some((d, p)) = formula.as_product_sum() {
        return product_sum::compile::<F>(d, p);
    }

    // Generic: compile directly from the normalized formula
    formula::compile::<F>(formula)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_compiler::{Factor, ProductTerm};
    use jolt_field::Fr;
    use num_traits::{One, Zero};

    fn eval_kernel(kernel: &CpuKernel<Fr>, lo: &[Fr], hi: &[Fr], n: usize) -> Vec<Fr> {
        let mut out = vec![Fr::zero(); n];
        kernel.evaluate(lo, hi, &[], &mut out);
        out
    }

    /// Helper: build a pure product-sum formula with `p` groups of `d` consecutive inputs.
    fn product_sum_formula(d: usize, p: usize) -> Formula {
        let terms: Vec<_> = (0..p)
            .map(|g| ProductTerm {
                coefficient: 1,
                factors: (0..d).map(|j| Factor::Input((g * d + j) as u32)).collect(),
            })
            .collect();
        Formula::from_terms(terms)
    }

    #[test]
    fn compile_product_sum_d4() {
        let formula = product_sum_formula(4, 1);
        let kernel: CpuKernel<Fr> = compile(&formula);

        let lo: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (5..=8).map(Fr::from_u64).collect();
        let result = eval_kernel(&kernel, &lo, &hi, formula.degree());

        assert_eq!(result.len(), 4);
        assert_eq!(result[0], Fr::from_u64(1680));
        assert_eq!(result[3], Fr::from_u64(256));
    }

    #[test]
    fn compile_product_sum_d8() {
        let formula = product_sum_formula(8, 1);
        let kernel: CpuKernel<Fr> = compile(&formula);

        let lo: Vec<Fr> = vec![Fr::one(); 8];
        let hi: Vec<Fr> = vec![Fr::from_u64(2); 8];
        let result = eval_kernel(&kernel, &lo, &hi, formula.degree());

        assert_eq!(result.len(), 8);
        assert_eq!(result[0], Fr::from_u64(256));
        assert_eq!(result[7], Fr::one());
    }

    #[test]
    fn compile_product_sum_d16() {
        let formula = product_sum_formula(16, 1);
        let kernel: CpuKernel<Fr> = compile(&formula);

        let lo: Vec<Fr> = vec![Fr::one(); 16];
        let hi: Vec<Fr> = vec![Fr::from_u64(2); 16];
        let result = eval_kernel(&kernel, &lo, &hi, formula.degree());

        assert_eq!(result.len(), 16);
        assert_eq!(result[0], Fr::from_u64(65536));
        assert_eq!(result[15], Fr::one());
    }

    #[test]
    fn compile_product_sum_multiple_groups() {
        let formula = product_sum_formula(4, 2);
        let kernel: CpuKernel<Fr> = compile(&formula);

        let lo: Vec<Fr> = (1..=8).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (9..=16).map(Fr::from_u64).collect();
        let result = eval_kernel(&kernel, &lo, &hi, formula.degree());
        assert_eq!(result.len(), 4);

        let p1_g0: u64 = 9 * 10 * 11 * 12;
        let p1_g1: u64 = 13 * 14 * 15 * 16;
        assert_eq!(result[0], Fr::from_u64(p1_g0 + p1_g1));
    }

    #[test]
    fn compile_product_sum_generic_d3() {
        let formula = product_sum_formula(3, 1);
        let kernel: CpuKernel<Fr> = compile(&formula);

        let lo: Vec<Fr> = (1..=3).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (4..=6).map(Fr::from_u64).collect();
        let result = eval_kernel(&kernel, &lo, &hi, formula.degree());

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Fr::from_u64(120));
        assert_eq!(result[1], Fr::from_u64(504));
        assert_eq!(result[2], Fr::from_u64(27));
    }

    #[test]
    fn compile_formula_simple_product() {
        // Input(0) * Input(1)
        let formula = Formula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(1)],
        }]);
        let kernel: CpuKernel<Fr> = compile(&formula);

        let lo = vec![Fr::from_u64(3), Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(7), Fr::from_u64(11)];
        let result = eval_kernel(&kernel, &lo, &hi, formula.degree());

        assert_eq!(result[0], Fr::from_u64(15));
        assert_eq!(result[1], Fr::from_u64(187));
    }

    #[test]
    fn compile_formula_booleanity() {
        // h^2 - h = [coeff:1 Input(0)*Input(0)] + [coeff:-1 Input(0)]
        let formula = Formula::from_terms(vec![
            ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Input(0), Factor::Input(0)],
            },
            ProductTerm {
                coefficient: -1,
                factors: vec![Factor::Input(0)],
            },
        ]);
        let kernel: CpuKernel<Fr> = compile(&formula);

        let lo = vec![Fr::from_u64(3)];
        let hi = vec![Fr::from_u64(7)];
        let result = eval_kernel(&kernel, &lo, &hi, formula.degree());

        assert_eq!(result[0], Fr::from_u64(6));
        assert_eq!(result[1], Fr::from_u64(110));
    }

    #[test]
    fn compile_formula_with_challenge() {
        // Challenge(0) * Input(0)
        let formula = Formula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Challenge(0), Factor::Input(0)],
        }]);

        let kernel: CpuKernel<Fr> = compile(&formula);
        let lo = vec![Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(10)];

        // No challenges → Challenge(0) defaults to zero
        let result = eval_kernel(&kernel, &lo, &hi, formula.degree());
        assert_eq!(result[0], Fr::zero());

        // With challenges passed at dispatch time
        let mut out = vec![Fr::zero(); formula.degree()];
        kernel.evaluate(&lo, &hi, &[Fr::from_u64(7)], &mut out);
        assert_eq!(out[0], Fr::from_u64(35));
    }

    #[test]
    fn eq_product_detection() {
        // Input(0) * Input(1) — should be detected as eq_product
        let formula = Formula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(1)],
        }]);
        assert!(formula.is_eq_product());

        // Three inputs — not eq_product
        let formula3 = Formula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(2)],
        }]);
        assert!(!formula3.is_eq_product());
    }

    #[test]
    fn hamming_booleanity_detection() {
        // Challenge(0)*Input(0)*Input(0) + [-1]*Challenge(0)*Input(0)
        let formula = Formula::from_terms(vec![
            ProductTerm {
                coefficient: 1,
                factors: vec![Factor::Challenge(0), Factor::Input(0), Factor::Input(0)],
            },
            ProductTerm {
                coefficient: -1,
                factors: vec![Factor::Challenge(0), Factor::Input(0)],
            },
        ]);
        assert!(formula.is_hamming_booleanity());
    }
}
