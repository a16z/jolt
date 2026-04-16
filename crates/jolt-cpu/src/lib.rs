//! CPU compute backend and kernel compiler for the Jolt zkVM.
//!
//! Provides [`CpuBackend`], the reference [`ComputeBackend`](jolt_compute::ComputeBackend)
//! implementation using `Vec<T>` buffers and Rayon parallelism. Also provides
//! kernel compilation from [`KernelSpec`]s into [`CpuKernel`]s.
//!
//! # Kernel compilation
//!
//! [`compile`] inspects the [`KernelSpec`]'s composition and dispatches to:
//!
//! - **ProductSum** — Toom-Cook evaluation on grid `{1, ..., D-1, ∞}`.
//!   Covers ~80% of prover time (instruction RA sumchecks and claim
//!   reductions).
//!
//! - **Generic** — direct sum-of-products evaluator on the standard grid
//!   `{0, 2, 3, ..., degree}`. Handles all other compositions.

mod backend;
mod generic;
mod product_sum;
pub mod toom_cook;

pub use backend::{BoxedEvalFn, CpuBackend, CpuKernel};

use jolt_compiler::{Iteration, KernelSpec};
use jolt_field::Field;

/// Compile a [`KernelSpec`] into a CPU kernel.
///
/// Inspects the spec's composition to select the best eval strategy:
/// product-sum compositions use Toom-Cook; everything else uses the
/// generic sum-of-products evaluator.
pub fn compile<F: Field>(spec: &KernelSpec) -> CpuKernel<F> {
    let composition = &spec.formula;

    let eval_fn: BoxedEvalFn<F> = if let Some((d, p)) = composition
        .as_product_sum()
        .filter(|&(d, _)| d == spec.num_evals)
    {
        product_sum::compile_fn::<F>(d, p)
    } else {
        Box::new(generic::compile_fn::<F>(composition))
    };

    let mut kernel = CpuKernel::from_boxed(
        eval_fn,
        spec.num_evals,
        spec.iteration.clone(),
        spec.binding_order,
    );

    if matches!(spec.iteration, Iteration::Domain { .. }) {
        kernel = kernel.with_domain_eval(Box::new(generic::compile_domain_fn::<F>(composition)));
    }

    kernel
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_compiler::BindingOrder;
    use jolt_compiler::{Factor, Formula, Iteration, ProductTerm};
    use jolt_field::Fr;
    use num_traits::{One, Zero};

    fn eval_kernel(kernel: &CpuKernel<Fr>, lo: &[Fr], hi: &[Fr], n: usize) -> Vec<Fr> {
        let mut out = vec![Fr::zero(); n];
        kernel.evaluate(lo, hi, &[], &mut out);
        out
    }

    fn product_sum_spec(d: usize, p: usize) -> KernelSpec {
        let terms: Vec<_> = (0..p)
            .map(|g| ProductTerm {
                coefficient: 1,
                factors: (0..d).map(|j| Factor::Input((g * d + j) as u32)).collect(),
            })
            .collect();
        let formula = Formula::from_terms(terms);
        let num_evals = formula.degree();
        KernelSpec {
            formula,
            num_evals,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
        }
    }

    #[test]
    fn compile_product_sum_d4() {
        let spec = product_sum_spec(4, 1);
        let kernel: CpuKernel<Fr> = compile(&spec);

        let lo: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (5..=8).map(Fr::from_u64).collect();
        let result = eval_kernel(&kernel, &lo, &hi, spec.num_evals);

        assert_eq!(result.len(), 4);
        assert_eq!(result[0], Fr::from_u64(1680));
        assert_eq!(result[3], Fr::from_u64(256));
    }

    #[test]
    fn compile_product_sum_d8() {
        let spec = product_sum_spec(8, 1);
        let kernel: CpuKernel<Fr> = compile(&spec);

        let lo: Vec<Fr> = vec![Fr::one(); 8];
        let hi: Vec<Fr> = vec![Fr::from_u64(2); 8];
        let result = eval_kernel(&kernel, &lo, &hi, spec.num_evals);

        assert_eq!(result.len(), 8);
        assert_eq!(result[0], Fr::from_u64(256));
        assert_eq!(result[7], Fr::one());
    }

    #[test]
    fn compile_product_sum_d16() {
        let spec = product_sum_spec(16, 1);
        let kernel: CpuKernel<Fr> = compile(&spec);

        let lo: Vec<Fr> = vec![Fr::one(); 16];
        let hi: Vec<Fr> = vec![Fr::from_u64(2); 16];
        let result = eval_kernel(&kernel, &lo, &hi, spec.num_evals);

        assert_eq!(result.len(), 16);
        assert_eq!(result[0], Fr::from_u64(65536));
        assert_eq!(result[15], Fr::one());
    }

    #[test]
    fn compile_product_sum_multiple_groups() {
        let spec = product_sum_spec(4, 2);
        let kernel: CpuKernel<Fr> = compile(&spec);

        let lo: Vec<Fr> = (1..=8).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (9..=16).map(Fr::from_u64).collect();
        let result = eval_kernel(&kernel, &lo, &hi, spec.num_evals);
        assert_eq!(result.len(), 4);

        let p1_g0: u64 = 9 * 10 * 11 * 12;
        let p1_g1: u64 = 13 * 14 * 15 * 16;
        assert_eq!(result[0], Fr::from_u64(p1_g0 + p1_g1));
    }

    #[test]
    fn compile_product_sum_generic_d3() {
        let spec = product_sum_spec(3, 1);
        let kernel: CpuKernel<Fr> = compile(&spec);

        let lo: Vec<Fr> = (1..=3).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (4..=6).map(Fr::from_u64).collect();
        let result = eval_kernel(&kernel, &lo, &hi, spec.num_evals);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Fr::from_u64(120));
        assert_eq!(result[1], Fr::from_u64(504));
        assert_eq!(result[2], Fr::from_u64(27));
    }

    #[test]
    fn compile_formula_simple_product() {
        let formula = Formula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(1)],
        }]);
        let spec = KernelSpec {
            num_evals: formula.degree(),
            formula,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
        };
        let kernel: CpuKernel<Fr> = compile(&spec);

        let lo = vec![Fr::from_u64(3), Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(7), Fr::from_u64(11)];
        let result = eval_kernel(&kernel, &lo, &hi, spec.num_evals);

        // Toom-Cook grid {1, ∞}: P(1) = 7*11 = 77, P(∞) = 4*6 = 24
        assert_eq!(result[0], Fr::from_u64(77));
        assert_eq!(result[1], Fr::from_u64(24));
    }

    #[test]
    fn compile_formula_booleanity() {
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
        let spec = KernelSpec {
            num_evals: formula.degree(),
            formula,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
        };
        let kernel: CpuKernel<Fr> = compile(&spec);

        let lo = vec![Fr::from_u64(3)];
        let hi = vec![Fr::from_u64(7)];
        let result = eval_kernel(&kernel, &lo, &hi, spec.num_evals);

        // t=0: 3^2 - 3 = 6, t=1: 7^2 - 7 = 42
        assert_eq!(result[0], Fr::from_u64(6));
        assert_eq!(result[1], Fr::from_u64(42));
    }

    #[test]
    fn compile_formula_with_challenge() {
        let formula = Formula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Challenge(0), Factor::Input(0)],
        }]);
        let spec = KernelSpec {
            num_evals: formula.degree(),
            formula,
            iteration: Iteration::Dense,
            binding_order: BindingOrder::LowToHigh,
        };

        let kernel: CpuKernel<Fr> = compile(&spec);
        let lo = vec![Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(10)];

        let result = eval_kernel(&kernel, &lo, &hi, spec.num_evals);
        assert_eq!(result[0], Fr::zero());

        let mut out = vec![Fr::zero(); spec.num_evals];
        kernel.evaluate(&lo, &hi, &[Fr::from_u64(7)], &mut out);
        assert_eq!(out[0], Fr::from_u64(35));
    }

    #[test]
    fn eq_product_detection() {
        let formula = Formula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(1)],
        }]);
        assert!(formula.is_eq_product());

        let formula3 = Formula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(1), Factor::Input(2)],
        }]);
        assert!(!formula3.is_eq_product());
    }

    #[test]
    fn hamming_booleanity_detection() {
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
