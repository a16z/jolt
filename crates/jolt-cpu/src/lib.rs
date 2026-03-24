//! CPU compute backend and kernel compiler for the Jolt zkVM.
//!
//! Provides [`CpuBackend`], the reference [`ComputeBackend`](jolt_compute::ComputeBackend)
//! implementation using `Vec<T>` buffers and Rayon parallelism. Also provides
//! kernel compilation from [`KernelDescriptor`]s into [`CpuKernel`]s.
//!
//! # Kernel compilation strategies
//!
//! - **`ProductSum` D∈{4,8,16,32}** — hand-optimized closures with fully unrolled
//!   product evaluation. These cover ~80% of prover time (instruction RA
//!   sumchecks and claim reductions).
//!
//! - **`ProductSum` generic** — loop-based fallback for other D values.
//!
//! - **`Custom`** — the [`Expr`](jolt_ir::Expr) is walked once at compile time
//!   to produce a closure that evaluates the expression at each grid point.

mod backend;
mod custom;
mod product_sum;
mod specialized;
pub use jolt_ir::toom_cook;

pub use backend::{CpuBackend, CpuKernel};

use jolt_field::Field;
use jolt_ir::{KernelDescriptor, KernelShape};

/// Compile a kernel descriptor into a CPU kernel.
///
/// Dispatches to specialized implementations based on the descriptor's shape
/// and degree. For `ProductSum` with D∈{4,8,16}, returns a hand-optimized
/// kernel with fully unrolled product evaluation. For other D values, returns
/// a loop-based generic kernel. For `Custom`, compiles the expression into
/// a closure via the stack-machine compiler, with challenges baked as zero.
///
/// For `Custom` expressions that use challenge variables, use
/// [`compile_with_challenges`] instead.
///
/// # Panics
///
/// Panics if `desc.is_valid()` returns false.
pub fn compile<F: Field>(desc: &KernelDescriptor) -> CpuKernel<F> {
    compile_with_challenges(desc, &[])
}

/// Compile a kernel descriptor into a CPU kernel with challenge values baked in.
///
/// Like [`compile`], but `Custom` expressions have their `Var::Challenge(i)`
/// nodes resolved to `challenges[i]`. Out-of-bounds indices are baked as
/// `F::zero()`. For `ProductSum` descriptors, `challenges` is ignored (there
/// are no challenge variables in the product-sum pattern).
///
/// This is the primary entry point for compiling kernels that use
/// Fiat-Shamir-derived values (gamma, tau, batching coefficients, etc.).
///
/// # Panics
///
/// Panics if `desc.is_valid()` returns false.
pub fn compile_with_challenges<F: Field>(
    desc: &KernelDescriptor,
    challenges: &[F],
) -> CpuKernel<F> {
    assert!(desc.is_valid(), "invalid kernel descriptor");

    match &desc.shape {
        KernelShape::ProductSum {
            num_inputs_per_product,
            num_products,
        } => product_sum::compile::<F>(*num_inputs_per_product, *num_products),
        KernelShape::EqProduct => specialized::eq_product::<F>(),
        KernelShape::HammingBooleanity => specialized::hamming_booleanity::<F>(),
        KernelShape::Custom { expr, num_inputs } => {
            custom::compile_with_challenges::<F>(expr, *num_inputs, desc.degree, challenges)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_ir::{ExprBuilder, TensorSplit};
    use num_traits::{One, Zero};

    fn eval_kernel(kernel: &CpuKernel<Fr>, lo: &[Fr], hi: &[Fr], n: usize) -> Vec<Fr> {
        let mut out = vec![Fr::zero(); n];
        kernel.evaluate(lo, hi, &mut out);
        out
    }

    #[test]
    fn compile_product_sum_d4() {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 4,
                num_products: 1,
            },
            degree: 4,
            tensor_split: None,
        };
        let kernel: CpuKernel<Fr> = compile(&desc);

        let lo: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (5..=8).map(Fr::from_u64).collect();
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());

        assert_eq!(result.len(), 4);
        assert_eq!(result[0], Fr::from_u64(1680));
        assert_eq!(result[3], Fr::from_u64(256));
    }

    #[test]
    fn compile_product_sum_d8() {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 8,
                num_products: 1,
            },
            degree: 8,
            tensor_split: None,
        };
        let kernel: CpuKernel<Fr> = compile(&desc);

        let lo: Vec<Fr> = vec![Fr::one(); 8];
        let hi: Vec<Fr> = vec![Fr::from_u64(2); 8];
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());

        assert_eq!(result.len(), 8);
        assert_eq!(result[0], Fr::from_u64(256));
        assert_eq!(result[7], Fr::one());
    }

    #[test]
    fn compile_product_sum_d16() {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 16,
                num_products: 1,
            },
            degree: 16,
            tensor_split: None,
        };
        let kernel: CpuKernel<Fr> = compile(&desc);

        let lo: Vec<Fr> = vec![Fr::one(); 16];
        let hi: Vec<Fr> = vec![Fr::from_u64(2); 16];
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());

        assert_eq!(result.len(), 16);
        assert_eq!(result[0], Fr::from_u64(65536));
        assert_eq!(result[15], Fr::one());
    }

    #[test]
    fn compile_product_sum_multiple_groups() {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 4,
                num_products: 2,
            },
            degree: 4,
            tensor_split: None,
        };
        let kernel: CpuKernel<Fr> = compile(&desc);

        let lo: Vec<Fr> = (1..=8).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (9..=16).map(Fr::from_u64).collect();
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());
        assert_eq!(result.len(), 4);

        let p1_g0: u64 = 9 * 10 * 11 * 12;
        let p1_g1: u64 = 13 * 14 * 15 * 16;
        assert_eq!(result[0], Fr::from_u64(p1_g0 + p1_g1));
    }

    #[test]
    fn compile_product_sum_generic_d3() {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 3,
                num_products: 1,
            },
            degree: 3,
            tensor_split: None,
        };
        let kernel: CpuKernel<Fr> = compile(&desc);

        let lo: Vec<Fr> = (1..=3).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (4..=6).map(Fr::from_u64).collect();
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Fr::from_u64(120));
        assert_eq!(result[1], Fr::from_u64(504));
        assert_eq!(result[2], Fr::from_u64(27));
    }

    #[test]
    fn compile_custom_simple_product() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);

        let desc = KernelDescriptor {
            shape: KernelShape::Custom {
                expr: b.build(a * bv),
                num_inputs: 2,
            },
            degree: 2,
            tensor_split: None,
        };
        let kernel: CpuKernel<Fr> = compile(&desc);

        let lo = vec![Fr::from_u64(3), Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(7), Fr::from_u64(11)];
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());

        assert_eq!(result[0], Fr::from_u64(15));
        assert_eq!(result[1], Fr::from_u64(187));
    }

    #[test]
    fn compile_custom_booleanity() {
        let b = ExprBuilder::new();
        let h = b.opening(0);

        let desc = KernelDescriptor {
            shape: KernelShape::Custom {
                expr: b.build(h * h - h),
                num_inputs: 1,
            },
            degree: 2,
            tensor_split: None,
        };
        let kernel: CpuKernel<Fr> = compile(&desc);

        let lo = vec![Fr::from_u64(3)];
        let hi = vec![Fr::from_u64(7)];
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());

        assert_eq!(result[0], Fr::from_u64(6));
        assert_eq!(result[1], Fr::from_u64(110));
    }

    #[test]
    fn compile_custom_with_challenge() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let gamma = b.challenge(0);

        let desc = KernelDescriptor {
            shape: KernelShape::Custom {
                expr: b.build(gamma * a),
                num_inputs: 1,
            },
            degree: 1,
            tensor_split: None,
        };

        let kernel_zero: CpuKernel<Fr> = compile(&desc);
        let lo = vec![Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(10)];
        let result = eval_kernel(&kernel_zero, &lo, &hi, desc.num_evals());
        assert_eq!(result[0], Fr::zero());

        let kernel: CpuKernel<Fr> = compile_with_challenges(&desc, &[Fr::from_u64(7)]);
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());
        assert_eq!(result[0], Fr::from_u64(35));
    }

    #[test]
    fn compile_with_tensor_split_ignored() {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 4,
                num_products: 1,
            },
            degree: 4,
            tensor_split: Some(TensorSplit::balanced(20)),
        };
        let kernel: CpuKernel<Fr> = compile(&desc);

        let lo: Vec<Fr> = vec![Fr::one(); 4];
        let hi: Vec<Fr> = vec![Fr::from_u64(2); 4];
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());
        assert_eq!(result[0], Fr::from_u64(16));
    }

    #[test]
    fn num_evals_product_sum_vs_custom() {
        let ps_desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 4,
                num_products: 1,
            },
            degree: 4,
            tensor_split: None,
        };
        assert_eq!(ps_desc.num_evals(), 4);

        let b = ExprBuilder::new();
        let a = b.opening(0);
        let custom_desc = KernelDescriptor {
            shape: KernelShape::Custom {
                expr: b.build(a * a),
                num_inputs: 1,
            },
            degree: 2,
            tensor_split: None,
        };
        assert_eq!(custom_desc.num_evals(), 2);
    }

    #[test]
    #[should_panic(expected = "invalid kernel descriptor")]
    fn compile_invalid_descriptor_panics() {
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 4,
                num_products: 0,
            },
            degree: 4,
            tensor_split: None,
        };
        let _: CpuKernel<Fr> = compile(&desc);
    }
}
