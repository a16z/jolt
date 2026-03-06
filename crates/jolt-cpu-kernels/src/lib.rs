//! CPU kernel compiler for the Jolt zkVM.
//!
//! Compiles [`jolt_ir::KernelDescriptor`]s into [`jolt_compute::CpuKernel`]s.
//! This crate is the bridge between the symbolic IR (field-agnostic) and the
//! concrete CPU backend (field-specific closures).
//!
//! # Compilation strategies
//!
//! - **`ProductSum` D∈{4,8,16}** — hand-optimized closures with fully unrolled
//!   product evaluation. These cover ~80% of prover time (instruction RA
//!   sumchecks and claim reductions).
//!
//! - **`ProductSum` generic** — loop-based fallback for other D values.
//!
//! - **`Custom`** — the [`Expr`](jolt_ir::Expr) is walked once at compile time
//!   to produce a closure that evaluates the expression at each grid point.
//!
//! # Usage
//!
//! ```ignore
//! use jolt_cpu_kernels::compile;
//! use jolt_ir::{KernelDescriptor, KernelShape};
//!
//! let desc = KernelDescriptor {
//!     shape: KernelShape::ProductSum {
//!         num_inputs_per_product: 4,
//!         num_products: 3,
//!     },
//!     degree: 4,
//!     tensor_split: None,
//! };
//!
//! let kernel = compile::<Fr>(&desc);
//! ```

mod custom;
mod product_sum;

use jolt_compute::CpuKernel;
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
    use num_traits::Zero;

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
        let result = kernel.evaluate(&lo, &hi, 4);
        assert_eq!(result.len(), 5);

        // t=0: product of lo = 1*2*3*4 = 24
        assert_eq!(result[0], Fr::from_u64(24));
        // t=1: product of hi = 5*6*7*8 = 1680
        assert_eq!(result[1], Fr::from_u64(1680));
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

        let lo: Vec<Fr> = (1..=8).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (9..=16).map(Fr::from_u64).collect();
        let result = kernel.evaluate(&lo, &hi, 8);
        assert_eq!(result.len(), 9);

        // t=0: 1*2*3*4*5*6*7*8 = 40320
        assert_eq!(result[0], Fr::from_u64(40320));
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

        // Just verify it produces the right number of outputs
        let lo: Vec<Fr> = (1..=16).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (17..=32).map(Fr::from_u64).collect();
        let result = kernel.evaluate(&lo, &hi, 16);
        assert_eq!(result.len(), 17);

        // t=0: product of 1..=16 = 16! = 20922789888000
        assert_eq!(result[0], Fr::from_u64(20_922_789_888_000));
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

        // Group 0: inputs 0..4, Group 1: inputs 4..8
        let lo: Vec<Fr> = (1..=8).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (9..=16).map(Fr::from_u64).collect();
        let result = kernel.evaluate(&lo, &hi, 4);
        assert_eq!(result.len(), 5);

        // t=0: prod(1,2,3,4) + prod(5,6,7,8) = 24 + 1680 = 1704
        assert_eq!(result[0], Fr::from_u64(24 + 1680));
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
        let result = kernel.evaluate(&lo, &hi, 3);
        assert_eq!(result.len(), 4);

        // t=0: 1*2*3 = 6
        assert_eq!(result[0], Fr::from_u64(6));
        // t=1: 4*5*6 = 120
        assert_eq!(result[1], Fr::from_u64(120));
    }

    #[test]
    fn compile_custom_simple_product() {
        // expr = o0 * o1
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
        let result = kernel.evaluate(&lo, &hi, 2);

        // t=0: 3*5 = 15
        assert_eq!(result[0], Fr::from_u64(15));
        // t=1: 7*11 = 77
        assert_eq!(result[1], Fr::from_u64(77));
        // t=2: (3+2*(7-3))*(5+2*(11-5)) = 11*17 = 187
        assert_eq!(result[2], Fr::from_u64(187));
    }

    #[test]
    fn compile_custom_booleanity() {
        // Booleanity: o0^2 - o0
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
        let result = kernel.evaluate(&lo, &hi, 2);

        // t=0: 3^2 - 3 = 6
        assert_eq!(result[0], Fr::from_u64(6));
        // t=1: 7^2 - 7 = 42
        assert_eq!(result[1], Fr::from_u64(42));
    }

    #[test]
    fn compile_custom_with_challenge() {
        // expr = c0 * o0  (challenge * opening)
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

        // Without challenge bindings: defaults to zero
        let kernel_zero: CpuKernel<Fr> = compile(&desc);
        let lo = vec![Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(10)];
        let result = kernel_zero.evaluate(&lo, &hi, 1);
        assert_eq!(result[0], Fr::zero());
        assert_eq!(result[1], Fr::zero());

        // With challenge binding: gamma = 7
        let kernel: CpuKernel<Fr> = compile_with_challenges(&desc, &[Fr::from_u64(7)]);
        let result = kernel.evaluate(&lo, &hi, 1);
        // t=0: 7*5 = 35, t=1: 7*10 = 70
        assert_eq!(result[0], Fr::from_u64(35));
        assert_eq!(result[1], Fr::from_u64(70));
    }

    #[test]
    fn compile_with_tensor_split_ignored() {
        // TensorSplit doesn't affect compilation — it's a scheduling hint
        let desc = KernelDescriptor {
            shape: KernelShape::ProductSum {
                num_inputs_per_product: 4,
                num_products: 1,
            },
            degree: 4,
            tensor_split: Some(TensorSplit::balanced(20)),
        };
        let kernel: CpuKernel<Fr> = compile(&desc);

        let lo: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (5..=8).map(Fr::from_u64).collect();
        let result = kernel.evaluate(&lo, &hi, 4);
        assert_eq!(result[0], Fr::from_u64(24));
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
