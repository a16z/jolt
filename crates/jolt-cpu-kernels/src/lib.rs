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
mod specialized;
pub mod toom_cook;

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

        // p_j(x) = lo[j] + (hi[j] - lo[j])*x
        let lo: Vec<Fr> = (1..=4).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (5..=8).map(Fr::from_u64).collect();
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());

        // Toom-Cook grid: [P(1), P(2), P(3), P(∞)], D=4 outputs
        assert_eq!(result.len(), 4);
        // P(1) = hi[0]*hi[1]*hi[2]*hi[3] = 5*6*7*8 = 1680
        assert_eq!(result[0], Fr::from_u64(1680));
        // P(∞) = Π(hi[j]-lo[j]) = 4*4*4*4 = 256
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

        // All p_j(x) = 1 + x => P(t) = (1+t)^8
        let lo: Vec<Fr> = vec![Fr::one(); 8];
        let hi: Vec<Fr> = vec![Fr::from_u64(2); 8];
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());

        // D=8 outputs on Toom-Cook grid
        assert_eq!(result.len(), 8);
        // P(1) = 2^8 = 256
        assert_eq!(result[0], Fr::from_u64(256));
        // P(∞) = 1^8 = 1
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

        // All p_j(x) = 1 + x => P(t) = (1+t)^16
        let lo: Vec<Fr> = vec![Fr::one(); 16];
        let hi: Vec<Fr> = vec![Fr::from_u64(2); 16];
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());

        // D=16 outputs
        assert_eq!(result.len(), 16);
        // P(1) = 2^16 = 65536
        assert_eq!(result[0], Fr::from_u64(65536));
        // P(∞) = 1^16 = 1
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

        // Group 0: lo=(1,2,3,4), hi=(9,10,11,12)
        // Group 1: lo=(5,6,7,8), hi=(13,14,15,16)
        let lo: Vec<Fr> = (1..=8).map(Fr::from_u64).collect();
        let hi: Vec<Fr> = (9..=16).map(Fr::from_u64).collect();
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());
        assert_eq!(result.len(), 4);

        // P(1) = prod(hi_group0) + prod(hi_group1) = 9*10*11*12 + 13*14*15*16
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

        // Toom-Cook grid: [P(1), P(2), P(∞)], D=3 outputs
        assert_eq!(result.len(), 3);
        // P(1) = 4*5*6 = 120
        assert_eq!(result[0], Fr::from_u64(120));
        // P(2) = (1+2*3)*(2+2*3)*(3+2*3) = 7*8*9 = 504
        assert_eq!(result[1], Fr::from_u64(504));
        // P(∞) = 3*3*3 = 27
        assert_eq!(result[2], Fr::from_u64(27));
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
        // Custom: num_evals = degree = 2, grid {0, 2}
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());

        // t=0: 3*5 = 15
        assert_eq!(result[0], Fr::from_u64(15));
        // t=2: (3+2*(7-3))*(5+2*(11-5)) = 11*17 = 187
        assert_eq!(result[1], Fr::from_u64(187));
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
        // Custom: num_evals = degree = 2, grid {0, 2}
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());

        // t=0: 3^2 - 3 = 6
        assert_eq!(result[0], Fr::from_u64(6));
        // t=2: h(2) = 3+2*4 = 11, 11^2 - 11 = 110
        assert_eq!(result[1], Fr::from_u64(110));
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

        // Custom degree=1: num_evals = 1, grid {0}
        // Without challenge bindings: defaults to zero
        let kernel_zero: CpuKernel<Fr> = compile(&desc);
        let lo = vec![Fr::from_u64(5)];
        let hi = vec![Fr::from_u64(10)];
        let result = eval_kernel(&kernel_zero, &lo, &hi, desc.num_evals());
        assert_eq!(result[0], Fr::zero());

        // With challenge binding: gamma = 7
        let kernel: CpuKernel<Fr> = compile_with_challenges(&desc, &[Fr::from_u64(7)]);
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());
        // t=0: 7*5 = 35
        assert_eq!(result[0], Fr::from_u64(35));
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

        // All p_j(x) = 1 + x => P(t) = (1+t)^4
        let lo: Vec<Fr> = vec![Fr::one(); 4];
        let hi: Vec<Fr> = vec![Fr::from_u64(2); 4];
        let result = eval_kernel(&kernel, &lo, &hi, desc.num_evals());
        // P(1) = 2^4 = 16
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
        // ProductSum: num_evals = D = 4
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
        // Custom: num_evals = degree = 2
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
