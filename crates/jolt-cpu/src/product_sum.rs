//! ProductSum eval closure using Toom-Cook evaluation.
//!
//! When a [`KernelSpec`]'s composition is a pure product-sum (every term is
//! a product of D inputs with coefficient 1), this module compiles an eval
//! closure that evaluates on the Toom-Cook grid `{1, 2, …, D-1, ∞}`.
//!
//! This grid enables `O(D log D)` multiplications via balanced binary
//! splitting with extrapolation, compared to `O(D²)` on the standard grid.

use crate::toom_cook;
use jolt_field::Field;

/// Compile a product-sum eval closure for `num_products` groups of `d` inputs.
///
/// Dispatches to const-generic specializations for common D values to avoid
/// heap allocation in the eval hot path. Falls back to a dynamic-allocation
/// path for uncommon D.
pub fn compile_fn<F: Field>(d: usize, num_products: usize) -> crate::BoxedEvalFn<F> {
    if num_products == 1 {
        // P=1 fast path: write directly to `out`, no group accumulation.
        match d {
            2 => compile_single_product::<F, 2>(),
            3 => compile_single_product::<F, 3>(),
            4 => compile_single_product::<F, 4>(),
            5 => compile_single_product::<F, 5>(),
            6 => compile_single_product::<F, 6>(),
            7 => compile_single_product::<F, 7>(),
            8 => compile_single_product::<F, 8>(),
            16 => compile_single_product::<F, 16>(),
            32 => compile_single_product::<F, 32>(),
            _ => compile_dynamic::<F>(d, 1),
        }
    } else {
        match d {
            2 => compile_multi_product::<F, 2>(num_products),
            3 => compile_multi_product::<F, 3>(num_products),
            4 => compile_multi_product::<F, 4>(num_products),
            5 => compile_multi_product::<F, 5>(num_products),
            6 => compile_multi_product::<F, 6>(num_products),
            7 => compile_multi_product::<F, 7>(num_products),
            8 => compile_multi_product::<F, 8>(num_products),
            16 => compile_multi_product::<F, 16>(num_products),
            32 => compile_multi_product::<F, 32>(num_products),
            _ => compile_dynamic::<F>(d, num_products),
        }
    }
}

/// P=1 fast path: stack-allocated pairs, output written directly (no group buffer).
fn compile_single_product<F: Field, const D: usize>() -> crate::BoxedEvalFn<F> {
    Box::new(
        move |lo: &[F], hi: &[F], _challenges: &[F], out: &mut [F]| {
            let mut pairs = [(F::zero(), F::zero()); D];
            for j in 0..D {
                pairs[j] = (lo[j], hi[j]);
            }
            toom_cook::eval_linear_prod_assign(&pairs, out);
        },
    )
}

/// P>1 path: stack-allocated pairs and group buffer, accumulate into `out`.
fn compile_multi_product<F: Field, const D: usize>(
    num_products: usize,
) -> crate::BoxedEvalFn<F> {
    Box::new(
        move |lo: &[F], hi: &[F], _challenges: &[F], out: &mut [F]| {
            for slot in out.iter_mut() {
                *slot = F::zero();
            }
            let mut pairs = [(F::zero(), F::zero()); D];
            let mut group = [F::zero(); D];
            for g in 0..num_products {
                let base = g * D;
                for j in 0..D {
                    pairs[j] = (lo[base + j], hi[base + j]);
                }
                toom_cook::eval_linear_prod_assign(&pairs, &mut group);
                for k in 0..D {
                    out[k] += group[k];
                }
            }
        },
    )
}

/// Fallback for uncommon D values: uses Vec allocation.
fn compile_dynamic<F: Field>(d: usize, num_products: usize) -> crate::BoxedEvalFn<F> {
    Box::new(
        move |lo: &[F], hi: &[F], _challenges: &[F], out: &mut [F]| {
            for slot in out.iter_mut() {
                *slot = F::zero();
            }
            let mut pairs = vec![(F::zero(), F::zero()); d];
            let mut group = vec![F::zero(); d];
            for g in 0..num_products {
                let base = g * d;
                for j in 0..d {
                    pairs[j] = (lo[base + j], hi[base + j]);
                }
                toom_cook::eval_linear_prod_assign(&pairs, &mut group);
                for k in 0..d {
                    out[k] += group[k];
                }
            }
        },
    )
}

#[cfg(test)]
fn compile_kernel<F: Field>(d: usize, num_products: usize) -> crate::CpuKernel<F> {
    use jolt_compiler::{BindingOrder, Iteration};
    crate::CpuKernel::new(
        compile_fn::<F>(d, num_products),
        d,
        Iteration::Dense,
        BindingOrder::LowToHigh,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuKernel;
    use jolt_field::Fr;
    use num_traits::{One, Zero};

    fn reference_toom_cook(lo: &[Fr], hi: &[Fr], d: usize, num_products: usize) -> Vec<Fr> {
        let mut evals = vec![Fr::zero(); d];

        for g in 0..num_products {
            let base = g * d;

            for t in 1..d {
                let t_f = Fr::from_u64(t as u64);
                let mut product = Fr::one();
                for j in 0..d {
                    let delta = hi[base + j] - lo[base + j];
                    product *= lo[base + j] + t_f * delta;
                }
                evals[t - 1] += product;
            }

            let mut prod_inf = Fr::one();
            for j in 0..d {
                prod_inf *= hi[base + j] - lo[base + j];
            }
            evals[d - 1] += prod_inf;
        }

        evals
    }

    fn random_vecs(n: usize, seed: u64) -> (Vec<Fr>, Vec<Fr>) {
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;

        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let lo: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let hi: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        (lo, hi)
    }

    fn eval_kernel(kernel: &CpuKernel<Fr>, lo: &[Fr], hi: &[Fr], d: usize) -> Vec<Fr> {
        let mut out = vec![Fr::zero(); d];
        kernel.evaluate(lo, hi, &[], &mut out);
        out
    }

    #[test]
    fn d4_matches_reference() {
        let (lo, hi) = random_vecs(12, 100);
        let kernel = compile_kernel::<Fr>(4, 3);
        assert_eq!(
            eval_kernel(&kernel, &lo, &hi, 4),
            reference_toom_cook(&lo, &hi, 4, 3)
        );
    }

    #[test]
    fn d8_matches_reference() {
        let (lo, hi) = random_vecs(16, 200);
        let kernel = compile_kernel::<Fr>(8, 2);
        assert_eq!(
            eval_kernel(&kernel, &lo, &hi, 8),
            reference_toom_cook(&lo, &hi, 8, 2)
        );
    }

    #[test]
    fn d16_matches_reference() {
        let (lo, hi) = random_vecs(16, 300);
        let kernel = compile_kernel::<Fr>(16, 1);
        assert_eq!(
            eval_kernel(&kernel, &lo, &hi, 16),
            reference_toom_cook(&lo, &hi, 16, 1)
        );
    }

    #[test]
    fn d16_multiple_groups_matches_reference() {
        let (lo, hi) = random_vecs(48, 301);
        let kernel = compile_kernel::<Fr>(16, 3);
        assert_eq!(
            eval_kernel(&kernel, &lo, &hi, 16),
            reference_toom_cook(&lo, &hi, 16, 3)
        );
    }

    #[test]
    fn d32_matches_reference() {
        let (lo, hi) = random_vecs(32, 350);
        let kernel = compile_kernel::<Fr>(32, 1);
        assert_eq!(
            eval_kernel(&kernel, &lo, &hi, 32),
            reference_toom_cook(&lo, &hi, 32, 1)
        );
    }

    #[test]
    fn generic_d5_matches_reference() {
        let (lo, hi) = random_vecs(10, 400);
        let kernel = compile_kernel::<Fr>(5, 2);
        assert_eq!(
            eval_kernel(&kernel, &lo, &hi, 5),
            reference_toom_cook(&lo, &hi, 5, 2)
        );
    }

    #[test]
    fn generic_d3_matches_reference() {
        let (lo, hi) = random_vecs(6, 410);
        let kernel = compile_kernel::<Fr>(3, 2);
        assert_eq!(
            eval_kernel(&kernel, &lo, &hi, 3),
            reference_toom_cook(&lo, &hi, 3, 2)
        );
    }

    #[test]
    fn d4_known_values() {
        let lo: Vec<Fr> = vec![Fr::one(); 4];
        let hi: Vec<Fr> = vec![Fr::from_u64(2); 4];
        let kernel = compile_kernel::<Fr>(4, 1);
        let result = eval_kernel(&kernel, &lo, &hi, 4);

        assert_eq!(result[0], Fr::from_u64(2u64.pow(4)));
        assert_eq!(result[1], Fr::from_u64(3u64.pow(4)));
        assert_eq!(result[2], Fr::from_u64(4u64.pow(4)));
        assert_eq!(result[3], Fr::one());
    }
}
