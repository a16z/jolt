//! ProductSum kernel compilation using Toom-Cook evaluation.
//!
//! For each pair position, the kernel evaluates:
//! $$P(t) = \sum_{g=0}^{P-1} \prod_{j=0}^{D-1} p_{g,j}(t)$$
//!
//! on the Toom-Cook grid $U_D = \{1, 2, \ldots, D-1, \infty\}$, producing $D$
//! evaluations per position. This grid enables $O(D \log D)$ multiplications
//! via balanced binary splitting with extrapolation, compared to $O(D^2)$ for
//! the standard grid $\{0, 1, \ldots, D\}$.
//!
//! Specialized kernels for D∈{4, 8, 16, 32} use the hand-optimized
//! Toom-Cook implementations from [`crate::toom_cook`].

use crate::toom_cook;
use crate::CpuKernel;
use jolt_field::Field;

/// Compile a ProductSum kernel with the given parameters.
///
/// Returns a [`CpuKernel`] that evaluates the sum-of-products composition
/// on the Toom-Cook grid, producing `D` evaluations (not `D+1`).
///
/// Dispatches to hand-optimized Toom-Cook closures for D∈{4, 8, 16, 32}
/// and falls back to the [`toom_cook::eval_linear_prod_assign`] dispatcher
/// for other values.
pub fn compile<F: Field>(d: usize, num_products: usize) -> CpuKernel<F> {
    match d {
        4 => compile_d4(num_products),
        8 => compile_d8(num_products),
        16 => compile_d16(num_products),
        32 => compile_d32(num_products),
        _ => compile_generic(d, num_products),
    }
}

macro_rules! compile_specialized {
    ($name:ident, $d:literal, $eval_fn:path) => {
        fn $name<F: Field>(num_products: usize) -> CpuKernel<F> {
            CpuKernel::new(
                move |lo: &[F], hi: &[F], _challenges: &[F], out: &mut [F]| {
                    for slot in out.iter_mut() {
                        *slot = F::zero();
                    }

                    for g in 0..num_products {
                        let base = g * $d;
                        let pairs: [(F, F); $d] =
                            core::array::from_fn(|j| (lo[base + j], hi[base + j]));
                        let mut group = [F::zero(); $d];
                        $eval_fn(&pairs, &mut group);
                        for k in 0..$d {
                            out[k] += group[k];
                        }
                    }
                },
            )
        }
    };
}

compile_specialized!(compile_d4, 4, toom_cook::eval_prod_4_assign);
compile_specialized!(compile_d8, 8, toom_cook::eval_prod_8_assign);
compile_specialized!(compile_d16, 16, toom_cook::eval_prod_16_assign);
compile_specialized!(compile_d32, 32, toom_cook::eval_prod_32_assign);

fn compile_generic<F: Field>(d: usize, num_products: usize) -> CpuKernel<F> {
    CpuKernel::new(
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
mod tests {
    use super::*;
    use jolt_field::Fr;
    use num_traits::{One, Zero};

    /// Reference implementation computing on the Toom-Cook grid.
    fn reference_toom_cook(lo: &[Fr], hi: &[Fr], d: usize, num_products: usize) -> Vec<Fr> {
        let mut evals = vec![Fr::zero(); d];

        for g in 0..num_products {
            let base = g * d;

            // t = 1, 2, ..., D-1
            for t in 1..d {
                let t_f = Fr::from_u64(t as u64);
                let mut product = Fr::one();
                for j in 0..d {
                    let delta = hi[base + j] - lo[base + j];
                    product *= lo[base + j] + t_f * delta;
                }
                evals[t - 1] += product;
            }

            // t = ∞: product of slopes
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
        let (lo, hi) = random_vecs(12, 100); // 3 product groups
        let kernel = compile_d4::<Fr>(3);
        assert_eq!(
            eval_kernel(&kernel, &lo, &hi, 4),
            reference_toom_cook(&lo, &hi, 4, 3)
        );
    }

    #[test]
    fn d8_matches_reference() {
        let (lo, hi) = random_vecs(16, 200); // 2 product groups
        let kernel = compile_d8::<Fr>(2);
        assert_eq!(
            eval_kernel(&kernel, &lo, &hi, 8),
            reference_toom_cook(&lo, &hi, 8, 2)
        );
    }

    #[test]
    fn d16_matches_reference() {
        let (lo, hi) = random_vecs(16, 300); // 1 product group
        let kernel = compile_d16::<Fr>(1);
        assert_eq!(
            eval_kernel(&kernel, &lo, &hi, 16),
            reference_toom_cook(&lo, &hi, 16, 1)
        );
    }

    #[test]
    fn d16_multiple_groups_matches_reference() {
        let (lo, hi) = random_vecs(48, 301); // 3 product groups
        let kernel = compile_d16::<Fr>(3);
        assert_eq!(
            eval_kernel(&kernel, &lo, &hi, 16),
            reference_toom_cook(&lo, &hi, 16, 3)
        );
    }

    #[test]
    fn d32_matches_reference() {
        let (lo, hi) = random_vecs(32, 350); // 1 product group
        let kernel = compile_d32::<Fr>(1);
        assert_eq!(
            eval_kernel(&kernel, &lo, &hi, 32),
            reference_toom_cook(&lo, &hi, 32, 1)
        );
    }

    #[test]
    fn generic_d5_matches_reference() {
        let (lo, hi) = random_vecs(10, 400); // 2 product groups
        let kernel = compile_generic::<Fr>(5, 2);
        assert_eq!(
            eval_kernel(&kernel, &lo, &hi, 5),
            reference_toom_cook(&lo, &hi, 5, 2)
        );
    }

    #[test]
    fn generic_d3_matches_reference() {
        let (lo, hi) = random_vecs(6, 410); // 2 product groups
        let kernel = compile_generic::<Fr>(3, 2);
        assert_eq!(
            eval_kernel(&kernel, &lo, &hi, 3),
            reference_toom_cook(&lo, &hi, 3, 2)
        );
    }

    #[test]
    fn specialized_and_generic_agree() {
        for d in [4, 8, 16, 32] {
            let (lo, hi) = random_vecs(d * 2, d as u64 * 1000); // 2 groups
            let specialized: CpuKernel<Fr> = compile(d, 2);
            let generic = compile_generic::<Fr>(d, 2);
            assert_eq!(
                eval_kernel(&specialized, &lo, &hi, d),
                eval_kernel(&generic, &lo, &hi, d),
                "mismatch for D={d}"
            );
        }
    }

    #[test]
    fn d4_known_values() {
        // All p_j(x) = 1 + x => P(t) = (1+t)^4
        let lo: Vec<Fr> = vec![Fr::one(); 4];
        let hi: Vec<Fr> = vec![Fr::from_u64(2); 4];
        let kernel = compile_d4::<Fr>(1);
        let result = eval_kernel(&kernel, &lo, &hi, 4);

        // Toom-Cook grid: [P(1), P(2), P(3), P(∞)]
        assert_eq!(result[0], Fr::from_u64(2u64.pow(4))); // P(1) = 16
        assert_eq!(result[1], Fr::from_u64(3u64.pow(4))); // P(2) = 81
        assert_eq!(result[2], Fr::from_u64(4u64.pow(4))); // P(3) = 256
        assert_eq!(result[3], Fr::one()); // P(∞) = 1^4 = 1
    }
}
