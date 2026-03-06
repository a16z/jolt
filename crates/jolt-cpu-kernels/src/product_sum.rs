//! ProductSum kernel compilation with specializations for D=4, 8, 16.
//!
//! For each grid point $t \in \{0, 1, \ldots, D\}$, the kernel evaluates:
//! $$\text{eval}[t] = \sum_{g=0}^{P-1} \prod_{j=0}^{D-1}
//!     \bigl(\text{lo}[g \cdot D + j] + t \cdot (\text{hi}[g \cdot D + j] - \text{lo}[g \cdot D + j])\bigr)$$
//!
//! Specialized kernels for D=4, 8, 16 avoid the inner loop overhead by unrolling
//! the product and precomputing the `delta = hi - lo` differences.

use jolt_compute::CpuKernel;
use jolt_field::Field;

/// Compile a ProductSum kernel with the given parameters.
///
/// Dispatches to hand-optimized closures for D∈{4,8,16} and falls back to a
/// generic loop-based kernel for other values.
pub fn compile<F: Field>(d: usize, num_products: usize) -> CpuKernel<F> {
    match d {
        4 => compile_d4(num_products),
        8 => compile_d8(num_products),
        16 => compile_d16(num_products),
        _ => compile_generic(d, num_products),
    }
}

/// D=4 specialized kernel. Fully unrolled 4-way product.
fn compile_d4<F: Field>(num_products: usize) -> CpuKernel<F> {
    CpuKernel::new(move |lo: &[F], hi: &[F], degree: usize| {
        let num_outputs = degree + 1;
        let mut evals = vec![F::zero(); num_outputs];

        for g in 0..num_products {
            let base = g * 4;
            let d0 = hi[base] - lo[base];
            let d1 = hi[base + 1] - lo[base + 1];
            let d2 = hi[base + 2] - lo[base + 2];
            let d3 = hi[base + 3] - lo[base + 3];

            for (t, eval) in evals.iter_mut().enumerate() {
                let t_f = F::from_u64(t as u64);
                let v0 = lo[base] + t_f * d0;
                let v1 = lo[base + 1] + t_f * d1;
                let v2 = lo[base + 2] + t_f * d2;
                let v3 = lo[base + 3] + t_f * d3;
                *eval += v0 * v1 * v2 * v3;
            }
        }

        evals
    })
}

/// D=8 specialized kernel. Fully unrolled 8-way product.
fn compile_d8<F: Field>(num_products: usize) -> CpuKernel<F> {
    CpuKernel::new(move |lo: &[F], hi: &[F], degree: usize| {
        let num_outputs = degree + 1;
        let mut evals = vec![F::zero(); num_outputs];

        for g in 0..num_products {
            let base = g * 8;
            let d0 = hi[base] - lo[base];
            let d1 = hi[base + 1] - lo[base + 1];
            let d2 = hi[base + 2] - lo[base + 2];
            let d3 = hi[base + 3] - lo[base + 3];
            let d4 = hi[base + 4] - lo[base + 4];
            let d5 = hi[base + 5] - lo[base + 5];
            let d6 = hi[base + 6] - lo[base + 6];
            let d7 = hi[base + 7] - lo[base + 7];

            for (t, eval) in evals.iter_mut().enumerate() {
                let t_f = F::from_u64(t as u64);
                let v0 = lo[base] + t_f * d0;
                let v1 = lo[base + 1] + t_f * d1;
                let v2 = lo[base + 2] + t_f * d2;
                let v3 = lo[base + 3] + t_f * d3;
                let v4 = lo[base + 4] + t_f * d4;
                let v5 = lo[base + 5] + t_f * d5;
                let v6 = lo[base + 6] + t_f * d6;
                let v7 = lo[base + 7] + t_f * d7;
                // Balanced tree multiplication to improve ILP
                let p01 = v0 * v1;
                let p23 = v2 * v3;
                let p45 = v4 * v5;
                let p67 = v6 * v7;
                let p0123 = p01 * p23;
                let p4567 = p45 * p67;
                *eval += p0123 * p4567;
            }
        }

        evals
    })
}

/// D=16 specialized kernel. Fully unrolled 16-way product with balanced tree.
fn compile_d16<F: Field>(num_products: usize) -> CpuKernel<F> {
    CpuKernel::new(move |lo: &[F], hi: &[F], degree: usize| {
        let num_outputs = degree + 1;
        let mut evals = vec![F::zero(); num_outputs];

        for g in 0..num_products {
            let base = g * 16;

            // Precompute deltas
            let mut deltas = [F::zero(); 16];
            for (j, delta) in deltas.iter_mut().enumerate() {
                *delta = hi[base + j] - lo[base + j];
            }

            for (t, eval) in evals.iter_mut().enumerate() {
                let t_f = F::from_u64(t as u64);

                // Compute interpolated values
                let mut v = [F::zero(); 16];
                for (j, vj) in v.iter_mut().enumerate() {
                    *vj = lo[base + j] + t_f * deltas[j];
                }

                // Balanced tree reduction: 16 → 8 → 4 → 2 → 1
                let p01 = v[0] * v[1];
                let p23 = v[2] * v[3];
                let p45 = v[4] * v[5];
                let p67 = v[6] * v[7];
                let p89 = v[8] * v[9];
                let p_ab = v[10] * v[11];
                let p_cd = v[12] * v[13];
                let p_ef = v[14] * v[15];

                let q0 = p01 * p23;
                let q1 = p45 * p67;
                let q2 = p89 * p_ab;
                let q3 = p_cd * p_ef;

                let r0 = q0 * q1;
                let r1 = q2 * q3;

                *eval += r0 * r1;
            }
        }

        evals
    })
}

/// Generic fallback for arbitrary D values.
fn compile_generic<F: Field>(d: usize, num_products: usize) -> CpuKernel<F> {
    CpuKernel::new(move |lo: &[F], hi: &[F], degree: usize| {
        let num_outputs = degree + 1;
        let mut evals = vec![F::zero(); num_outputs];

        for g in 0..num_products {
            let base = g * d;

            for (t, eval) in evals.iter_mut().enumerate() {
                let t_f = F::from_u64(t as u64);
                let mut product = F::one();
                for j in 0..d {
                    product *= lo[base + j] + t_f * (hi[base + j] - lo[base + j]);
                }
                *eval += product;
            }
        }

        evals
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use num_traits::{One, Zero};

    /// Reference implementation for cross-validation.
    fn reference_product_sum(
        lo: &[Fr],
        hi: &[Fr],
        d: usize,
        num_products: usize,
        degree: usize,
    ) -> Vec<Fr> {
        let num_outputs = degree + 1;
        let mut evals = vec![Fr::zero(); num_outputs];

        for g in 0..num_products {
            let base = g * d;
            for (t, eval) in evals.iter_mut().enumerate() {
                let t_f = Fr::from_u64(t as u64);
                let mut product = Fr::one();
                for j in 0..d {
                    product *= lo[base + j] + t_f * (hi[base + j] - lo[base + j]);
                }
                *eval += product;
            }
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

    #[test]
    fn d4_matches_reference() {
        let (lo, hi) = random_vecs(12, 100); // 3 product groups
        let kernel = compile_d4::<Fr>(3);
        let result = kernel.evaluate(&lo, &hi, 4);
        let expected = reference_product_sum(&lo, &hi, 4, 3, 4);
        assert_eq!(result, expected);
    }

    #[test]
    fn d8_matches_reference() {
        let (lo, hi) = random_vecs(16, 200); // 2 product groups
        let kernel = compile_d8::<Fr>(2);
        let result = kernel.evaluate(&lo, &hi, 8);
        let expected = reference_product_sum(&lo, &hi, 8, 2, 8);
        assert_eq!(result, expected);
    }

    #[test]
    fn d16_matches_reference() {
        let (lo, hi) = random_vecs(16, 300); // 1 product group
        let kernel = compile_d16::<Fr>(1);
        let result = kernel.evaluate(&lo, &hi, 16);
        let expected = reference_product_sum(&lo, &hi, 16, 1, 16);
        assert_eq!(result, expected);
    }

    #[test]
    fn d16_multiple_groups_matches_reference() {
        let (lo, hi) = random_vecs(48, 301); // 3 product groups
        let kernel = compile_d16::<Fr>(3);
        let result = kernel.evaluate(&lo, &hi, 16);
        let expected = reference_product_sum(&lo, &hi, 16, 3, 16);
        assert_eq!(result, expected);
    }

    #[test]
    fn generic_d5_matches_reference() {
        let (lo, hi) = random_vecs(10, 400); // 2 product groups
        let kernel = compile_generic::<Fr>(5, 2);
        let result = kernel.evaluate(&lo, &hi, 5);
        let expected = reference_product_sum(&lo, &hi, 5, 2, 5);
        assert_eq!(result, expected);
    }

    #[test]
    fn generic_d1_is_sum() {
        // D=1: product of 1 linear function = linear function itself
        let lo = vec![Fr::from_u64(3)];
        let hi = vec![Fr::from_u64(7)];
        let kernel = compile_generic::<Fr>(1, 1);
        let result = kernel.evaluate(&lo, &hi, 1);

        assert_eq!(result[0], Fr::from_u64(3)); // t=0: lo
        assert_eq!(result[1], Fr::from_u64(7)); // t=1: hi
    }

    #[test]
    fn specialized_and_generic_agree() {
        for d in [4, 8, 16] {
            let (lo, hi) = random_vecs(d * 2, d as u64 * 1000); // 2 groups
            let specialized: CpuKernel<Fr> = compile(d, 2);
            let generic = compile_generic::<Fr>(d, 2);

            let result_s = specialized.evaluate(&lo, &hi, d);
            let result_g = generic.evaluate(&lo, &hi, d);
            assert_eq!(result_s, result_g, "mismatch for D={d}");
        }
    }
}
