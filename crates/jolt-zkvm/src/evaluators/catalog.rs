//! Kernel descriptor catalog for sumcheck compositions.
//!
//! Each function returns a [`KernelDescriptor`] for a specific sumcheck
//! formula. Stages compile these descriptors into backend-specific kernels
//! and pass them to [`KernelEvaluator`](super::kernel::KernelEvaluator).
//!
//! Named shapes are hand-coded kernels that eliminate stack-VM dispatch
//! overhead. The IR's `compile_descriptor` handles arbitrary formulas.

use jolt_ir::{KernelDescriptor, KernelShape};

/// Descriptor for a sum-of-products composition on the Toom-Cook grid.
///
/// Produces D evaluations at `{1, ..., D-1, ∞}` for each pair position,
/// summing across `num_products` product groups. Used with
/// [`KernelEvaluator::with_toom_cook_eq`](super::kernel::KernelEvaluator::with_toom_cook_eq)
/// for RA virtual sumchecks.
///
/// Input layout: `opening(g*D + k)` for group `g`, factor `k`.
pub fn product_sum(d: usize, num_products: usize) -> KernelDescriptor {
    KernelDescriptor {
        shape: KernelShape::ProductSum {
            num_inputs_per_product: d,
            num_products,
        },
        degree: d,
        tensor_split: None,
    }
}

/// Descriptor for `eq(x) · g(x)` — degree 2, 2 inputs.
///
/// Uses a hand-coded kernel that eliminates stack-VM dispatch overhead.
///
/// Input layout: `opening(0) = eq`, `opening(1) = g`.
pub fn eq_product() -> KernelDescriptor {
    KernelDescriptor {
        shape: KernelShape::EqProduct,
        degree: 2,
        tensor_split: None,
    }
}

/// Descriptor for `eq(x) · h(x) · (h(x) − 1)` — degree 3, 2 inputs.
///
/// Uses a hand-coded kernel that eliminates stack-VM dispatch overhead.
///
/// Input layout: `opening(0) = eq`, `opening(1) = h`.
pub fn hamming_booleanity() -> KernelDescriptor {
    KernelDescriptor {
        shape: KernelShape::HammingBooleanity,
        degree: 3,
        tensor_split: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use jolt_compute::ComputeBackend;
    use jolt_cpu::CpuBackend;
    use jolt_field::{Field, Fr};
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::prover::SumcheckCompute;
    use jolt_sumcheck::{SumcheckClaim, SumcheckProver, SumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use crate::evaluators::kernel::KernelEvaluator;

    fn cpu() -> Arc<CpuBackend> {
        Arc::new(CpuBackend)
    }

    #[test]
    fn eq_product_round_polynomial_consistency() {
        let backend = cpu();
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let g_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let s0: Fr = (0..n / 2).map(|j| eq_table[2 * j] * g_table[2 * j]).sum();
        let s1: Fr = (0..n / 2)
            .map(|j| eq_table[2 * j + 1] * g_table[2 * j + 1])
            .sum();

        let desc = eq_product();
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&g_table)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        // P(1) is derived from the claim, so set it before round_polynomial()
        witness.set_claim(s0 + s1);
        let poly = witness.round_polynomial();

        assert_eq!(poly.evaluate(Fr::zero()), s0);
        assert_eq!(poly.evaluate(Fr::one()), s1);
    }

    #[test]
    fn eq_product_full_prove_verify() {
        let backend = cpu();
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let g_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let claimed_sum: Fr = eq_table
            .iter()
            .zip(g_table.iter())
            .map(|(&e, &g)| e * g)
            .sum();

        let claim = SumcheckClaim {
            num_vars,
            degree: 2,
            claimed_sum,
        };

        let desc = eq_product();
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&g_table)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let mut pt = Blake2bTranscript::new(b"eq_product_test");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"eq_product_test");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    fn brute_force_booleanity_sum(h: &[Fr], eq: &[Fr]) -> Fr {
        h.iter()
            .zip(eq.iter())
            .map(|(&h_val, &eq_val)| eq_val * h_val * (h_val - Fr::one()))
            .sum()
    }

    #[test]
    fn boolean_polynomial_has_zero_sum() {
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();

        let h_table: Vec<Fr> = vec![0, 1, 1, 0, 1, 0, 0, 1]
            .into_iter()
            .map(Fr::from_u64)
            .collect();

        let sum = brute_force_booleanity_sum(&h_table, &eq_table);
        assert!(sum.is_zero(), "boolean polynomial should have sum 0");
    }

    #[test]
    fn hamming_round_polynomial_consistency() {
        let backend = cpu();
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();

        let h_table: Vec<Fr> = vec![0, 1, 1, 0, 1, 0, 0, 1]
            .into_iter()
            .map(Fr::from_u64)
            .collect();

        let one = Fr::one();
        let s0: Fr = (0..4)
            .map(|j| eq_table[2 * j] * h_table[2 * j] * (h_table[2 * j] - one))
            .sum();
        let s1: Fr = (0..4)
            .map(|j| eq_table[2 * j + 1] * h_table[2 * j + 1] * (h_table[2 * j + 1] - one))
            .sum();

        let desc = hamming_booleanity();
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&h_table)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        witness.set_claim(s0 + s1);
        let poly = witness.round_polynomial();

        assert_eq!(poly.evaluate(Fr::zero()), s0);
        assert_eq!(poly.evaluate(Fr::one()), s1);

        let total = s0 + s1;
        assert!(total.is_zero());
    }

    #[test]
    fn hamming_full_prove_verify() {
        let backend = cpu();
        let num_vars = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let h_table: Vec<Fr> = (0..(1 << num_vars)).map(|i| Fr::from_u64(i % 2)).collect();

        let claimed_sum = brute_force_booleanity_sum(&h_table, &eq_table);

        let claim = SumcheckClaim {
            num_vars,
            degree: 3,
            claimed_sum,
        };

        let desc = hamming_booleanity();
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&h_table)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let mut pt = Blake2bTranscript::new(b"test_hamming_booleanity");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"test_hamming_booleanity");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }
}
