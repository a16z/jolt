//! Generic eq · g product [`SumcheckCompute`] evaluator (degree 2).
//!
//! Proves $\sum_{x \in \{0,1\}^n} \widetilde{eq}(r, x) \cdot g(x) = v$ where
//! $g$ is a pre-computed multilinear polynomial and $v$ is the claimed sum.
//!
//! Used by claim reduction stages where $g$ is a random linear combination
//! of the input polynomials.
//!
//! Thin wrapper around [`KernelEvaluator`](super::kernel::KernelEvaluator)
//! with a degree-2 `Custom` kernel (`eq · g`).

use std::sync::Arc;

use jolt_compute::ComputeBackend;
use jolt_field::Field;
use jolt_ir::{ExprBuilder, KernelDescriptor, KernelShape};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::prover::SumcheckCompute;

use super::kernel::KernelEvaluator;

/// Sumcheck evaluator for the product $\widetilde{eq}(r, x) \cdot g(x)$.
///
/// Degree 2 per variable (1 from eq + 1 from g). Delegates all computation
/// to [`KernelEvaluator`] via a compiled `eq · g` kernel.
pub struct EqProductEvaluator<F: Field, B: ComputeBackend> {
    inner: KernelEvaluator<F, B>,
}

impl<F: Field, B: ComputeBackend> EqProductEvaluator<F, B> {
    /// Creates an evaluator from pre-uploaded eq and g buffers.
    ///
    /// `kernel` must be compiled from [`descriptor()`](Self::descriptor).
    pub fn new(
        eq_table: B::Buffer<F>,
        g_table: B::Buffer<F>,
        kernel: B::CompiledKernel<F>,
        backend: Arc<B>,
    ) -> Self {
        let num_evals = Self::descriptor().num_evals();
        Self {
            inner: KernelEvaluator::with_unit_weights(
                vec![eq_table, g_table],
                kernel,
                num_evals,
                backend,
            ),
        }
    }

    /// Canonical kernel descriptor for `eq · g`.
    pub fn descriptor() -> KernelDescriptor {
        let b = ExprBuilder::new();
        let eq = b.opening(0);
        let g = b.opening(1);
        KernelDescriptor {
            shape: KernelShape::Custom {
                expr: b.build(eq * g),
                num_inputs: 2,
            },
            degree: 2,
            tensor_split: None,
        }
    }
}

impl<F: Field, B: ComputeBackend> SumcheckCompute<F> for EqProductEvaluator<F, B> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        self.inner.round_polynomial()
    }

    fn bind(&mut self, challenge: F) {
        self.inner.bind(challenge);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_compute::CpuBackend;
    use jolt_field::{Field, Fr};
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::{SumcheckClaim, SumcheckProver, SumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn cpu() -> Arc<CpuBackend> {
        Arc::new(CpuBackend)
    }

    fn compile_kernel() -> <CpuBackend as ComputeBackend>::CompiledKernel<Fr> {
        let desc = EqProductEvaluator::<Fr, CpuBackend>::descriptor();
        jolt_cpu_kernels::compile::<Fr>(&desc)
    }

    #[test]
    fn round_polynomial_consistency_degree_2() {
        let backend = cpu();
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let g_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let kernel = compile_kernel();
        let witness = EqProductEvaluator::new(
            backend.upload(&eq_table),
            backend.upload(&g_table),
            kernel,
            backend,
        );
        let poly = witness.round_polynomial();

        let s0: Fr = (0..n / 2).map(|j| eq_table[2 * j] * g_table[2 * j]).sum();
        let s1: Fr = (0..n / 2)
            .map(|j| eq_table[2 * j + 1] * g_table[2 * j + 1])
            .sum();

        assert_eq!(poly.evaluate(Fr::zero()), s0);
        assert_eq!(poly.evaluate(Fr::one()), s1);
    }

    #[test]
    fn full_prove_verify() {
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

        let kernel = compile_kernel();
        let mut witness = EqProductEvaluator::new(
            backend.upload(&eq_table),
            backend.upload(&g_table),
            kernel,
            backend,
        );

        let mut pt = Blake2bTranscript::new(b"eq_product_test");
        let proof = SumcheckProver::prove(
            &claim,
            &mut witness,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"eq_product_test");
        let result = SumcheckVerifier::verify(
            &claim,
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );
        assert!(result.is_ok(), "verification failed: {result:?}");
    }
}
