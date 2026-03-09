//! Hamming booleanity [`SumcheckCompute`] evaluator.
//!
//! Proves that a polynomial $h$ is Boolean-valued on the hypercube:
//!
//! $$\sum_{x \in \{0,1\}^n} \widetilde{eq}(r, x) \cdot h(x) \cdot (h(x) - 1) = 0$$
//!
//! where $r$ is a random evaluation point.
//!
//! Thin wrapper around [`KernelEvaluator`](super::kernel::KernelEvaluator)
//! with a degree-3 `Custom` kernel (`eq · h · (h − 1)`).

use std::sync::Arc;

use jolt_compute::ComputeBackend;
use jolt_field::Field;
use jolt_ir::{ExprBuilder, KernelDescriptor, KernelShape};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::prover::SumcheckCompute;

use super::kernel::KernelEvaluator;

/// Sumcheck evaluator for the Hamming booleanity check.
///
/// The per-variable polynomial has degree 3:
/// - degree 1 from $\widetilde{eq}$
/// - degree 2 from $h \cdot (h - 1)$
///
/// Delegates all computation to [`KernelEvaluator`] via a compiled
/// `eq · (h² − h)` kernel.
pub struct HammingBooleanityEvaluator<F: Field, B: ComputeBackend> {
    inner: KernelEvaluator<F, B>,
}

impl<F: Field, B: ComputeBackend> HammingBooleanityEvaluator<F, B> {
    /// Creates a new evaluator from pre-uploaded eq and h buffers.
    ///
    /// `kernel` must be compiled from [`descriptor()`](Self::descriptor).
    pub fn new(
        eq_table: B::Buffer<F>,
        h_table: B::Buffer<F>,
        kernel: B::CompiledKernel<F>,
        backend: Arc<B>,
    ) -> Self {
        let num_evals = Self::descriptor().num_evals();
        Self {
            inner: KernelEvaluator::with_unit_weights(
                vec![eq_table, h_table],
                kernel,
                num_evals,
                backend,
            ),
        }
    }

    /// Canonical kernel descriptor for `eq · h · (h − 1)`.
    pub fn descriptor() -> KernelDescriptor {
        let b = ExprBuilder::new();
        let eq = b.opening(0);
        let h = b.opening(1);
        KernelDescriptor {
            shape: KernelShape::Custom {
                expr: b.build(eq * (h * h - h)),
                num_inputs: 2,
            },
            degree: 3,
            tensor_split: None,
        }
    }
}

impl<F: Field, B: ComputeBackend> SumcheckCompute<F> for HammingBooleanityEvaluator<F, B> {
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
    use jolt_sumcheck::{SumcheckClaim, SumcheckProver};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn cpu() -> Arc<CpuBackend> {
        Arc::new(CpuBackend)
    }

    fn compile_kernel() -> <CpuBackend as ComputeBackend>::CompiledKernel<Fr> {
        let desc = HammingBooleanityEvaluator::<Fr, CpuBackend>::descriptor();
        jolt_cpu_kernels::compile::<Fr>(&desc)
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
    fn non_boolean_polynomial_has_nonzero_sum() {
        let num_vars = 2;
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();

        let h_table: Vec<Fr> = vec![0, 2, 1, 0].into_iter().map(Fr::from_u64).collect();

        let sum = brute_force_booleanity_sum(&h_table, &eq_table);
        assert!(
            !sum.is_zero(),
            "non-boolean polynomial should have nonzero sum"
        );
    }

    #[test]
    fn round_polynomial_consistency() {
        let backend = cpu();
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();

        let h_table: Vec<Fr> = vec![0, 1, 1, 0, 1, 0, 0, 1]
            .into_iter()
            .map(Fr::from_u64)
            .collect();

        let kernel = compile_kernel();
        let witness = HammingBooleanityEvaluator::new(
            backend.upload(&eq_table),
            backend.upload(&h_table),
            kernel,
            backend,
        );
        let poly = witness.round_polynomial();

        let one = Fr::one();

        let s0: Fr = (0..4)
            .map(|j| eq_table[2 * j] * h_table[2 * j] * (h_table[2 * j] - one))
            .sum();

        let s1: Fr = (0..4)
            .map(|j| eq_table[2 * j + 1] * h_table[2 * j + 1] * (h_table[2 * j + 1] - one))
            .sum();

        assert_eq!(poly.evaluate(Fr::zero()), s0);
        assert_eq!(poly.evaluate(Fr::one()), s1);

        let total = s0 + s1;
        assert!(total.is_zero());
    }

    #[test]
    fn full_sumcheck_proof_boolean_polynomial() {
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

        let kernel = compile_kernel();
        let mut witness = HammingBooleanityEvaluator::new(
            backend.upload(&eq_table),
            backend.upload(&h_table),
            kernel,
            backend,
        );

        let mut prover_transcript = Blake2bTranscript::new(b"test_hamming_booleanity");
        let proof = SumcheckProver::prove(
            &claim,
            &mut witness,
            &mut prover_transcript,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        assert_eq!(proof.round_polynomials.len(), num_vars);

        let mut verifier_transcript = Blake2bTranscript::new(b"test_hamming_booleanity");
        let result = jolt_sumcheck::SumcheckVerifier::verify(
            &claim,
            &proof,
            &mut verifier_transcript,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    #[test]
    fn full_sumcheck_proof_random_polynomial() {
        let backend = cpu();
        let num_vars = 3;
        let mut rng = ChaCha20Rng::seed_from_u64(456);
        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();

        let h_table: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::random(&mut rng)).collect();

        let claimed_sum = brute_force_booleanity_sum(&h_table, &eq_table);

        let claim = SumcheckClaim {
            num_vars,
            degree: 3,
            claimed_sum,
        };

        let kernel = compile_kernel();
        let mut witness = HammingBooleanityEvaluator::new(
            backend.upload(&eq_table),
            backend.upload(&h_table),
            kernel,
            backend,
        );

        let mut prover_transcript = Blake2bTranscript::new(b"test_random");
        let proof = SumcheckProver::prove(
            &claim,
            &mut witness,
            &mut prover_transcript,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut verifier_transcript = Blake2bTranscript::new(b"test_random");
        let result = jolt_sumcheck::SumcheckVerifier::verify(
            &claim,
            &proof,
            &mut verifier_transcript,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    #[test]
    fn bind_halves_table_size() {
        let backend = cpu();
        let num_vars = 3;
        let h_table = vec![Fr::from_u64(1); 8];
        let eq_table = vec![Fr::from_u64(1); 8];

        let kernel = compile_kernel();
        let mut witness = HammingBooleanityEvaluator::new(
            backend.upload(&eq_table),
            backend.upload(&h_table),
            kernel,
            Arc::clone(&backend),
        );

        assert_eq!(witness.inner.current_len(), 8);
        witness.bind(Fr::from_u64(5));
        assert_eq!(witness.inner.current_len(), 4);
        witness.bind(Fr::from_u64(3));
        assert_eq!(witness.inner.current_len(), 2);
        witness.bind(Fr::from_u64(7));
        assert_eq!(witness.inner.current_len(), 1);
    }
}
