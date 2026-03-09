//! Backend-generic [`SumcheckCompute`] via compiled kernels.
//!
//! [`KernelEvaluator`] implements the sumcheck round-polynomial evaluation by
//! delegating all heavy computation to a [`ComputeBackend`]. The composition
//! formula lives in a [`CompiledKernel`](jolt_compute::ComputeBackend::CompiledKernel),
//! compiled from a [`KernelDescriptor`](jolt_ir::KernelDescriptor) at setup
//! time.
//!
//! This is the universal `SumcheckCompute` implementation that works on any
//! backend (CPU, GPU). Composition formulas come from [`catalog`](super::catalog)
//! descriptors.
//!
//! # Data layout
//!
//! All buffers use interleaved lo/hi layout: for a multilinear polynomial
//! with $2^n$ evaluations, position $2j$ holds the evaluation at the hypercube
//! point with leading bit 0, and $2j+1$ the evaluation with leading bit 1.
//! This is the natural evaluation table order.
//!
//! # Eq polynomial
//!
//! The eq polynomial is included as a regular input buffer to the kernel
//! (typically the first opening). The `weights` buffer holds unit scalars
//! (all ones) — it exists for future split-eq optimization where the outer
//! eq factor becomes the weight.

use std::sync::Arc;

use jolt_compute::{BindingOrder, ComputeBackend};
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::prover::SumcheckCompute;

/// Backend-generic sumcheck evaluator using a compiled kernel.
///
/// Stores polynomial evaluation tables as backend buffers and delegates
/// `round_polynomial()` to [`pairwise_reduce`](ComputeBackend::pairwise_reduce)
/// and `bind()` to [`interpolate_pairs_batch`](ComputeBackend::interpolate_pairs_batch).
///
/// After monomorphization over [`CpuBackend`](jolt_compute::CpuBackend), this
/// compiles to the same code as a hand-written evaluator — the `ComputeBackend`
/// trait calls become direct function calls with no indirection.
pub struct KernelEvaluator<F: Field, B: ComputeBackend> {
    /// Input buffers (interleaved lo/hi pairs).
    ///
    /// The ordering matches the kernel's `Opening(i)` indices. Typically
    /// `inputs[0]` is the eq polynomial and `inputs[1..]` are the payload
    /// polynomials.
    inputs: Vec<B::Buffer<F>>,

    /// Weight buffer: one scalar per pair position (`len = buffer_len / 2`).
    ///
    /// For non-split-eq mode this is all ones. Interpolated alongside inputs
    /// on each `bind()` so it shrinks correctly.
    weights: B::Buffer<F>,

    /// Compiled composition kernel.
    kernel: B::CompiledKernel<F>,

    /// Number of evaluations produced per pair position.
    ///
    /// For `Custom` kernels: `degree + 1` (standard grid `{0, 1, ..., degree}`).
    /// For `ProductSum` kernels: `D` (Toom-Cook grid `{1, ..., D-1, ∞}`).
    num_evals: usize,

    backend: Arc<B>,
}

impl<F: Field, B: ComputeBackend> KernelEvaluator<F, B> {
    /// Creates a kernel evaluator from pre-uploaded buffers.
    ///
    /// # Arguments
    ///
    /// * `inputs` — Polynomial evaluation tables as backend buffers. All must
    ///   have the same even length $2^n$.
    /// * `weights` — Weight buffer with length $2^{n-1}$ (one per pair position).
    /// * `kernel` — Compiled kernel matching the composition formula.
    /// * `num_evals` — Number of grid-point evaluations the kernel produces.
    /// * `backend` — Handle to the compute backend.
    ///
    /// # Panics
    ///
    /// Panics if inputs are empty or have mismatched lengths.
    pub fn new(
        inputs: Vec<B::Buffer<F>>,
        weights: B::Buffer<F>,
        kernel: B::CompiledKernel<F>,
        num_evals: usize,
        backend: Arc<B>,
    ) -> Self {
        assert!(
            !inputs.is_empty(),
            "KernelEvaluator requires at least one input"
        );
        let n = backend.len(&inputs[0]);
        assert!(
            n > 0 && n % 2 == 0,
            "input buffer length must be positive and even"
        );
        for (i, buf) in inputs.iter().enumerate().skip(1) {
            assert_eq!(
                backend.len(buf),
                n,
                "input[{i}] length {} != input[0] length {n}",
                backend.len(buf)
            );
        }
        assert_eq!(
            backend.len(&weights),
            n / 2,
            "weights length {} != expected {}",
            backend.len(&weights),
            n / 2
        );
        Self {
            inputs,
            weights,
            kernel,
            num_evals,
            backend,
        }
    }

    /// Creates a kernel evaluator with unit weights (all ones).
    ///
    /// Convenience constructor that allocates a weight buffer filled with
    /// `F::one()`. Use this for non-split-eq sumcheck instances where the
    /// eq polynomial is included as a regular input buffer.
    pub fn with_unit_weights(
        inputs: Vec<B::Buffer<F>>,
        kernel: B::CompiledKernel<F>,
        num_evals: usize,
        backend: Arc<B>,
    ) -> Self {
        assert!(
            !inputs.is_empty(),
            "KernelEvaluator requires at least one input"
        );
        let n = backend.len(&inputs[0]);
        let half = n / 2;
        let ones = vec![F::one(); half];
        let weights = backend.upload(&ones);
        Self::new(inputs, weights, kernel, num_evals, backend)
    }

    /// Number of input buffers.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Current buffer length (halves each round).
    pub fn current_len(&self) -> usize {
        self.backend.len(&self.inputs[0])
    }
}

impl<F: Field, B: ComputeBackend> SumcheckCompute<F> for KernelEvaluator<F, B> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let input_refs: Vec<&B::Buffer<F>> = self.inputs.iter().collect();
        let evals = self.backend.pairwise_reduce(
            &input_refs,
            &self.weights,
            &self.kernel,
            self.num_evals,
            BindingOrder::LowToHigh,
        );
        UnivariatePoly::interpolate_over_integers(&evals)
    }

    fn bind(&mut self, challenge: F) {
        // Batch-interpolate all input buffers and the weight buffer together
        // for maximum parallelism.
        let mut all_bufs = std::mem::take(&mut self.inputs);
        let empty_weights = self.backend.alloc(0);
        all_bufs.push(std::mem::replace(&mut self.weights, empty_weights));

        let mut bound = self.backend.interpolate_pairs_batch(all_bufs, challenge);
        self.weights = bound
            .pop()
            .expect("interpolate_pairs_batch returned fewer buffers than given");
        self.inputs = bound;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluators::catalog;
    use jolt_compute::CpuBackend;
    use jolt_field::{Field, Fr};
    use jolt_ir::{ExprBuilder, KernelDescriptor, KernelShape};
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::{SumcheckClaim, SumcheckProver, SumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::One;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn cpu() -> Arc<CpuBackend> {
        Arc::new(CpuBackend)
    }

    /// Full sumcheck prove/verify using KernelEvaluator for eq · g.
    #[test]
    fn kernel_witness_eq_product_prove_verify() {
        let backend = cpu();
        let num_vars = 5;
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

        let desc = catalog::eq_product();
        let kernel = jolt_cpu_kernels::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&g_table)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let mut pt = Blake2bTranscript::new(b"kw_eq_product");
        let proof = SumcheckProver::prove(
            &claim,
            &mut witness,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"kw_eq_product");
        let result = SumcheckVerifier::verify(
            &claim,
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// Full sumcheck prove/verify using KernelEvaluator for hamming booleanity.
    #[test]
    fn kernel_witness_hamming_prove_verify() {
        let backend = cpu();
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(123);

        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        // Boolean-valued h: the claimed sum should be zero
        let h_table: Vec<Fr> = (0..n).map(|i| Fr::from_u64(i as u64 % 2)).collect();

        let claimed_sum: Fr = eq_table
            .iter()
            .zip(h_table.iter())
            .map(|(&e, &h)| e * h * (h - Fr::one()))
            .sum();

        let claim = SumcheckClaim {
            num_vars,
            degree: 3,
            claimed_sum,
        };

        let desc = catalog::hamming_booleanity();
        let kernel = jolt_cpu_kernels::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&h_table)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let mut pt = Blake2bTranscript::new(b"kw_hamming");
        let proof = SumcheckProver::prove(
            &claim,
            &mut witness,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"kw_hamming");
        let result = SumcheckVerifier::verify(
            &claim,
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// Full sumcheck prove/verify with a non-boolean hamming polynomial
    /// (nonzero claimed sum).
    #[test]
    fn kernel_witness_hamming_random_prove_verify() {
        let backend = cpu();
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(456);

        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let h_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let claimed_sum: Fr = eq_table
            .iter()
            .zip(h_table.iter())
            .map(|(&e, &h)| e * h * (h - Fr::one()))
            .sum();

        let claim = SumcheckClaim {
            num_vars,
            degree: 3,
            claimed_sum,
        };

        let desc = catalog::hamming_booleanity();
        let kernel = jolt_cpu_kernels::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&h_table)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let mut pt = Blake2bTranscript::new(b"kw_hamming_random");
        let proof = SumcheckProver::prove(
            &claim,
            &mut witness,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"kw_hamming_random");
        let result = SumcheckVerifier::verify(
            &claim,
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// Custom 3-input formula: eq · (c0·a·b + c1·a·c).
    #[test]
    fn kernel_witness_custom_formula_prove_verify() {
        let backend = cpu();
        let num_vars = 3;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(789);

        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let eq_table = EqPolynomial::new(r).evaluations();
        let a: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let b_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let c_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let c0 = Fr::from_u64(3);
        let c1 = Fr::from_u64(7);

        // Brute-force sum: Σ eq[i] · (c0·a[i]·b[i] + c1·a[i]·c[i])
        let claimed_sum: Fr = (0..n)
            .map(|i| eq_table[i] * (c0 * a[i] * b_table[i] + c1 * a[i] * c_table[i]))
            .sum();

        // Kernel: eq · (c0·a·b + c1·a·c)
        // = eq · a · (c0·b + c1·c), degree 3 (eq × a × one-of-{b,c})
        // But we write it as the sum-of-products directly:
        let eb = ExprBuilder::new();
        let eq_v = eb.opening(0);
        let a_v = eb.opening(1);
        let b_v = eb.opening(2);
        let c_v = eb.opening(3);
        let c0_v = eb.challenge(0);
        let c1_v = eb.challenge(1);
        let expr = eb.build(eq_v * (c0_v * a_v * b_v + c1_v * a_v * c_v));

        let desc = KernelDescriptor {
            shape: KernelShape::Custom {
                expr,
                num_inputs: 4,
            },
            degree: 3,
            tensor_split: None,
        };
        let kernel = jolt_cpu_kernels::compile_with_challenges::<Fr>(&desc, &[c0, c1]);

        let inputs = vec![
            backend.upload(&eq_table),
            backend.upload(&a),
            backend.upload(&b_table),
            backend.upload(&c_table),
        ];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let claim = SumcheckClaim {
            num_vars,
            degree: 3,
            claimed_sum,
        };

        let mut pt = Blake2bTranscript::new(b"kw_formula");
        let proof = SumcheckProver::prove(
            &claim,
            &mut witness,
            &mut pt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );

        let mut vt = Blake2bTranscript::new(b"kw_formula");
        let result = SumcheckVerifier::verify(
            &claim,
            &proof,
            &mut vt,
            |c: <Blake2bTranscript as Transcript>::Challenge| c.into(),
        );
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// Verify that unit weights stay all-ones through multiple bind rounds.
    #[test]
    fn unit_weights_preserved_through_binds() {
        let backend = cpu();
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(333);

        let eq_table = vec![Fr::one(); n];
        let g_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let desc = catalog::eq_product();
        let kernel = jolt_cpu_kernels::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&g_table)];
        let mut kw = KernelEvaluator::with_unit_weights(
            inputs,
            kernel,
            desc.num_evals(),
            Arc::clone(&backend),
        );

        for _ in 0..num_vars - 1 {
            let challenge = Fr::random(&mut rng);
            kw.bind(challenge);

            // Weights should still be all ones
            let weights = backend.download(&kw.weights);
            assert!(
                weights.iter().all(|&w| w == Fr::one()),
                "weights deviated from 1.0 after bind"
            );
        }
    }

    /// Verify that bind halves the buffer length each round.
    #[test]
    fn bind_halves_buffers() {
        let backend = cpu();
        let num_vars = 5;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(555);

        let eq_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let g_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let desc = catalog::eq_product();
        let kernel = jolt_cpu_kernels::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&g_table)];
        let mut kw = KernelEvaluator::with_unit_weights(
            inputs,
            kernel,
            desc.num_evals(),
            Arc::clone(&backend),
        );

        assert_eq!(kw.current_len(), n);
        for round in 1..=num_vars {
            kw.bind(Fr::random(&mut rng));
            assert_eq!(kw.current_len(), n >> round);
        }
    }
}
