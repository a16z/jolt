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
//! # Interpolation modes
//!
//! Two modes are supported:
//!
//! - **Standard grid** (`Custom`/`EqProduct`/`HammingBooleanity` kernels):
//!   eq is included as a regular input buffer, weights are unit. The kernel
//!   evaluates on `{0, 2, ..., degree}` (skipping `t=1`), and `P(1)` is
//!   derived as `claim - P(0)` before interpolation.
//!
//! - **Toom-Cook** (`ProductSum` kernels): eq is factored into a partial eq
//!   weight buffer and per-round linear factors. The kernel produces D
//!   evaluations on `{1, ..., D-1, ∞}`, and the round polynomial is
//!   reconstructed by recovering `h(0)` from the claim, interpolating via
//!   `from_evals_toom`, and multiplying by the leading eq factor.

use std::sync::Arc;

use jolt_compute::{BindingOrder, ComputeBackend};
use jolt_field::Field;
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::prover::SumcheckCompute;

/// `eq_single(w, x) = (1 - w)(1 - x) + w·x`.
#[inline]
fn eq_single<F: Field>(w: F, x: F) -> F {
    let one = F::one();
    (one - w) * (one - x) + w * x
}

/// Interpolation strategy for converting `pairwise_reduce` output to a
/// round polynomial.
enum InterpolationMode<F: Field> {
    /// Standard grid `{0, 2, ..., degree}` (skipping `t=1`): the kernel
    /// includes eq as an input. `P(1) = claim - P(0)` is derived before
    /// interpolation via `interpolate_over_integers`.
    StandardGrid {
        /// Running sumcheck claim, set by [`SumcheckCompute::set_claim`]
        /// before each round.
        claim: F,
    },

    /// Toom-Cook grid `{1, ..., D-1, ∞}`: eq is factored out. Output needs
    /// claim-based `h(0)` recovery, `from_evals_toom`, and eq multiplication.
    ToomCook(ToomCookState<F>),
}

/// Tracked state for Toom-Cook reconstruction across sumcheck rounds.
///
/// The round polynomial is `q_k(X) = outer_scalar · eq_single(w_k, X) · h_k(X)`,
/// where `h_k` is degree D (the product of D linear interpolants summed
/// over remaining hypercube positions). We recover `h_k(0)` from the running
/// claim and reconstruct h via `from_evals_toom`.
struct ToomCookState<F: Field> {
    /// Eq challenge point `[w_0, w_1, ..., w_{n-1}]`.
    eq_w: Vec<F>,
    /// Current round index (0-based).
    round: usize,
    /// Running sumcheck claim: `q_{k-1}(r_{k-1})` after round k-1.
    claim: F,
    /// Accumulated outer eq scalar: `Π_{i<round} eq_single(w_i, r_i)`.
    outer_scalar: F,
    /// Accumulated weight buffer scalar: `Π_{i=1}^{round} eq_single(w_i, r_{i-1})`.
    /// Arises because the weight buffer (partial eq) picks up one eq factor
    /// per interpolation round.
    weight_scalar: F,
}

/// Reconstructs the round polynomial from Toom-Cook grid evaluations.
///
/// Given D raw evaluations (weighted by `weight_scalar`) on `{1, ..., D-1, ∞}`,
/// recovers `h(0)` from the claim, interpolates h via `from_evals_toom`, and
/// multiplies by `outer_scalar · eq_single(w_k, X)`.
fn toom_cook_reconstruct<F: Field>(raw_evals: &[F], state: &ToomCookState<F>) -> UnivariatePoly<F> {
    let d = raw_evals.len();
    let w_k = state.eq_w[state.round];

    // Unscale raw evaluations by accumulated weight scalar to get h(grid).
    let inv_weight = state
        .weight_scalar
        .inverse()
        .expect("weight_scalar must be nonzero");
    let h_toom: Vec<F> = raw_evals.iter().map(|&e| e * inv_weight).collect();
    // h_toom = [h(1), h(2), ..., h(D-1), h(∞)]

    // Recover h(0) from the claim.
    // reduced_claim = claim / outer_scalar = (1-w_k)*h(0) + w_k*h(1)
    let eq_at_0 = F::one() - w_k;
    let inv_outer = state
        .outer_scalar
        .inverse()
        .expect("outer_scalar must be nonzero");
    let reduced_claim = state.claim * inv_outer;
    let h_at_1 = h_toom[0];
    let h_at_0 =
        (reduced_claim - w_k * h_at_1) * eq_at_0.inverse().expect("eq(w_k, 0) must be nonzero");

    // Full evaluations: [h(0), h(1), ..., h(D-1), h(∞)]
    let mut full_evals = Vec::with_capacity(d + 1);
    full_evals.push(h_at_0);
    full_evals.extend_from_slice(&h_toom);

    // Interpolate h polynomial from Toom-Cook grid.
    let h_coeffs = UnivariatePoly::from_evals_toom(&full_evals).into_coefficients();

    // Multiply by outer_scalar · eq_single(w_k, X).
    // eq_single(w_k, X) = (1-w_k) + (2w_k - 1)·X
    let scale = state.outer_scalar;
    let constant_coeff = scale * (F::one() - w_k);
    let x_coeff = scale * (w_k + w_k - F::one());

    let mut coeffs = vec![F::zero(); h_coeffs.len() + 1];
    for (i, c) in h_coeffs.into_iter().enumerate() {
        coeffs[i] += c * constant_coeff;
        coeffs[i + 1] += c * x_coeff;
    }

    UnivariatePoly::new(coeffs)
}

/// Backend-generic sumcheck evaluator using a compiled kernel.
///
/// Stores polynomial evaluation tables as backend buffers and delegates
/// `round_polynomial()` to [`pairwise_reduce`](ComputeBackend::pairwise_reduce)
/// and `bind()` to [`interpolate_pairs_batch`](ComputeBackend::interpolate_pairs_batch).
///
/// After monomorphization over [`CpuBackend`](jolt_cpu::CpuBackend), this
/// compiles to the same code as a hand-written evaluator — the `ComputeBackend`
/// trait calls become direct function calls with no indirection.
pub struct KernelEvaluator<F: Field, B: ComputeBackend> {
    /// Input buffers (interleaved lo/hi pairs).
    ///
    /// For standard grid: `inputs[0]` is eq, `inputs[1..]` are payload polys.
    /// For Toom-Cook: all inputs are payload polys (no eq).
    inputs: Vec<B::Buffer<F>>,

    /// Weight buffer: one scalar per pair position (`len = buffer_len / 2`).
    ///
    /// `None` for standard-grid mode (unit weights) — the backend dispatches
    /// an unweighted kernel that skips the per-element weight multiply.
    /// `Some` for Toom-Cook mode: partial eq table `eq(w[1..], ·)`,
    /// interpolated alongside inputs on each bind.
    weights: Option<B::Buffer<F>>,

    /// Compiled composition kernel.
    kernel: B::CompiledKernel<F>,

    /// Number of evaluations produced per pair position.
    ///
    /// For standard-grid kernels: `degree` (grid `{0, 2, ..., degree}`,
    /// skipping `t=1`).
    /// For `ProductSum` kernels: `D` (Toom-Cook grid `{1, ..., D-1, ∞}`).
    num_evals: usize,

    backend: Arc<B>,

    /// Controls how `pairwise_reduce` output is converted to a round polynomial.
    mode: InterpolationMode<F>,

    /// Variable binding order for pairwise operations.
    ///
    /// - `LowToHigh`: interleaved pairs `(buf[2i], buf[2i+1])`. Default for
    ///   most sumcheck instances.
    /// - `HighToLow`: split-half pairs `(buf[i], buf[i+n/2])`. Used by Spartan
    ///   outer sumcheck and address-phase ReadRaf stages.
    binding_order: BindingOrder,

    /// Precomputed first-round polynomial for univariate skip optimization.
    ///
    /// When set, [`SumcheckCompute::first_round_polynomial`] returns this
    /// instead of `None`, and the sumcheck prover uses it for round 0.
    /// Consumed on first access (set to `None` after returning).
    first_round_override: Option<UnivariatePoly<F>>,
}

impl<F: Field, B: ComputeBackend> KernelEvaluator<F, B> {
    /// Creates a kernel evaluator with standard-grid interpolation.
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
        Self::new_with_mode(
            inputs,
            Some(weights),
            kernel,
            num_evals,
            backend,
            InterpolationMode::StandardGrid { claim: F::zero() },
        )
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
        Self::new_with_mode(
            inputs,
            None,
            kernel,
            num_evals,
            backend,
            InterpolationMode::StandardGrid { claim: F::zero() },
        )
    }

    /// Creates a kernel evaluator with Toom-Cook eq factoring.
    ///
    /// The eq polynomial is split: `eq(w, x) = eq_single(w_0, x_0) · eq(w[1..], x[1..])`.
    /// The partial eq `eq(w[1..], ·)` becomes the weight buffer, and the
    /// per-round `eq_single(w_k, X)` factor is applied during reconstruction.
    ///
    /// # Arguments
    ///
    /// * `inputs` — Polynomial buffers (no eq). For a sum of P products of D
    ///   linear interpolants, provide D·P buffers.
    /// * `kernel` — Compiled `ProductSum` kernel producing D evaluations.
    /// * `num_evals` — D (the number of Toom-Cook grid evaluations).
    /// * `eq_w` — Full eq challenge point `[w_0, ..., w_{n-1}]`.
    /// * `claimed_sum` — Initial sumcheck claim `Σ_x eq(w, x) · f(x)`.
    /// * `backend` — Handle to the compute backend.
    ///
    /// # Panics
    ///
    /// Panics if `eq_w` is empty or inputs have mismatched lengths.
    pub fn with_toom_cook_eq(
        inputs: Vec<B::Buffer<F>>,
        kernel: B::CompiledKernel<F>,
        num_evals: usize,
        eq_w: Vec<F>,
        claimed_sum: F,
        backend: Arc<B>,
    ) -> Self {
        assert!(!eq_w.is_empty(), "eq_w must have at least one element");

        let partial_eq = EqPolynomial::new(eq_w[1..].to_vec()).evaluations();
        let weights = backend.upload(&partial_eq);

        let state = ToomCookState {
            eq_w,
            round: 0,
            claim: claimed_sum,
            outer_scalar: F::one(),
            weight_scalar: F::one(),
        };

        Self::new_with_mode(
            inputs,
            Some(weights),
            kernel,
            num_evals,
            backend,
            InterpolationMode::ToomCook(state),
        )
    }

    fn new_with_mode(
        inputs: Vec<B::Buffer<F>>,
        weights: Option<B::Buffer<F>>,
        kernel: B::CompiledKernel<F>,
        num_evals: usize,
        backend: Arc<B>,
        mode: InterpolationMode<F>,
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
        if let Some(ref w) = weights {
            assert_eq!(
                backend.len(w),
                n / 2,
                "weights length {} != expected {}",
                backend.len(w),
                n / 2
            );
        }
        Self {
            inputs,
            weights,
            kernel,
            num_evals,
            backend,
            mode,
            binding_order: BindingOrder::LowToHigh,
            first_round_override: None,
        }
    }

    /// Sets a precomputed first-round polynomial for univariate skip.
    ///
    /// The caller computes $t_1(2)$ using formula-specific logic (e.g., by
    /// iterating over the raw evaluation tables before upload), then calls
    /// [`uniskip_round_poly`](jolt_spartan::uniskip_round_poly) to build the
    /// analytical degree-3 polynomial, and passes it here.
    ///
    /// The polynomial is consumed on the first call to
    /// [`first_round_polynomial`](SumcheckCompute::first_round_polynomial).
    pub fn set_first_round_override(&mut self, poly: UnivariatePoly<F>) {
        self.first_round_override = Some(poly);
    }

    /// Sets the variable binding order for pairwise operations.
    ///
    /// Defaults to `LowToHigh`. Set to `HighToLow` for address-phase
    /// ReadRaf stages or Spartan-style sumchecks that bind the MSB first.
    pub fn with_binding_order(mut self, order: BindingOrder) -> Self {
        self.binding_order = order;
        self
    }

    /// Replaces the compiled kernel with one using updated parameters.
    ///
    /// Used for intra-segment re-parameterization (e.g., recompiling with
    /// updated checkpoint challenge values every 2 rounds). On CPU this is
    /// ~free (new closure capture). On GPU, use push constants to avoid
    /// shader recompilation.
    pub fn update_kernel(&mut self, kernel: B::CompiledKernel<F>) {
        self.kernel = kernel;
    }

    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Active element count of the first input buffer. Halves each bind round.
    pub fn current_len(&self) -> usize {
        self.backend.len(&self.inputs[0])
    }

    /// Returns a reference to the compute backend.
    pub fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    /// Runs `pairwise_reduce` and returns the raw evaluation vector.
    ///
    /// When weights are `None` (unit weights / StandardGrid mode), dispatches
    /// the unweighted kernel variant which skips the per-element weight multiply.
    fn reduce_raw(&self) -> Vec<F> {
        let input_refs: Vec<&B::Buffer<F>> = self.inputs.iter().collect();
        match &self.weights {
            Some(w) => self.backend.pairwise_reduce(
                &input_refs,
                w,
                &self.kernel,
                self.num_evals,
                self.binding_order,
            ),
            None => self.backend.pairwise_reduce_unweighted(
                &input_refs,
                &self.kernel,
                self.num_evals,
                self.binding_order,
            ),
        }
    }
}

impl<F: Field, B: ComputeBackend> SumcheckCompute<F> for KernelEvaluator<F, B> {
    fn set_claim(&mut self, claim: F) {
        match &mut self.mode {
            InterpolationMode::StandardGrid { claim: stored } => *stored = claim,
            InterpolationMode::ToomCook(state) => state.claim = claim,
        }
    }

    fn first_round_polynomial(&self) -> Option<UnivariatePoly<F>> {
        self.first_round_override.clone()
    }

    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let raw_evals = self.reduce_raw();
        match &self.mode {
            InterpolationMode::StandardGrid { claim } => {
                // raw_evals = [P(0), P(2), P(3), ..., P(d)]
                // Derive P(1) = claim - P(0)
                let p0 = raw_evals[0];
                let p1 = *claim - p0;
                let mut full_evals = Vec::with_capacity(raw_evals.len() + 1);
                full_evals.push(p0);
                full_evals.push(p1);
                full_evals.extend_from_slice(&raw_evals[1..]);
                UnivariatePoly::interpolate_over_integers(&full_evals)
            }
            InterpolationMode::ToomCook(state) => toom_cook_reconstruct(&raw_evals, state),
        }
    }

    fn bind(&mut self, c: F) {
        // Update Toom-Cook state (claim is now set externally via set_claim).
        if let InterpolationMode::ToomCook(state) = &mut self.mode {
            let w_k = state.eq_w[state.round];
            state.outer_scalar *= eq_single(w_k, c);
            if state.round + 1 < state.eq_w.len() {
                state.weight_scalar *= eq_single(state.eq_w[state.round + 1], c);
            }
            state.round += 1;
        }

        match self.binding_order {
            BindingOrder::LowToHigh => {
                if let Some(weights) = self.weights.take() {
                    // Weighted mode (Toom-Cook): interpolate inputs + weights together.
                    let mut all_bufs = std::mem::take(&mut self.inputs);
                    all_bufs.push(weights);

                    let mut bound = self.backend.interpolate_pairs_batch(all_bufs, c);
                    self.weights = Some(
                        bound
                            .pop()
                            .expect("interpolate_pairs_batch returned fewer buffers than given"),
                    );
                    self.inputs = bound;
                } else {
                    // Unit-weights mode (StandardGrid): skip weights entirely.
                    let inputs = std::mem::take(&mut self.inputs);
                    self.inputs = self.backend.interpolate_pairs_batch(inputs, c);
                }
            }
            BindingOrder::HighToLow => {
                self.backend.interpolate_pairs_batch_inplace(
                    &mut self.inputs,
                    c,
                    BindingOrder::HighToLow,
                );
                if let Some(ref mut w) = self.weights {
                    self.backend
                        .interpolate_pairs_inplace(w, c, BindingOrder::HighToLow);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluators::catalog;
    use jolt_cpu::CpuBackend;
    use jolt_field::{Field, Fr};
    use jolt_ir::{ExprBuilder, KernelDescriptor, KernelShape};
    use jolt_sumcheck::{SumcheckClaim, SumcheckProver, SumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::{One, Zero};
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
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&g_table)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let mut pt = Blake2bTranscript::new(b"kw_eq_product");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"kw_eq_product");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
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
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&h_table)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let mut pt = Blake2bTranscript::new(b"kw_hamming");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"kw_hamming");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
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
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&h_table)];
        let mut witness =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), backend);

        let mut pt = Blake2bTranscript::new(b"kw_hamming_random");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"kw_hamming_random");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
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
        let kernel = jolt_cpu::compile_with_challenges::<Fr>(&desc, &[c0, c1]);

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
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"kw_formula");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// Verify that unit-weights mode keeps weights as None through bind rounds.
    #[test]
    fn unit_weights_none_through_binds() {
        let backend = cpu();
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(333);

        let eq_table = vec![Fr::one(); n];
        let g_table: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let desc = catalog::eq_product();
        let kernel = jolt_cpu::compile::<Fr>(&desc);
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

            assert!(
                kw.weights.is_none(),
                "unit-weights mode should keep weights as None after bind"
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
        let kernel = jolt_cpu::compile::<Fr>(&desc);
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

    /// Toom-Cook mode: single product group (D=4, P=1).
    #[test]
    fn toom_cook_single_product_prove_verify() {
        let backend = cpu();
        let num_vars = 5;
        let n = 1usize << num_vars;
        let d = 4;
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let eq_w: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let polys: Vec<Vec<Fr>> = (0..d)
            .map(|_| (0..n).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let eq_table = EqPolynomial::new(eq_w.clone()).evaluations();
        let claimed_sum: Fr = (0..n)
            .map(|x| {
                let mut product = Fr::one();
                for poly in &polys {
                    product *= poly[x];
                }
                eq_table[x] * product
            })
            .sum();

        let desc = catalog::product_sum(d, 1);
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs: Vec<_> = polys.iter().map(|p| backend.upload(p)).collect();
        let mut witness = KernelEvaluator::with_toom_cook_eq(
            inputs,
            kernel,
            desc.num_evals(),
            eq_w,
            claimed_sum,
            Arc::clone(&backend),
        );

        let claim = SumcheckClaim {
            num_vars,
            degree: d + 1,
            claimed_sum,
        };

        let mut pt = Blake2bTranscript::new(b"tc_single");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"tc_single");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// Toom-Cook mode: multiple product groups with gamma pre-scaling (D=4, P=3).
    #[test]
    fn toom_cook_multi_product_gamma_prove_verify() {
        let backend = cpu();
        let num_vars = 4;
        let n = 1usize << num_vars;
        let d = 4;
        let n_virtual = 3;
        let total = d * n_virtual;
        let mut rng = ChaCha20Rng::seed_from_u64(99);

        let eq_w: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let polys: Vec<Vec<Fr>> = (0..total)
            .map(|_| (0..n).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let gamma = Fr::random(&mut rng);
        let gamma_powers: Vec<Fr> = {
            let mut g = Fr::one();
            (0..n_virtual)
                .map(|_| {
                    let val = g;
                    g *= gamma;
                    val
                })
                .collect()
        };

        // Brute-force sum: Σ_x eq(w, x) · Σ_t γ^t · Π_k p_{t*d+k}(x)
        let eq_table = EqPolynomial::new(eq_w.clone()).evaluations();
        let claimed_sum: Fr = (0..n)
            .map(|x| {
                let mut formula = Fr::zero();
                for t in 0..n_virtual {
                    let mut product = gamma_powers[t];
                    for k in 0..d {
                        product *= polys[t * d + k][x];
                    }
                    formula += product;
                }
                eq_table[x] * formula
            })
            .sum();

        // Pre-scale first poly of each group by γ^t
        let mut scaled_polys = polys;
        for t in 0..n_virtual {
            for j in 0..n {
                scaled_polys[t * d][j] *= gamma_powers[t];
            }
        }

        let desc = catalog::product_sum(d, n_virtual);
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs: Vec<_> = scaled_polys.iter().map(|p| backend.upload(p)).collect();
        let mut witness = KernelEvaluator::with_toom_cook_eq(
            inputs,
            kernel,
            desc.num_evals(),
            eq_w,
            claimed_sum,
            Arc::clone(&backend),
        );

        let claim = SumcheckClaim {
            num_vars,
            degree: d + 1,
            claimed_sum,
        };

        let mut pt = Blake2bTranscript::new(b"tc_multi_gamma");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"tc_multi_gamma");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// Toom-Cook mode with D=8 to test a larger Toom-Cook grid.
    #[test]
    fn toom_cook_d8_prove_verify() {
        let backend = cpu();
        let num_vars = 3;
        let n = 1usize << num_vars;
        let d = 8;
        let mut rng = ChaCha20Rng::seed_from_u64(200);

        let eq_w: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let polys: Vec<Vec<Fr>> = (0..d)
            .map(|_| (0..n).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let eq_table = EqPolynomial::new(eq_w.clone()).evaluations();
        let claimed_sum: Fr = (0..n)
            .map(|x| {
                let mut product = Fr::one();
                for poly in &polys {
                    product *= poly[x];
                }
                eq_table[x] * product
            })
            .sum();

        let desc = catalog::product_sum(d, 1);
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs: Vec<_> = polys.iter().map(|p| backend.upload(p)).collect();
        let mut witness = KernelEvaluator::with_toom_cook_eq(
            inputs,
            kernel,
            desc.num_evals(),
            eq_w,
            claimed_sum,
            Arc::clone(&backend),
        );

        let claim = SumcheckClaim {
            num_vars,
            degree: d + 1,
            claimed_sum,
        };

        let mut pt = Blake2bTranscript::new(b"tc_d8");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"tc_d8");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }
}
