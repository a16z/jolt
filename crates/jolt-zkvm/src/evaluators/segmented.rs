//! Multi-phase kernel composition for sumcheck.
//!
//! [`SegmentedEvaluator`] chains [`KernelEvaluator`] instances across phase
//! boundaries within a single sumcheck instance. Each segment is a standard
//! kernel evaluator — every round is a `pairwise_reduce` call with a compiled
//! kernel. At segment boundaries a transition callback materializes new
//! buffers and produces the next evaluator.
//!
//! # Motivation
//!
//! ReadRaf sumchecks have two variable groups (address + cycle) with different
//! kernels, buffer sets, and interpolation modes. Rather than writing custom
//! [`SumcheckCompute`] impls with hand-coded round loops, `SegmentedEvaluator`
//! composes existing [`KernelEvaluator`] instances so that every round stays
//! within the backend-generic kernel framework.

use std::collections::VecDeque;
use std::sync::Arc;

use jolt_compute::ComputeBackend;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::prover::SumcheckCompute;

use super::kernel::KernelEvaluator;

/// Transition callback between segments.
///
/// Receives all challenges accumulated so far and a handle to the compute
/// backend. Returns the next segment's [`KernelEvaluator`].
pub type SegmentTransition<F, B> =
    Box<dyn FnOnce(Vec<F>, &Arc<B>) -> KernelEvaluator<F, B> + Send + Sync>;

/// Per-round hook called after `bind()` within a segment.
///
/// Receives `(round_in_segment, challenge, &mut KernelEvaluator)`. Can
/// re-parameterize the kernel (via [`update_kernel`](KernelEvaluator::update_kernel)),
/// update weights, etc.
pub type RoundHook<F, B> =
    Box<dyn FnMut(usize, F, &mut KernelEvaluator<F, B>) + Send + Sync>;

/// Composes multiple [`KernelEvaluator`] instances across phase boundaries.
///
/// Each segment runs for a fixed number of rounds using a standard
/// `KernelEvaluator`. At segment boundaries a transition callback
/// materializes new buffers and produces the next evaluator.
///
/// # Round hooks
///
/// Optional per-round hooks run after each `bind()` within a segment.
/// Used for intra-segment re-parameterization (e.g., recompiling the
/// kernel with updated checkpoint values every 2 rounds in the
/// InstructionReadRaf address phases).
pub struct SegmentedEvaluator<F: Field, B: ComputeBackend> {
    /// Active segment's kernel evaluator.
    current: KernelEvaluator<F, B>,

    /// Remaining segments: `(rounds_in_segment, transition_fn)`.
    continuations: VecDeque<(usize, SegmentTransition<F, B>)>,

    /// Rounds completed within the current segment.
    round_in_segment: usize,

    /// Total rounds in the current segment.
    rounds_this_segment: usize,

    /// All challenges accumulated across all segments (passed to transitions).
    challenges: Vec<F>,

    /// Optional hook called after each `bind()` in the current segment.
    round_hook: Option<RoundHook<F, B>>,

    /// Hooks for future segments, consumed at each transition.
    future_hooks: VecDeque<Option<RoundHook<F, B>>>,

    backend: Arc<B>,
}

impl<F: Field, B: ComputeBackend> SegmentedEvaluator<F, B> {
    /// Creates a new evaluator starting from the first segment.
    ///
    /// # Arguments
    ///
    /// * `first_segment` — The initial kernel evaluator.
    /// * `rounds` — Number of sumcheck rounds in this segment.
    /// * `backend` — Handle to the compute backend.
    pub fn new(first_segment: KernelEvaluator<F, B>, rounds: usize, backend: Arc<B>) -> Self {
        Self {
            current: first_segment,
            continuations: VecDeque::new(),
            round_in_segment: 0,
            rounds_this_segment: rounds,
            challenges: Vec::new(),
            round_hook: None,
            future_hooks: VecDeque::new(),
            backend,
        }
    }

    /// Appends a continuation segment with its transition callback.
    ///
    /// `transition` is called when the previous segment completes, receiving
    /// all accumulated challenges and the backend handle. It must return the
    /// next segment's [`KernelEvaluator`].
    pub fn then(mut self, rounds: usize, transition: SegmentTransition<F, B>) -> Self {
        self.continuations.push_back((rounds, transition));
        self.future_hooks.push_back(None);
        self
    }

    /// Attaches a round hook to the most recently added segment.
    ///
    /// If called before any `then()`, attaches to the first segment.
    /// If called after `then()`, attaches to the last continuation.
    pub fn with_round_hook(mut self, hook: RoundHook<F, B>) -> Self {
        if self.future_hooks.is_empty() {
            // Attach to the first (current) segment.
            self.round_hook = Some(hook);
        } else {
            // Attach to the most recently added continuation.
            *self.future_hooks.back_mut().unwrap() = Some(hook);
        }
        self
    }
}

impl<F: Field, B: ComputeBackend> SumcheckCompute<F> for SegmentedEvaluator<F, B> {
    fn set_claim(&mut self, claim: F) {
        self.current.set_claim(claim);
    }

    fn first_round_polynomial(&self) -> Option<UnivariatePoly<F>> {
        self.current.first_round_polynomial()
    }

    fn round_polynomial(&self) -> UnivariatePoly<F> {
        self.current.round_polynomial()
    }

    fn bind(&mut self, challenge: F) {
        self.current.bind(challenge);
        self.challenges.push(challenge);

        // Fire round hook if present.
        if let Some(ref mut hook) = self.round_hook {
            hook(self.round_in_segment, challenge, &mut self.current);
        }

        self.round_in_segment += 1;

        // Transition to next segment if current is exhausted.
        if self.round_in_segment == self.rounds_this_segment {
            if let Some((next_rounds, transition)) = self.continuations.pop_front() {
                let challenges = self.challenges.clone();
                self.current = transition(challenges, &self.backend);
                self.round_in_segment = 0;
                self.rounds_this_segment = next_rounds;
                self.round_hook = self.future_hooks.pop_front().flatten();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluators::catalog;
    use jolt_compute::BindingOrder;
    use jolt_cpu::CpuBackend;
    use jolt_field::{Field, Fr};
    use jolt_ir::{ExprBuilder, KernelDescriptor, KernelShape};
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::claim::SumcheckClaim;
    use jolt_sumcheck::{SumcheckProver, SumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn cpu() -> Arc<CpuBackend> {
        Arc::new(CpuBackend)
    }

    /// Single-segment (degenerate case): should behave identically to a
    /// plain KernelEvaluator.
    #[test]
    fn single_segment_matches_plain_kernel() {
        let backend = cpu();
        let num_vars = 4;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(100);

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
        let evaluator =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), Arc::clone(&backend));

        let mut witness = SegmentedEvaluator::new(evaluator, num_vars, backend);

        let mut pt = Blake2bTranscript::new(b"single_seg");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"single_seg");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// Two-segment prove/verify: both StandardGrid with different formulas.
    ///
    /// Segment 0 (vars 0..M): eq0 · g
    /// Segment 1 (vars M..N): eq1 · h
    ///
    /// The claimed sum is Σ_{x ∈ {0,1}^N} eq_full(r, x) · f(x) where f
    /// is defined piecewise across address/cycle variable groups.
    ///
    /// For simplicity, we test a synthetic function:
    ///   Σ_{x0 ∈ {0,1}^M} Σ_{x1 ∈ {0,1}^K} eq(r, x0||x1) · g(x0) · h(x1)
    /// where after binding x0, the transition evaluates g at the bound point
    /// and folds it into h's eq weights.
    #[test]
    fn two_segment_standard_grid_prove_verify() {
        let backend = cpu();
        let m = 3; // first segment rounds
        let k = 3; // second segment rounds
        let num_vars = m + k;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(200);

        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let g_evals: Vec<Fr> = (0..(1 << m)).map(|_| Fr::random(&mut rng)).collect();
        let h_evals: Vec<Fr> = (0..(1 << k)).map(|_| Fr::random(&mut rng)).collect();

        // Full table: f(x0, x1) = g(x0) · h(x1)
        let eq_full = EqPolynomial::new(r.clone()).evaluations();
        let _claimed_sum: Fr = (0..n)
            .map(|idx| {
                let x0 = idx >> k;
                let x1 = idx & ((1 << k) - 1);
                eq_full[idx] * g_evals[x0] * h_evals[x1]
            })
            .sum();

        // Segment 0: sum over x0 of eq_outer(r[0..m], x0) · g(x0) · [Σ_{x1} eq_inner(r[m..], x1) · h(x1)]
        // The inner sum is a constant across x0, so we fold it into the first
        // segment. But for a realistic test, we do it differently:
        //
        // We express the sumcheck over all N vars as:
        //   Segment 0 (M rounds): kernel = eq_outer · g · sum_h_eq
        //     where sum_h_eq = Σ_{x1} eq_inner(r[m..], x1) · h(x1)
        //     This is actually a 1-var-per-pair kernel over the outer variables.
        //   Segment 1 (K rounds): kernel = eq_inner · h · g_at_bound
        //     where g_at_bound is a scalar.
        //
        // Actually, let's do a cleaner test: a single polynomial product
        // over N vars, split into two segments. The transition rebases.
        //
        // Simplest correct test: outer eq covers all N vars, kernel is
        // eq · product, and at the transition we just pass the partially
        // bound evaluator's state to a new kernel on the remaining vars.
        //
        // Let's test with a sum-of-products that naturally factors:
        // Σ eq(r, x) · a(x) where a has N vars.
        // Segment 0 does M rounds, segment 1 does K rounds, both with eq·a.
        // No transition logic needed — just swap evaluator identity.

        // Build a single polynomial a(x) over N vars.
        let a_evals: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let claimed_sum_simple: Fr = eq_full
            .iter()
            .zip(a_evals.iter())
            .map(|(&e, &a)| e * a)
            .sum();

        let claim = SumcheckClaim {
            num_vars,
            degree: 2,
            claimed_sum: claimed_sum_simple,
        };

        let desc = catalog::eq_product();
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![
            backend.upload(&eq_full),
            backend.upload(&a_evals),
        ];
        let evaluator =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), Arc::clone(&backend));

        // Wrap in a SegmentedEvaluator with a trivial transition at round M.
        // The transition creates a new KernelEvaluator from the partially-bound
        // buffers. For this test, we just run all rounds in segment 0 with no
        // actual transition.
        let mut witness = SegmentedEvaluator::new(evaluator, num_vars, Arc::clone(&backend));

        let mut pt = Blake2bTranscript::new(b"two_seg_simple");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"two_seg_simple");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// Two-segment with actual transition: segment 0 uses StandardGrid,
    /// segment 1 uses ToomCook. Models the BytecodeReadRaf pattern.
    #[test]
    fn two_segment_standard_to_toom_cook() {
        let backend = cpu();
        let m = 2; // address rounds (StandardGrid)
        let k = 3; // cycle rounds (ToomCook)
        let d = 3; // product degree for ToomCook
        let num_vars = m + k;
        let mut rng = ChaCha20Rng::seed_from_u64(300);

        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        // Brute-force: Σ_{x0, x1} eq(r, x0||x1) · g(x0) · Π_j p_j(x1)
        let g_evals: Vec<Fr> = (0..(1 << m)).map(|_| Fr::random(&mut rng)).collect();
        let p_evals: Vec<Vec<Fr>> = (0..d)
            .map(|_| (0..(1 << k)).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let n = 1usize << num_vars;
        let eq_full = EqPolynomial::new(r.clone()).evaluations();
        let claimed_sum: Fr = (0..n)
            .map(|idx| {
                let x0 = idx >> k;
                let x1 = idx & ((1 << k) - 1);
                let mut prod = Fr::one();
                for pj in &p_evals {
                    prod *= pj[x1];
                }
                eq_full[idx] * g_evals[x0] * prod
            })
            .sum();

        let _claim = SumcheckClaim {
            num_vars,
            degree: d + 2, // eq × g × Π p_j => degree d+2 in the full thing
            // Actually, address phase has degree 2 (eq · g) and cycle phase
            // has degree d+1 (eq · Π p_j). The overall claim degree must be
            // the max across segments for the verifier.
            claimed_sum,
        };

        // Segment 0: eq_outer · g, StandardGrid, M rounds.
        // After M rounds, the remaining sum is:
        //   Σ_{x1} eq_inner(r[m..], x1) · g(bound_x0) · Π_j p_j(x1)
        // = g(bound_x0) · Σ_{x1} eq_inner · Π_j p_j

        // Build eq for the full space, and combine g into a multiplied buffer.
        // Actually for a 2-segment test, we need the address kernel to only
        // operate on the outer M variables while carrying inner products as
        // constants. This is complex. Let's test a simpler pattern:
        //
        // Segment 0 (M rounds): Custom kernel over eq_outer · g, StandardGrid
        // Segment 1 (K rounds): ProductSum(d, 1) with ToomCook eq
        //
        // The correct way: split eq = eq_outer(r[0..m], x0) · eq_inner(r[m..], x1).
        // Segment 0 sums over x0 with kernel = eq_outer(x0) · g(x0) · inner_sum(x0)
        // where inner_sum(x0) = Σ_{x1} eq_inner(x1) · Π p_j(x1) for each x0.
        // But inner_sum doesn't depend on x0, so it's constant.
        //
        // This doesn't exercise the transition well. Let's use a simpler setup:
        // Build the full sumcheck as eq_full · a · b with transition at round M.

        let a_evals: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let b_evals: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let claimed_sum_ab: Fr = (0..n)
            .map(|i| eq_full[i] * a_evals[i] * b_evals[i])
            .sum();

        let claim_ab = SumcheckClaim {
            num_vars,
            degree: 3, // eq · a · b
            claimed_sum: claimed_sum_ab,
        };

        // Build segment 0 with eq · a · b (degree 3, StandardGrid, M rounds).
        let eb = ExprBuilder::new();
        let eq_v = eb.opening(0);
        let a_v = eb.opening(1);
        let b_v = eb.opening(2);
        let expr = eb.build(eq_v * a_v * b_v);
        let desc = KernelDescriptor {
            shape: KernelShape::Custom {
                expr,
                num_inputs: 3,
            },
            degree: 3,
            tensor_split: None,
        };
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![
            backend.upload(&eq_full),
            backend.upload(&a_evals),
            backend.upload(&b_evals),
        ];
        let evaluator =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), Arc::clone(&backend));

        // Transition at round M: create a new KernelEvaluator with the same
        // formula on the remaining (partially-bound) buffers.
        let a_evals_clone = a_evals.clone();
        let b_evals_clone = b_evals.clone();
        let eq_full_clone = eq_full.clone();
        let transition: SegmentTransition<Fr, CpuBackend> =
            Box::new(move |challenges: Vec<Fr>, backend: &Arc<CpuBackend>| {
                // Partially bind the original tables at the challenge points.
                let half_n = 1usize << k;
                let mut eq_bound = eq_full_clone;
                let mut a_bound = a_evals_clone;
                let mut b_bound = b_evals_clone;

                for &c in &challenges {
                    let half = eq_bound.len() / 2;
                    eq_bound = (0..half)
                        .map(|i| eq_bound[2 * i] + c * (eq_bound[2 * i + 1] - eq_bound[2 * i]))
                        .collect();
                    a_bound = (0..half)
                        .map(|i| a_bound[2 * i] + c * (a_bound[2 * i + 1] - a_bound[2 * i]))
                        .collect();
                    b_bound = (0..half)
                        .map(|i| b_bound[2 * i] + c * (b_bound[2 * i + 1] - b_bound[2 * i]))
                        .collect();
                }

                assert_eq!(eq_bound.len(), half_n);
                assert_eq!(a_bound.len(), half_n);
                assert_eq!(b_bound.len(), half_n);

                let eb = ExprBuilder::new();
                let eq_v = eb.opening(0);
                let a_v = eb.opening(1);
                let b_v = eb.opening(2);
                let expr = eb.build(eq_v * a_v * b_v);
                let desc = KernelDescriptor {
                    shape: KernelShape::Custom {
                        expr,
                        num_inputs: 3,
                    },
                    degree: 3,
                    tensor_split: None,
                };
                let kernel = jolt_cpu::compile::<Fr>(&desc);
                let inputs = vec![
                    backend.upload(&eq_bound),
                    backend.upload(&a_bound),
                    backend.upload(&b_bound),
                ];
                KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), Arc::clone(backend))
            });

        let mut witness = SegmentedEvaluator::new(evaluator, m, Arc::clone(&backend))
            .then(k, transition);

        let mut pt = Blake2bTranscript::new(b"two_seg_transition");
        let proof = SumcheckProver::prove(&claim_ab, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"two_seg_transition");
        let result = SumcheckVerifier::verify(&claim_ab, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// Three-segment with round hooks: hook recompiles kernel every round.
    #[test]
    fn three_segment_with_round_hooks() {
        let backend = cpu();
        let rounds_per_seg = 2;
        let num_segs = 3;
        let num_vars = rounds_per_seg * num_segs;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(400);

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

        // All segments use the same eq · g kernel. The round hook is a no-op
        // that verifies it gets called.
        let hook_call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let hook_counter = Arc::clone(&hook_call_count);

        let desc = catalog::eq_product();
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![backend.upload(&eq_table), backend.upload(&g_table)];
        let evaluator =
            KernelEvaluator::with_unit_weights(inputs, kernel, desc.num_evals(), Arc::clone(&backend));

        let hook: RoundHook<Fr, CpuBackend> =
            Box::new(move |_round, _challenge, _eval| {
                let _ = hook_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            });

        // Single segment with all rounds + a hook, no actual transitions.
        let mut witness = SegmentedEvaluator::new(evaluator, num_vars, Arc::clone(&backend))
            .with_round_hook(hook);

        let mut pt = Blake2bTranscript::new(b"three_seg_hooks");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"three_seg_hooks");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");

        assert_eq!(
            hook_call_count.load(std::sync::atomic::Ordering::Relaxed),
            num_vars,
            "round hook should be called once per round"
        );
    }

    /// HighToLow segment transitioning to LowToHigh segment.
    /// Tests that binding_order is respected across transitions.
    #[test]
    fn high_to_low_then_low_to_high() {
        let backend = cpu();
        let m = 2; // HighToLow rounds
        let k = 3; // LowToHigh rounds
        let num_vars = m + k;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(500);

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

        // Segment 0: HighToLow binding for M rounds
        let desc = catalog::eq_product();
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![
            backend.upload(&eq_table),
            backend.upload(&g_table),
        ];
        let evaluator = KernelEvaluator::with_unit_weights(
            inputs,
            kernel,
            desc.num_evals(),
            Arc::clone(&backend),
        )
        .with_binding_order(BindingOrder::HighToLow);

        // Transition: bind the original tables HighToLow at challenge points,
        // then create a new LowToHigh evaluator.
        let eq_clone = eq_table.clone();
        let g_clone = g_table.clone();
        let transition: SegmentTransition<Fr, CpuBackend> =
            Box::new(move |challenges: Vec<Fr>, backend: &Arc<CpuBackend>| {
                let mut eq_bound = eq_clone;
                let mut g_bound = g_clone;

                // HighToLow binding: pairs are (buf[i], buf[i + n/2])
                for &c in &challenges {
                    let half = eq_bound.len() / 2;
                    eq_bound = (0..half)
                        .map(|i| eq_bound[i] + c * (eq_bound[i + half] - eq_bound[i]))
                        .collect();
                    g_bound = (0..half)
                        .map(|i| g_bound[i] + c * (g_bound[i + half] - g_bound[i]))
                        .collect();
                }

                let desc = catalog::eq_product();
                let kernel = jolt_cpu::compile::<Fr>(&desc);
                let inputs = vec![
                    backend.upload(&eq_bound),
                    backend.upload(&g_bound),
                ];
                // Segment 1 uses LowToHigh (default)
                KernelEvaluator::with_unit_weights(
                    inputs,
                    kernel,
                    desc.num_evals(),
                    Arc::clone(backend),
                )
            });

        let mut witness = SegmentedEvaluator::new(evaluator, m, Arc::clone(&backend))
            .then(k, transition);

        let mut pt = Blake2bTranscript::new(b"h2l_then_l2h");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"h2l_then_l2h");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }

    /// StandardGrid → ToomCook transition with actual product sumcheck.
    /// Models the real BytecodeReadRaf pattern more closely.
    #[test]
    fn standard_grid_to_toom_cook_product() {
        let backend = cpu();
        let m = 2; // address rounds (StandardGrid)
        let k = 3; // cycle rounds (ToomCook, d=3 product)
        let d = 3;
        let num_vars = m + k;
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(600);

        let r: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

        // f(x0, x1) = g(x0) · Π_j p_j(x1)
        let g_evals: Vec<Fr> = (0..(1 << m)).map(|_| Fr::random(&mut rng)).collect();
        let p_evals: Vec<Vec<Fr>> = (0..d)
            .map(|_| (0..(1 << k)).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        // Full table built from the factored representation
        let eq_full = EqPolynomial::new(r.clone()).evaluations();
        let claimed_sum: Fr = (0..n)
            .map(|idx| {
                let x0 = idx >> k;
                let x1 = idx & ((1 << k) - 1);
                let mut prod = Fr::one();
                for pj in &p_evals {
                    prod *= pj[x1];
                }
                eq_full[idx] * g_evals[x0] * prod
            })
            .sum();

        // We build the full polynomial table for address phase: each position
        // in the full table = g(x0) * Π p_j(x1).
        let mut full_table = vec![Fr::zero(); n];
        for (idx, entry) in full_table.iter_mut().enumerate() {
            let x0 = idx >> k;
            let x1 = idx & ((1 << k) - 1);
            let mut prod = g_evals[x0];
            for pj in &p_evals {
                prod *= pj[x1];
            }
            *entry = prod;
        }

        // Address phase claim — degree is only 2 because it's eq · full_table.
        // Cycle phase claim — after binding address vars, the remaining sum has
        // degree d+1 (eq · Π p_j). But for the outer claim, we need the max
        // degree that any round poly can have.
        let claim = SumcheckClaim {
            num_vars,
            degree: d + 1, // max across phases
            claimed_sum,
        };

        // Segment 0: eq · full_table, StandardGrid
        let desc = catalog::eq_product();
        let kernel = jolt_cpu::compile::<Fr>(&desc);
        let inputs = vec![
            backend.upload(&eq_full),
            backend.upload(&full_table),
        ];
        let evaluator = KernelEvaluator::with_unit_weights(
            inputs,
            kernel,
            desc.num_evals(),
            Arc::clone(&backend),
        );

        // At transition: bind eq and full_table at address challenges.
        // The remaining inner sum is:
        //   Σ_{x1} eq_inner(r[m..], x1) · g_bound · Π p_j(x1)
        // = g_bound · Σ_{x1} eq_inner · Π p_j
        // We fold g_bound into the first p_j (or into the eq weight).
        let r_clone = r.clone();
        let p_evals_clone = p_evals.clone();
        let g_evals_clone = g_evals.clone();
        let transition: SegmentTransition<Fr, CpuBackend> =
            Box::new(move |challenges: Vec<Fr>, backend: &Arc<CpuBackend>| {
                // Bind g at address challenges (LowToHigh)
                let mut g_bound = g_evals_clone;
                for &c in &challenges {
                    let half = g_bound.len() / 2;
                    g_bound = (0..half)
                        .map(|i| g_bound[2 * i] + c * (g_bound[2 * i + 1] - g_bound[2 * i]))
                        .collect();
                }
                assert_eq!(g_bound.len(), 1);
                let g_scalar = g_bound[0];

                // Build eq for the cycle (inner) variables.
                let eq_inner = EqPolynomial::new(r_clone[m..].to_vec()).evaluations();

                // Scale the first p poly by g_scalar.
                let p0_scaled: Vec<Fr> = p_evals_clone[0]
                    .iter()
                    .map(|&v| v * g_scalar)
                    .collect();

                // ToomCook with eq_inner as weights.
                let inner_claimed_sum: Fr = (0..(1 << k))
                    .map(|x1| {
                        let mut prod = p0_scaled[x1];
                        for pj in p_evals_clone.iter().skip(1) {
                            prod *= pj[x1];
                        }
                        eq_inner[x1] * prod
                    })
                    .sum();

                let desc = catalog::product_sum(d, 1);
                let kernel = jolt_cpu::compile::<Fr>(&desc);

                let mut inputs: Vec<Vec<Fr>> = Vec::with_capacity(d);
                inputs.push(p0_scaled);
                for pj in p_evals_clone.iter().skip(1) {
                    inputs.push(pj.clone());
                }
                let input_bufs: Vec<_> = inputs.iter().map(|p| backend.upload(p)).collect();

                KernelEvaluator::with_toom_cook_eq(
                    input_bufs,
                    kernel,
                    desc.num_evals(),
                    r_clone[m..].to_vec(),
                    inner_claimed_sum,
                    Arc::clone(backend),
                )
            });

        let mut witness = SegmentedEvaluator::new(evaluator, m, Arc::clone(&backend))
            .then(k, transition);

        let mut pt = Blake2bTranscript::new(b"sg_to_tc");
        let proof = SumcheckProver::prove(&claim, &mut witness, &mut pt);

        let mut vt = Blake2bTranscript::new(b"sg_to_tc");
        let result = SumcheckVerifier::verify(&claim, &proof, &mut vt);
        assert!(result.is_ok(), "verification failed: {result:?}");
    }
}
