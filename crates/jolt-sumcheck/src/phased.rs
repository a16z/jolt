//! N-phase sumcheck evaluator compositor.
//!
//! [`PhasedEvaluator`] chains multiple [`SumcheckCompute`] implementations
//! across phase boundaries, triggering transition closures at each boundary.
//!
//! This subsumes [`PrefixSuffixEvaluator`] (which is a 2-phase special case)
//! and supports the 3-phase RAM/register read-write checking pattern
//! (sparse-cycle → sparse-address → dense kernel).
//!
//! # Example: 3-phase RAM read-write checking
//!
//! ```text
//! PhasedEvaluator::builder()
//!     .first_phase(sparse_cycle_witness, phase1_rounds)
//!     .transition(|challenges| build_address_phase(challenges))
//!     .transition(|challenges| build_dense_phase(challenges))
//!     .build()
//! ```

use std::collections::VecDeque;

use jolt_field::Field;
use jolt_poly::UnivariatePoly;

use crate::prover::SumcheckCompute;

/// Transition closure: receives challenges from the completed phase,
/// returns the next phase's evaluator.
pub type TransitionFn<F> =
    Box<dyn FnOnce(PhaseTransition<F>) -> Box<dyn SumcheckCompute<F>> + Send + Sync>;

/// State at a phase boundary, passed to the transition closure.
pub struct PhaseTransition<F> {
    /// All challenges accumulated during the completed phase.
    pub challenges: Vec<F>,
    /// All challenges from ALL prior phases (including the just-completed one).
    pub all_challenges: Vec<F>,
}

/// Pending phase: rounds to run + transition to build the evaluator.
struct PendingPhase<F: Field> {
    num_rounds: usize,
    build: TransitionFn<F>,
}

/// N-phase sumcheck evaluator.
///
/// Implements [`SumcheckCompute`] by delegating to the active phase's
/// evaluator. When the active phase exhausts its rounds, the next
/// transition closure fires to build the subsequent evaluator.
///
/// All phases share the same `SumcheckCompute` interface — each phase
/// can be a `KernelEvaluator` (dense/GPU), `SparseEvaluator` (sparse/CPU),
/// `PrefixSuffixEvaluator`, or any other impl. The `PhasedEvaluator`
/// doesn't know or care about the inner representation.
pub struct PhasedEvaluator<F: Field> {
    active: Box<dyn SumcheckCompute<F>>,
    rounds_left_in_phase: usize,
    phase_challenges: Vec<F>,
    all_challenges: Vec<F>,
    pending: VecDeque<PendingPhase<F>>,
}

impl<F: Field> PhasedEvaluator<F> {
    /// Start building a phased evaluator.
    pub fn builder() -> PhasedBuilder<F> {
        PhasedBuilder {
            first: None,
            transitions: VecDeque::new(),
        }
    }
}

impl<F: Field> SumcheckCompute<F> for PhasedEvaluator<F> {
    fn set_claim(&mut self, claim: F) {
        self.active.set_claim(claim);
    }

    fn round_polynomial(&self) -> UnivariatePoly<F> {
        self.active.round_polynomial()
    }

    fn bind(&mut self, challenge: F) {
        self.active.bind(challenge);
        self.phase_challenges.push(challenge);
        self.all_challenges.push(challenge);
        self.rounds_left_in_phase -= 1;

        if self.rounds_left_in_phase == 0 {
            if let Some(next) = self.pending.pop_front() {
                let transition = PhaseTransition {
                    challenges: std::mem::take(&mut self.phase_challenges),
                    all_challenges: self.all_challenges.clone(),
                };
                self.active = (next.build)(transition);
                self.rounds_left_in_phase = next.num_rounds;
            }
        }
    }

    fn first_round_polynomial(&self) -> Option<UnivariatePoly<F>> {
        self.active.first_round_polynomial()
    }

    fn produced_evaluations(&self) -> Vec<(usize, F)> {
        self.active.produced_evaluations()
    }
}

/// Builder for [`PhasedEvaluator`].
pub struct PhasedBuilder<F: Field> {
    first: Option<(Box<dyn SumcheckCompute<F>>, usize)>,
    transitions: VecDeque<PendingPhase<F>>,
}

impl<F: Field> PhasedBuilder<F> {
    /// Set the first phase's evaluator and round count.
    pub fn first_phase(
        mut self,
        witness: Box<dyn SumcheckCompute<F>>,
        num_rounds: usize,
    ) -> Self {
        self.first = Some((witness, num_rounds));
        self
    }

    /// Add a transition to the next phase. The closure receives accumulated
    /// challenges and must return the next phase's evaluator.
    ///
    /// `num_rounds` is the number of rounds for the phase AFTER this transition.
    pub fn then(
        mut self,
        num_rounds: usize,
        build: impl FnOnce(PhaseTransition<F>) -> Box<dyn SumcheckCompute<F>> + Send + Sync + 'static,
    ) -> Self {
        self.transitions.push_back(PendingPhase {
            num_rounds,
            build: Box::new(build),
        });
        self
    }

    /// Build the phased evaluator.
    ///
    /// # Panics
    ///
    /// Panics if no first phase was set.
    pub fn build(self) -> PhasedEvaluator<F> {
        let (active, rounds) = self.first.expect("first_phase must be set");
        PhasedEvaluator {
            active,
            rounds_left_in_phase: rounds,
            phase_challenges: Vec::new(),
            all_challenges: Vec::new(),
            pending: self.transitions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SumcheckClaim, SumcheckProver, SumcheckVerifier};
    use jolt_field::{Field, Fr};
    use jolt_poly::{EqPolynomial, Polynomial};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::Zero;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// Minimal dense evaluator for testing: computes Σ eq(r, x) · g(x).
    /// Uses **HighToLow** binding (MSB first) — matching `Polynomial::bind()`.
    struct DenseEqWitness<F: Field> {
        eq_table: Vec<F>,
        g_table: Vec<F>,
        claim: F,
    }

    impl<F: Field> SumcheckCompute<F> for DenseEqWitness<F> {
        fn set_claim(&mut self, claim: F) {
            self.claim = claim;
        }

        fn round_polynomial(&self) -> UnivariatePoly<F> {
            let half = self.eq_table.len() / 2;
            let mut eval_0 = F::zero();
            let mut eval_2 = F::zero();
            for j in 0..half {
                // HighToLow: table[j] = MSB=0, table[j+half] = MSB=1
                let eq_lo = self.eq_table[j];
                let eq_hi = self.eq_table[j + half];
                let g_lo = self.g_table[j];
                let g_hi = self.g_table[j + half];
                eval_0 += eq_lo * g_lo;
                let eq2 = eq_hi + eq_hi - eq_lo;
                let g2 = g_hi + g_hi - g_lo;
                eval_2 += eq2 * g2;
            }
            UnivariatePoly::from_evals_and_hint(self.claim, &[eval_0, eval_2])
        }

        fn bind(&mut self, r: F) {
            let half = self.eq_table.len() / 2;
            for j in 0..half {
                let lo = self.eq_table[j];
                let hi = self.eq_table[j + half];
                self.eq_table[j] = lo + r * (hi - lo);
                let lo = self.g_table[j];
                let hi = self.g_table[j + half];
                self.g_table[j] = lo + r * (hi - lo);
            }
            self.eq_table.truncate(half);
            self.g_table.truncate(half);
        }
    }

    fn eval_mle(evals: &[Fr], point: &[Fr]) -> Fr {
        Polynomial::new(evals.to_vec()).evaluate(point)
    }

    /// 1-phase PhasedEvaluator should behave identically to raw SumcheckCompute.
    #[test]
    fn single_phase_matches_direct() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 4;
        let size = 1 << n;

        let r: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let g: Vec<Fr> = (0..size).map(|_| Fr::random(&mut rng)).collect();
        let eq = EqPolynomial::new(r.clone()).evaluations();
        let claimed_sum: Fr = eq.iter().zip(g.iter()).map(|(&e, &gv)| e * gv).sum();

        let witness = DenseEqWitness {
            eq_table: eq.clone(),
            g_table: g.clone(),
            claim: Fr::zero(),
        };

        let mut phased = PhasedEvaluator::builder()
            .first_phase(Box::new(witness), n)
            .build();

        let claim = SumcheckClaim { num_vars: n, degree: 2, claimed_sum };
        let mut pt = Blake2bTranscript::new(b"single_phase");
        let proof = SumcheckProver::prove(&claim, &mut phased, &mut pt);

        let mut vt = Blake2bTranscript::new(b"single_phase");
        let (final_eval, challenges) =
            SumcheckVerifier::verify(&claim, &proof, &mut vt).unwrap();

        let expected = eval_mle(&eq, &challenges) * eval_mle(&g, &challenges);
        assert_eq!(final_eval, expected);
    }

    /// 2-phase PhasedEvaluator using PrefixSuffixEvaluator's proven pattern.
    /// HighToLow binding: Phase 1 = prefix vars, Phase 2 = suffix vars.
    /// Same decomposition as prefix_suffix::tests::single_pair_prove_verify.
    #[test]
    fn two_phase_high_to_low() {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let total = 6;
        let prefix_vars = 3;
        let suffix_vars = total - prefix_vars;
        let prefix_size = 1 << prefix_vars;
        let suffix_size = 1 << suffix_vars;
        let n = 1 << total;

        let r: Vec<Fr> = (0..total).map(|_| Fr::random(&mut rng)).collect();
        let g: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let eq_full = EqPolynomial::new(r.clone()).evaluations();
        let claimed_sum: Fr = eq_full.iter().zip(g.iter()).map(|(&e, &gv)| e * gv).sum();

        let r_prefix = r[..prefix_vars].to_vec();
        let r_suffix = r[prefix_vars..].to_vec();

        let p_table = EqPolynomial::new(r_prefix).evaluations();
        let suffix_evals = EqPolynomial::new(r_suffix).evaluations();

        // Fold g into prefix domain: Q[j] = Σ_s suffix_eq[s] * g[j * suffix_size + s]
        let mut q_table = vec![Fr::zero(); prefix_size];
        for j in 0..prefix_size {
            for (s, &sv) in suffix_evals.iter().enumerate() {
                q_table[j] += sv * g[j * suffix_size + s];
            }
        }

        let phase1_witness = DenseEqWitness {
            eq_table: p_table,
            g_table: q_table,
            claim: Fr::zero(),
        };

        let g_clone = g.clone();
        let suffix_evals_clone = suffix_evals.clone();
        let r_prefix_clone = r[..prefix_vars].to_vec();

        let mut phased = PhasedEvaluator::builder()
            .first_phase(Box::new(phase1_witness), prefix_vars)
            .then(suffix_vars, move |transition| {
                // After phase 1 fully bound the prefix eq table, the residual
                // is a scalar: eq(r_prefix, challenges). Scale the suffix eq.
                let prefix_scalar = EqPolynomial::new(r_prefix_clone).evaluate(&transition.challenges);
                let eq_table: Vec<Fr> = suffix_evals_clone.iter().map(|&s| prefix_scalar * s).collect();

                // Partially evaluate g at the prefix challenges:
                // g_bound(s) = Σ_p eq(challenges, p) * g[p * suffix_size + s]
                let eq_prefix_bound = EqPolynomial::new(transition.challenges).evaluations();
                let mut g_suffix = vec![Fr::zero(); suffix_size];
                for s in 0..suffix_size {
                    for (p, &eq_val) in eq_prefix_bound.iter().enumerate() {
                        g_suffix[s] += eq_val * g_clone[p * suffix_size + s];
                    }
                }

                Box::new(DenseEqWitness {
                    eq_table,
                    g_table: g_suffix,
                    claim: Fr::zero(),
                }) as Box<dyn SumcheckCompute<Fr>>
            })
            .build();

        let claim = SumcheckClaim { num_vars: total, degree: 2, claimed_sum };
        let mut pt = Blake2bTranscript::new(b"two_phase");
        let proof = SumcheckProver::prove(&claim, &mut phased, &mut pt);

        let mut vt = Blake2bTranscript::new(b"two_phase");
        let (final_eval, challenges) =
            SumcheckVerifier::verify(&claim, &proof, &mut vt).unwrap();

        let expected = eval_mle(&eq_full, &challenges) * eval_mle(&g, &challenges);
        assert_eq!(final_eval, expected);
    }

    /// 3-phase PhasedEvaluator with HighToLow binding.
    /// Same factored decomposition pattern as 2-phase, extended to 3 groups.
    #[test]
    fn three_phase_transitions() {
        let mut rng = ChaCha20Rng::seed_from_u64(777);
        let pv1 = 2;
        let pv2 = 2;
        let pv3 = 2;
        let total = pv1 + pv2 + pv3;
        let n = 1 << total;
        let s1 = 1 << pv1;
        let s2 = 1 << pv2;
        let s3 = 1 << pv3;

        let r: Vec<Fr> = (0..total).map(|_| Fr::random(&mut rng)).collect();
        let g: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let eq_full = EqPolynomial::new(r.clone()).evaluations();
        let claimed_sum: Fr = eq_full.iter().zip(g.iter()).map(|(&e, &gv)| e * gv).sum();

        // Phase 1 eq: r[0..pv1], phase 2 eq: r[pv1..pv1+pv2], phase 3 eq: r[pv1+pv2..]
        let eq1 = EqPolynomial::new(r[..pv1].to_vec()).evaluations();
        let eq_rest = EqPolynomial::new(r[pv1..].to_vec()).evaluations();

        // Fold g into phase 1 domain via eq_rest
        let rest_size = s2 * s3;
        let mut g_folded1 = vec![Fr::zero(); s1];
        for i in 0..s1 {
            for j in 0..rest_size {
                g_folded1[i] += eq_rest[j] * g[i * rest_size + j];
            }
        }

        let phase1_witness = DenseEqWitness {
            eq_table: eq1,
            g_table: g_folded1,
            claim: Fr::zero(),
        };

        let g_clone = g.clone();
        let r_clone = r.clone();

        let mut phased = PhasedEvaluator::builder()
            .first_phase(Box::new(phase1_witness), pv1)
            .then(pv2, {
                let g = g_clone.clone();
                let r = r_clone.clone();
                move |t1| {
                    let r_p1 = r[..pv1].to_vec();
                    let r_p2 = r[pv1..pv1 + pv2].to_vec();
                    let r_p3 = r[pv1 + pv2..].to_vec();
                    let p1_scalar = EqPolynomial::new(r_p1).evaluate(&t1.challenges);
                    let eq2 = EqPolynomial::new(r_p2).evaluations();
                    let eq3 = EqPolynomial::new(r_p3).evaluations();

                    let eq1_bound = EqPolynomial::new(t1.challenges).evaluations();
                    let mut g_folded2 = vec![Fr::zero(); s2];
                    for mid in 0..s2 {
                        for lo in 0..s3 {
                            for (hi, &e1) in eq1_bound.iter().enumerate() {
                                g_folded2[mid] += e1 * eq3[lo] * g[hi * rest_size + mid * s3 + lo];
                            }
                        }
                    }
                    // Scale eq2 by p1 scalar
                    let eq2_scaled: Vec<Fr> = eq2.iter().map(|&e| p1_scalar * e).collect();

                    Box::new(DenseEqWitness {
                        eq_table: eq2_scaled,
                        g_table: g_folded2,
                        claim: Fr::zero(),
                    }) as Box<dyn SumcheckCompute<Fr>>
                }
            })
            .then(pv3, {
                let g = g_clone;
                let r = r_clone;
                move |t2| {
                    let r_p12 = r[..pv1 + pv2].to_vec();
                    let r_p3 = r[pv1 + pv2..].to_vec();
                    let p12_scalar = EqPolynomial::new(r_p12).evaluate(&t2.all_challenges);
                    let eq3 = EqPolynomial::new(r_p3).evaluations();
                    let eq3_scaled: Vec<Fr> = eq3.iter().map(|&e| p12_scalar * e).collect();

                    let eq_bound_all = EqPolynomial::new(t2.all_challenges).evaluations();
                    let mut g_folded3 = vec![Fr::zero(); s3];
                    for lo in 0..s3 {
                        for (hi_mid, &eq_val) in eq_bound_all.iter().enumerate() {
                            let hi = hi_mid / s2;
                            let mid = hi_mid % s2;
                            g_folded3[lo] += eq_val * g[hi * rest_size + mid * s3 + lo];
                        }
                    }

                    Box::new(DenseEqWitness {
                        eq_table: eq3_scaled,
                        g_table: g_folded3,
                        claim: Fr::zero(),
                    }) as Box<dyn SumcheckCompute<Fr>>
                }
            })
            .build();

        let claim = SumcheckClaim { num_vars: total, degree: 2, claimed_sum };
        let mut pt = Blake2bTranscript::new(b"three_phase");
        let proof = SumcheckProver::prove(&claim, &mut phased, &mut pt);

        let mut vt = Blake2bTranscript::new(b"three_phase");
        let (final_eval, challenges) =
            SumcheckVerifier::verify(&claim, &proof, &mut vt).unwrap();

        let expected = eval_mle(&eq_full, &challenges) * eval_mle(&g, &challenges);
        assert_eq!(final_eval, expected);
    }
}
