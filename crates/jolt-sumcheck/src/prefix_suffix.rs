//! Generic prefix-suffix sumcheck evaluator for tensor-decomposed polynomials.
//!
//! When a multilinear expression decomposes as
//!
//! ```text
//! f(x) = Σ_i P_i(x_prefix) · S_i(x_suffix)
//! ```
//!
//! the prover can avoid materializing the full N-sized table by working
//! with √N-sized "pair" buffers during Phase 1, then materializing only
//! √N-sized tables for Phase 2.
//!
//! # Variable convention
//!
//! Uses **HighToLow** binding (MSB first), matching `Polynomial::bind`.
//! Prefix variables are the first `m` variables (`point[0..m]`), which
//! occupy the **high** bits of the evaluation table index:
//!
//! ```text
//! table[prefix_idx * suffix_size + suffix_idx]
//! ```
//!
//! # Pair construction (caller responsibility)
//!
//! The caller folds witness data into the Q tables during construction:
//!
//! ```text
//! P_i[j] = prefix_i(j)                                          (eq factor, √N entries)
//! Q_i[j] = Σ_{s} suffix_i(s) · g[j * suffix_size + s]          (witness-folded, √N entries)
//! ```
//!
//! Both P_i and Q_i are indexed over the **prefix domain**.
//! Phase 1 computes `Σ_i ⟨P_i, Q_i⟩` = `Σ_x f(x) · g(x)`.
//!
//! # Phase 2
//!
//! After all prefix variables are bound, the [`Phase2Builder`] callback
//! receives a [`PrefixSuffixTransition`] containing the prefix challenges
//! and the scalar evaluations of each P_i. It materializes √N-sized suffix
//! tables and returns a [`SumcheckCompute`] impl for the remaining rounds.
//!
//! See <https://eprint.iacr.org/2025/611.pdf> (Appendix A).

use jolt_field::{Field, WithChallenge};
use jolt_poly::UnivariatePoly;

use crate::prover::SumcheckCompute;

/// State at the Phase 1 → Phase 2 transition.
///
/// Passed to the [`Phase2Builder`] callback so it can construct the
/// suffix-phase witness from the bound prefix state.
pub struct PrefixSuffixTransition<F: Field> {
    /// All prefix challenges in binding order (HighToLow: `point[0]` first).
    pub challenges: Vec<F>,
    /// `P_i(challenges)` — the scalar evaluation of each prefix table
    /// after all prefix variables have been bound.
    pub prefix_evals: Vec<F>,
}

impl<F: Field> PrefixSuffixTransition<F> {
    /// Combines suffix evaluation tables weighted by `prefix_evals`:
    ///
    /// ```text
    /// combined[j] = Σ_i prefix_evals[i] · suffix_tables[i][j]
    /// ```
    ///
    /// This is the most common Phase 2 operation: building the combined
    /// eq table for the suffix sumcheck from the original (un-folded)
    /// suffix decomposition tables and the prefix scalars.
    pub fn combined_suffix(&self, suffix_tables: &[&[F]]) -> Vec<F> {
        assert_eq!(
            self.prefix_evals.len(),
            suffix_tables.len(),
            "mismatch: {} prefix evals vs {} suffix tables",
            self.prefix_evals.len(),
            suffix_tables.len(),
        );
        let len = suffix_tables[0].len();
        let mut result = vec![F::zero(); len];
        for (scalar, table) in self.prefix_evals.iter().zip(suffix_tables.iter()) {
            debug_assert_eq!(table.len(), len);
            for (r, &t) in result.iter_mut().zip(table.iter()) {
                *r += *scalar * t;
            }
        }
        result
    }
}

/// Builder that produces the Phase 2 witness.
///
/// Called exactly once at the Phase 1 → Phase 2 transition. Receives
/// a [`PrefixSuffixTransition`] and must return a [`SumcheckCompute`]
/// impl for the remaining suffix rounds.
pub type Phase2Builder<F> =
    Box<dyn FnOnce(PrefixSuffixTransition<F>) -> Box<dyn SumcheckCompute<F>> + Send + Sync>;

/// Generic prefix-suffix sumcheck evaluator.
///
/// Implements [`SumcheckCompute`] in two phases:
///
/// **Phase 1** (prefix variables, HighToLow): Operates on `(P_i, Q_i)` pairs,
/// both √N-sized evaluation tables over the prefix domain. The round polynomial
/// is `s(X) = Σ_i Σ_j P_i(X, j) · Q_i(X, j)`, degree 2. Both tables are
/// bound (MSB first) and halved each round.
///
/// **Phase 2** (suffix variables): Delegates to a user-provided
/// [`SumcheckCompute`] produced by the [`Phase2Builder`] callback, which
/// typically materializes √N-sized tables for the suffix domain.
pub struct PrefixSuffixEvaluator<F: WithChallenge> {
    pairs: Vec<(Vec<F>, Vec<F>)>,
    prefix_vars: usize,
    claim: F,
    challenges: Vec<F>,
    phase2: Option<Box<dyn SumcheckCompute<F>>>,
    phase2_builder: Option<Phase2Builder<F>>,
}

impl<F: WithChallenge> PrefixSuffixEvaluator<F> {
    /// Creates a new prefix-suffix evaluator.
    ///
    /// # Arguments
    ///
    /// * `pairs` — `(P_i, Q_i)` evaluation table pairs over the prefix
    ///   domain. All tables must have the same power-of-2 length. P_i is
    ///   typically the eq/eq+1 prefix factor; Q_i has witness data folded
    ///   in: `Q_i[j] = Σ_{s} suffix_i(s) · g[j * suffix_size + s]`.
    /// * `phase2_builder` — Closure called at the Phase 1 → Phase 2
    ///   transition with a [`PrefixSuffixTransition`].
    ///
    /// # Panics
    ///
    /// Panics if `pairs` is empty or if any table has mismatched or
    /// non-power-of-2 length.
    pub fn new(pairs: Vec<(Vec<F>, Vec<F>)>, phase2_builder: Phase2Builder<F>) -> Self {
        assert!(!pairs.is_empty(), "must have at least one (P, Q) pair");
        let len = pairs[0].0.len();
        assert!(len.is_power_of_two(), "pair length must be a power of 2");
        for (i, (p, q)) in pairs.iter().enumerate() {
            assert_eq!(p.len(), len, "P[{i}].len() = {} != {len}", p.len());
            assert_eq!(q.len(), len, "Q[{i}].len() = {} != {len}", q.len());
        }

        let prefix_vars = len.trailing_zeros() as usize;
        Self {
            pairs,
            prefix_vars,
            claim: F::zero(),
            challenges: Vec::with_capacity(prefix_vars),
            phase2: None,
            phase2_builder: Some(phase2_builder),
        }
    }

    #[inline]
    pub fn in_prefix_phase(&self) -> bool {
        self.phase2.is_none()
    }

    #[inline]
    pub fn prefix_vars(&self) -> usize {
        self.prefix_vars
    }
}

/// Binds a multilinear evaluation table by the highest variable (HighToLow / MSB first).
///
/// Pairs `table[j]` (x_msb=0) with `table[j + half]` (x_msb=1):
/// `table[j] = table[j] + r · (table[j + half] - table[j])` for `j ∈ [0, len/2)`.
#[inline]
fn bind_high_to_low<F: WithChallenge>(table: &mut Vec<F>, r: F::Challenge) {
    let half = table.len() / 2;
    for j in 0..half {
        let lo = table[j];
        let hi = table[j + half];
        table[j] = lo + r * (hi - lo);
    }
    table.truncate(half);
}

impl<F: WithChallenge> SumcheckCompute<F> for PrefixSuffixEvaluator<F> {
    fn set_claim(&mut self, claim: F) {
        if let Some(ref mut inner) = self.phase2 {
            inner.set_claim(claim);
        } else {
            self.claim = claim;
        }
    }

    fn round_polynomial(&self) -> UnivariatePoly<F> {
        if let Some(ref inner) = self.phase2 {
            return inner.round_polynomial();
        }

        // Phase 1: accumulate P_i(t) · Q_i(t) at standard grid {0, 2}.
        // HighToLow: first half = MSB=0, second half = MSB=1.
        let half = self.pairs[0].0.len() / 2;
        let mut eval_0 = F::zero();
        let mut eval_2 = F::zero();

        for (p, q) in &self.pairs {
            for j in 0..half {
                let p_lo = p[j];
                let p_hi = p[j + half];
                let q_lo = q[j];
                let q_hi = q[j + half];

                eval_0 += p_lo * q_lo;

                let p_at_2 = p_hi + p_hi - p_lo;
                let q_at_2 = q_hi + q_hi - q_lo;
                eval_2 += p_at_2 * q_at_2;
            }
        }

        // Degree 2: P(1) = claim − P(0).
        UnivariatePoly::from_evals_and_hint(self.claim, &[eval_0, eval_2])
    }

    fn bind(&mut self, challenge: F::Challenge) {
        if let Some(ref mut inner) = self.phase2 {
            inner.bind(challenge);
            return;
        }

        for (p, q) in &mut self.pairs {
            bind_high_to_low(p, challenge);
            bind_high_to_low(q, challenge);
        }
        self.challenges.push(challenge.into());

        if self.challenges.len() == self.prefix_vars {
            let prefix_evals = self.pairs.iter().map(|(p, _)| p[0]).collect();
            let transition = PrefixSuffixTransition {
                challenges: std::mem::take(&mut self.challenges),
                prefix_evals,
            };
            let builder = self
                .phase2_builder
                .take()
                .expect("phase2_builder consumed twice");
            self.phase2 = Some(builder(transition));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr, WithChallenge};
    use jolt_poly::{EqPolynomial, Polynomial};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use crate::{SumcheckClaim, SumcheckProver, SumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};

    /// Phase 2 witness: eq·g product over suffix-domain tables.
    /// Uses HighToLow binding to match the overall convention.
    struct SimplePhase2<F: WithChallenge> {
        eq_table: Vec<F>,
        g_table: Vec<F>,
        claim: F,
    }

    impl<F: WithChallenge> SumcheckCompute<F> for SimplePhase2<F> {
        fn set_claim(&mut self, claim: F) {
            self.claim = claim;
        }

        fn round_polynomial(&self) -> UnivariatePoly<F> {
            let half = self.eq_table.len() / 2;
            let mut eval_0 = F::zero();
            let mut eval_2 = F::zero();
            for j in 0..half {
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

        fn bind(&mut self, challenge: F::Challenge) {
            bind_high_to_low(&mut self.eq_table, challenge);
            bind_high_to_low(&mut self.g_table, challenge);
        }
    }

    /// Folds witness `g` into the suffix dimension to produce a Q table
    /// over the prefix domain (HighToLow layout):
    ///
    /// `Q[j] = Σ_{s} suffix[s] · g[j * suffix_size + s]`
    fn fold_witness_into_q(
        suffix_table: &[Fr],
        g: &[Fr],
        prefix_size: usize,
        suffix_size: usize,
    ) -> Vec<Fr> {
        let mut q = vec![Fr::zero(); prefix_size];
        for j in 0..prefix_size {
            for (s, &sv) in suffix_table.iter().enumerate() {
                q[j] += sv * g[j * suffix_size + s];
            }
        }
        q
    }

    /// Brute-force evaluation of `f(point)` for a multilinear polynomial
    /// given by its evaluation table.
    fn eval_mle(evals: &[Fr], point: &[Fr]) -> Fr {
        Polynomial::new(evals.to_vec()).evaluate(point)
    }

    #[test]
    fn phase_transition_timing() {
        let prefix_size = 8; // 3 vars
        let p: Vec<Fr> = (0..prefix_size).map(|i| Fr::from_u64(i as u64)).collect();
        let q: Vec<Fr> = (0..prefix_size)
            .map(|i| Fr::from_u64(i as u64 + 10))
            .collect();

        let mut evaluator = PrefixSuffixEvaluator::new(
            vec![(p, q)],
            Box::new(|transition: PrefixSuffixTransition<Fr>| {
                assert_eq!(
                    transition.challenges.len(),
                    3,
                    "should receive 3 prefix challenges"
                );
                assert_eq!(transition.prefix_evals.len(), 1, "1 pair → 1 prefix eval");
                struct Dummy;
                impl SumcheckCompute<Fr> for Dummy {
                    fn set_claim(&mut self, _: Fr) {}
                    fn round_polynomial(&self) -> UnivariatePoly<Fr> {
                        UnivariatePoly::zero()
                    }
                    fn bind(&mut self, _: <Fr as WithChallenge>::Challenge) {}
                }
                Box::new(Dummy) as Box<dyn SumcheckCompute<Fr>>
            }),
        );

        assert!(evaluator.in_prefix_phase());
        assert_eq!(evaluator.prefix_vars(), 3);

        evaluator.set_claim(Fr::zero());
        let _ = evaluator.round_polynomial();
        evaluator.bind(<Fr as WithChallenge>::Challenge::from(1u128));
        assert!(evaluator.in_prefix_phase());

        let _ = evaluator.round_polynomial();
        evaluator.bind(<Fr as WithChallenge>::Challenge::from(2u128));
        assert!(evaluator.in_prefix_phase());

        let _ = evaluator.round_polynomial();
        evaluator.bind(<Fr as WithChallenge>::Challenge::from(3u128));
        assert!(!evaluator.in_prefix_phase());
    }

    #[test]
    fn single_pair_prove_verify() {
        // Prove Σ_x eq(r, x) · g(x) via 1-pair prefix-suffix.
        //
        // HighToLow layout: prefix = first m vars (high bits of index),
        // suffix = last (n-m) vars (low bits of index).
        // Index: prefix_idx * suffix_size + suffix_idx
        let mut rng = ChaCha20Rng::seed_from_u64(200);
        let total_vars = 6;
        let prefix_vars = 3;
        let prefix_size = 1usize << prefix_vars;
        let suffix_size = 1usize << (total_vars - prefix_vars);
        let n = 1usize << total_vars;

        let r: Vec<Fr> = (0..total_vars).map(|_| Fr::random(&mut rng)).collect();
        let g: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let eq_full = EqPolynomial::new(r.clone()).evaluations();
        let claimed_sum: Fr = eq_full.iter().zip(g.iter()).map(|(&e, &gv)| e * gv).sum();

        let r_prefix = r[..prefix_vars].to_vec();
        let r_suffix = r[prefix_vars..].to_vec();

        let p = EqPolynomial::new(r_prefix).evaluations();
        let suffix_evals = EqPolynomial::new(r_suffix).evaluations();
        let q = fold_witness_into_q(&suffix_evals, &g, prefix_size, suffix_size);

        let g_clone = g.clone();
        let suffix_evals_clone = suffix_evals.clone();

        let mut evaluator = PrefixSuffixEvaluator::new(
            vec![(p, q)],
            Box::new(move |transition: PrefixSuffixTransition<Fr>| {
                // eq_combined(s) = prefix_eval · suffix_evals(s)
                let eq_table = transition.combined_suffix(&[&suffix_evals_clone]);

                // g_bound(s) = g(challenges, s) = Σ_{p} eq(challenges, p) · g[p * suffix_size + s]
                let eq_prefix = EqPolynomial::new(transition.challenges).evaluations();
                let mut g_suffix = vec![Fr::zero(); suffix_size];
                for s in 0..suffix_size {
                    for (p_idx, &eq_val) in eq_prefix.iter().enumerate() {
                        g_suffix[s] += eq_val * g_clone[p_idx * suffix_size + s];
                    }
                }

                Box::new(SimplePhase2 {
                    eq_table,
                    g_table: g_suffix,
                    claim: Fr::zero(),
                }) as Box<dyn SumcheckCompute<Fr>>
            }),
        );

        let claim = SumcheckClaim {
            num_vars: total_vars,
            degree: 2,
            claimed_sum,
        };

        let mut pt = Blake2bTranscript::new(b"single_pair");
        let proof = SumcheckProver::prove(&claim, &mut evaluator, &mut pt);

        let mut vt = Blake2bTranscript::new(b"single_pair");
        let (final_eval, challenges) =
            SumcheckVerifier::verify(&claim, &proof, &mut vt).unwrap();

        // Oracle check: final_eval must equal eq(r, challenges) · g(challenges).
        let eq_at_r = eval_mle(&eq_full, &challenges);
        let g_at_r = eval_mle(&g, &challenges);
        assert_eq!(final_eval, eq_at_r * g_at_r, "oracle check failed");
    }

    #[test]
    fn two_pair_prove_verify_with_witness() {
        // Prove Σ_x [a·eq(r1,x) + b·eq(r2,x)] · g(x) via 2-pair prefix-suffix.
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let total_vars = 8;
        let prefix_vars = total_vars / 2;
        let prefix_size = 1usize << prefix_vars;
        let suffix_size = 1usize << (total_vars - prefix_vars);
        let n = 1usize << total_vars;

        let r1: Vec<Fr> = (0..total_vars).map(|_| Fr::random(&mut rng)).collect();
        let r2: Vec<Fr> = (0..total_vars).map(|_| Fr::random(&mut rng)).collect();
        let g: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let a = Fr::from_u64(3);
        let b = Fr::from_u64(7);

        let eq1 = EqPolynomial::new(r1.clone()).evaluations();
        let eq2 = EqPolynomial::new(r2.clone()).evaluations();
        let claimed_sum: Fr = (0..n).map(|x| (a * eq1[x] + b * eq2[x]) * g[x]).sum();

        let r1_prefix = r1[..prefix_vars].to_vec();
        let r1_suffix = r1[prefix_vars..].to_vec();
        let r2_prefix = r2[..prefix_vars].to_vec();
        let r2_suffix = r2[prefix_vars..].to_vec();

        let suffix1 = EqPolynomial::new(r1_suffix).evaluations();
        let suffix2 = EqPolynomial::new(r2_suffix).evaluations();

        // P_i = scaled prefix eq, Q_i = witness-folded
        let p0: Vec<Fr> = EqPolynomial::new(r1_prefix)
            .evaluations()
            .into_iter()
            .map(|e| a * e)
            .collect();
        let q0 = fold_witness_into_q(&suffix1, &g, prefix_size, suffix_size);
        let p1: Vec<Fr> = EqPolynomial::new(r2_prefix)
            .evaluations()
            .into_iter()
            .map(|e| b * e)
            .collect();
        let q1 = fold_witness_into_q(&suffix2, &g, prefix_size, suffix_size);

        let g_clone = g.clone();
        let suffix1_clone = suffix1.clone();
        let suffix2_clone = suffix2.clone();

        let mut evaluator = PrefixSuffixEvaluator::new(
            vec![(p0, q0), (p1, q1)],
            Box::new(move |transition: PrefixSuffixTransition<Fr>| {
                let eq_table = transition.combined_suffix(&[&suffix1_clone, &suffix2_clone]);

                let eq_prefix = EqPolynomial::new(transition.challenges).evaluations();
                let mut g_suffix = vec![Fr::zero(); suffix_size];
                for s in 0..suffix_size {
                    for (p_idx, &eq_val) in eq_prefix.iter().enumerate() {
                        g_suffix[s] += eq_val * g_clone[p_idx * suffix_size + s];
                    }
                }

                Box::new(SimplePhase2 {
                    eq_table,
                    g_table: g_suffix,
                    claim: Fr::zero(),
                }) as Box<dyn SumcheckCompute<Fr>>
            }),
        );

        let claim = SumcheckClaim {
            num_vars: total_vars,
            degree: 2,
            claimed_sum,
        };

        let mut pt = Blake2bTranscript::new(b"two_pair_eq");
        let proof = SumcheckProver::prove(&claim, &mut evaluator, &mut pt);

        let mut vt = Blake2bTranscript::new(b"two_pair_eq");
        let (final_eval, challenges) =
            SumcheckVerifier::verify(&claim, &proof, &mut vt).unwrap();

        // Oracle check
        let expected = (a * eval_mle(&eq1, &challenges) + b * eval_mle(&eq2, &challenges))
            * eval_mle(&g, &challenges);
        assert_eq!(final_eval, expected, "oracle check failed");
    }

    #[test]
    fn combined_suffix_matches_manual() {
        let suffix_a = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(4)];
        let suffix_b = vec![
            Fr::from_u64(10),
            Fr::from_u64(20),
            Fr::from_u64(30),
            Fr::from_u64(40),
        ];

        let transition = PrefixSuffixTransition {
            challenges: vec![],
            prefix_evals: vec![Fr::from_u64(3), Fr::from_u64(7)],
        };

        let combined = transition.combined_suffix(&[&suffix_a, &suffix_b]);
        // 3*1 + 7*10 = 73, 3*2 + 7*20 = 146, 3*3 + 7*30 = 219, 3*4 + 7*40 = 292
        assert_eq!(combined[0], Fr::from_u64(73));
        assert_eq!(combined[1], Fr::from_u64(146));
        assert_eq!(combined[2], Fr::from_u64(219));
        assert_eq!(combined[3], Fr::from_u64(292));
    }

    #[test]
    fn no_witness_identity_sum() {
        // Σ_x eq(r, x) = 1, trivial case where g = 1 everywhere.
        // Q[j] = Σ_{s} eq(r_suffix, s) · 1 = 1 for all j.
        let mut rng = ChaCha20Rng::seed_from_u64(400);
        let total_vars = 6;
        let prefix_vars = 3;
        let suffix_size = 1usize << (total_vars - prefix_vars);

        let r: Vec<Fr> = (0..total_vars).map(|_| Fr::random(&mut rng)).collect();
        let r_prefix = r[..prefix_vars].to_vec();
        let r_suffix = r[prefix_vars..].to_vec();

        let p = EqPolynomial::new(r_prefix).evaluations();
        let q = vec![Fr::one(); p.len()]; // Σ eq(r_suffix, s) = 1

        let suffix_evals = EqPolynomial::new(r_suffix).evaluations();

        let mut evaluator = PrefixSuffixEvaluator::new(
            vec![(p, q)],
            Box::new(move |transition: PrefixSuffixTransition<Fr>| {
                let eq_table = transition.combined_suffix(&[&suffix_evals]);
                let g_ones = vec![Fr::one(); suffix_size];
                Box::new(SimplePhase2 {
                    eq_table,
                    g_table: g_ones,
                    claim: Fr::zero(),
                }) as Box<dyn SumcheckCompute<Fr>>
            }),
        );

        let claim = SumcheckClaim {
            num_vars: total_vars,
            degree: 2,
            claimed_sum: Fr::one(),
        };

        let mut pt = Blake2bTranscript::new(b"identity");
        let proof = SumcheckProver::prove(&claim, &mut evaluator, &mut pt);

        let mut vt = Blake2bTranscript::new(b"identity");
        let (final_eval, challenges) =
            SumcheckVerifier::verify(&claim, &proof, &mut vt).unwrap();

        // eq(r, challenges) · 1 should equal final_eval
        let eq_full = EqPolynomial::new(r).evaluations();
        assert_eq!(final_eval, eval_mle(&eq_full, &challenges));
    }
}
