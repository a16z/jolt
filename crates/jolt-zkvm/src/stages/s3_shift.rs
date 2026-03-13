//! Stage 3a: Shift sumcheck.
//!
//! Proves the PC-shift invariant via EqPlusOne prefix-suffix decomposition:
//!
//! $$\sum_j \text{eq+1}(r_{\text{outer}}, j) \cdot \bigl(\gamma^0 \cdot p_0(j) + \gamma^1 \cdot p_1(j) + \gamma^2 \cdot p_2(j) + \gamma^3 \cdot p_3(j)\bigr) + \gamma^4 \cdot \text{eq+1}(r_{\text{product}}, j) \cdot (1 - p_4(j)) = \text{claim}$$
//!
//! Phase 1 operates on √T-sized pair buffers (4 pairs from the rank-2
//! decomposition of two `eq+1` instances). Phase 2 materializes √T-sized
//! suffix-domain tables. Uses **HighToLow** binding (MSB first).
//!
//! The 5 shifted witness polynomials are:
//!
//! | Index | Polynomial | Meaning |
//! |-------|-----------|---------|
//! | 0 | UnexpandedPC | Normalized instruction address at cycle j+1 |
//! | 1 | PC | Bytecode offset at cycle j+1 |
//! | 2 | IsVirtual | VirtualInstruction flag at cycle j+1 |
//! | 3 | IsFirstInSequence | Sequence start flag at cycle j+1 |
//! | 4 | IsNoop | Noop flag at cycle j+1 |

use jolt_field::Field;
use jolt_ir::ClaimDefinition;
use jolt_openings::ProverClaim;
use jolt_poly::{EqPlusOnePrefixSuffix, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::claim::SumcheckClaim;
use jolt_sumcheck::prefix_suffix::Phase2Builder;
use jolt_sumcheck::prover::SumcheckCompute;
use jolt_sumcheck::{PrefixSuffixEvaluator, PrefixSuffixTransition};
use jolt_transcript::Transcript;

use crate::stage::{ProverStage, StageBatch};

/// Number of shifted witness polynomials.
pub const NUM_SHIFT_POLYS: usize = 5;

/// Shift sumcheck prover stage.
///
/// Uses EqPlusOne prefix-suffix decomposition to avoid materializing full
/// N-sized eq+1 tables. Two eq+1 instances (`r_outer`, `r_product`) each
/// decompose into 2 rank-1 terms, giving 4 pairs for Phase 1.
pub struct ShiftSumcheckStage<F: Field> {
    /// Full-size evaluation tables, preserved for [`extract_claims`].
    /// Order: [UnexpandedPC, PC, IsVirtual, IsFirstInSequence, IsNoop].
    shift_polys: Option<[Vec<F>; NUM_SHIFT_POLYS]>,
    /// Challenge point from outer Spartan sumcheck (big-endian).
    r_outer: Vec<F>,
    /// Challenge point from product virtual sumcheck (big-endian).
    r_product: Vec<F>,
    /// Batching coefficients [γ^0, ..., γ^4].
    gammas: [F; NUM_SHIFT_POLYS],
    /// Expected sum value (reconstructed from prior opening claims).
    claimed_sum: F,
    num_vars: usize,
}

impl<F: Field> ShiftSumcheckStage<F> {
    /// Creates a new shift sumcheck stage.
    ///
    /// # Arguments
    ///
    /// * `shift_polys` — 5 shifted polynomial evaluation tables, each of
    ///   length `2^num_vars`. Order: [UnexpandedPC, PC, IsVirtual,
    ///   IsFirstInSequence, IsNoop].
    /// * `r_outer` — Challenge point from outer Spartan sumcheck (big-endian).
    /// * `r_product` — Challenge point from product virtual sumcheck (big-endian).
    /// * `gammas` — Batching coefficients `[γ^0, ..., γ^4]`.
    /// * `claimed_sum` — Expected sum value.
    pub fn new(
        shift_polys: [Vec<F>; NUM_SHIFT_POLYS],
        r_outer: Vec<F>,
        r_product: Vec<F>,
        gammas: [F; NUM_SHIFT_POLYS],
        claimed_sum: F,
    ) -> Self {
        let num_vars = r_outer.len();
        assert_eq!(r_product.len(), num_vars);
        let expected_len = 1usize << num_vars;
        for (i, poly) in shift_polys.iter().enumerate() {
            assert_eq!(
                poly.len(),
                expected_len,
                "shift_polys[{i}].len() = {} != {expected_len}",
                poly.len(),
            );
        }

        Self {
            shift_polys: Some(shift_polys),
            r_outer,
            r_product,
            gammas,
            claimed_sum,
            num_vars,
        }
    }
}

impl<F: Field, T: Transcript> ProverStage<F, T> for ShiftSumcheckStage<F> {
    fn name(&self) -> &'static str {
        "S3_shift"
    }

    fn build(&mut self, _prior_claims: &[ProverClaim<F>], _transcript: &mut T) -> StageBatch<F> {
        let polys = self
            .shift_polys
            .as_ref()
            .expect("build() called after extract_claims()");

        let mid = self.num_vars / 2;
        let hi_size = 1usize << mid;
        let lo_size = 1usize << (self.num_vars - mid);
        let n = 1usize << self.num_vars;
        let g = self.gammas;

        // Pre-compute combined witness tables (N entries each).
        // v_witness: outer term's polynomial linear combination.
        // noop_witness: product term's (1 − IsNoop) contribution.
        let mut v_witness = Vec::with_capacity(n);
        let mut noop_witness = Vec::with_capacity(n);
        for j in 0..n {
            v_witness.push(
                g[0] * polys[0][j] + g[1] * polys[1][j] + g[2] * polys[2][j] + g[3] * polys[3][j],
            );
            noop_witness.push(g[4] * (F::one() - polys[4][j]));
        }

        // Decompose eq+1(r_outer) and eq+1(r_product) into prefix-suffix pairs.
        //
        // EqPlusOnePrefixSuffix convention (named by variable position):
        //   prefix_k = tables over y_lo domain (low bits)
        //   suffix_k = tables over y_hi domain (high bits)
        //
        // PrefixSuffixEvaluator (HighToLow, named by binding order):
        //   Phase 1 P tables = y_hi domain = EqPlusOne suffix tables
        //   Phase 2 tables   = y_lo domain = EqPlusOne prefix tables
        let ps_outer = EqPlusOnePrefixSuffix::new(&self.r_outer);
        let ps_product = EqPlusOnePrefixSuffix::new(&self.r_product);

        // Fold witness into Q tables over y_hi domain:
        // Q_k[hi] = Σ_{lo} prefix_k[lo] · witness[hi * lo_size + lo]
        let fold_hi = |prefix_table: &[F], witness: &[F]| -> Vec<F> {
            let mut q = vec![F::zero(); hi_size];
            for (hi, q_hi) in q.iter_mut().enumerate() {
                let base = hi * lo_size;
                for (lo, &pv) in prefix_table.iter().enumerate() {
                    *q_hi += pv * witness[base + lo];
                }
            }
            q
        };

        let pairs = vec![
            (
                ps_outer.suffix_0.clone(),
                fold_hi(&ps_outer.prefix_0, &v_witness),
            ),
            (
                ps_outer.suffix_1.clone(),
                fold_hi(&ps_outer.prefix_1, &v_witness),
            ),
            (
                ps_product.suffix_0.clone(),
                fold_hi(&ps_product.prefix_0, &noop_witness),
            ),
            (
                ps_product.suffix_1.clone(),
                fold_hi(&ps_product.prefix_1, &noop_witness),
            ),
        ];

        // Capture EqPlusOne prefix tables for Phase 2 materialization.
        let prefix_0_outer = ps_outer.prefix_0;
        let prefix_1_outer = ps_outer.prefix_1;
        let prefix_0_product = ps_product.prefix_0;
        let prefix_1_product = ps_product.prefix_1;

        let phase2_builder: Phase2Builder<F> =
            Box::new(move |transition: PrefixSuffixTransition<F>| {
                let pe = &transition.prefix_evals;
                debug_assert_eq!(pe.len(), 4);

                // Combined eq+1 tables over y_lo domain, weighted by bound suffix scalars.
                // eq_outer_lo[lo] = pe[0]·prefix_0_outer[lo] + pe[1]·prefix_1_outer[lo]
                let mut eq_outer_lo = vec![F::zero(); lo_size];
                let mut eq_product_lo = vec![F::zero(); lo_size];
                for lo in 0..lo_size {
                    eq_outer_lo[lo] = pe[0] * prefix_0_outer[lo] + pe[1] * prefix_1_outer[lo];
                    eq_product_lo[lo] = pe[2] * prefix_0_product[lo] + pe[3] * prefix_1_product[lo];
                }

                // Fold full witness tables over y_hi using eq(hi_challenges, ·).
                let eq_hi = EqPolynomial::new(transition.challenges).evaluations();
                let mut v_lo = vec![F::zero(); lo_size];
                let mut noop_lo = vec![F::zero(); lo_size];
                for lo in 0..lo_size {
                    for (hi, &eq_val) in eq_hi.iter().enumerate() {
                        v_lo[lo] += eq_val * v_witness[hi * lo_size + lo];
                        noop_lo[lo] += eq_val * noop_witness[hi * lo_size + lo];
                    }
                }

                Box::new(ShiftPhase2 {
                    eq_outer: Polynomial::new(eq_outer_lo),
                    eq_product: Polynomial::new(eq_product_lo),
                    v_witness: Polynomial::new(v_lo),
                    noop_witness: Polynomial::new(noop_lo),
                    claim: F::zero(), // Overwritten by set_claim before first round.
                }) as Box<dyn SumcheckCompute<F>>
            });

        let evaluator = PrefixSuffixEvaluator::new(pairs, phase2_builder);

        StageBatch {
            claims: vec![SumcheckClaim {
                num_vars: self.num_vars,
                degree: 2,
                claimed_sum: self.claimed_sum,
            }],
            witnesses: vec![Box::new(evaluator)],
        }
    }

    fn extract_claims(&mut self, challenges: &[F], _final_eval: F) -> Vec<ProverClaim<F>> {
        let polys = self
            .shift_polys
            .take()
            .expect("extract_claims() called twice");

        // HighToLow binding: challenge[0] bound MSB = point[0].
        // No reversal needed (unlike LowToHigh stages).
        let eval_point = challenges.to_vec();

        polys
            .into_iter()
            .map(|evals| {
                let poly = Polynomial::new(evals.clone());
                let eval = poly.evaluate(&eval_point);
                ProverClaim {
                    evaluations: evals,
                    point: eval_point.clone(),
                    eval,
                }
            })
            .collect()
    }

    fn claim_definitions(&self) -> Vec<ClaimDefinition> {
        // Will be wired when IR shift claim definitions are added.
        vec![]
    }
}

/// Phase 2 witness for the shift sumcheck (y_lo / suffix domain).
///
/// Computes round polynomials for the degree-2 identity:
/// ```text
/// s(X) = Σ_lo [ eq_outer(X, lo) · v_witness(X, lo) + eq_product(X, lo) · noop_witness(X, lo) ]
/// ```
///
/// Uses HighToLow binding (`Polynomial::bind`) to match Phase 1.
struct ShiftPhase2<F: Field> {
    eq_outer: Polynomial<F>,
    eq_product: Polynomial<F>,
    v_witness: Polynomial<F>,
    noop_witness: Polynomial<F>,
    claim: F,
}

impl<F: Field> SumcheckCompute<F> for ShiftPhase2<F> {
    fn set_claim(&mut self, claim: F) {
        self.claim = claim;
    }

    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let eo = self.eq_outer.evaluations();
        let ep = self.eq_product.evaluations();
        let vw = self.v_witness.evaluations();
        let nw = self.noop_witness.evaluations();
        let half = eo.len() / 2;

        let mut eval_0 = F::zero();
        let mut eval_2 = F::zero();

        // HighToLow: pair [j] (leading bit 0) with [j + half] (leading bit 1).
        for j in 0..half {
            let eo_lo = eo[j];
            let eo_hi = eo[j + half];
            let ep_lo = ep[j];
            let ep_hi = ep[j + half];
            let vw_lo = vw[j];
            let vw_hi = vw[j + half];
            let nw_lo = nw[j];
            let nw_hi = nw[j + half];

            eval_0 += eo_lo * vw_lo + ep_lo * nw_lo;

            // Extrapolate at t=2: p(2) = 2·p(1) − p(0).
            let eo_2 = eo_hi + eo_hi - eo_lo;
            let ep_2 = ep_hi + ep_hi - ep_lo;
            let vw_2 = vw_hi + vw_hi - vw_lo;
            let nw_2 = nw_hi + nw_hi - nw_lo;
            eval_2 += eo_2 * vw_2 + ep_2 * nw_2;
        }

        // Degree 2: P(1) = claim − P(0), then interpolate on {0, 1, 2}.
        UnivariatePoly::from_evals_and_hint(self.claim, &[eval_0, eval_2])
    }

    fn bind(&mut self, challenge: F) {
        self.eq_outer.bind(challenge);
        self.eq_product.bind(challenge);
        self.v_witness.bind(challenge);
        self.noop_witness.bind(challenge);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_poly::EqPlusOnePolynomial;
    use jolt_sumcheck::{BatchedSumcheckProver, BatchedSumcheckVerifier};
    use jolt_transcript::{Blake2bTranscript, Transcript};
    use num_traits::One;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// Brute-force computation of the shift sumcheck claimed sum.
    fn brute_force_shift_sum(
        polys: &[Vec<Fr>; NUM_SHIFT_POLYS],
        r_outer: &[Fr],
        r_product: &[Fr],
        gammas: &[Fr; NUM_SHIFT_POLYS],
    ) -> Fr {
        let n = polys[0].len();
        let (_, epo_outer) = EqPlusOnePolynomial::evals(r_outer, None);
        let (_, epo_product) = EqPlusOnePolynomial::evals(r_product, None);
        let one = Fr::one();

        (0..n)
            .map(|j| {
                let v = gammas[0] * polys[0][j]
                    + gammas[1] * polys[1][j]
                    + gammas[2] * polys[2][j]
                    + gammas[3] * polys[3][j];
                let noop = gammas[4] * (one - polys[4][j]);
                epo_outer[j] * v + epo_product[j] * noop
            })
            .sum()
    }

    fn run_shift_test(num_vars: usize, seed: u64) {
        let n = 1usize << num_vars;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        let r_outer: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let r_product: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
        let gamma = Fr::random(&mut rng);
        let gammas: [Fr; NUM_SHIFT_POLYS] = {
            let mut g = Fr::one();
            core::array::from_fn(|_| {
                let v = g;
                g *= gamma;
                v
            })
        };

        let shift_polys: [Vec<Fr>; NUM_SHIFT_POLYS] =
            core::array::from_fn(|_| (0..n).map(|_| Fr::random(&mut rng)).collect());

        let claimed_sum = brute_force_shift_sum(&shift_polys, &r_outer, &r_product, &gammas);

        let polys_copy = shift_polys.clone();
        let mut stage = ShiftSumcheckStage::new(
            shift_polys,
            r_outer.clone(),
            r_product.clone(),
            gammas,
            claimed_sum,
        );

        let mut pt = Blake2bTranscript::new(b"shift");
        let mut batch = stage.build(&[], &mut pt);

        assert_eq!(batch.claims.len(), 1);
        assert_eq!(batch.claims[0].degree, 2);
        assert_eq!(batch.claims[0].num_vars, num_vars);
        assert_eq!(batch.claims[0].claimed_sum, claimed_sum);

        let claims_snapshot = batch.claims.clone();
        let proof = BatchedSumcheckProver::prove(&batch.claims, &mut batch.witnesses, &mut pt);

        let mut vt = Blake2bTranscript::new(b"shift");
        let (final_eval, challenges) =
            BatchedSumcheckVerifier::verify(&claims_snapshot, &proof, &mut vt)
                .expect("verification should succeed");

        // Oracle check: final_eval = eq+1(r_outer, r)·v(r) + eq+1(r_product, r)·noop(r).
        let eq_outer_r = EqPlusOnePolynomial::new(r_outer).evaluate(&challenges);
        let eq_product_r = EqPlusOnePolynomial::new(r_product).evaluate(&challenges);
        let v_at_r: Fr = (0..4)
            .map(|i| gammas[i] * Polynomial::new(polys_copy[i].clone()).evaluate(&challenges))
            .sum();
        let noop_at_r =
            gammas[4] * (Fr::one() - Polynomial::new(polys_copy[4].clone()).evaluate(&challenges));
        let expected = eq_outer_r * v_at_r + eq_product_r * noop_at_r;
        assert_eq!(final_eval, expected, "oracle check failed");

        // Extract claims and verify evaluations.
        let claims = <ShiftSumcheckStage<Fr> as ProverStage<Fr, Blake2bTranscript>>::extract_claims(
            &mut stage,
            &challenges,
            final_eval,
        );
        assert_eq!(claims.len(), NUM_SHIFT_POLYS);
        for (i, claim) in claims.iter().enumerate() {
            let expected_eval = Polynomial::new(polys_copy[i].clone()).evaluate(&challenges);
            assert_eq!(claim.eval, expected_eval, "claim {i} eval mismatch");
            assert_eq!(claim.point, challenges, "claim {i} point mismatch");
        }
    }

    #[test]
    fn shift_sumcheck_even_vars() {
        run_shift_test(6, 42);
    }

    #[test]
    fn shift_sumcheck_odd_vars() {
        run_shift_test(7, 123);
    }

    #[test]
    fn shift_sumcheck_small() {
        run_shift_test(4, 777);
    }
}
